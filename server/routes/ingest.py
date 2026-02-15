"""
Document ingestion endpoint for RAG pipeline.

Handles:
- Document upload and processing
- Background embedding generation
- User-isolated document storage
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from server.deps import get_current_user, get_user_id
from server.schemas import UploadRequest, UploadResponse, ErrorResponse
from core.rag_retriever import RAGRetriever
from langchain_core.documents import Document
from config import CONFIG
import logging
from uuid import uuid4
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["ingest"])

# In-memory task tracking (use Redis/DB in production)
ingestion_tasks = {}


def ingest_document_background(
    task_id: str,
    user_id: str,
    content: str,
    filename: str,
    metadata: dict
):
    """
    Background task for document ingestion and embedding.
    
    Args:
        task_id: Unique task identifier
        user_id: User who uploaded the document
        content: Document text content
        filename: Original filename
        metadata: Additional metadata
    """
    try:
        logger.info(f"Starting background ingestion: task={task_id}, user={user_id}")
        
        # Task entry already created in upload_document, just update status
        if task_id in ingestion_tasks:
            ingestion_tasks[task_id]["status"] = "processing"
        else:
            # Fallback: create entry if somehow missing
            ingestion_tasks[task_id] = {
                "status": "processing",
                "user_id": user_id,
                "filename": filename,
                "started_at": datetime.utcnow()
            }
        
        # Ingest into the same vector store that tools query (rag/vectorstore).
        # This avoids writing into a separate FAISS singleton that tools don't see.
        from rag.vectorstore import add_documents
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "source": filename,
                "user_id": user_id,
                "uploaded_at": datetime.utcnow().isoformat(),
                **metadata
            }
        )

        # IMPORTANT: Tools (flashcards, quiz, etc.) use the actual MongoDB notebook_id
        # for retrieval. We must prioritize storing documents under this ID.
        notebook_id = (metadata or {}).get("notebook_id") or (metadata or {}).get("session_id") or "default"

        # Chunk before writing, so retrieval works well and chunk metadata is preserved.
        chunk_size = int(CONFIG.get("chunk_size", 1000))
        chunk_overlap = int(CONFIG.get("chunk_overlap", 200))
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents([doc])

        # Record processed chunk count for status visibility
        if task_id in ingestion_tasks:
            ingestion_tasks[task_id]["chunks_processed"] = len(chunks)

        # Ensure metadata is present on every chunk (splitter should preserve it).
        for c in chunks:
            c.metadata.setdefault("user_id", user_id)
            c.metadata.setdefault("notebook_id", notebook_id)

        add_documents(documents=chunks, user_id=user_id, notebook_id=str(notebook_id))
        
        # Update task status
        ingestion_tasks[task_id]["status"] = "completed"
        ingestion_tasks[task_id]["completed_at"] = datetime.utcnow()
        
        logger.info(f"Background ingestion completed: task={task_id}")
        
    except Exception as e:
        logger.exception(f"Background ingestion failed: task={task_id}, error={e}")
        # Preserve user_id and other fields when marking as failed
        if task_id in ingestion_tasks:
            ingestion_tasks[task_id]["status"] = "failed"
            ingestion_tasks[task_id]["error"] = str(e)
            ingestion_tasks[task_id]["failed_at"] = datetime.utcnow()
        else:
            # Fallback if task entry is missing
            ingestion_tasks[task_id] = {
                "status": "failed",
                "user_id": user_id,
                "filename": filename,
                "error": str(e),
                "failed_at": datetime.utcnow()
            }


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Upload document for RAG",
    description="Upload document content for embedding and retrieval (processed in background)"
)
async def upload_document(
    request: UploadRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_user_id)
) -> UploadResponse:
    """
    Upload document for user-isolated RAG retrieval.
    
    - Extracts user_id from JWT token
    - Processes document in background (chunking + embedding)
    - Returns task_id for status tracking
    """
    try:
        # Generate unique task ID
        task_id = str(uuid4())
        
        logger.info(f"Document upload initiated: user={user_id}, file={request.filename}, task={task_id}")
        
        # Create task entry immediately (before background task runs) to avoid race condition
        ingestion_tasks[task_id] = {
            "status": "processing",
            "user_id": user_id,
            "filename": request.filename,
            "started_at": datetime.utcnow()
        }
        
        logger.info(f"✅ Task created: task_id={task_id}, user_id='{user_id}' (type={type(user_id).__name__})")
        
        # Add background task for ingestion
        background_tasks.add_task(
            ingest_document_background,
            task_id=task_id,
            user_id=user_id,
            content=request.content,
            filename=request.filename,
            metadata=request.metadata
        )
        
        return UploadResponse(
            task_id=task_id,
            status="processing",
            filename=request.filename,
            user_id=user_id,
            message=f"Document upload initiated. Track progress with task_id: {task_id}"
        )
        
    except Exception as e:
        logger.exception(f"Upload endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate document upload: {str(e)}"
        )


@router.get(
    "/upload/status/{task_id}",
    summary="Check upload status",
    description="Get status of background document ingestion task"
)
async def get_upload_status(
    task_id: str,
    user_id: str = Depends(get_user_id)
):
    """
    Check status of document ingestion task.
    
    Args:
        task_id: Task identifier from upload response
        user_id: Current user (from JWT)
        
    Returns:
        Task status (processing, completed, failed)
    """
    if task_id not in ingestion_tasks:
        # Task not found - likely server was restarted or task expired
        # Return a default "completed" status to avoid blocking the frontend
        logger.warning(f"Task {task_id} not found in memory (server may have restarted)")
        return {
            "task_id": task_id,
            "status": "unknown",
            "message": "Task not found - server may have restarted. Please check if document was ingested.",
            "filename": None,
            "started_at": None,
            "completed_at": None,
            "error": None
        }
    
    task = ingestion_tasks[task_id]
    
    # Verify task belongs to current user (normalize to string for comparison)
    task_user_id_raw = task.get("user_id")
    task_user_id = str(task_user_id_raw) if task_user_id_raw else ""
    current_user_id = str(user_id)
    
    logger.info(
        f"🔍 Status check: task_id={task_id}, "
        f"stored_user='{task_user_id}' (raw_type={type(task_user_id_raw).__name__}), "
        f"current_user='{current_user_id}' (type={type(user_id).__name__}), "
        f"match={task_user_id == current_user_id}"
    )
    
    if task_user_id != current_user_id:
        logger.error(
            f"❌ User ID mismatch for task {task_id}:\n"
            f"  Stored:  '{task_user_id}' (len={len(task_user_id)}, bytes={task_user_id.encode('utf-8')})\n"
            f"  Current: '{current_user_id}' (len={len(current_user_id)}, bytes={current_user_id.encode('utf-8')})\n"
            f"  Task data: {task}"
        )
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: task user '{task_user_id}' != current user '{current_user_id}'"
        )
    
    return {
        "task_id": task_id,
        "status": task.get("status"),
        "filename": task.get("filename"),
        "started_at": task.get("started_at"),
        "completed_at": task.get("completed_at"),
        "chunks_processed": task.get("chunks_processed"),
        "error": task.get("error")
    }


@router.delete(
    "/documents/source/{source_id}",
    summary="Delete source documents",
    description="Delete all documents associated with a source from FAISS index"
)
async def delete_source_documents(
    source_id: str,
    user_id: str = Depends(get_user_id)
):
    """
    Delete all documents for a specific source.
    
    - Used when a source is removed from a notebook
    - Removes all chunks associated with source_id
    """
    try:
        logger.info(f"Delete request for source={source_id}, user={user_id}")
        
        rag = RAGRetriever(CONFIG, user_id=user_id)
        deleted_count = rag.delete_documents_by_metadata(
            user_id=user_id,
            source_id=source_id,
            save_index=True
        )
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "source_id": source_id,
            "message": f"Deleted {deleted_count} document chunks"
        }
        
    except Exception as e:
        logger.exception(f"Failed to delete source documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete source documents: {str(e)}"
        )


@router.delete(
    "/documents/session/{session_id}",
    summary="Delete session documents",
    description="Delete all documents associated with a session/notebook from FAISS index"
)
async def delete_session_documents(
    session_id: str,
    user_id: str = Depends(get_user_id)
):
    """
    Delete all documents for a specific session (notebook).
    
    - Used when a notebook is deleted
    - Removes all chunks associated with session_id
    """
    try:
        logger.info(f"Delete request for session={session_id}, user={user_id}")
        
        rag = RAGRetriever(CONFIG, user_id=user_id)
        deleted_count = rag.delete_documents_by_metadata(
            user_id=user_id,
            session_id=session_id,
            save_index=True
        )
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "session_id": session_id,
            "message": f"Deleted {deleted_count} document chunks"
        }
        
    except Exception as e:
        logger.exception(f"Failed to delete session documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete session documents: {str(e)}"
        )
