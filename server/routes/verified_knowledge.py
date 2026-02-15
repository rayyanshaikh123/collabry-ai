"""
Verified Knowledge API Routes

Admin endpoints for managing verified knowledge base.
"""

import logging
import os
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from pydantic import BaseModel, HttpUrl

from server.deps import get_current_user
from tools.core.verified_knowledge import VerifiedKnowledgeIngestionService, IngestionResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/verified-knowledge", tags=["verified-knowledge"])


# Request/Response Models
class IngestURLRequest(BaseModel):
    url: HttpUrl
    authority_level: Optional[str] = None


class IngestTextRequest(BaseModel):
    text: str
    title: str
    authority_level: str = 'low'
    subject: Optional[str] = None
    topic: Optional[str] = None


class IngestionResponse(BaseModel):
    success: bool
    documents_added: int = 0
    authority_level: str = 'low'
    errors: list[str] = []
    metadata: dict = {}


@router.post("/ingest/url", response_model=IngestionResponse)
async def ingest_url(
    request: IngestURLRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Ingest content from URL into verified knowledge base.
    
    **Requires authentication** (TODO: Add admin-only restriction)
    
    Args:
        request: URL and optional authority level
        current_user: Current user
        
    Returns:
        Ingestion result
    """
    logger.info(f"User {current_user} ingesting URL: {request.url}")
    
    service = VerifiedKnowledgeIngestionService()
    
    try:
        result = await service.ingest_from_url(
            str(request.url),
            user_authority=request.authority_level
        )
        
        return IngestionResponse(
            success=result.success,
            documents_added=result.documents_added,
            authority_level=result.authority_level,
            errors=result.errors,
            metadata=result.metadata
        )
    
    except Exception as e:
        logger.error(f"Failed to ingest URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


@router.post("/ingest/text", response_model=IngestionResponse)
async def ingest_text(
    request: IngestTextRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Ingest plain text into verified knowledge base.
    
    **Requires authentication** (TODO: Add admin-only restriction)
    
    Args:
        request: Text content and metadata
        current_user: Current user
        
    Returns:
        Ingestion result
    """
    logger.info(f"User {current_user} ingesting text: {request.title}")
    
    # Validate authority level
    if request.authority_level not in ['high', 'medium', 'low']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="authority_level must be 'high', 'medium', or 'low'"
        )
    
    service = VerifiedKnowledgeIngestionService()
    
    try:
        extra_metadata = {}
        if request.subject:
            extra_metadata['subject'] = request.subject
        if request.topic:
            extra_metadata['topic'] = request.topic
        
        result = service.ingest_from_text(
            request.text,
            request.title,
            authority_level=request.authority_level,
            **extra_metadata
        )
        
        return IngestionResponse(
            success=result.success,
            documents_added=result.documents_added,
            authority_level=result.authority_level,
            errors=result.errors,
            metadata=result.metadata
        )
    
    except Exception as e:
        logger.error(f"Failed to ingest text: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


@router.post("/ingest/pdf", response_model=IngestionResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    authority_level: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Ingest PDF file into verified knowledge base.
    
    **Requires authentication** (TODO: Add admin-only restriction)
    
    Args:
        file: PDF file upload
        authority_level: Optional authority level override
        current_user: Current user
        
    Returns:
        Ingestion result
    """
    logger.info(f"User {current_user} ingesting PDF: {file.filename}")
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    # Validate authority level if provided
    if authority_level and authority_level not in ['high', 'medium', 'low']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="authority_level must be 'high', 'medium', or 'low'"
        )
    
    # Save temporary file
    temp_path = f"/tmp/{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        service = VerifiedKnowledgeIngestionService()
        result = await service.ingest_from_pdf(
            temp_path,
            user_authority=authority_level
        )
        
        return IngestionResponse(
            success=result.success,
            documents_added=result.documents_added,
            authority_level=result.authority_level,
            errors=result.errors,
            metadata=result.metadata
        )
    
    except Exception as e:
        logger.error(f"Failed to ingest PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )
    
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/stats")
async def get_stats(current_user: dict = Depends(get_current_user)):
    """
    Get verified knowledge base statistics.
    
    Returns:
        Coverage statistics
    """
    from tools.core.verified_knowledge import get_verified_knowledge_store
    
    vkb = get_verified_knowledge_store()
    stats = vkb.get_coverage_stats()
    
    return stats
