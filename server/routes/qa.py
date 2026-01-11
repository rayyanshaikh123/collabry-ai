"""
Question Answering endpoint using RAG.

Handles:
- RAG-based question answering over user documents
- File upload for context
- Streaming responses
- Configurable retrieval parameters
- Source document tracking
- Quiz question generation from content
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from server.deps import get_current_user
from server.schemas import QARequest, QAResponse, QAGenerateRequest, QAGenerateResponse, QuizQuestion, ErrorResponse
from core.agent import create_agent
from core.rag_retriever import RAGRetriever
from config import CONFIG
import logging
from datetime import datetime
from uuid import uuid4
from typing import Optional, List
import PyPDF2
import io
import json
import ast
import re

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["qa"])

# File size limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.md', '.doc', '.docx'}


@router.post(
    "/qa",
    response_model=QAResponse,
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Question answering with RAG",
    description="Answer questions using user documents via Retrieval-Augmented Generation"
)
async def question_answering(
    request: QARequest,
    user_id: str = Depends(get_current_user)
) -> QAResponse:
    """
    Answer questions using RAG over user documents.
    
    - Extracts user_id from JWT token
    - Retrieves relevant documents from user's RAG index
    - Generates answer using retrieved context
    - Returns answer with source documents
    """
    try:
        logger.info(f"QA request from user={user_id}, question={request.question[:50]}...")
        
        sources = []
        context_text = ""
        
        # Retrieve relevant documents if RAG enabled
        if request.use_rag:
            rag = RAGRetriever(CONFIG, user_id=user_id)
            
            # Retrieve documents
            docs = rag.get_relevant_documents(
                request.question,
                user_id=user_id
            )
            
            # Build context from retrieved documents
            context_parts = []
            for i, doc in enumerate(docs[:request.top_k]):
                context_parts.append(f"[Document {i+1}]:\n{doc.page_content}")
                sources.append({
                    "source": doc.metadata.get("source", "unknown"),
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                })
            
            context_text = "\n\n".join(context_parts)
            logger.info(f"Retrieved {len(docs)} documents for QA")
        
        # Use provided context if not using RAG
        elif request.context:
            context_text = request.context
        
        # Create agent for QA
        agent, _, _, _ = create_agent(
            user_id=user_id,
            session_id=str(uuid4()),  # Temporary session
            config=CONFIG
        )
        
        # Build QA prompt
        if context_text:
            prompt = f"""Answer the following question using the provided context.

Context:
{context_text}

Question: {request.question}

Answer (be specific and cite sources if available):"""
        else:
            prompt = f"""Answer the following question to the best of your knowledge.

Question: {request.question}

Answer:"""
        
        # Collect response
        response_chunks = []
        
        def collect_chunk(chunk: str):
            response_chunks.append(chunk)
        
        # Execute agent
        agent.handle_user_input_stream(prompt, collect_chunk)
        
        answer = "".join(response_chunks).strip()
        
        logger.info(f"QA answer generated: {len(answer)} chars, sources={len(sources)}")
        
        return QAResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            user_id=user_id,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.exception(f"QA error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {str(e)}"
        )


@router.post(
    "/qa/stream",
    summary="Streaming QA with RAG",
    description="Answer questions with streaming response using RAG",
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def question_answering_stream(
    request: QARequest,
    user_id: str = Depends(get_current_user)
):
    """
    Streaming QA endpoint with RAG support.
    
    Returns:
        StreamingResponse with Server-Sent Events
    """
    try:
        logger.info(f"Streaming QA request from user={user_id}, question={request.question[:50]}...")
        
        sources = []
        context_text = ""
        
        # Retrieve relevant documents if RAG enabled
        if request.use_rag:
            rag = RAGRetriever(CONFIG, user_id=user_id)
            
            # Retrieve documents
            docs = rag.get_relevant_documents(
                request.question,
                user_id=user_id
            )
            
            # Build context from retrieved documents
            context_parts = []
            for i, doc in enumerate(docs[:request.top_k]):
                context_parts.append(f"[Document {i+1}]:\n{doc.page_content}")
                sources.append({
                    "source": doc.metadata.get("source", "unknown"),
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                })
            
            context_text = "\n\n".join(context_parts)
            logger.info(f"Retrieved {len(docs)} documents for streaming QA")
        
        # Use provided context if not using RAG
        elif request.context:
            context_text = request.context
        
        # Create agent for QA
        agent, _, _, _ = create_agent(
            user_id=user_id,
            session_id=str(uuid4()),
            config=CONFIG
        )
        
        # Build QA prompt
        if context_text:
            prompt = f"""Answer the following question using the provided context.

Context:
{context_text}

Question: {request.question}

Answer (be specific and cite sources if available):"""
        else:
            prompt = f"""Answer the following question to the best of your knowledge.

Question: {request.question}

Answer:"""
        
        async def event_generator():
            """Generate SSE events for streaming QA."""
            has_data = False
            
            # Stream sources first if available
            if sources:
                yield f"event: sources\ndata: {len(sources)}\n\n"
            
            # Execute agent with streaming
            chunks_buffer = []
            def collect_chunk(chunk: str):
                chunks_buffer.append(chunk)
            
            agent.handle_user_input_stream(prompt, collect_chunk)
            
            # Stream the answer chunks
            for chunk in chunks_buffer:
                if chunk.strip():
                    has_data = True
                    yield f"data: {chunk}\n\n"
            
            # Send completion event
            if has_data:
                yield f"event: done\ndata: \n\n"
            else:
                yield f"data: No answer generated\n\n"
                yield f"event: done\ndata: \n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        logger.exception(f"Streaming QA error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stream QA response: {str(e)}"
        )


@router.post(
    "/qa/file",
    response_model=QAResponse,
    summary="QA with file upload",
    description="Answer questions about uploaded file content (max 10MB, PDF/TXT/MD)",
    responses={
        401: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def question_answering_with_file(
    question: str = Form(...),
    file: UploadFile = File(...),
    use_rag: bool = Form(False),
    user_id: str = Depends(get_current_user)
) -> QAResponse:
    """
    QA with file upload support - Accepts PDF, TXT, MD files up to 10MB.
    """
    try:
        logger.info(f"File QA request from user={user_id}, file={file.filename}")
        
        # Validate file size
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large. Max: {MAX_FILE_SIZE / 1024 / 1024}MB")
        
        # Extract file extension
        file_ext = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
        
        # Extract text based on file type
        text_content = ""
        if file_ext == '.pdf':
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                text_content = "\n".join([page.extract_text() for page in pdf_reader.pages])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")
        elif file_ext in {'.txt', '.md'}:
            text_content = content.decode('utf-8', errors='ignore')
        
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No text content found in file")
        
        logger.info(f"Extracted {len(text_content)} characters from {file.filename}")
        
        # Get RAG context if enabled
        rag_context = ""
        sources = []
        if use_rag:
            try:
                rag = RAGRetriever(CONFIG, user_id=user_id)
                docs = rag.get_relevant_documents(question, user_id=user_id)
                if docs:
                    rag_context = "\n\n[Additional context from your documents]:\n"
                    for doc in docs[:2]:
                        rag_context += f"\n{doc.page_content[:300]}..."
                        sources.append({"source": doc.metadata.get("source", "unknown"), "content": doc.page_content[:200] + "..."})
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        # Create agent
        agent, _, _, _ = create_agent(user_id=user_id, session_id=str(uuid4()), config=CONFIG)
        
        # Build prompt
        prompt = f"""Answer the following question based on the uploaded file content.

File: {file.filename}

Content:
{text_content[:4000]}

{rag_context}

Question: {question}

Answer (be specific and reference the file content):"""
        
        response_chunks = []
        def collect_chunk(chunk: str):
            response_chunks.append(chunk)
        
        agent.handle_user_input_stream(prompt, collect_chunk)
        answer = "".join(response_chunks).strip()
        
        logger.info(f"File QA answer generated: {len(answer)} chars")
        
        return QAResponse(
            question=question,
            answer=answer,
            sources=sources,
            user_id=user_id,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"File QA error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file QA: {str(e)}")


@router.post(
    "/qa/file/stream",
    summary="Streaming QA with file upload",
    description="Answer questions about uploaded file with streaming response (max 10MB)",
    responses={
        401: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def question_answering_with_file_stream(
    question: str = Form(...),
    file: UploadFile = File(...),
    use_rag: bool = Form(False),
    user_id: str = Depends(get_current_user)
):
    """Streaming QA with file upload."""
    try:
        logger.info(f"Streaming file QA request from user={user_id}, file={file.filename}")
        
        # Validate file
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large. Max: {MAX_FILE_SIZE / 1024 / 1024}MB")
        
        file_ext = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
        
        # Extract text
        text_content = ""
        if file_ext == '.pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text_content = "\n".join([page.extract_text() for page in pdf_reader.pages])
        elif file_ext in {'.txt', '.md'}:
            text_content = content.decode('utf-8', errors='ignore')
        
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No text content found in file")
        
        # Get RAG context
        rag_context = ""
        sources_count = 0
        if use_rag:
            try:
                rag = RAGRetriever(CONFIG, user_id=user_id)
                docs = rag.get_relevant_documents(question, user_id=user_id)
                if docs:
                    rag_context = "\n\n[Additional context]:\n" + "\n".join([d.page_content[:300] for d in docs[:2]])
                    sources_count = len(docs[:2])
            except:
                pass
        
        agent, _, _, _ = create_agent(user_id=user_id, session_id=str(uuid4()), config=CONFIG)
        
        prompt = f"""Answer based on the uploaded file.

File: {file.filename}
Content:
{text_content[:4000]}

{rag_context}

Question: {question}

Answer:"""
        
        async def event_generator():
            yield f"event: file\ndata: {file.filename}\n\n"
            if sources_count > 0:
                yield f"event: sources\ndata: {sources_count}\n\n"
            
            chunks_buffer = []
            def collect_chunk(chunk: str):
                chunks_buffer.append(chunk)
            
            agent.handle_user_input_stream(prompt, collect_chunk)
            
            for chunk in chunks_buffer:
                if chunk.strip():
                    yield f"data: {chunk}\n\n"
            
            yield f"event: done\ndata: \n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Streaming file QA error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Q&A Generation Endpoints (Quiz/Test Generation)
# ============================================================================

@router.post(
    "/qa/generate",
    response_model=QAGenerateResponse,
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Generate quiz questions from content",
    description="Generate quiz/test questions and answers from provided text content"
)
async def generate_qa(
    request: QAGenerateRequest,
    user_id: str = Depends(get_current_user)
) -> QAGenerateResponse:
    """
    Generate quiz questions from text content.
    
    Args:
        request: QAGenerateRequest with text and generation parameters
        user_id: User ID from JWT token
    
    Returns:
        QAGenerateResponse with generated quiz questions
    """
    try:
        logger.info(f"Generating {request.num_questions} questions for user {user_id} (use_rag={request.use_rag})")
        
        # Retrieve relevant documents if RAG enabled
        context_text = request.text
        rag_sources = []
        
        if request.use_rag:
            try:
                rag = RAGRetriever(CONFIG, user_id=user_id)
                
                # Use topic or first 100 chars of text as query
                query = request.topic or request.text[:100]
                
                # Retrieve relevant documents
                docs = rag.get_relevant_documents(query, user_id=user_id)
                
                if docs:
                    # Prepend retrieved context to provided text
                    context_parts = ["[Retrieved Context from Your Documents]\n"]
                    for i, doc in enumerate(docs[:3]):
                        context_parts.append(f"Document {i+1}: {doc.page_content}")
                        rag_sources.append(doc.metadata.get("source", "unknown"))
                    
                    context_text = "\n\n".join(context_parts) + f"\n\n[Provided Content]\n{request.text}"
                    logger.info(f"RAG: Retrieved {len(docs)} documents for quiz generation")
            except Exception as e:
                logger.warning(f"RAG retrieval failed, continuing with provided text only: {e}")
        
        # Create agent for generation (using generic session for quiz generation)
        agent, _, _, _ = create_agent(user_id, session_id="quiz_generation")
        
        # Build generation prompt
        difficulty_instruction = ""
        if request.difficulty and request.difficulty != "mixed":
            difficulty_instruction = f"Make all questions {request.difficulty} difficulty. "
        elif request.difficulty == "mixed":
            difficulty_instruction = "Vary the difficulty from easy to hard. "
            
        options_instruction = ""
        if request.include_options:
            options_instruction = "For each question, provide 4 multiple choice options (A, B, C, D). "
        
        # Truncate very long texts for prompt (keep first 10k chars for better context)
        content_for_prompt = context_text
        if len(context_text) > 10000:
            content_for_prompt = context_text[:10000]
            logger.info(f"Truncated content from {len(context_text)} to {len(content_for_prompt)} chars for prompt")
        
        logger.info(f"Content preview (first 500 chars): {content_for_prompt[:500]}")
        
        prompt = f"""Read this content carefully and generate {request.num_questions} quiz questions based ONLY on the information provided below.

CONTENT:
{content_for_prompt}

INSTRUCTIONS:
- Create questions about specific facts, names, dates, or concepts from the content above
- Each question must be answerable using ONLY the content provided
- Provide 4 different answer options for each question
- One option must be the correct answer from the content
- The other 3 options should be plausible but incorrect

{difficulty_instruction}

CRITICAL: Return ONLY a valid JSON array. Use DOUBLE QUOTES for all strings, not single quotes.
Return this exact format:
[
  {{
    "question": "What specific detail from the content?",
    "answer": "Correct answer from content",
    "options": ["Correct answer", "Wrong option 1", "Wrong option 2", "Wrong option 3"],
    "explanation": "Why this is correct based on the content",
    "difficulty": "medium"
  }}
]

Generate exactly {request.num_questions} questions. Return ONLY valid JSON with double quotes, no extra text, no markdown code blocks."""

        # Generate response using streaming collection
        response_text = ""
        def collect_chunk(chunk: str):
            nonlocal response_text
            response_text += chunk
        
        agent.handle_user_input_stream(prompt, collect_chunk)
        
        logger.info(f"Generated response length: {len(response_text)} chars")
        logger.debug(f"Response preview: {response_text[:500]}...")
        
        # Parse questions from response
        questions = []
        try:
            # Try to extract JSON from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                logger.info(f"Attempting to parse JSON array of {len(json_str)} chars")
                
                try:
                    # First try: parse as valid JSON
                    questions_data = json.loads(json_str)
                except json.JSONDecodeError:
                    # Second try: handle Python dict syntax (single quotes)
                    logger.info("JSON parsing failed, trying Python literal_eval")
                    try:
                        questions_data = ast.literal_eval(json_str)
                    except (ValueError, SyntaxError):
                        # Third try: simple quote replacement (risky but last resort)
                        logger.info("literal_eval failed, trying quote replacement")
                        json_str = json_str.replace("'", '"')
                        questions_data = json.loads(json_str)
                
                for q_data in questions_data:
                    questions.append(QuizQuestion(
                        question=str(q_data.get('question', '')),
                        answer=str(q_data.get('answer', '')),
                        options=q_data.get('options'),
                        explanation=str(q_data.get('explanation', '')) if q_data.get('explanation') else None,
                        difficulty=str(q_data.get('difficulty', 'medium'))
                    ))
                logger.info(f"Successfully parsed {len(questions)} questions from JSON")
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse structured response: {e}, falling back to text parsing")
            # Fallback: parse text format
            lines = response_text.split('\n')
            current_q = {}
            for line in lines:
                line = line.strip()
                if line.startswith('Question:') or line.startswith('Q:') or line.startswith('**Question'):
                    if current_q.get('question') and current_q.get('answer'):
                        questions.append(QuizQuestion(**current_q))
                    current_q = {'question': line.split(':', 1)[1].strip() if ':' in line else line}
                elif line.startswith('Answer:') or line.startswith('A:') or line.startswith('**Answer'):
                    current_q['answer'] = line.split(':', 1)[1].strip() if ':' in line else line
                elif line.startswith('Explanation:') or line.startswith('**Explanation'):
                    current_q['explanation'] = line.split(':', 1)[1].strip() if ':' in line else line
            
            if current_q.get('question') and current_q.get('answer'):
                questions.append(QuizQuestion(**current_q))
            
            logger.info(f"Parsed {len(questions)} questions from text format")
        
        if not questions:
            logger.error(f"No questions generated. Response text: {response_text[:1000]}")
            raise HTTPException(status_code=500, detail="Failed to generate valid questions from AI response")
        
        return QAGenerateResponse(
            questions=questions[:request.num_questions],  # Ensure we don't exceed requested amount
            source_length=len(request.text),
            user_id=user_id,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"QA generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/qa/generate/stream",
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Generate quiz questions with streaming",
    description="Stream quiz question generation as they are created"
)
async def generate_qa_stream(
    request: QAGenerateRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Stream quiz question generation.
    
    Each question is sent as a separate SSE event as it's generated.
    """
    try:
        logger.info(f"Streaming generation of {request.num_questions} questions for user {user_id}")
        
        # Create agent for generation (using generic session for quiz generation)
        agent, _, _, _ = create_agent(user_id, session_id="quiz_generation")
        
        # Build generation prompt
        difficulty_instruction = ""
        if request.difficulty and request.difficulty != "mixed":
            difficulty_instruction = f"Make all questions {request.difficulty} difficulty. "
        elif request.difficulty == "mixed":
            difficulty_instruction = "Vary the difficulty from easy to hard. "
            
        options_instruction = ""
        if request.include_options:
            options_instruction = "For each question, provide 4 multiple choice options (A, B, C, D). "
        
        prompt = f"""Based on the following content, generate {request.num_questions} quiz questions with answers.

{difficulty_instruction}{options_instruction}

For each question, provide:
1. A clear, well-formed question
2. The correct answer
3. {'Four multiple choice options (A, B, C, D) ' if request.include_options else ''}A brief explanation

Format each question clearly separated:

Question 1: [question text]
Answer: [answer]
{f'Options: A) ... B) ... C) ... D) ...' if request.include_options else ''}
Explanation: [explanation]

Question 2: [question text]
...

Content:
{request.text}

Generate exactly {request.num_questions} questions."""

        async def event_generator():
            try:
                # Generate response using streaming collection
                response_text = ""
                def collect_chunk(chunk: str):
                    nonlocal response_text
                    response_text += chunk
                
                agent.handle_user_input_stream(prompt, collect_chunk)
                
                # Parse and stream questions one by one
                questions_sent = 0
                lines = response_text.split('\n')
                current_q = {}
                
                for line in lines:
                    line = line.strip()
                    
                    # Detect question start
                    if (line.startswith('Question') and ':' in line) or (line.startswith('Q') and line[1:3] in ['. ', ': ', '1:', '2:']):
                        if current_q.get('question') and current_q.get('answer'):
                            # Send previous question
                            question_data = {
                                'question': current_q['question'],
                                'answer': current_q['answer'],
                                'options': current_q.get('options'),
                                'explanation': current_q.get('explanation'),
                                'difficulty': current_q.get('difficulty', 'medium')
                            }
                            yield f"data: {json.dumps(question_data)}\n\n"
                            questions_sent += 1
                            current_q = {}
                        
                        # Extract question text
                        if ':' in line:
                            current_q['question'] = line.split(':', 1)[1].strip()
                        else:
                            current_q['question'] = line
                    
                    elif line.startswith('Answer:') or (line.startswith('A:') and len(line) > 3):
                        current_q['answer'] = line.split(':', 1)[1].strip()
                    
                    elif line.startswith('Options:'):
                        options_text = line.split(':', 1)[1].strip()
                        # Parse A) B) C) D) format
                        current_q['options'] = [opt.strip() for opt in options_text.replace('A)', '|A)').replace('B)', '|B)').replace('C)', '|C)').replace('D)', '|D)').split('|') if opt.strip()]
                    
                    elif line.startswith('Explanation:'):
                        current_q['explanation'] = line.split(':', 1)[1].strip()
                
                # Send last question
                if current_q.get('question') and current_q.get('answer'):
                    question_data = {
                        'question': current_q['question'],
                        'answer': current_q['answer'],
                        'options': current_q.get('options'),
                        'explanation': current_q.get('explanation'),
                        'difficulty': current_q.get('difficulty', 'medium')
                    }
                    yield f"data: {json.dumps(question_data)}\n\n"
                    questions_sent += 1
                
                logger.info(f"Sent {questions_sent} questions via stream")
                yield f"event: done\ndata: \n\n"
                
            except Exception as e:
                logger.exception(f"Stream generation error: {e}")
                error_data = {"error": str(e)}
                yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Streaming QA generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/qa/generate/file",
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Generate quiz questions from uploaded file",
    description="Upload a file and generate quiz questions from its content"
)
async def generate_qa_from_file(
    file: UploadFile = File(...),
    num_questions: int = Form(5),
    difficulty: Optional[str] = Form("medium"),
    include_options: bool = Form(False),
    user_id: str = Depends(get_current_user)
):
    """
    Generate quiz questions from uploaded file content.
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_ext = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Read file content
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024}MB")
        
        # Extract text based on file type
        text = ""
        if file_ext == '.pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() for page in pdf_reader.pages)
        else:
            text = content.decode('utf-8', errors='ignore')
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text content found in file")
        
        # Create request and generate
        qa_request = QAGenerateRequest(
            text=text,
            num_questions=num_questions,
            difficulty=difficulty,
            include_options=include_options
        )
        
        return await generate_qa(qa_request, user_id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"File QA generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to stream file QA: {str(e)}")
