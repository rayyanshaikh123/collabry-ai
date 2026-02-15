"""
Summarize Notes Tool - Generate comprehensive summaries.

This tool creates structured summaries of study materials
from a user's notebook.
"""

from typing import Optional, List
from langchain_core.tools import tool
from rag.retriever import get_retriever
from core.llm import get_async_openai_client, get_llm_config


@tool
async def summarize_notes(
    topic: Optional[str] = "all notes",
    notebook_id: Optional[str] = None,
    user_id: str = "default",
    source_ids: Optional[List[str]] = None,
    retrieval_policy: str = "PREFER_SELECTED",
    retrieval_mode: str = "CHUNK_SEARCH",
    token: Optional[str] = None
) -> str:
    """
    Generate a comprehensive summary of notes in a notebook.
    """
    try:
        from core.retrieval_service import get_hybrid_context
        
        # Use the intelligent grounding service
        combined_text = await get_hybrid_context(
            user_id=user_id,
            notebook_id=notebook_id,
            policy=retrieval_policy,
            mode=retrieval_mode,
            source_ids=source_ids or [],
            query=topic,
            token=token
        )

        if not combined_text:
            return "No documents found. Please upload some study materials first."
        
        # Generate summary using LLM
        client = get_async_openai_client()
        config = get_llm_config()
        
        system_prompt = """You are a study assistant creating a comprehensive summary.
        
Generate a well-structured summary with:
1. Main Topics - key subjects covered
2. Key Concepts - important ideas and definitions
3. Important Details - facts, dates, examples
4. Study Tips - suggested focus areas

Format in clear sections with bullet points."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Summarize these study materials:\n\n{combined_text[:8000]}"}  # Limit context
        ]
        
        response = await client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        
        summary = response.choices[0].message.content
        
        return f"# Summary of {notebook_id or 'Your Notes'}\n\n{summary}"
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"
