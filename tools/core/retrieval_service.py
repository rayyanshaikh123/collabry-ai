"""
Retrieval Service - Orchestrates semantic search (RAG) and full-text retrieval.
"""

import logging
import httpx
import json
import hashlib
import re
from typing import List, Optional, Dict, Any
from rag.vectorstore import similarity_search
from config import CONFIG
from core.redis_client import get_redis

logger = logging.getLogger(__name__)

def validate_source_ids(source_ids: List[str]) -> List[str]:
    """
    SECURITY FIX - Phase 1: Validate and sanitize source IDs to prevent injection attacks.
    
    Args:
        source_ids: List of source ID strings to validate
        
    Returns:
        List of validated source IDs
        
    Raises:
        ValueError: If any source ID is invalid/malicious
    """
    if not source_ids:
        return []
        
    validated_ids = []
    
    for source_id in source_ids:
        # Check for basic type safety
        if not isinstance(source_id, str):
            logger.warning(f"Invalid source_id type: {type(source_id)}")
            raise ValueError(f"Invalid source_id type: {type(source_id)}")
            
        # Remove whitespace
        source_id = source_id.strip()
        
        # Check minimum length
        if len(source_id) < 12:
            logger.warning(f"Source ID too short: {source_id}")
            raise ValueError(f"Invalid source_id format: {source_id}")
        
        # Reject obviously malicious patterns
        malicious_patterns = [
            r'[./\\]',  # Path traversal attempts
            r'[;\'"{}()]',  # SQL/NoSQL injection characters
            r'\s*(OR|AND|DROP|SELECT|INSERT|UPDATE|DELETE|UNION)\s*',  # SQL keywords
            r'<script>|javascript:|eval\(',  # XSS attempts
            r'\$\w+',  # MongoDB operators like $ne, $gt
            r'\.\./',  # Directory traversal
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, source_id, re.IGNORECASE):
                logger.warning(f"Malicious pattern detected in source_id: {source_id}")
                raise ValueError(f"Malicious pattern detected in source_id: {source_id}")
        
        # Validate ObjectId/MongoDB format (24 hex characters)
        if len(source_id) == 24 and all(c in '0123456789abcdefABCDEF' for c in source_id):
            validated_ids.append(source_id)
        else:
            # Could be UUID or other valid format - validate more strictly
            if re.match(r'^[a-zA-Z0-9_-]{12,50}$', source_id):
                validated_ids.append(source_id)
            else:
                logger.warning(f"Invalid source_id format: {source_id}")
                raise ValueError(f"Invalid source_id format: {source_id}")
    
    logger.info(f"✅ Validated {len(validated_ids)} source IDs successfully")
    return validated_ids

BACKEND_URL = CONFIG.get("backend_url", "http://localhost:5000")

async def get_hybrid_context(
    user_id: str,
    notebook_id: str,
    policy: str,
    mode: str,
    source_ids: List[str],
    query: str,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieves context based on the validated plan.
    SECURITY FIX - Phase 1: Enhanced with source ID validation.
    PHASE 4: Now returns citations alongside context.
    
    Returns:
        Dictionary with 'context' (str) and 'citations' (list)
    """
    from core.citation_manager import CitationManager
    
    logger.info(f"🚀 Retrieval: policy={policy}, mode={mode}, sources={len(source_ids)}")
    
    citation_mgr = CitationManager()
    
    if mode == "NONE":
        return {"context": "", "citations": []}

    # SECURITY FIX: Validate source IDs before ANY processing
    try:
        validated_source_ids = validate_source_ids(source_ids)
    except ValueError as e:
        logger.error(f"🚨 Security: Invalid source_ids provided: {e}")
        return {"context": "", "citations": []}  # Fail safe

    context_parts: List[str] = []

    # PHASE 4 — RETRIEVAL CACHE
    # Cache RAG retrieval results keyed by normalized query and selected sources.
    cache_key = None
    redis = None
    try:
        redis = await get_redis()
        if redis is not None:
            cache_payload = {
                "q": query,
                "policy": policy,
                "mode": mode,
                "notebook_id": notebook_id,
                "source_ids": sorted(validated_source_ids),
            }
            digest = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()
            cache_key = f"rag:{digest}"
            cached = await redis.get(cache_key)
            if isinstance(cached, str) and cached:
                logger.debug(f"✅ Retrieval cache hit for key={cache_key}")
                # Note: Cached results don't have citations - we'll need to regenerate them
                # For now, return cached context without citations
                return {"context": cached, "citations": []}
    except Exception as e:
        logger.warning(f"Redis retrieval cache error (read): {e}")
        redis = None

    # 1. Handle FULL_DOCUMENT (High fidelity recap/summary)
    if mode == "FULL_DOCUMENT" or mode == "MULTI_DOC_SYNTHESIS":
        if validated_source_ids and token:
            full_texts = await fetch_full_sources(notebook_id, validated_source_ids, token)
            for st in full_texts:
                # Add citation for full document
                citation_id = citation_mgr.add_citation(
                    source_id=st['id'],
                    source_name=st['name'],
                    excerpt=st['content'][:300],  # First 300 chars as excerpt
                    source_type=st['type']
                )
                context_parts.append(f"--- SOURCE: {st['name']} (Type: {st['type']}) [{citation_id}] ---\n{st['content']}")
        
        # If we got full text, we might skip chunk search for summaries, 
        # but for synthesis we might want both.
        if mode == "FULL_DOCUMENT" and context_parts:
            context = "\n\n".join(context_parts)
            return {"context": context, "citations": citation_mgr.get_all()}

    # 2. Handle CHUNK_SEARCH (Standard RAG)
    # We always do this for specific queries or if full text failed/wasn't requested.
    search_policy_notebook = notebook_id if policy != "AUTO_EXPAND" else None
    search_source_ids = validated_source_ids if policy == "STRICT_SELECTED" or policy == "PREFER_SELECTED" else None
    
    docs = similarity_search(
        query=query,
        user_id=user_id,
        notebook_id=search_policy_notebook,
        source_ids=search_source_ids,
        k=15 if mode == "MULTI_DOC_SYNTHESIS" else 8
    )
    
    if docs:
        if context_parts: context_parts.append("\n--- RELEVANT EXCERPTS ---")
        for doc in docs:
            # Add citation for each chunk
            citation_id = citation_mgr.add_from_document(doc)
            source_name = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[{source_name}]: {doc.page_content} [{citation_id}]")
    
    context = "\n\n".join(context_parts)

    # Best-effort cache write with 6-hour TTL
    if redis is not None and cache_key and context:
        try:
            await redis.set(cache_key, context, ex=6 * 60 * 60)
        except Exception as e:
            logger.warning(f"Redis retrieval cache error (write): {e}")

    return {"context": context, "citations": citation_mgr.get_all()}

async def fetch_full_sources(
    notebook_id: str, 
    source_ids: List[str], 
    token: str
) -> List[Dict[str, Any]]:
    """
    Call the Node.js backend to get the full raw text of specific sources.
    SECURITY FIX - Phase 1: Enhanced with source ID validation.
    """
    # SECURITY FIX: Double-validate source IDs (defense in depth)
    try:
        validated_source_ids = validate_source_ids(source_ids)
    except ValueError as e:
        logger.error(f"🚨 Security: Invalid source_ids in fetch_full_sources: {e}")
        return []  # Fail safe
    
    results = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for sid in validated_source_ids:
            try:
                # Use the existing backend endpoint
                url = f"{BACKEND_URL}/api/notebook/notebooks/{notebook_id}/sources/{sid}/content"
                headers = {"Authorization": f"Bearer {token}"}
                
                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json().get("data", {})
                    if data.get("content"):
                        results.append({
                            "id": sid,
                            "name": data.get("name", "Unknown"),
                            "type": data.get("type", "unknown"),
                            "content": data.get("content")
                        })
            except Exception as e:
                logger.error(f"Error fetching source {sid}: {e}")
                
    return results

async def fetch_source_metadata(
    notebook_id: str,
    source_ids: List[str],
    token: str
) -> List[Dict[str, Any]]:
    """
    Fetch names and types for the provided source IDs.
    SECURITY FIX - Phase 1: Enhanced with source ID validation.
    """
    if not notebook_id or not token:
        return []

    # SECURITY FIX: Validate source IDs before metadata fetch
    try:
        validated_source_ids = validate_source_ids(source_ids)
    except ValueError as e:
        logger.error(f"🚨 Security: Invalid source_ids in fetch_source_metadata: {e}")
        return []  # Fail safe

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            url = f"{BACKEND_URL}/api/notebook/notebooks/{notebook_id}"
            headers = {"Authorization": f"Bearer {token}"}
            
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                notebook = response.json().get("data", {})
                sources = notebook.get("sources", [])
                
                # Filter to requested IDs (using validated IDs)
                meta_list = []
                for s in sources:
                    sid = str(s.get("_id") or s.get("id"))
                    if sid in [str(requested_id) for requested_id in validated_source_ids]:
                        meta_list.append({
                            "id": sid,
                            "name": s.get("name", "Unknown"),
                            "type": s.get("type", "unknown")
                        })
                return meta_list
    except Exception as e:
        logger.error(f"Error fetching source metadata: {e}")
        
    return []
