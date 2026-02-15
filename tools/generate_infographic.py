"""Generate Infographic Tool - Create structured infographic JSON.

Returns JSON in the exact shape expected by the frontend:
{
  "title": string,
  "subtitle": string,
  "sections": [...]
}

Optionally grounds content in the user's uploaded notebook sources via RAG.
"""

import json
from typing import Optional, Any, List

from langchain_core.tools import tool

from core.llm import get_async_openai_client, get_llm_config
from core.artifact_templates import format_infographic_prompt
from rag.retriever import get_retriever


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if not t.startswith("```"):
        return t
    parts = t.split("```")
    if len(parts) < 2:
        return t
    inner = parts[1].strip()
    if inner.lower().startswith("json"):
        inner = inner[len("json") :].strip()
    return inner


def _coerce_infographic_schema(obj: Any, fallback_title: str) -> dict:
    if not isinstance(obj, dict):
        return {
            "title": fallback_title,
            "subtitle": "",
            "sections": [],
        }

    title = obj.get("title") if isinstance(obj.get("title"), str) else fallback_title
    subtitle = obj.get("subtitle") if isinstance(obj.get("subtitle"), str) else ""
    sections = obj.get("sections") if isinstance(obj.get("sections"), list) else []

    return {
        "title": title,
        "subtitle": subtitle,
        "sections": sections,
    }


@tool
async def generate_infographic(
    topics: str,
    user_id: str = "default",
    notebook_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None,
    include_sources: bool = True,
) -> str:
    """Generate infographic JSON for the given topics.

    Args:
        topics: Topic(s) to build an infographic from.
        user_id: User identifier (injected by agent).
        notebook_id: Optional notebook context for retrieval.
        include_sources: Whether to ground the infographic in the user's uploaded documents.

    Returns:
        A JSON string (no markdown/code fences).
    """

    llm_config = get_llm_config()
    client = get_async_openai_client()

    topics = (topics or "").strip() or "key concepts"

    context = ""
    if include_sources and notebook_id:
        try:
            retriever = get_retriever(user_id=user_id, notebook_id=notebook_id, source_ids=source_ids)
            docs = retriever.invoke(topics)
            if docs:
                context = "\n\n".join([d.page_content for d in docs[:4]])
        except Exception:
            context = ""

    base_prompt = format_infographic_prompt(topics)
    prompt = base_prompt
    if context:
        prompt = (
            f"{base_prompt}\n\n"
            "SOURCE MATERIAL (use as the only basis of the infographic):\n"
            f"{context}\n"
        )

    resp = await client.chat.completions.create(
        model=llm_config.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a study assistant. Return ONLY valid JSON. "
                    "No markdown code fences. No extra commentary. "
                    "If source material is provided, do not invent data beyond it."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=llm_config.temperature,
        max_tokens=min(int(getattr(llm_config, "max_tokens", 1600)), 1600),
    )

    content = _strip_code_fences(resp.choices[0].message.content or "")

    try:
        data = json.loads(content)
    except Exception as e:
        return json.dumps(
            {
                "error": "Failed to parse infographic JSON",
                "message": str(e),
                "topic": topics,
            },
            ensure_ascii=False,
        )

    normalized = _coerce_infographic_schema(data, fallback_title=topics)
    return json.dumps(normalized, indent=2, ensure_ascii=False)
