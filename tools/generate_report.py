"""Generate Report Tool - Create comprehensive markdown study reports.

This tool generates a structured markdown report, optionally grounded
in the user's uploaded notebook sources via RAG.
"""

import json
from typing import Optional, List

from langchain_core.tools import tool

from core.llm import get_async_openai_client, get_llm_config
from core.artifact_templates import format_reports_prompt
from rag.retriever import get_retriever


@tool
async def generate_report(
    topics: str,
    user_id: str = "default",
    notebook_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None,
    include_sources: bool = True,
) -> str:
    """Generate a comprehensive study report in Markdown.

    Args:
        topics: The subject/topics to report on.
        user_id: User identifier (injected by agent).
        notebook_id: Optional notebook context for retrieval.
        include_sources: Whether to ground the report in the user's uploaded documents.

    Returns:
        Markdown string with headers and bullet points (no JSON wrapping).
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

    base_prompt = format_reports_prompt(topics)

    prompt = base_prompt
    if context:
        prompt = (
            f"{base_prompt}\n\n"
            "SOURCE MATERIAL (use as the only basis of the report):\n"
            f"{context}\n"
        )

    resp = await client.chat.completions.create(
        model=llm_config.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a study assistant. Produce ONLY the markdown report. "
                    "Do not include code fences or extra commentary. "
                    "If source material is provided, do not invent facts beyond it."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=llm_config.temperature,
        max_tokens=min(int(getattr(llm_config, "max_tokens", 1500)), 1500),
    )

    content = (resp.choices[0].message.content or "").strip()

    # Defensive: some models wrap markdown in ``` blocks
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1].strip()
            if content.lower().startswith("markdown"):
                content = content[len("markdown") :].strip()

    # If the model accidentally returned JSON, unwrap 'report' field if present.
    try:
        maybe_json = json.loads(content)
        if isinstance(maybe_json, dict):
            for key in ("report", "markdown", "content", "answer"):
                if isinstance(maybe_json.get(key), str) and maybe_json[key].strip():
                    return maybe_json[key].strip()
    except Exception:
        pass

    return content
