"""
Generate Mind Map Tool - Create visual concept maps from topics.

This tool generates mind maps to help visualize relationships
between concepts and ideas.
"""

import json
from typing import Optional, List
from langchain_core.tools import tool
from core.llm import get_async_openai_client
from core.llm import get_llm_config
from rag.retriever import get_retriever


@tool
async def generate_mindmap(
    topic: Optional[str] = "main concepts",
    user_id: str = "default",
    notebook_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None,
  include_sources: bool = True
) -> str:
    """
    Generate a mind map for visualizing concepts and relationships.
    
    Use this tool when the user wants to:
    - Create a concept map
    - Visualize relationships between ideas
    - See the structure of a topic
    
    Examples of when to use:
    - "Create a mind map for machine learning"
    - "Show me a concept map of the causes of World War I"
    - "Visualize the relationships in photosynthesis"
    
    Args:
        topic: The main topic or concept to map
        user_id: User identifier (injected by agent)
        notebook_id: Optional notebook context
        include_sources: Whether to use user's uploaded documents
    
    Returns:
        JSON mind map with nodes and edges in the format:
        {
          "title": "Topic Name",
          "root": {
            "label": "Root Concept",
            "children": [
              {"label": "Child 1", "children": [...]},
              {"label": "Child 2", "children": [...]}
            ]
          }
        }
    """
    try:
        # Get OpenAI client
        client = get_async_openai_client()
        llm_config = get_llm_config()
        
        # Build context from sources if requested
        context = ""
        if include_sources and notebook_id:
          try:
            retriever = get_retriever(
              user_id=user_id,
              notebook_id=notebook_id,
              source_ids=source_ids,
            )
            docs = retriever.invoke(topic)
            if docs:
              context = "\n\n".join([doc.page_content for doc in docs[:3]])
          except Exception as e:
            # Continue without context if retrieval fails
            context = ""
        
        # Create prompt for mind map generation
        prompt = f"""Generate a hierarchical mind map for the topic: {topic}

{f"Use this context from the user's documents:\n{context}\n" if context else ""}

Create a structured JSON mind map with the following format (hierarchical tree):
{{
  "id": "root",
  "label": "Central Concept",
  "children": [
    {{
      "id": "node_1",
      "label": "Main Branch 1",
      "children": [
        {{"id": "node_1_1", "label": "Sub-concept 1.1", "children": []}},
        {{"id": "node_1_2", "label": "Sub-concept 1.2", "children": []}}
      ]
    }},
    {{
      "id": "node_2",
      "label": "Main Branch 2",
      "children": []
    }}
  ]
}}

Guidelines:
- Create 3-5 main branches from the root
- Each main branch should have 2-4 sub-concepts
- Keep labels concise (2-5 words)
- Focus on key concepts and relationships
- Ensure logical hierarchy

Return ONLY the JSON, no additional text."""

        # Generate mind map
        response = await client.chat.completions.create(
          model=llm_config.model,
          messages=[
            {
              "role": "system",
              "content": "You are an expert at creating educational mind maps. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
          ],
          temperature=llm_config.temperature,
          max_tokens=min(int(getattr(llm_config, "max_tokens", 1500)), 1500),
        )
        
        # Extract and validate JSON
        content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        # Validate JSON
        mindmap_data = json.loads(content)
        
        # Normalize output to a frontend-parseable hierarchical tree shape.
        # Frontend parser accepts either {label, children} or {nodes, edges}.
        # If the model returned {title, root:{label, children}}, unwrap to {id,label,children}.
        if isinstance(mindmap_data, dict):
            if "root" in mindmap_data and isinstance(mindmap_data.get("root"), dict):
                root = mindmap_data.get("root") or {}
                normalized = {
                    "id": root.get("id") or "root",
                    "label": root.get("label") or topic,
                    "children": root.get("children") or [],
                }
                return json.dumps(normalized, indent=2, ensure_ascii=False)

            # Already hierarchical
            if "label" in mindmap_data and "children" in mindmap_data:
                if "id" not in mindmap_data:
                    mindmap_data["id"] = "root"
                return json.dumps(mindmap_data, indent=2, ensure_ascii=False)

        # If we got here, shape isn't something the frontend can reliably parse.
        return json.dumps(
            {
                "error": "Invalid mind map structure",
                "message": "Generated mind map JSON missing expected fields",
                "topic": topic,
            },
            indent=2,
            ensure_ascii=False,
        )
        
    except json.JSONDecodeError as e:
        return json.dumps({
            "error": "Failed to parse mind map JSON",
            "message": str(e),
            "topic": topic
        })
    except Exception as e:
        return json.dumps({
            "error": "Failed to generate mind map",
            "message": str(e),
            "topic": topic
        })
