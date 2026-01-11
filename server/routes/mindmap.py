"""
Mind Map generation endpoint.

Handles:
- Hierarchical concept mapping
- Document-informed topic expansion
- Structured graph output
"""
from fastapi import APIRouter, Depends, HTTPException
from server.deps import get_current_user
from server.schemas import MindMapRequest, MindMapResponse, MindMapNode, ErrorResponse
from core.agent import create_agent
from core.rag_retriever import RAGRetriever
from config import CONFIG
import logging
from datetime import datetime
from uuid import uuid4
import json
from fastapi import Body, Query
from fastapi.responses import JSONResponse
from typing import Optional
import tools

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["mindmap"])


def parse_mindmap_from_text(text: str, topic: str) -> MindMapNode:
    """
    Parse mind map structure from LLM text output.
    
    Expected format:
    - Main Topic
      - Subtopic 1
        - Detail 1.1
        - Detail 1.2
      - Subtopic 2
        - Detail 2.1
    
    Returns:
        MindMapNode tree structure
    """
    lines = text.strip().split('\n')
    root = MindMapNode(id="root", label=topic, level=0, children=[])
    
    # Stack to track parent nodes at each level
    stack = [(0, root)]
    node_counter = [1]  # Use list for mutable counter
    
    for line in lines:
        if not line.strip() or line.strip().startswith('#'):
            continue
        
        # Calculate indentation level (number of leading spaces or dashes)
        stripped = line.lstrip(' -•*')
        indent_level = len(line) - len(stripped)
        level = indent_level // 2 + 1  # Convert spaces to level
        
        label = stripped.strip()
        if not label:
            continue
        
        # Create new node
        node = MindMapNode(
            id=f"node_{node_counter[0]}",
            label=label,
            level=level,
            children=[]
        )
        node_counter[0] += 1
        
        # Find parent at previous level
        while stack and stack[-1][0] >= level:
            stack.pop()
        
        if stack:
            parent = stack[-1][1]
            parent.children.append(node)
        
        stack.append((level, node))
    
    return root


@router.post(
    "/mindmap",
    response_model=MindMapResponse,
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Generate mind map",
    description="Create hierarchical mind map for a topic using AI and user documents"
)
async def generate_mindmap(
    request: MindMapRequest,
    user_id: str = Depends(get_current_user)
) -> MindMapResponse:
    """
    Generate mind map for a topic.
    
    - Extracts user_id from JWT token
    - Retrieves relevant documents if requested
    - Generates hierarchical concept map
    - Returns structured graph data
    """
    try:
        logger.info(f"Mind map request from user={user_id}, topic={request.topic}")
        
        context_text = ""
        
        # Retrieve relevant documents if enabled
        if request.use_documents:
            rag = RAGRetriever(CONFIG, user_id=user_id)
            docs = rag.get_relevant_documents(request.topic, user_id=user_id)
            
            if docs:
                context_parts = []
                for doc in docs[:3]:  # Use top 3 documents
                    context_parts.append(doc.page_content[:500])
                context_text = "\n\n".join(context_parts)
                logger.info(f"Retrieved {len(docs)} documents for mind map context")
        
        # Create agent
        agent, _, _, _ = create_agent(
            user_id=user_id,
            session_id=str(uuid4()),
            config=CONFIG
        )
        
        # Build mind map prompt
        prompt = f"""Create a hierarchical mind map for the topic: "{request.topic}"

Requirements:
- Generate {request.depth} levels of depth
- Use bullet points with indentation to show hierarchy
- Start with main topic, then subtopics, then details
- Be concise (5-10 items per level)
"""
        
        if context_text:
            prompt += f"""
Context from user documents:
{context_text}

Use this context to inform your mind map structure.
"""
        
        prompt += """
Output format (use bullet points with indentation):
- Main Topic
  - Subtopic 1
    - Detail 1.1
    - Detail 1.2
  - Subtopic 2
    - Detail 2.1

Mind map:"""
        
        # Collect response
        response_chunks = []
        
        def collect_chunk(chunk: str):
            response_chunks.append(chunk)
        
        # Execute agent
        agent.handle_user_input_stream(prompt, collect_chunk)
        
        mindmap_text = "".join(response_chunks).strip()
        
        # Parse mind map structure
        root = parse_mindmap_from_text(mindmap_text, request.topic)
        
        # Count total nodes
        def count_nodes(node: MindMapNode) -> int:
            return 1 + sum(count_nodes(child) for child in node.children)
        
        total_nodes = count_nodes(root)
        
        logger.info(f"Mind map generated: {total_nodes} nodes")
        
        return MindMapResponse(
            topic=request.topic,
            root=root,
            total_nodes=total_nodes,
            user_id=user_id,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.exception(f"Mind map generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate mind map: {str(e)}"
        )



@router.post(
    "/mindmap/render",
    summary="Render mindmap (Mermaid or SVG)",
    description="Accepts a JSON mindmap tree and returns Mermaid code and optionally an SVG (base64)."
)
async def render_mindmap(
    mindmap: dict = Body(..., description="Mind map JSON structure"),
    format: Optional[str] = Query('svg', description="Output format: 'svg' or 'mermaid' or 'both'"),
    user_id: str = Depends(get_current_user)
) -> JSONResponse:
    """Render a provided mindmap JSON into Mermaid and/or Graphviz SVG using the internal tool.

    Expects the mindmap shape compatible with `MindMapNode` (id, label, children).
    """
    try:
        logger.info(f"Mindmap render request from user={user_id}, format={format}")

        # Load tools and locate mindmap_generator
        tools_registry = tools.load_tools()
        mg_tool = tools_registry.get('mindmap_generator')
        if not mg_tool:
            return JSONResponse({"error": "mindmap_generator tool not available"}, status_code=500)

        func = mg_tool.get('func')
        if not callable(func):
            return JSONResponse({"error": "mindmap_generator tool invalid"}, status_code=500)

        # Call the tool - it returns mermaid and svg_base64 when possible
        result = func(mindmap)

        # Prepare response based on requested format
        out: dict = {"mermaid": result.get('mermaid'), "json": result.get('json')}
        if format in ('svg', 'both'):
            out['svg_base64'] = result.get('svg_base64')

        return JSONResponse(out)
    except Exception as e:
        logger.exception(f"Failed to render mindmap: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
