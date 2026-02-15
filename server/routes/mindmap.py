"""Mindmap endpoints.

This module intentionally only exposes rendering/conversion endpoints used by the frontend viewer.
Mindmap *generation* should happen via the chat/tool pipeline (tools.generate_mindmap) for consistent
provider behavior (OpenAI vs OpenAI-compatible) and consistent JSON output.
"""

from fastapi import APIRouter, Depends
from server.deps import get_current_user, get_user_id
import logging
from fastapi import Body, Query
from fastapi.responses import JSONResponse
from typing import Optional
from tools.mindmap_generator import TOOL as mindmap_tool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["mindmap"])
@router.post(
    "/mindmap/render",
    summary="Render mindmap (Mermaid or SVG)",
    description="Accepts a JSON mindmap tree and returns Mermaid code and optionally an SVG (base64)."
)
async def render_mindmap(
    mindmap: dict = Body(..., description="Mind map JSON structure"),
    format: Optional[str] = Query('svg', description="Output format: 'svg' or 'mermaid' or 'both'"),
    user_id: str = Depends(get_user_id)
) -> JSONResponse:
    """Render a provided mindmap JSON into Mermaid and/or Graphviz SVG using the internal tool.

    Expects the mindmap shape compatible with `MindMapNode` (id, label, children).
    """
    try:
        logger.info(f"Mindmap render request from user={user_id}, format={format}")

        # Get the mindmap generator function from tool registry
        generate_mindmap = mindmap_tool.get('func')
        if not generate_mindmap or not callable(generate_mindmap):
            return JSONResponse({"error": "mindmap_generator tool not available"}, status_code=500)

        # Call the mindmap generator function
        result = generate_mindmap(mindmap, render_svg=(format in ('svg', 'both')))
        
        # Check for errors
        if 'error' in result:
            return JSONResponse({"error": result['error']}, status_code=500)

        # Prepare response based on requested format
        out: dict = {"mermaid": result.get('mermaid'), "json": result.get('json')}
        if format in ('svg', 'both'):
            out['svg_base64'] = result.get('svg_base64')

        return JSONResponse(out)
    except Exception as e:
        logger.exception(f"Failed to render mindmap: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
