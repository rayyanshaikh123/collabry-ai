import json
from typing import Any, Dict, List, Optional

try:
    import graphviz
except Exception:
    graphviz = None  # handled at runtime

import base64


def _normalize_node(n: Any) -> Dict[str, Any]:
    # Accept either string or dict nodes
    if isinstance(n, str):
        return {"label": n, "children": []}
    if isinstance(n, dict):
        return {
            "label": n.get("label") or n.get("text") or n.get("name") or "",
            "children": n.get("children") or n.get("nodes") or []
        }
    return {"label": str(n), "children": []}


def _traverse_build(graph, mermaid_lines: List[str], node, parent_id: Optional[str], id_prefix: str, counter: Dict[str, int], node_declarations: Optional[set] = None):
    if node_declarations is None:
        node_declarations = set()
    
    norm = _normalize_node(node)
    counter["i"] += 1
    # Ensure valid Mermaid node ID (alphanumeric only, no underscores in prefix to avoid issues)
    node_id = f"{id_prefix}{counter['i']}"
    label = norm["label"].replace('"', "'")

    # Graphviz node
    if graph is not None:
        graph.node(node_id, label)

    # Mermaid line
    # Mermaid nodes must be declared before edges
    # Format: nodeId["Label"]
    mermaid_label = label.replace("\n", " ").strip()
    # Escape quotes in label
    mermaid_label = mermaid_label.replace('"', "'")
    
    # Declare node first (only once)
    if node_id not in node_declarations:
        mermaid_lines.append(f'{node_id}["{mermaid_label}"]')
        node_declarations.add(node_id)
    
    # Then add edge from parent if exists
    if parent_id:
        mermaid_lines.append(f"{parent_id} --> {node_id}")

    for child in norm["children"]:
        _traverse_build(graph, mermaid_lines, child, node_id, id_prefix, counter, node_declarations)


def json_to_mermaid(mindmap: Any) -> str:
    """Convert a JSON mindmap structure to a Mermaid `graph TD` representation.

    Expected JSON shape: {"label": "Root", "children": [{...}, ...]}
    or a list with a single root.
    """
    mermaid_lines: List[str] = ["graph TD"]
    counter = {"i": 0}
    node_declarations = set()
    # Allow either a dict root or a list
    if isinstance(mindmap, list):
        for root in mindmap:
            _traverse_build(None, mermaid_lines, root, None, "n", counter, node_declarations)
    else:
        _traverse_build(None, mermaid_lines, mindmap, None, "n", counter, node_declarations)

    return "\n".join(mermaid_lines)


def generate_mindmap(mindmap_json: Any, output_path: Optional[str] = None, render_svg: bool = True) -> Dict[str, Any]:
    """Generate mindmap artifacts.

    - `mindmap_json` may be a JSON string or a Python dict/list describing the mindmap.
    - Returns dict with keys: json (the input normalized), mermaid (string), svg_base64 (if rendered), saved_path (if written)
    """
    try:
        if isinstance(mindmap_json, str):
            mindmap = json.loads(mindmap_json)
        else:
            mindmap = mindmap_json
    except Exception as e:
        return {"error": f"Invalid JSON: {e}"}

    # Build Graphviz graph if library available
    graph = None
    svg_b64 = None
    try:
        if graphviz is not None and render_svg:
            graph = graphviz.Digraph(format="svg")
    except Exception:
        graph = None

    mermaid = json_to_mermaid(mindmap)

    # Populate graph and optionally render
    counter = {"i": 0}
    try:
        _traverse_build(graph, [], mindmap, None, "n", counter, set())
        if graph is not None and render_svg:
            # Render to bytes (SVG)
            svg_bytes = graph.pipe(format="svg")
            if svg_bytes:
                svg_b64 = base64.b64encode(svg_bytes).decode("utf-8")
                if output_path:
                    with open(output_path, "wb") as f:
                        f.write(svg_bytes)
    except Exception as e:
        return {"error": f"Failed to build/render graph: {e}"}

    result: Dict[str, Any] = {
        "json": mindmap,
        "mermaid": mermaid,
        "svg_base64": svg_b64,
    }

    if output_path:
        result["saved_path"] = output_path

    return result


TOOL = {
    "name": "mindmap_generator",
    "func": generate_mindmap,
    "description": "Generate mind maps: accepts JSON mindmap input, produces Graphviz SVG (if Graphviz installed), Mermaid code for web, and returns JSON for storage. Requires python-graphviz package and Graphviz system binary."
}
