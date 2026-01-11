# tools/read_file.py
from typing import Dict, Any
from pathlib import Path

def read_file(path: str) -> Dict[str, Any]:
    try:
        p = Path(path)
        if not p.exists():
            return {"error": f"File not found: {path}"}
        text = p.read_text(encoding="utf-8", errors="replace")
        # return trimmed
        return {"path": str(p), "text": text[:20000]}
    except Exception as e:
        return {"error": str(e)}

TOOL = {"name": "read_file", "func": read_file, "description": "Read a local file and return contents (truncated)."}
