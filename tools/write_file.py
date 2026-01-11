# tools/write_file.py
from typing import Dict, Any
from pathlib import Path

def write_file(path: str, contents: str) -> Dict[str, Any]:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(contents, encoding="utf-8")
        return {"path": str(p), "status": "ok"}
    except Exception as e:
        return {"error": str(e)}

TOOL = {
    "name": "write_file", 
    "func": write_file, 
    "description": "Save/write text content to a file. Use when user asks to save, note down, or create a file. Args: path (filename), contents (text to write)."
}
