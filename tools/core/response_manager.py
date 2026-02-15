import json
import re
from typing import Any, Dict, List, Optional
from core.tool_registry import ToolMetadata
from urllib.parse import urlparse

class ResponseManager:
    @staticmethod
    async def format_response(tool_name: str, output: Any, metadata: ToolMetadata, params: Dict[str, Any]) -> str:
        """Central formatting entry point."""
        fmt = metadata.output_format
        
        if fmt == "json_quiz":
            return ResponseManager._coerce_quiz(str(output))
        if fmt == "json_flashcards":
            topic = params.get("topic") or "Study Flashcards"
            return f"[FLASHCARDS_GENERATION_REQUEST]\n{ResponseManager._coerce_flashcards(str(output), topic)}"
        if fmt == "json_mindmap":
            return f"[MINDMAP_GENERATION_REQUEST]\n{str(output)}"
        if tool_name == "generate_infographic":
            return f"[INFOGRAPHIC_GENERATION_REQUEST]\n{str(output)}"
        if tool_name == "search_web":
            return ResponseManager._format_web_search(str(output))
        
        # Default fallback strategies
        if isinstance(output, str):
            return output
        
        return json.dumps(output, ensure_ascii=False)

    @staticmethod
    def _coerce_quiz(tool_output: str) -> str:
        raw = (tool_output or "").strip()
        try:
            data = json.loads(raw)
            if isinstance(data, list): return raw
        except: return raw

        if isinstance(data, dict) and isinstance(data.get("questions"), list):
            out = []
            for q in data.get("questions", []):
                if not isinstance(q, dict): continue
                opts = q.get("options") if isinstance(q.get("options"), dict) else {}
                options = [opts.get(k, "") for k in ("A", "B", "C", "D")]
                letter = str(q.get("correct_answer") or "A").strip().upper()[:1]
                idx = ord(letter) - 65 if 'A' <= letter <= 'D' else 0
                out.append({
                    "question": str(q.get("question", "")),
                    "options": options,
                    "correctAnswer": idx,
                    "explanation": str(q.get("explanation", "")),
                })
            if out: return json.dumps(out, ensure_ascii=False)
        return raw

    @staticmethod
    def _coerce_flashcards(tool_output: str, fallback_title: str) -> str:
        raw = (tool_output or "").strip()
        try:
            data = json.loads(raw)
        except: return raw

        if isinstance(data, dict) and isinstance(data.get("cards"), list):
            title = data.get("title") or fallback_title
            cards = []
            for i, c in enumerate(data.get("cards", [])):
                if not isinstance(c, dict): continue
                f, b = c.get("front"), c.get("back")
                if not f or not b: continue
                card = {"id": f"card-{i+1}", "front": str(f).strip(), "back": str(b).strip()}
                if isinstance(c.get("tags"), list) and c["tags"]:
                    card["category"] = str(c["tags"][0])[:40]
                cards.append(card)
            if cards: return json.dumps({"title": title, "cards": cards}, ensure_ascii=False)
        return raw

    @staticmethod
    def _format_web_search(output: str) -> str:
        pattern = re.compile(r"\*\*(?P<title>[^*]+)\*\*\s*\n(?P<snippet>.*?)\n🔗\s*(?P<url>\S+)", re.DOTALL)
        items = []
        for match in pattern.finditer(output):
            title, url = match.group("title").strip(), match.group("url").strip()
            host = urlparse(url).netloc.lower()
            platform = "Online"
            for p in ("coursera", "udemy", "edx", "youtube"):
                if p in host: platform = p.capitalize(); break
            items.append(f"- [{title}]({url}) - Platform: {platform}")
        
        if not items: return output.strip()
        return "I found some great resources for you:\n\n" + "\n".join(items[:8])
