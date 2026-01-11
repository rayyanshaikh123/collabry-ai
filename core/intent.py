"""
Clean & Stable Intent / NER recognizer for COLLABRY

Priorities:
1. Trained Intent Classifier (Sentence-Transformers)
2. Rule-based shortcuts (regex)
3. Fallback: none

Supported Intents:
- open_app
- open_in_browser
- search
- web_scrape
- read_file
- play_youtube
- schedule_task
- none
"""

from typing import Dict, Any, Optional
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ============================================================
#  Load Trainable Classifier
# ============================================================
try:
    from core.intent_classifier import IntentClassifier
    _TRAINED_CLASSIFIER = None

    try:
        default_prefix = str(Path(__file__).parent.parent / "models" / "intent_classifier" / "intentclf")
        ic = IntentClassifier()
        ic.load(default_prefix)
        _TRAINED_CLASSIFIER = ic
        logger.info(f"[intent] Loaded trained classifier from {default_prefix}")
    except Exception as e:
        logger.warning(f"[intent] No trained classifier found: {e}")
        _TRAINED_CLASSIFIER = None

except Exception as e:
    logger.warning(f"[intent] Classifier import failed: {e}")
    _TRAINED_CLASSIFIER = None


# ============================================================
#  Optional spaCy NER
# ============================================================
try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm")
    except Exception:
        _NLP = spacy.blank("en")
    _HAS_NER = "ner" in _NLP.pipe_names
except Exception:
    _NLP = None
    _HAS_NER = False


# ============================================================
#  Entity Extraction
# ============================================================
def _extract_entities(text: str) -> Dict[str, Any]:
    text = text.strip()
    tl = text.lower()
    ents = {}

    # URL
    for tok in text.split():
        if tok.startswith(("http://", "https://", "www.")):
            ents["url"] = tok.strip(".,;")
            break

    # File paths
    for tok in text.split():
        if tok.lower().endswith((".txt", ".md", ".json", ".py", ".pdf")):
            ents["path"] = tok.strip("\"'")
            break

    # App names
    m = re.search(r"\bopen\s+([a-zA-Z0-9 ._-]+)", tl)
    if m:
        ents["app"] = m.group(1).strip()

    # Browser
    if "chrome" in tl:
        ents["browser"] = "chrome"

    # Query
    ents["query"] = text

    # spaCy NER
    if _NLP:
        try:
            doc = _NLP(text)
            if _HAS_NER:
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        ents["name"] = ent.text
                    if ent.label_ in ("GPE", "LOC"):
                        ents["location"] = ent.text
                    if ent.label_ in ("DATE", "TIME"):
                        ents["date"] = ent.text
        except Exception:
            pass

    return ents


# ============================================================
#  Intent Detection
# ============================================================
def detect_intent(text: str) -> Dict[str, Any]:
    """
    Clean + hybrid intent classification
    Priority:
    1. Trained classifier (if installed)
    2. Rule-based shortcuts
    3. Fallback none
    """

    if not text or not text.strip():
        return {"intent": "none", "entities": {}}

    t = text.strip()
    tl = t.lower()
    ents = _extract_entities(t)

    # ----------------------------------------------------------
    # 1. TRAINED CLASSIFIER (highest priority)
    # ----------------------------------------------------------
    try:
        if _TRAINED_CLASSIFIER is not None:
            pred = _TRAINED_CLASSIFIER.predict(t)
            if pred and pred.get("score", 0.0) >= 0.35:   # threshold
                return {
                    "intent": pred["intent"],
                    "entities": ents,
                    "score": pred["score"]
                }
    except Exception as e:
        logger.warning(f"[intent] classifier failed: {e}")

    # ----------------------------------------------------------
    # 2. RULE-BASED INTENTS (fallback)
    # ----------------------------------------------------------

    # open app
    m = re.match(r"^(please\s+)?(open|launch|run|start)\s+(.+)$", tl)
    if m:
        return {"intent": "open_app", "entities": {"app": m.group(3).strip()}}

    # youtube
    if "youtube" in tl or "youtu.be" in tl or tl.startswith("play "):
        return {"intent": "play_youtube", "entities": {"query": ents["query"]}}

    # url
    if "url" in ents:
        if "browser" in ents:
            return {"intent": "open_in_browser", "entities": {"url": ents["url"], "browser": ents["browser"]}}
        return {"intent": "web_scrape", "entities": {"url": ents["url"]}}

    # search
    if any(w in tl for w in ["search", "google", "look up", "find"]):
        return {"intent": "search", "entities": {"query": ents["query"]}}

    # read file
    if "path" in ents and any(w in tl for w in ["read", "open file", "open the file"]):
        return {"intent": "read_file", "entities": {"path": ents["path"]}}

    # schedule task
    if "schedule" in tl or "remind me" in tl:
        return {"intent": "schedule_task", "entities": {"text": t}}

    # ----------------------------------------------------------
    # 3. FALLBACK
    # ----------------------------------------------------------
    return {"intent": "none", "entities": ents}
