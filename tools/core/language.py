from typing import Optional, Tuple
import hashlib
import logging

from langdetect import detect_langs  # type: ignore

from core.redis_client import get_redis

logger = logging.getLogger(__name__)


def _safe_detect(text: str):
    """Wrapper around langdetect.detect_langs with robust error handling."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        return detect_langs(text)
    except Exception as e:
        logger.debug(f"Language detection failed: {e}")
        return []


def detect_session_language(
    message: str,
    previous_language: Optional[str] = None,
    confidence_threshold: float = 0.7,
) -> Tuple[str, float, bool]:
    """
    Detect the primary language of the latest user message.

    Returns (language_code, confidence, is_mixed).
    If confidence is low, fall back to previous_language (or 'en').
    """
    detections = _safe_detect(message)
    default_lang = previous_language or "en"

    if not detections:
        return default_lang, 0.0, False

    detections_sorted = sorted(detections, key=lambda d: d.prob, reverse=True)
    top = detections_sorted[0]
    lang_code = top.lang
    confidence = float(top.prob or 0.0)

    # Mixed-language heuristic: if we see multiple plausible languages with
    # non-trivial probability (e.g. Hinglish or other blends).
    is_mixed = False
    if len(detections_sorted) > 1:
        second = detections_sorted[1]
        if float(second.prob or 0.0) >= 0.2:
            is_mixed = True

    # Confidence fallback to previous session language
    if confidence < confidence_threshold:
        logger.debug(
            f"Low language detection confidence ({confidence:.2f}) for '{lang_code}', "
            f"falling back to previous={default_lang}"
        )
        return default_lang, confidence, is_mixed

    return lang_code, confidence, is_mixed


def language_display_name(code: str) -> str:
    """Map ISO language codes to human-readable names for prompt wording."""
    mapping = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "pt": "Portuguese",
        "it": "Italian",
        "hi": "Hindi",
        "bn": "Bengali",
        "ur": "Urdu",
        "ta": "Tamil",
        "te": "Telugu",
        "zh-cn": "Chinese",
        "zh-tw": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
    }
    return mapping.get(code.lower(), code)


def build_language_instructions(
    session_language: str,
    is_mixed: bool,
) -> str:
    """
    Build a compact system instruction that steers the assistant to respond
    in the user's language while keeping technical tokens in English.
    """
    lang_code = (session_language or "en").lower()
    lang_name = language_display_name(lang_code)

    base = (
        "Always respond in the same language as the user's last message "
        "unless they explicitly ask for a translation or a different language."
    )

    # Default English: still keep explicit rule for mixed content.
    if lang_code == "en":
        return (
            base
            + "\n\nWhen the user mixes languages, mirror their style but keep code, "
              "API names, error messages, and mathematical symbols in clear English."
        )

    if is_mixed:
        # Hinglish / mixed-language guidance – explanations vs technical tokens.
        return (
            base
            + f"\n\nThe user appears to be writing in a mix of {lang_name} and English."
              f" Use natural {lang_name} for explanations and narrative text, but keep"
              " formulas, code, algorithm names, library/package names, and most"
              " technical keywords in English for clarity."
        )

    # Pure non-English language guidance.
    return (
        base
        + f"\n\nThe user's primary language for this session is {lang_name}."
          f" Write explanations and prose in {lang_name}, but keep code, function and"
          " class names, library/package names, error messages, and mathematical"
          " expressions in English when that is clearer or more standard."
    )


async def normalize_query_with_cache(user_message: str) -> str:
    """
    Normalize user queries in a language-agnostic way with Redis caching.

    PHASE 3 — QUERY NORMALIZATION CACHE:
    - Key:  norm:{hash(user_message)}
    - Store: normalized_query
    - TTL:  24 hours

    For now, normalization is a pass-through (identity) so we don't change
    behavior. An LLM-based normalizer can later be inserted here without
    touching the callers.
    """
    text = (user_message or "").strip()
    if not text:
        return text

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    key = f"norm:{digest}"

    try:
        redis = await get_redis()
    except Exception as e:
        logger.warning(f"Redis error during query normalization (get client): {e}")
        redis = None

    if redis is not None:
        try:
            cached = await redis.get(key)
            if isinstance(cached, str) and cached:
                return cached
        except Exception as e:
            logger.warning(f"Redis error during query normalization (read): {e}")

    # CURRENT BEHAVIOR: identity normalization so semantics are unchanged.
    normalized = text

    if redis is not None:
        try:
            await redis.set(key, normalized, ex=24 * 60 * 60)
        except Exception as e:
            logger.warning(f"Redis error during query normalization (write): {e}")

    return normalized

