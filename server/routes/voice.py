"""
Voice Tutor endpoint — Whisper (STT) → RAG Chat → TTS.

Flow:
  1. Receive audio blob from browser (webm/wav)
  2. Transcribe via OpenAI Whisper
  3. Run through existing agent pipeline (with RAG)
  4. Synthesise reply with OpenAI TTS
  5. Return JSON with transcript + audio (base64) + text response
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from server.deps import get_current_user
from core.agent import chat as agent_chat
from config import CONFIG
from openai import OpenAI
import base64
import io
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai/voice", tags=["voice-tutor"])


def _get_openai_direct_client() -> OpenAI:
    """
    Return an OpenAI client that always points to api.openai.com.
    Whisper STT and TTS are OpenAI-only features — they don't work on
    Groq, Ollama, Together, etc. — so we need a dedicated client.
    """
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url="https://api.openai.com/v1",
    )


# ── Whisper STT ────────────────────────────────────────────────────────
async def _transcribe(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """Transcribe audio bytes using OpenAI Whisper."""
    client = _get_openai_direct_client()
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename  # Whisper needs a filename hint

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    return transcript.text


# ── TTS ────────────────────────────────────────────────────────────────
async def _synthesise(text: str, voice: str = "nova") -> bytes:
    """Convert text to speech using OpenAI TTS."""
    client = _get_openai_direct_client()
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text[:4096],  # TTS input limit
    )
    return response.content  # raw mp3 bytes


# ── Main endpoint ──────────────────────────────────────────────────────
@router.post("/chat")
async def voice_chat(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    notebook_id: Optional[str] = Form(None),
    voice: Optional[str] = Form("nova"),
    user_info: dict = Depends(get_current_user),
):
    """
    Full voice-to-voice tutoring turn.

    Accepts: multipart form with audio file + optional session/notebook IDs.
    Returns: JSON { transcript, response_text, audio_base64, session_id }
    """
    user_id = user_info["user_id"]
    start = time.time()

    # 1. Read audio blob ---------------------------------------------------
    audio_bytes = await audio.read()
    if len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail="Audio too short or empty")
    if len(audio_bytes) > 25 * 1024 * 1024:  # 25 MB Whisper limit
        raise HTTPException(status_code=400, detail="Audio file too large (max 25 MB)")

    logger.info(f"🎙️ Voice chat from user {user_id} — {len(audio_bytes)} bytes")

    # 2. Transcribe (STT) --------------------------------------------------
    try:
        transcript = await _transcribe(audio_bytes, filename=audio.filename or "audio.webm")
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    if not transcript or not transcript.strip():
        return JSONResponse(content={
            "transcript": "",
            "response_text": "I couldn't hear anything. Could you try speaking again?",
            "audio_base64": "",
            "session_id": session_id,
        })

    logger.info(f"📝 Transcript: {transcript[:120]}...")

    # 3. RAG chat (reuse existing agent) -----------------------------------
    try:
        # Build a message that instructs the agent to reply in Hinglish style
        system_hint = (
            "[Voice Tutor Mode] The student is speaking aloud. "
            "Reply in simple, conversational Hinglish (mix of Hindi and English). "
            "Keep sentences short and natural. "
            "Use examples and analogies a friendly tutor would use. "
            "Do NOT use markdown formatting — plain spoken text only."
        )
        full_message = f"{system_hint}\n\nStudent says: {transcript}"

        # Ensure session_id is never None — agent requires a string
        effective_session_id = session_id or f"voice-{user_id}"

        logger.info(f"🧠 Calling agent_chat for user={user_id}, session={effective_session_id}, notebook={notebook_id}")

        response_text = await agent_chat(
            user_id=user_id,
            session_id=effective_session_id,
            message=full_message,
            notebook_id=notebook_id,
        )

        if not response_text or not response_text.strip():
            response_text = "I understood your question but couldn't generate a response. Please try again."

        logger.info(f"🧠 Agent returned {len(response_text)} chars")
    except Exception as e:
        logger.error(f"Agent chat failed: {e}", exc_info=True)
        response_text = "Sorry, I couldn't process your question. Please try again."

    # Strip any accidental markdown from the response
    response_text = response_text.replace("**", "").replace("##", "").replace("* ", "").strip()

    logger.info(f"🤖 Response: {response_text[:120]}...")

    # 4. Synthesise (TTS) --------------------------------------------------
    audio_b64 = ""
    try:
        tts_voice = voice if voice in ("alloy", "echo", "fable", "onyx", "nova", "shimmer") else "nova"
        tts_bytes = await _synthesise(response_text, voice=tts_voice)
        audio_b64 = base64.b64encode(tts_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        # Non-fatal: return text even if TTS fails

    elapsed = time.time() - start
    logger.info(f"✅ Voice turn completed in {elapsed:.1f}s")

    return JSONResponse(content={
        "transcript": transcript,
        "response_text": response_text,
        "audio_base64": audio_b64,
        "session_id": session_id,
        "processing_time": round(elapsed, 2),
    })


# ── Lightweight TTS-only endpoint (for text→speech fallback) ───────────
@router.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    voice: Optional[str] = Form("nova"),
    user_info: dict = Depends(get_current_user),
):
    """
    Convert text to speech. Returns raw MP3 audio stream.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    tts_voice = voice if voice in ("alloy", "echo", "fable", "onyx", "nova", "shimmer") else "nova"
    try:
        tts_bytes = await _synthesise(text.strip(), voice=tts_voice)
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

    return StreamingResponse(
        io.BytesIO(tts_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=response.mp3"},
    )


# ── STT-only endpoint (for transcript preview) ────────────────────────
@router.post("/stt")
async def speech_to_text(
    audio: UploadFile = File(...),
    user_info: dict = Depends(get_current_user),
):
    """Transcribe audio using Whisper. Returns plain text."""
    audio_bytes = await audio.read()
    if len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail="Audio too short")

    try:
        transcript = await _transcribe(audio_bytes, filename=audio.filename or "audio.webm")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    return {"transcript": transcript}
