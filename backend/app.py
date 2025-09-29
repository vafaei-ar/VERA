import os
import asyncio
import uuid
import time
import json
import zipfile
import io
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse, RedirectResponse
from pydantic import BaseModel

# Import our custom modules (will implement these next)
from dialog import Dialog
from asr import ASR
from tts_backends.base import TTSBackend
from tts_factory import create_tts_backend
from vad import StreamingVAD
from audio_io import WavAppendWriter

# Load configuration
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Set up logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=getattr(logging, config["logging"]["level"]),
    format=config["logging"]["format"],
    handlers=[
        logging.FileHandler(log_dir / "vera.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

# Simple in-memory cache for precompiled TTS audio
PRECOMPILED_AUDIO: Dict[Tuple[str, float, str], bytes] = {}

def _cache_key(text: str, voice: Optional[str], rate: float) -> Tuple[str, float, str]:
    return (voice or "", float(rate or 1.0), text)

def synth_cached(tts_engine: TTSBackend, text: str, voice: Optional[str], rate: float) -> bytes:
    """Return cached audio if available; otherwise synthesize (and cache if enabled)."""
    if not text:
        return b""
    key = _cache_key(text, voice, rate)
    if PRECOMPILED_AUDIO and key in PRECOMPILED_AUDIO:
        return PRECOMPILED_AUDIO[key]
    audio = tts_engine.synthesize_with_metrics(text, voice=voice, speaking_rate=rate)
    try:
        if config.get("tts", {}).get("cache_audio", True):
            PRECOMPILED_AUDIO[key] = audio
    except Exception:
        # Cache is best-effort; ignore failures
        pass
    return audio

# -------- Pause-aware synthesis helpers --------
def _split_text_with_pauses(text: str) -> list:
    """Split text into a list of parts: either dicts {type:'pause', ms:int} or {type:'text', text:str}."""
    import re
    parts = []
    idx = 0
    for m in re.finditer(r"\[pause=(\d+)\]", text):
        if m.start() > idx:
            chunk = text[idx:m.start()].strip()
            if chunk:
                parts.append({"type": "text", "text": chunk})
        try:
            ms = int(m.group(1))
        except Exception:
            ms = 0
        parts.append({"type": "pause", "ms": max(0, ms)})
        idx = m.end()
    if idx < len(text):
        tail = text[idx:].strip()
        if tail:
            parts.append({"type": "text", "text": tail})
    if not parts:
        parts = [{"type": "text", "text": text}]
    return parts

def _extract_wav_pcm(wav_bytes: bytes) -> tuple[int, int, bytes]:
    """Return (sample_rate, bytes_per_sample, pcm_bytes). Supports mono PCM16 WAV.
    If format unexpected, raise ValueError.
    """
    import struct
    if len(wav_bytes) < 44 or not wav_bytes.startswith(b"RIFF"):
        raise ValueError("Invalid WAV data")
    fmt_chunk_size = int.from_bytes(wav_bytes[16:20], "little")
    audio_format = int.from_bytes(wav_bytes[20:22], "little")
    num_channels = int.from_bytes(wav_bytes[22:24], "little")
    sample_rate = int.from_bytes(wav_bytes[24:28], "little")
    bits_per_sample = int.from_bytes(wav_bytes[34:36], "little")
    # Find 'data' subchunk
    # Minimum header up to 44 bytes, but fmt may be larger
    offset = 12
    data_offset = None
    while offset + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[offset:offset+4]
        chunk_size = int.from_bytes(wav_bytes[offset+4:offset+8], "little")
        if chunk_id == b"data":
            data_offset = offset + 8
            data_size = chunk_size
            break
        offset += 8 + chunk_size
    if data_offset is None:
        # Fallback to typical header location
        data_offset = 44
        data_size = len(wav_bytes) - 44
    if audio_format != 1 or num_channels != 1:
        raise ValueError("Only PCM mono WAV supported for concat")
    bps = bits_per_sample // 8
    pcm = wav_bytes[data_offset:data_offset+data_size]
    return sample_rate, bps, pcm

def _build_wav(pcm_bytes: bytes, sample_rate: int, bits_per_sample: int = 16, channels: int = 1) -> bytes:
    import struct
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_bytes)
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header + pcm_bytes

def _silence_pcm(duration_ms: int, sample_rate: int, bytes_per_sample: int) -> bytes:
    num_samples = int(sample_rate * (max(0, duration_ms) / 1000.0))
    return b"\x00" * (num_samples * bytes_per_sample)

def synth_with_pauses(tts_engine: TTSBackend, text: str, voice: Optional[str], rate: float) -> bytes:
    parts = _split_text_with_pauses(text)
    if len(parts) == 1 and parts[0]['type'] == 'text':
        return synth_cached(tts_engine, parts[0]['text'], voice, rate)
    wavs: list[bytes] = []
    for part in parts:
        if part['type'] == 'text':
            wavs.append(synth_cached(tts_engine, part['text'], voice, rate))
        else:
            # create tiny silent wav with 100ms to capture sample rate later, we'll convert to pcm
            # we'll handle silence during concat using determined sample rate
            wavs.append(b"")
    # Concatenate by extracting PCM and injecting silence PCM where needed
    base_sr = None
    bps = None
    pcm_acc = bytearray()
    for part, wav in zip(parts, wavs):
        if part['type'] == 'text':
            sr, bytes_per_sample, pcm = _extract_wav_pcm(wav)
            if base_sr is None:
                base_sr = sr
                bps = bytes_per_sample
            elif sr != base_sr or bytes_per_sample != bps:
                # Incompatible sample rates; fall back to plain synth
                return synth_cached(tts_engine, text, voice, rate)
            pcm_acc.extend(pcm)
        else:
            if base_sr is None:
                # No audio yet; synthesize a tiny empty to set sample rate
                temp = synth_cached(tts_engine, " ", voice, rate)
                sr, bytes_per_sample, _ = _extract_wav_pcm(temp)
                base_sr = sr
                bps = bytes_per_sample
            pcm_acc.extend(_silence_pcm(part['ms'], base_sr, bps))
    if base_sr is None or bps is None:
        return synth_cached(tts_engine, text, voice, rate)
    return _build_wav(bytes(pcm_acc), base_sr, bits_per_sample=bps*8, channels=1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting VERA application...")
    
    # Load Whisper ASR model
    logger.info("Loading Whisper ASR model...")
    global ASR_ENGINE
    ASR_ENGINE = ASR()
    logger.info("Whisper ASR model loaded successfully")
    
    # Initialize TTS engine
    global TTS_ENGINE
    TTS_ENGINE = create_tts_backend(config)
    logger.info("TTS backend initialized: %s", config.get("tts", {}).get("backend", "piper"))
    
    # Optionally precompile voices for default scenario
    try:
        tts_cfg = config.get("tts", {})
        if tts_cfg.get("precompile_on_start", False):
            scenario_file = config.get("session", {}).get("default_scenario", "default.yml")
            scenario_path = Path(__file__).parent / "scenarios" / scenario_file
            # Build greeting via Dialog to resolve variables
            dialog = Dialog(
                scenario_path=str(scenario_path),
                honorific="Mr.",
                patient_name="Patient"
            )
            voice = tts_cfg.get("voice")
            rate = float(tts_cfg.get("speaking_rate", 1.0))

            texts_to_compile = []
            # Greeting
            try:
                texts_to_compile.append(dialog.build_greeting())
            except Exception as e:
                logger.warning(f"Precompile: failed to build greeting: {e}")

            # Load YAML to collect prompts and wrapup
            with open(scenario_path, "r", encoding="utf-8") as f:
                scenario_yaml = yaml.safe_load(f)
            for item in scenario_yaml.get("flow", []) or []:
                if isinstance(item, dict) and "prompt" in item:
                    p = item.get("prompt")
                    if isinstance(p, str) and p.strip():
                        texts_to_compile.append(p.strip())
            wrapup_msg = (scenario_yaml.get("wrapup") or {}).get("message")
            if isinstance(wrapup_msg, str) and wrapup_msg.strip():
                texts_to_compile.append(wrapup_msg.strip())

            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for t in texts_to_compile:
                if t not in seen:
                    seen.add(t)
                    deduped.append(t)

            compiled = 0
            for text in deduped:
                try:
                    audio = TTS_ENGINE.synthesize_with_metrics(text, voice=voice, speaking_rate=rate)
                    if tts_cfg.get("cache_audio", True):
                        PRECOMPILED_AUDIO[_cache_key(text, voice, rate)] = audio
                    compiled += 1
                except Exception as e:
                    snippet = (text or "")[:80].replace("\n", " ")
                    logger.warning(
                        f"Precompile: failed for voice={voice or 'default'} text_len={len(text)} snippet='{snippet}' error={e}"
                    )

            logger.info(f"Precompile complete: cached {compiled}/{len(deduped)} utterances for scenario '{scenario_file}'")
    except Exception as e:
        logger.warning(f"Precompile step skipped due to error: {e}")

    logger.info("VERA application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down VERA application...")

app = FastAPI(
    title="VERA - Voice-Enabled Recovery Assistant",
    description="AI-powered post-discharge stroke care follow-up system",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global instances - will be initialized on startup
SESSIONS: Dict[str, dict] = {}
ASR_ENGINE: Optional[ASR] = None
TTS_ENGINE: Optional[TTSBackend] = None

class StartRequest(BaseModel):
    honorific: str = "Mr."
    patient_name: str = "Patient"
    scenario: str = "default.yml"
    backend: Optional[str] = None
    voice: Optional[str] = None  # Allow runtime voice selection
    rate: float = 1.0  # TTS speaking rate

class HealthResponse(BaseModel):
    status: str
    whisper_loaded: bool
    piper_available: bool
    gpu_available: bool
    message: str

@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse(static_dir / "index.html")

@app.get("/favicon.ico")
async def favicon():
    """Serve favicon (redirect to static SVG)."""
    svg_path = static_dir / "favicon.svg"
    if svg_path.exists():
        return RedirectResponse(url="/static/favicon.svg")
    # If svg is missing, return 204 to avoid 404 noise
    return JSONResponse(status_code=204, content=None)

@app.get("/health")
async def health_check() -> HealthResponse:
    """Health check endpoint to verify system status"""
    whisper_loaded = ASR_ENGINE is not None
    tts_available = TTS_ENGINE is not None
    
    # Check GPU availability
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass
    
    # Test TTS functionality if available
    tts_working = False
    tts_backend_info = "unknown"
    voice_count = 0
    if tts_available:
        try:
            # Get TTS backend health info
            tts_health = TTS_ENGINE.health()
            tts_working = tts_health.get("ok", False)
            tts_backend_info = tts_health.get("message", "unknown")
            
            # Get voice count
            voices = TTS_ENGINE.list_voices()
            voice_count = len(voices) if voices else 0
            
            # Get TTS metrics
            tts_metrics = TTS_ENGINE.get_metrics()
            
            # Test synthesis if backend is working
            if tts_working:
                test_audio = TTS_ENGINE.synthesize_with_metrics("test", voice=None)
                tts_working = len(test_audio) > 0
        except Exception as e:
            logger.warning(f"TTS health check failed: {e}")
            tts_backend_info = f"error: {str(e)}"
    
    # Test ASR functionality if available
    asr_working = False
    if whisper_loaded:
        try:
            # Create a small test audio buffer (silence)
            import numpy as np
            test_audio = np.zeros(1600, dtype=np.int16).tobytes()  # 0.1s of silence
            text, confidence = ASR_ENGINE.transcribe(test_audio)
            asr_working = True  # If no exception, ASR is working
        except Exception as e:
            logger.warning(f"ASR health check failed: {e}")
    
    status = "healthy" if whisper_loaded and tts_available and tts_working else "degraded"
    message = f"All systems operational. TTS: {tts_backend_info}, Voices: {voice_count}"
    
    # Add metrics to message if available
    if tts_available and 'tts_metrics' in locals():
        avg_time = tts_metrics.get('average_synthesis_time', 0)
        count = tts_metrics.get('synthesis_count', 0)
        message += f", Avg synthesis: {avg_time:.3f}s, Count: {count}"
    
    if not whisper_loaded:
        message = "Whisper ASR not loaded"
    elif not tts_available:
        message = "TTS backend not available"
    elif not tts_working:
        message = f"TTS synthesis not working: {tts_backend_info}"
    elif not asr_working:
        message = "ASR transcription not working"
    
    return HealthResponse(
        status=status,
        whisper_loaded=whisper_loaded,
        piper_available=tts_available,  # Keep for backward compatibility
        gpu_available=gpu_available,
        message=message
    )

@app.post("/api/start")
async def start_session(request: StartRequest):
    """Create a new session and initialize dialog"""
    if ASR_ENGINE is None or TTS_ENGINE is None:
        raise HTTPException(status_code=503, detail="Engines not initialized")

    # Per-session backend override
    # Use global TTS engine from config only (UI override removed)
    session_tts = TTS_ENGINE

    # Generate session ID
    session_id = str(uuid.uuid4())

    # Create session directory
    session_dir = Path(__file__).parent.parent / "data" / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting new session {session_id} for {request.honorific} {request.patient_name}")

    try:
        scenario_path = Path(__file__).parent / "scenarios" / request.scenario
        dialog = Dialog(
            scenario_path=str(scenario_path),
            honorific=request.honorific,
            patient_name=request.patient_name
        )

        greeting_text = dialog.build_greeting()
        logger.info(f"Session {session_id}: Greeting prepared")

        wav_writer = WavAppendWriter(
            path=str(session_dir / "full_audio.wav"),
            sample_rate=config["audio"]["sample_rate"]
        )

        # Optionally precompute greeting TTS to avoid WS idle timeouts on connect
        precomputed_greeting_audio = None
        greeting_duration = 0.0
        try:
            precomputed_greeting_audio = synth_with_pauses(
                session_tts,
                greeting_text,
                voice=request.voice or config.get("tts", {}).get("voice") or config["models"]["piper"]["default_voice"],
                rate=request.rate,
            )
            greeting_duration = session_tts.get_audio_duration(precomputed_greeting_audio)
            logger.info(
                f"Session {session_id}: Precomputed greeting audio ({len(precomputed_greeting_audio)} bytes, {greeting_duration:.2f}s)"
            )
        except Exception as e:
            logger.warning(f"Session {session_id}: Failed to precompute greeting TTS ({e}), will synthesize on WS connect")

        SESSIONS[session_id] = {
            "dialog": dialog,
            "writer": wav_writer,
            "session_dir": session_dir,
            "created": time.time(),
            "transcript": [],
            "finished": False,
            "voice": request.voice or config["models"]["piper"]["default_voice"],
            "rate": request.rate,
            "tts": session_tts,
            "precomputed_greeting_audio": precomputed_greeting_audio,
            "precomputed_greeting_duration": greeting_duration,
        }

        return JSONResponse({
            "session_id": session_id,
            "greeting": greeting_text,
            "status": "ready"
        })

    except Exception as e:
        logger.error(f"Failed to start session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize session: {str(e)}")

@app.websocket("/ws/audio/{session_id}")
async def audio_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for bidirectional audio streaming"""
    if session_id not in SESSIONS:
        await websocket.close(code=1008, reason="Session not found")
        return
    
    await websocket.accept()
    logger.info(f"WebSocket connected for session {session_id}")
    
    session_state = SESSIONS[session_id]
    dialog: Dialog = session_state["dialog"]
    writer: WavAppendWriter = session_state["writer"]
    voice = session_state["voice"]
    rate = session_state["rate"]
    tts_engine: TTSBackend = session_state.get("tts", TTS_ENGINE)

    # Initialize VAD
    vad = StreamingVAD(
        sample_rate=config["vad"]["sample_rate"],
        frame_length_ms=config["vad"]["frame_length_ms"],
        energy_threshold=config["vad"]["energy_threshold"],
        silence_duration_ms=config["vad"]["silence_duration_ms"],
        max_speech_duration_ms=config["vad"].get("max_speech_duration_ms", 15000)
    )
    
    # Small keep-alive ping to prevent idle close while preparing audio
    try:
        await websocket.send_text(json.dumps({"type": "progress_update", "progress": {"status": "connecting"}}))
    except Exception:
        pass

    # Helper: chunked audio sender to reduce frame size and improve time-to-first-byte
    async def send_audio_chunked(ws: WebSocket, audio_bytes: bytes, chunk_size: int = 65536) -> None:
        for i in range(0, len(audio_bytes), chunk_size):
            await ws.send_bytes(audio_bytes[i:i + chunk_size])
            await asyncio.sleep(0)  # yield to event loop

    # Send initial greeting
    try:
        precomputed = session_state.get("precomputed_greeting_audio")
        if precomputed:
            greeting_audio = precomputed
            greeting_duration = float(session_state.get("precomputed_greeting_duration") or 0.0)
        else:
            try:
                greeting_audio = synth_with_pauses(tts_engine, dialog.last_prompt_text, voice=voice, rate=rate)
            except Exception as e:
                logger.warning(f"Session {session_id}: Primary TTS failed ({e}); falling back to Piper if possible")
                # Fallback to Piper backend
                try:
                    fallback_cfg = dict(config)
                    fallback_cfg.setdefault('tts', {})
                    fallback_cfg['tts']['backend'] = 'piper'
                    tts_engine = create_tts_backend(fallback_cfg)
                    greeting_audio = tts_engine.synthesize_with_metrics(dialog.last_prompt_text, voice=voice, speaking_rate=rate)
                except Exception as e2:
                    raise RuntimeError(f"TTS fallback failed: {e2}")
        
        # Validate audio data before sending
        if len(greeting_audio) == 0:
            raise RuntimeError("Generated greeting audio is empty")
        
        # Validate WAV header
        if not greeting_audio.startswith(b'RIFF') or b'WAVE' not in greeting_audio[:12]:
            logger.warning(f"Session {session_id}: Generated greeting audio may not be valid WAV format")
        
        # Send as a single WS binary frame - client expects a complete WAV blob per message
        await websocket.send_bytes(greeting_audio)
        logger.info(f"Session {session_id}: Sent greeting audio ({len(greeting_audio)} bytes)")
        
        # Send greeting to transcript
        greeting_message = {
            "type": "ai_question",
            "text": dialog.last_prompt_text,
            "question_key": "greeting"
        }
        await websocket.send_text(json.dumps(greeting_message))
        logger.info(f"Session {session_id}: Sent greeting to transcript")
        
        # Calculate actual audio duration for proper timing
        if not precomputed:
            greeting_duration = tts_engine.get_audio_duration(greeting_audio)
            if greeting_duration > 0:
                logger.info(f"Session {session_id}: Greeting duration: {greeting_duration:.2f}s")
            else:
                # Fallback to estimation if duration calculation fails
                greeting_duration = tts_engine.estimate_duration(dialog.last_prompt_text, voice=voice)
                logger.info(f"Session {session_id}: Greeting duration (estimated): {greeting_duration:.2f}s")
    except Exception as e:
        logger.error(f"Session {session_id}: Failed to send greeting: {e}")
        await websocket.close(code=1011, reason="TTS error")
        return

    # Wait for greeting to complete before sending first prompt
    try:
        # Wait for greeting audio to finish playing (add grace period)
        # Periodic keep-alive while waiting for greeting to play client-side
        total_wait = max(0.5, greeting_duration + 0.3)
        waited = 0.0
        while waited < total_wait:
            try:
                await websocket.send_text(json.dumps({"type": "progress_update", "progress": {"status": "playing_greeting"}}))
            except Exception:
                pass
            step = min(1.0, total_wait - waited)
            await asyncio.sleep(step)
            waited += step
        
        # Send initial progress update (0% at start)
        progress = dialog.get_progress()
        progress_message = {
            "type": "progress_update",
            "progress": progress
        }
        await websocket.send_text(json.dumps(progress_message))
        logger.info(f"Session {session_id}: Sent initial progress update: {progress['progress_percent']:.1f}% (Q{progress['answered_questions']+1}/{progress['total_questions']})")
        
        first_prompt = dialog.next_prompt()
        if first_prompt:
            # Send AI question to transcript
            ai_question_message = {
                "type": "ai_question",
                "text": first_prompt,
                "question_key": dialog.current_key
            }
            await websocket.send_text(json.dumps(ai_question_message))
            logger.info(f"Session {session_id}: Sent AI question to transcript: {dialog.current_key}")
            
            try:
                # Keep-alive during potentially long TTS synthesis
                # Run synthesis in a thread to avoid blocking keep-alives
                loop = asyncio.get_running_loop()
                synth_task = loop.run_in_executor(None, lambda: synth_with_pauses(tts_engine, first_prompt, voice=voice, rate=rate))
                while True:
                    done = synth_task.done()
                    try:
                        await websocket.send_text(json.dumps({"type": "progress_update", "progress": {"status": "synthesizing"}}))
                    except Exception:
                        pass
                    if done:
                        break
                    await asyncio.sleep(1.0)
                prompt_audio = await synth_task
            except Exception as e:
                logger.warning(f"Session {session_id}: TTS failed for first prompt ({e}); attempting Piper fallback")
                fallback_cfg = dict(config)
                fallback_cfg.setdefault('tts', {})
                fallback_cfg['tts']['backend'] = 'piper'
                tts_engine = create_tts_backend(fallback_cfg)
                prompt_audio = synth_with_pauses(tts_engine, first_prompt, voice=voice, rate=rate)
            # Send as a single complete WAV blob
            await websocket.send_bytes(prompt_audio)
            
            # Calculate prompt duration and add grace period before listening
            prompt_duration = tts_engine.get_audio_duration(prompt_audio)
            if prompt_duration > 0:
                logger.info(f"Session {session_id}: Sent first prompt ({prompt_duration:.2f}s)")
                # Add grace period after prompt before VAD starts listening
                await asyncio.sleep(prompt_duration + 1.0)
            else:
                logger.info(f"Session {session_id}: Sent first prompt")
                # Default grace period if duration calculation fails
                await asyncio.sleep(2.0)
    except Exception as e:
        logger.error(f"Session {session_id}: Failed to send first prompt: {e}")
    
    # Audio processing loop
    pcm_buffer = bytearray()
    frames_received = 0
    
    try:
        while True:
            # Receive audio frame from client
            frame_data = await websocket.receive_bytes()
            frames_received += 1
            
            # Write to full session recording
            writer.append(frame_data)
            
            # Skip VAD processing for first few frames to avoid TTS audio interference
            if frames_received < 10:  # Skip first ~300ms of audio
                continue
            
            # Process with VAD
            speech_state = vad.accept_frame(frame_data)
            
            # Buffer audio for transcription
            pcm_buffer.extend(frame_data)
            
            # Limit buffer size to prevent memory issues (keep last 30 seconds)
            max_buffer_size = 16000 * 2 * 30  # 30 seconds of audio
            if len(pcm_buffer) > max_buffer_size:
                pcm_buffer = pcm_buffer[-max_buffer_size:]
            
            # Check for question timeout and repeat if needed
            # Handle explicit reprompt requests from dialog (e.g., unclear consent)
            reprompt_text = dialog.get_and_clear_reprompt() if hasattr(dialog, 'get_and_clear_reprompt') else None
            if reprompt_text:
                try:
                    try:
                        reprompt_audio = synth_with_pauses(tts_engine, reprompt_text, voice=voice, rate=rate)
                    except Exception as e:
                        logger.warning(f"Session {session_id}: TTS failed for reprompt ({e}); attempting Piper fallback")
                        fallback_cfg = dict(config)
                        fallback_cfg.setdefault('tts', {})
                        fallback_cfg['tts']['backend'] = 'piper'
                        tts_engine = create_tts_backend(fallback_cfg)
                        reprompt_audio = tts_engine.synthesize_with_metrics(reprompt_text, voice=voice, speaking_rate=rate)
                    await websocket.send_bytes(reprompt_audio)
                    logger.info(f"Session {session_id}: Sent reprompt")
                except Exception as e:
                    logger.error(f"Session {session_id}: Failed to send reprompt: {e}")
                    # Send text fallback
                    await websocket.send_text(json.dumps({
                        "error": "TTS Error",
                        "message": reprompt_text
                    }))

            if dialog.should_repeat_question():
                repeat_prompt = dialog.repeat_question()
                if repeat_prompt:
                    logger.info(f"Session {session_id}: Repeating question due to timeout")
                    try:
                        try:
                            repeat_audio = synth_with_pauses(tts_engine, repeat_prompt, voice=voice, rate=rate)
                        except Exception as e:
                            logger.warning(f"Session {session_id}: TTS failed for repeat ({e}); attempting Piper fallback")
                            fallback_cfg = dict(config)
                            fallback_cfg.setdefault('tts', {})
                            fallback_cfg['tts']['backend'] = 'piper'
                            tts_engine = create_tts_backend(fallback_cfg)
                            repeat_audio = tts_engine.synthesize_with_metrics(repeat_prompt, voice=voice, speaking_rate=rate)
                        await websocket.send_bytes(repeat_audio)
                        logger.info(f"Session {session_id}: Sent repeat question")
                    except Exception as e:
                        logger.error(f"Session {session_id}: Failed to send repeat: {e}")
                        # Send text fallback
                        await websocket.send_text(json.dumps({
                            "error": "TTS Error", 
                            "message": repeat_prompt
                        }))
                else:
                    # Max repeats reached; keep waiting on the same question without auto-advancing
                    logger.info(f"Session {session_id}: Max repeats reached; waiting for response to '{dialog.current_key}'")
            
            if speech_state == "finalize_utterance":
                logger.info(f"Session {session_id}: Finalizing utterance")
                logger.info(f"Session {session_id}: Buffer size: {len(pcm_buffer)} bytes")
                logger.info(f"Session {session_id}: VAD stats: {vad.get_stats()}")
                
                # Transcribe the buffered audio
                try:
                    logger.info(f"Session {session_id}: Starting ASR transcription...")
                    text, confidence = ASR_ENGINE.transcribe(bytes(pcm_buffer))
                    logger.info(f"Session {session_id}: ASR completed - Text: '{text}' (confidence: {confidence:.3f})")
                    
                    # Clear buffer
                    pcm_buffer.clear()
                    
                    # Store transcript entry
                    transcript_entry = {
                        "key": dialog.current_key,
                        "text": text,
                        "confidence": confidence,
                        "timestamp": time.time()
                    }
                    session_state["transcript"].append(transcript_entry)
                    logger.info(f"Session {session_id}: Stored transcript entry for key: {dialog.current_key}")
                    
                    # Submit answer to dialog
                    logger.info(f"Session {session_id}: Submitting answer to dialog: '{text}'")
                    dialog.submit_answer(text)
                    logger.info(f"Session {session_id}: Dialog state after answer - finished: {dialog.finished}, consent_given: {dialog.consent_given}")
                    
                    # If the question was confirm and queued a follow-up message, send it now
                    try:
                        followup = dialog.get_and_clear_post_answer() if hasattr(dialog, 'get_and_clear_post_answer') else None
                        if followup:
                            logger.info(f"Session {session_id}: Sending post-answer follow-up: '{followup[:100]}...'")
                            try:
                                follow_audio = synth_with_pauses(tts_engine, followup, voice=voice, rate=rate)
                                follow_duration = tts_engine.get_audio_duration(follow_audio)
                                logger.info(f"Session {session_id}: Follow-up audio generated ({len(follow_audio)} bytes, {follow_duration:.2f}s)")
                            except Exception as e:
                                logger.warning(f"Session {session_id}: TTS failed for follow-up ({e}); attempting Piper fallback")
                                fallback_cfg = dict(config)
                                fallback_cfg.setdefault('tts', {})
                                fallback_cfg['tts']['backend'] = 'piper'
                                tts_engine = create_tts_backend(fallback_cfg)
                                follow_audio = tts_engine.synthesize_with_metrics(followup, voice=voice, speaking_rate=rate)
                                follow_duration = tts_engine.get_audio_duration(follow_audio)
                                logger.info(f"Session {session_id}: Follow-up audio generated via fallback ({len(follow_audio)} bytes, {follow_duration:.2f}s)")
                            await websocket.send_bytes(follow_audio)
                            logger.info(f"Session {session_id}: Follow-up audio sent to client")
                            
                            # Add grace period for follow-up audio to play
                            if follow_duration > 0:
                                await asyncio.sleep(follow_duration + 0.5)
                                logger.info(f"Session {session_id}: Follow-up grace period completed")
                    except Exception as e:
                        logger.warning(f"Session {session_id}: Failed to send follow-up: {e}")

                    # Send progress update to frontend
                    progress = dialog.get_progress()
                    progress_message = {
                        "type": "progress_update",
                        "progress": progress
                    }
                    await websocket.send_text(json.dumps(progress_message))
                    logger.info(f"Session {session_id}: Sent progress update: {progress['progress_percent']:.1f}% (Q{progress['answered_questions']+1}/{progress['total_questions']})")
                    
                    # Send transcript update to frontend
                    transcript_message = {
                        "type": "transcript_update",
                        "text": text,
                        "confidence": confidence,
                        "is_final": True,
                        "question_key": dialog.current_key
                    }
                    await websocket.send_text(json.dumps(transcript_message))
                    logger.info(f"Session {session_id}: Sent transcript update: '{text}'")
                    
                    # Determine next step: if consent unclear, do not advance
                    logger.info(f"Session {session_id}: Getting next prompt...")
                    if dialog.current_key == 'consent' and not dialog.consent_given and hasattr(dialog, 'get_and_clear_reprompt'):
                        next_prompt = None
                        logger.info(f"Session {session_id}: Consent unclear, staying on current question")
                    else:
                        next_prompt = dialog.next_prompt()
                        logger.info(f"Session {session_id}: Next prompt result: {next_prompt}")
                    
                    if next_prompt is None:
                        # Only finish if dialog explicitly marked finished
                        if dialog.finished:
                            logger.info(f"Session {session_id}: Dialog completed")
                            session_state["finished"] = True
                            writer.close()
                            
                            # Send wrap-up message
                            logger.info(f"Session {session_id}: Sending wrap-up message")
                            try:
                                wrapup_audio = synth_with_pauses(tts_engine, dialog.wrapup_text, voice=voice, rate=rate)
                            except Exception as e:
                                logger.warning(f"Session {session_id}: TTS failed for wrap-up ({e}); attempting Piper fallback")
                                fallback_cfg = dict(config)
                                fallback_cfg.setdefault('tts', {})
                                fallback_cfg['tts']['backend'] = 'piper'
                                tts_engine = create_tts_backend(fallback_cfg)
                                wrapup_audio = tts_engine.synthesize_with_metrics(dialog.wrapup_text, voice=voice, speaking_rate=rate)
                            await websocket.send_bytes(wrapup_audio)
                            
                            await websocket.close(code=1000, reason="Dialog completed")
                            break
                        else:
                            # Stay on current question (e.g., unclear consent) and continue listening
                            logger.info(f"Session {session_id}: Staying on current question, continuing to listen")
                            continue
                    else:
                        # Send next prompt
                        try:
                            logger.info(f"Session {session_id}: Synthesizing next prompt...")
                            
                            # Send AI question to transcript
                            ai_question_message = {
                                "type": "ai_question",
                                "text": next_prompt,
                                "question_key": dialog.current_key
                            }
                            await websocket.send_text(json.dumps(ai_question_message))
                            logger.info(f"Session {session_id}: Sent AI question to transcript: {dialog.current_key}")
                            
                            try:
                                prompt_audio = synth_with_pauses(tts_engine, next_prompt, voice=voice, rate=rate)
                            except Exception as e:
                                logger.warning(f"Session {session_id}: TTS failed for next prompt ({e}); attempting Piper fallback")
                                fallback_cfg = dict(config)
                                fallback_cfg.setdefault('tts', {})
                                fallback_cfg['tts']['backend'] = 'piper'
                                tts_engine = create_tts_backend(fallback_cfg)
                                prompt_audio = tts_engine.synthesize_with_metrics(next_prompt, voice=voice, speaking_rate=rate)
                            
                            # Validate audio data before sending
                            if len(prompt_audio) == 0:
                                raise RuntimeError("Generated prompt audio is empty")
                            
                            await websocket.send_bytes(prompt_audio)
                            logger.info(f"Session {session_id}: Sent next prompt ({len(prompt_audio)} bytes)")
                            
                            # Send progress update for new question
                            progress = dialog.get_progress()
                            progress_message = {
                                "type": "progress_update",
                                "progress": progress
                            }
                            await websocket.send_text(json.dumps(progress_message))
                            logger.info(f"Session {session_id}: Sent progress update for new question: {progress['progress_percent']:.1f}%")
                            
                            # Reset VAD for new question
                            vad.reset_for_new_question()
                            pcm_buffer.clear()  # Clear audio buffer for new question
                            
                            # Log prompt duration and add grace period
                            prompt_duration = tts_engine.get_audio_duration(prompt_audio)
                            if prompt_duration > 0:
                                logger.info(f"Session {session_id}: Prompt duration: {prompt_duration:.2f}s")
                                # Add grace period after prompt before VAD starts listening
                                await asyncio.sleep(prompt_duration + 1.0)
                                logger.info(f"Session {session_id}: Grace period completed, VAD now active")
                            else:
                                logger.info(f"Session {session_id}: Prompt duration unknown, using default grace period")
                                # Default grace period if duration calculation fails
                                await asyncio.sleep(2.0)
                                logger.info(f"Session {session_id}: Default grace period completed, VAD now active")
                        except Exception as tts_error:
                            logger.error(f"Session {session_id}: TTS failed for next prompt: {tts_error}")
                            # Graceful degradation: send text message instead of audio
                            error_msg = "I'm having trouble speaking right now. Please continue with your response."
                            await websocket.send_text(json.dumps({
                                "type": "text_message",
                                "message": error_msg,
                                "prompt": next_prompt  # Include the original prompt as text
                            }))
                            logger.info(f"Session {session_id}: Sent text fallback for failed TTS")
                
                except Exception as e:
                    logger.error(f"Session {session_id}: ASR/TTS error: {e}")
                    await websocket.send_text(json.dumps({
                        "error": "Processing error",
                        "message": str(e)
                    }))
    
    except WebSocketDisconnect:
        logger.info(f"Session {session_id}: WebSocket disconnected")
        writer.close()
    except Exception as e:
        logger.error(f"Session {session_id}: WebSocket error: {e}")
        writer.close()
        await websocket.close(code=1011, reason=f"Server error: {str(e)}")

@app.get("/api/download/{session_id}")
async def download_transcript(session_id: str):
    """Download session transcript and audio as ZIP file"""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_state = SESSIONS[session_id]
    session_dir = session_state["session_dir"]
    
    if not session_state["finished"]:
        raise HTTPException(status_code=400, detail="Session not yet completed")
    
    logger.info(f"Preparing download for session {session_id}")
    
    # Create ZIP file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add transcript JSON
        transcript_json = {
            "session_id": session_id,
            "created": session_state["created"],
            "completed": time.time(),
            "dialog_scenario": session_state["dialog"].scenario_name,
            "transcript": session_state["transcript"]
        }
        
        zip_file.writestr("transcript.json", json.dumps(transcript_json, indent=2))
        
        # Add human-readable transcript
        transcript_text = f"VERA Session Transcript\\n"
        transcript_text += f"Session ID: {session_id}\\n"
        transcript_text += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_state['created']))}\\n\\n"
        
        for entry in session_state["transcript"]:
            transcript_text += f"Q: {entry['key']}\\n"
            transcript_text += f"A: {entry['text']} (confidence: {entry['confidence']:.3f})\\n\\n"
        
        zip_file.writestr("transcript.txt", transcript_text)
        
        # Add full audio file if it exists
        audio_path = session_dir / "full_audio.wav"
        if audio_path.exists():
            zip_file.write(audio_path, "full_audio.wav")
    
    zip_buffer.seek(0)
    
    return StreamingResponse(
        io.BytesIO(zip_buffer.read()),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=vera_session_{session_id}.zip"}
    )

@app.get("/api/voices")
async def list_voices():
    try:
        voices = TTS_ENGINE.list_voices() if TTS_ENGINE else []
        return JSONResponse({"backend": config.get("tts", {}).get("backend", "piper"), "voices": voices})
    except Exception as e:
        logger.error("Failed to list voices: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config["app"]["host"],
        port=config["app"]["port"],
        log_level=config["logging"]["level"].lower()
    )

