import os
import asyncio
import uuid
import time
import json
import zipfile
import io
import logging
from pathlib import Path
from typing import Dict, Optional

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse, RedirectResponse
from pydantic import BaseModel

# Import our custom modules (will implement these next)
from dialog import Dialog
from asr import ASR
from tts import TTS
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
TTS_ENGINE: Optional[TTS] = None

class StartRequest(BaseModel):
    honorific: str = "Mr."
    patient_name: str = "Patient"
    scenario: str = "default.yml"
    voice: Optional[str] = None  # Allow runtime voice selection
    rate: float = 1.0  # TTS speaking rate

class HealthResponse(BaseModel):
    status: str
    whisper_loaded: bool
    piper_available: bool
    gpu_available: bool
    message: str

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting VERA application...")
    
    # Load Whisper ASR model
    logger.info("Loading Whisper ASR model...")
    global ASR_ENGINE
    ASR_ENGINE = WhisperASR()
    logger.info("Whisper ASR model loaded successfully")
    
    # Initialize TTS engine
    logger.info("Initializing Piper TTS engine...")
    global TTS_ENGINE
    TTS_ENGINE = TTS()
    logger.info("Piper TTS engine initialized successfully")
    
    logger.info("VERA application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down VERA application...")

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
    piper_available = TTS_ENGINE is not None
    
    # Check GPU availability
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass
    
    # Test TTS functionality if available
    tts_working = False
    if piper_available:
        try:
            test_audio = TTS_ENGINE.synthesize("test", voice=None)
            tts_working = len(test_audio) > 0
        except Exception as e:
            logger.warning(f"TTS health check failed: {e}")
    
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
    
    status = "healthy" if whisper_loaded and piper_available and tts_working else "degraded"
    message = "All systems operational"
    
    if not whisper_loaded:
        message = "Whisper ASR not loaded"
    elif not piper_available:
        message = "Piper TTS not available"
    elif not tts_working:
        message = "TTS synthesis not working"
    elif not asr_working:
        message = "ASR transcription not working"
    
    return HealthResponse(
        status=status,
        whisper_loaded=whisper_loaded,
        piper_available=piper_available,
        gpu_available=gpu_available,
        message=message
    )

@app.post("/api/start")
async def start_session(request: StartRequest):
    """Create a new session and initialize dialog"""
    if ASR_ENGINE is None or TTS_ENGINE is None:
        raise HTTPException(status_code=503, detail="Engines not initialized")
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Create session directory
    session_dir = Path(__file__).parent.parent / "data" / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting new session {session_id} for {request.honorific} {request.patient_name}")
    
    try:
        # Initialize dialog
        scenario_path = Path(__file__).parent / "scenarios" / request.scenario
        dialog = Dialog(
            scenario_path=str(scenario_path),
            honorific=request.honorific,
            patient_name=request.patient_name
        )
        
        # Build greeting text
        greeting_text = dialog.build_greeting()
        logger.info(f"Session {session_id}: Greeting prepared")
        
        # Initialize WAV writer for full session recording
        wav_writer = WavAppendWriter(
            path=str(session_dir / "full_audio.wav"),
            sample_rate=config["audio"]["sample_rate"]
        )
        
        # Store session state
        SESSIONS[session_id] = {
            "dialog": dialog,
            "writer": wav_writer,
            "session_dir": session_dir,
            "created": time.time(),
            "transcript": [],
            "finished": False,
            "voice": request.voice or config["models"]["piper"]["default_voice"],
            "rate": request.rate
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
    
    # Initialize VAD
    vad = StreamingVAD(
        sample_rate=config["vad"]["sample_rate"],
        frame_length_ms=config["vad"]["frame_length_ms"],
        energy_threshold=config["vad"]["energy_threshold"],
        silence_duration_ms=config["vad"]["silence_duration_ms"],
        max_speech_duration_ms=config["vad"].get("max_speech_duration_ms", 15000)
    )
    
    # Send initial greeting
    try:
        greeting_audio = TTS_ENGINE.synthesize(dialog.last_prompt_text, voice=voice, speaking_rate=rate)
        
        # Validate audio data before sending
        if len(greeting_audio) == 0:
            raise RuntimeError("Generated greeting audio is empty")
        
        # Validate WAV header
        if not greeting_audio.startswith(b'RIFF') or b'WAVE' not in greeting_audio[:12]:
            logger.warning(f"Session {session_id}: Generated greeting audio may not be valid WAV format")
        
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
        greeting_duration = TTS_ENGINE.get_audio_duration(greeting_audio)
        if greeting_duration > 0:
            logger.info(f"Session {session_id}: Greeting duration: {greeting_duration:.2f}s")
        else:
            # Fallback to estimation if duration calculation fails
            greeting_duration = TTS_ENGINE.estimate_duration(dialog.last_prompt_text, voice=voice)
            logger.info(f"Session {session_id}: Greeting duration (estimated): {greeting_duration:.2f}s")
    except Exception as e:
        logger.error(f"Session {session_id}: Failed to send greeting: {e}")
        await websocket.close(code=1011, reason="TTS error")
        return

    # Wait for greeting to complete before sending first prompt
    try:
        # Wait for greeting audio to finish playing (add grace period)
        await asyncio.sleep(max(0.5, greeting_duration + 1.0))  # Increased grace period
        
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
            
            prompt_audio = TTS_ENGINE.synthesize(first_prompt, voice=voice, speaking_rate=rate)
            await websocket.send_bytes(prompt_audio)
            
            # Calculate prompt duration and add grace period before listening
            prompt_duration = TTS_ENGINE.get_audio_duration(prompt_audio)
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
                    reprompt_audio = TTS_ENGINE.synthesize(reprompt_text, voice=voice, speaking_rate=rate)
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
                        repeat_audio = TTS_ENGINE.synthesize(repeat_prompt, voice=voice, speaking_rate=rate)
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
                            wrapup_audio = TTS_ENGINE.synthesize(dialog.wrapup_text, voice=voice, speaking_rate=rate)
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
                            
                            prompt_audio = TTS_ENGINE.synthesize(next_prompt, voice=voice, speaking_rate=rate)
                            
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
                            prompt_duration = TTS_ENGINE.get_audio_duration(prompt_audio)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config["app"]["host"],
        port=config["app"]["port"],
        log_level=config["logging"]["level"].lower()
    )

