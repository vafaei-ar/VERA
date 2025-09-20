import os
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

app = FastAPI(title="VERA - Voice-Enabled Recovery Assistant", version="1.0.0")

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

@app.on_event("startup")
async def startup_event():
    """Initialize models and engines on startup"""
    global ASR_ENGINE, TTS_ENGINE
    
    logger.info("Starting VERA application...")
    
    try:
        # Initialize ASR engine
        logger.info("Loading Whisper ASR model...")
        ASR_ENGINE = ASR(
            model_size=config["models"]["whisper"]["model_size"],
            device=config["models"]["whisper"]["device"],
            compute_type=config["models"]["whisper"]["compute_type"],
            device_index=config["models"]["whisper"]["device_index"]
        )
        logger.info("Whisper ASR model loaded successfully")
        
        # Initialize TTS engine
        logger.info("Initializing Piper TTS engine...")
        TTS_ENGINE = TTS(
            binary_path=config["models"]["piper"]["binary_path"],
            voice_dir=config["models"]["piper"]["voice_dir"],
            default_voice=config["models"]["piper"]["default_voice"],
            speaking_rate=config["models"]["piper"]["speaking_rate"],
            noise_scale=config["models"]["piper"]["noise_scale"],
            noise_w=config["models"]["piper"]["noise_w"]
        )
        logger.info("Piper TTS engine initialized successfully")
        
        logger.info("VERA application startup complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize engines: {e}")
        raise

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
    
    status = "healthy" if whisper_loaded and piper_available else "degraded"
    message = "All systems operational"
    
    if not whisper_loaded:
        message = "Whisper ASR not loaded"
    elif not piper_available:
        message = "Piper TTS not available"
    
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
        await websocket.send_bytes(greeting_audio)
        logger.info(f"Session {session_id}: Sent greeting audio")
    except Exception as e:
        logger.error(f"Session {session_id}: Failed to send greeting: {e}")
        await websocket.close(code=1011, reason="TTS error")
        return
    
    # Audio processing loop
    pcm_buffer = bytearray()
    
    try:
        while True:
            # Receive audio frame from client
            frame_data = await websocket.receive_bytes()
            
            # Write to full session recording
            writer.append(frame_data)
            
            # Process with VAD
            speech_state = vad.accept_frame(frame_data)
            
            # Buffer audio for transcription
            pcm_buffer.extend(frame_data)
            
            # Limit buffer size to prevent memory issues (keep last 30 seconds)
            max_buffer_size = 16000 * 2 * 30  # 30 seconds of audio
            if len(pcm_buffer) > max_buffer_size:
                pcm_buffer = pcm_buffer[-max_buffer_size:]
            
            # Check for question timeout and repeat if needed
            if dialog.should_repeat_question():
                repeat_prompt = dialog.repeat_question()
                if repeat_prompt:
                    logger.info(f"Session {session_id}: Repeating question due to timeout")
                    try:
                        repeat_audio = TTS_ENGINE.synthesize(repeat_prompt, voice=voice, speaking_rate=rate)
                        await websocket.send_bytes(repeat_audio)
                    except Exception as e:
                        logger.error(f"Session {session_id}: Failed to send repeat: {e}")
                else:
                    # Max repeats reached, move to next question
                    logger.info(f"Session {session_id}: Max repeats reached, moving to next question")
                    next_prompt = dialog.next_prompt()
                    if next_prompt:
                        try:
                            prompt_audio = TTS_ENGINE.synthesize(next_prompt, voice=voice, speaking_rate=rate)
                            await websocket.send_bytes(prompt_audio)
                        except Exception as e:
                            logger.error(f"Session {session_id}: Failed to send next prompt: {e}")
                    else:
                        # Conversation finished
                        session_state["finished"] = True
                        wrapup_audio = TTS_ENGINE.synthesize(dialog.wrapup_text, voice=voice, speaking_rate=rate)
                        await websocket.send_bytes(wrapup_audio)
                        break
            
            if speech_state == "finalize_utterance":
                logger.info(f"Session {session_id}: Finalizing utterance")
                logger.info(f"Session {session_id}: Buffer size: {len(pcm_buffer)} bytes")
                
                # Transcribe the buffered audio
                try:
                    text, confidence = ASR_ENGINE.transcribe(bytes(pcm_buffer))
                    logger.info(f"Session {session_id}: Transcribed: '{text}' (confidence: {confidence:.3f})")
                    
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
                    
                    # Submit answer to dialog
                    logger.info(f"Session {session_id}: Submitting answer to dialog: '{text}'")
                    dialog.submit_answer(text)
                    
                    # Get next prompt
                    logger.info(f"Session {session_id}: Getting next prompt...")
                    next_prompt = dialog.next_prompt()
                    logger.info(f"Session {session_id}: Next prompt result: {next_prompt}")
                    
                    if next_prompt is None:
                        # Dialog finished
                        logger.info(f"Session {session_id}: Dialog completed")
                        session_state["finished"] = True
                        writer.close()
                        
                        # Send wrap-up message
                        wrapup_audio = TTS_ENGINE.synthesize(dialog.wrapup_text, voice=voice, speaking_rate=rate)
                        await websocket.send_bytes(wrapup_audio)
                        
                        await websocket.close(code=1000, reason="Dialog completed")
                        break
                    else:
                        # Send next prompt
                        prompt_audio = TTS_ENGINE.synthesize(next_prompt, voice=voice, speaking_rate=rate)
                        await websocket.send_bytes(prompt_audio)
                        logger.info(f"Session {session_id}: Sent next prompt")
                
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

