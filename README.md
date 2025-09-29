# VERA - Voice-Enabled Recovery Assistant

VERA is an AI-powered post-discharge stroke care follow-up system that conducts structured voice interviews with patients to assess their recovery progress and care needs.

## Features

- **Voice-First Interface**: Natural conversation flow with AI-powered speech recognition and synthesis
- **Structured Assessment**: Comprehensive post-stroke follow-up questionnaire covering:
  - General well-being and symptoms
  - Medication adherence and side effects
  - Follow-up care appointments
  - Lifestyle management
  - Daily activities and support needs
- **Local Processing**: Runs entirely on local infrastructure with no external API dependencies
- **Privacy-First**: All audio processing and storage happens locally
- **Multi-Voice Support**: Choice of AI voices for personalized experience
- **Multiple TTS Backends**: Support for Piper (local), Kokoro, VibeVoice, and ResembleAI Chatterbox
- **Complete Transcripts**: Downloadable session transcripts with audio recordings
- **Web-Based**: Simple browser interface accessible on any device

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 12GB+ VRAM (for Whisper large-v3)
- **RAM**: 16GB+ system RAM recommended
- **Storage**: 10GB+ free space for models and sessions
- **Network**: Local network access for browser clients

### Software
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.11+ (3.13+ recommended)
- **Conda**: Miniconda or Anaconda
- **CUDA**: 11.8+ with compatible drivers
- **Browser**: Chrome 88+ or Edge 88+ (for WebRTC support)

## Quick Setup

### Automated Installation
```bash
# Clone the repository
git clone https://github.com/your-org/VERA.git
cd VERA

# Run the setup script (this handles everything!)
./setup.sh
```

The setup script will:
- ✅ Create conda environment with Python 3.11
- ✅ Install all system dependencies (ffmpeg, CUDA support)
- ✅ Install Python packages from requirements.txt
- ✅ Download Piper TTS binary and voices
- ✅ **Optionally install additional TTS backends** (Kokoro, VibeVoice, Chatterbox)
- ✅ Set up logging and data directories
- ✅ Verify the installation

### Manual Installation (Alternative)
If you prefer manual setup or the script fails:

```bash
# 1. Clone repository
git clone https://github.com/your-org/VERA.git
cd VERA

# 2. Create conda environment
conda create -n vera python=3.11
conda activate vera

# 3. Install dependencies
pip install -r requirements.txt
conda install -c conda-forge ffmpeg

# 4. Download models and voices
cd backend
python download_models.py
```

### What Gets Downloaded
- **Whisper large-v3 model** (~3GB) - Downloaded on first run
- **Piper TTS voices** (~150MB total):
  - Amy (Female, Medium Quality)
  - Lessac (Female, Medium Quality) 
  - Ryan (Male, High Quality)
- **Piper TTS binary** (~50MB) - Linux x86_64 version

### Optional TTS Backends (Interactive Setup)
The setup script will ask if you want to install additional TTS backends:

- **Kokoro TTS** (~500MB) - Neural TTS with multiple speakers
- **VibeVoice TTS** (~1.5GB) - High-quality Microsoft TTS (requires 8GB+ VRAM)
- **ResembleAI Chatterbox** - Cloud-based TTS (requires API key)

## Configuration

### TTS Backend Selection
VERA supports multiple Text-to-Speech backends. Edit `backend/config.yaml` to choose your preferred backend:

```yaml
tts:
  backend: "piper"  # piper | kokoro | vibevoice | chatterbox
```

#### Available TTS Backends

**1. Piper (Default - Recommended)**
- ✅ **Local processing** - No internet required
- ✅ **Fast synthesis** - Optimized for real-time use
- ✅ **Multiple voices** - Amy, Lessac, Ryan voices included
- ✅ **Low resource usage** - Works on CPU-only systems
- ❌ **Limited voice variety** - Fixed set of pre-trained voices

**2. Kokoro TTS (ONNX)**
- ✅ **Local processing** - No internet required after download
- ✅ **High quality** - Neural TTS with excellent quality
- ✅ **Multiple speakers** - 11 voices (American/British, Male/Female)
- ✅ **ONNX optimized** - Fast inference with ONNX Runtime
- ✅ **Quantized models** - Smaller file sizes for better performance
- ❌ **Large download** - ~400MB for model + voices
- ❌ **Complex setup** - Requires phoneme tokenization (currently mock implementation)

**3. VibeVoice (Experimental)**
- ✅ **High quality** - Microsoft's advanced TTS model
- ✅ **Multiple speakers** - Rich voice variety
- ❌ **High VRAM requirement** - Needs 8GB+ VRAM
- ❌ **Language limited** - English and Chinese only
- ❌ **Slow loading** - Large model takes time to load

**4. ResembleAI Chatterbox**
- ✅ **Professional quality** - Commercial-grade TTS
- ✅ **Custom voices** - Can use your own voice models
- ❌ **Requires API key** - Needs ResembleAI subscription
- ❌ **Internet required** - No offline processing
- ❌ **Cost** - Pay-per-use pricing

### Voice Selection
Edit `backend/config.yaml` to configure available voices for each backend:

```yaml
models:
  piper:
    default_voice: "en_US-amy-medium"
    available_voices:
      - "en_US-amy-medium"
      - "en_US-lessac-medium"
      - "en_US-ryan-high"
  
  kokoro:
    model: "onnx-community/Kokoro-82M-ONNX"
    default_voice: "af"  # Default American Female voice
    available_voices:
      - "af"           # Default American Female
      - "af_bella"     # Bella - American Female
      - "af_nicole"    # Nicole - American Female
      - "af_sarah"     # Sarah - American Female
      - "af_sky"       # Sky - American Female
      - "am_adam"      # Adam - American Male
      - "am_michael"   # Michael - American Male
      - "bf_emma"      # Emma - British Female
      - "bf_isabella"  # Isabella - British Female
      - "bm_george"    # George - British Male
      - "bm_lewis"     # Lewis - British Male
  
  vibevoice:
    model: "microsoft/VibeVoice-1.5B"
  
  chatterbox:
    api_key: null  # Set RESEMBLE_API_KEY environment variable
```

### Changing TTS Backends
To switch between different TTS backends:

1. **Edit Configuration**: Modify `backend/config.yaml`:
   ```yaml
   tts:
     backend: "kokoro"  # Change to desired backend
   ```

2. **Restart Server**: Stop and restart the VERA server:
   ```bash
   # Stop server (Ctrl+C)
   cd backend
   python app.py
   ```

3. **Verify Health**: Check `/health` endpoint to confirm backend is working:
   ```bash
   curl http://localhost:8000/health
   ```

**Backend Requirements:**
- **Piper**: No additional setup (default)
- **Kokoro**: Requires `onnxruntime` package and model download (run `python download_models.py`)
- **VibeVoice**: Requires 8GB+ VRAM and `transformers` package
- **Chatterbox**: Requires `RESEMBLE_API_KEY` environment variable

### Changing Kokoro Voices
To change the Kokoro voice, edit `backend/config.yaml`:

```yaml
models:
  kokoro:
    default_voice: "am_adam"  # Change to desired voice
```

**Available Kokoro Voices:**
- **American Female**: `af`, `af_bella`, `af_nicole`, `af_sarah`, `af_sky`
- **American Male**: `am_adam`, `am_michael`
- **British Female**: `bf_emma`, `bf_isabella`
- **British Male**: `bm_george`, `bm_lewis`

**Note**: Currently, Kokoro uses a mock implementation that generates silence. Full phoneme tokenization and voice synthesis will be implemented in future versions.

### Scenario Customization
Modify `backend/scenarios/default.yml` to customize:
- Greeting message and organization details
- Assessment questions and flow
- Consent handling and emergency disclaimers

### System Tuning
Adjust `backend/config.yaml` for your hardware:

```yaml
models:
  whisper:
    model_size: "large-v3"  # or "medium.en" for lower VRAM
    device_index: 0         # GPU index for multi-GPU systems

vad:
  energy_threshold: 200     # Adjust for microphone sensitivity
  silence_duration_ms: 900  # Time to wait before finalizing speech
```

## Running VERA

### Start the Server
```bash
conda activate vera
cd backend
python app.py
```

The server will start on `http://localhost:8000`

### Access the Interface
1. Open Chrome or Edge browser
2. Navigate to `http://YOUR_SERVER_IP:8000`
3. Allow microphone access when prompted
4. Fill in patient information and start the call

### Network Access
For LAN access, the server binds to `0.0.0.0:8000` by default. Clients can access via:
- `http://192.168.1.100:8000` (replace with actual server IP)
- Ensure firewall allows port 8000

## Usage Guide

### Starting a Session
1. **Patient Information**: Enter title (Mr./Ms.) and patient name
2. **Voice Selection**: Choose preferred AI voice
3. **Start Call**: Click to begin the assessment
4. **Microphone**: Allow browser microphone access

### During the Call
- **Listen**: AI will greet patient and explain the process
- **Respond**: Speak naturally when prompted
- **Wait**: Allow brief pauses for speech processing
- **Emergency**: End call immediately if medical emergency

### After Completion
- **Download**: Get transcript package (ZIP file contains):
  - `transcript.json` - Structured responses with timestamps
  - `transcript.txt` - Human-readable summary
  - `full_audio.wav` - Complete call recording

## Troubleshooting

### Common Issues

**"System unavailable" on health check**
- Verify conda environment is activated
- Check GPU memory: `nvidia-smi`
- Restart server if models failed to load

**"Microphone access denied"**
- Use HTTPS or localhost for microphone access
- Check browser permissions in Settings
- Try different browser (Chrome/Edge recommended)

**"Connection timeout" during call**
- Check server logs for errors
- Verify WebSocket support in browser
- Ensure stable network connection

**Poor speech recognition**
- Adjust microphone levels
- Reduce background noise
- Speak clearly with brief pauses
- Check VAD sensitivity in config

**TTS Backend Issues**

**"Kokoro dependencies not installed"**
- Run setup script and select Kokoro installation
- Or manually install: `pip install transformers accelerate`

**"VibeVoice model loading failed"**
- Ensure you have 8GB+ VRAM available
- Check GPU memory: `nvidia-smi`
- Try reducing model precision in config
- Fall back to Piper if insufficient VRAM

**"Chatterbox API connection failed"**
- Set `RESEMBLE_API_KEY` environment variable
- Verify API key is valid and has credits
- Check internet connection
- Fall back to local backend if needed

**"TTS synthesis not working"**
- Check `/health` endpoint for backend status
- Verify selected backend is properly installed
- Check server logs for specific error messages
- Try switching to Piper backend as fallback

### Performance Optimization

**High GPU Memory Usage**
```yaml
models:
  whisper:
    model_size: "medium.en"  # Use smaller model
    compute_type: "int8"     # Reduce precision
```

**Slow Response Times**
- Ensure CUDA is properly installed
- Check GPU utilization: `nvidia-smi`
- Consider faster SSD for model storage
- Reduce VAD silence duration for quicker responses

### Logs and Debugging

Server logs are written to:
- `logs/vera.log` - Application logs
- `data/sessions/{session_id}/` - Per-session data

Enable debug logging:
```yaml
logging:
  level: "DEBUG"
```

### Health Monitoring

Check system status via the health endpoint:
```bash
curl http://localhost:8000/health
```

The health endpoint reports:
- **TTS Backend Status**: Active backend and voice count
- **ASR Status**: Whisper model loading status
- **GPU Availability**: CUDA support detection
- **Synthesis Metrics**: Performance statistics (count, timing)
- **Overall Status**: Healthy/Degraded based on component availability

Example health response:
```json
{
  "status": "healthy",
  "whisper_loaded": true,
  "piper_available": true,
  "gpu_available": true,
  "message": "All systems operational. TTS: Piper ready with 3 voices, Voices: 3, Avg synthesis: 0.234s, Count: 15"
}
```

## Security Considerations

### Data Privacy
- All processing occurs locally
- No data sent to external services
- Audio files stored in `data/sessions/`
- Consider encryption for sensitive deployments

### Network Security
- Run on isolated network for PHI compliance
- Use HTTPS in production (requires SSL certificates)
- Implement authentication if needed
- Regular security updates for dependencies

### PHI Compliance
- Configure appropriate data retention policies
- Implement access controls as needed
- Consider de-identification of stored data
- Follow organizational HIPAA guidelines

## Development

### Project Structure
```
VERA/
├── backend/
│   ├── app.py                    # FastAPI application
│   ├── config.yaml               # Configuration
│   ├── asr.py                   # Speech recognition
│   ├── tts.py                   # Legacy TTS (deprecated)
│   ├── tts_factory.py           # TTS backend factory
│   ├── tts_backends/            # TTS backend implementations
│   │   ├── base.py              # Abstract base class
│   │   ├── piper_backend.py     # Piper TTS backend
│   │   ├── kokoro_backend.py    # Kokoro TTS backend
│   │   ├── vibevoice_backend.py # VibeVoice TTS backend
│   │   └── chatterbox_backend.py # ResembleAI backend
│   ├── vad.py                   # Voice activity detection
│   ├── dialog.py                # Conversation flow
│   ├── audio_io.py              # Audio processing
│   ├── download_models.py       # Model management
│   ├── scenarios/               # Conversation scripts
│   ├── static/                  # Web interface
│   └── models/                  # Downloaded models
├── data/
│   └── sessions/                # Session recordings
├── logs/                        # Application logs
└── requirements.txt             # Python dependencies
```

### Adding New Voices

**For Piper Backend:**
1. Download voice files to `backend/models/piper/`
2. Add to `config.yaml` available voices list
3. Restart server to detect new voices

**For Other Backends:**
- **Kokoro**: Voices are automatically discovered from the model
- **VibeVoice**: Voices are predefined by the model
- **Chatterbox**: Voices are managed through ResembleAI dashboard

### Customizing Scenarios
1. Copy `default.yml` to new scenario file
2. Modify questions, flow, and metadata
3. Add to frontend dropdown in `index.html`

## Docker Deployment (Optional)

```bash
# Build container
docker build -t vera .

# Run with GPU support
docker run --gpus all -p 8000:8000 -v ./data:/app/data vera
```

## Support

### Getting Help
- Check logs in `logs/vera.log`
- Review configuration in `backend/config.yaml`
- Test individual components (ASR, TTS, VAD)
- Verify model files are downloaded correctly

### Known Limitations
- Requires NVIDIA GPU for optimal performance
- Chrome/Edge browsers only (WebRTC requirements)
- **Language Support**:
  - Piper: English only
  - Kokoro: English (with some multilingual support)
  - VibeVoice: English and Chinese only
  - Chatterbox: Depends on configured voices
- Single concurrent session per server instance
- VibeVoice requires 8GB+ VRAM
- Chatterbox requires internet connection and API key

### Contributing
- Follow Python PEP 8 style guidelines
- Add tests for new features
- Update documentation for changes
- Ensure HIPAA compliance considerations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Whisper**: OpenAI's speech recognition model
- **Piper**: Rhasspy's neural text-to-speech
- **FastAPI**: Modern Python web framework
- **Silero VAD**: Voice activity detection models

---

**Important Medical Disclaimer**: VERA is a technology demonstration for post-discharge follow-up assessment. It is not intended to replace professional medical care. Patients experiencing medical emergencies should call 911 immediately.