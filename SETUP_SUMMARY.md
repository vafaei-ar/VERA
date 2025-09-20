# VERA Repository Setup Summary

## What's Included in Git

### Core Application Files
- `backend/app.py` - Main FastAPI application
- `backend/asr.py` - Speech recognition (Whisper)
- `backend/tts.py` - Text-to-speech (Piper)
- `backend/vad.py` - Voice activity detection
- `backend/dialog.py` - Conversation flow management
- `backend/audio_io.py` - Audio processing utilities
- `backend/config.yaml` - Configuration file
- `backend/scenarios/default.yml` - Default conversation scenario

### Frontend Files
- `backend/static/index.html` - Web interface
- `backend/static/app.js` - Frontend JavaScript
- `backend/static/pcm-worklet.js` - Audio processing worklet
- `backend/static/styles.css` - Styling
- `backend/static/favicon.svg` - Favicon

### Setup and Configuration
- `setup.sh` - Automated setup script
- `requirements.txt` - Python dependencies
- `README.md` - Comprehensive documentation
- `.gitignore` - Git ignore rules
- `.gitattributes` - Line ending handling

### Directory Structure
- `backend/models/.gitkeep` - Ensures models directory exists
- `data/.gitkeep` - Ensures data directory exists
- `logs/.gitkeep` - Ensures logs directory exists

## What's Excluded from Git (Downloaded by Setup)

### Large Model Files
- `backend/models/piper/*.onnx` - Piper TTS voice models (~150MB)
- `backend/models/piper/*.onnx.json` - Piper voice configurations
- `backend/piper/` - Piper TTS binary and libraries (~50MB)
- `backend/models/whisper/` - Whisper ASR models (~3GB, downloaded on first run)

### Runtime Data
- `data/sessions/*` - Session recordings and transcripts
- `logs/*.log` - Application logs
- `backend/__pycache__/` - Python cache files

## For New Users

### Quick Start
```bash
git clone <repository-url>
cd VERA
./setup.sh
```

### What the Setup Script Does
1. ✅ Creates conda environment `vera` with Python 3.10
2. ✅ Installs system dependencies (ffmpeg, CUDA support)
3. ✅ Installs Python packages from `requirements.txt`
4. ✅ Downloads Piper TTS binary and voice models
5. ✅ Sets up logging and data directories
6. ✅ Verifies installation with test script

### Manual Setup (Alternative)
```bash
conda create -n vera python=3.10
conda activate vera
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
cd backend
python download_models.py
```

## File Sizes

### In Git Repository
- **Total**: ~500KB (source code only)
- **Core Python files**: ~50KB
- **Frontend files**: ~20KB
- **Documentation**: ~15KB
- **Configuration**: ~5KB

### Downloaded by Setup
- **Piper TTS binary**: ~50MB
- **Piper voices**: ~150MB
- **Whisper models**: ~3GB (on first run)

## Benefits of This Approach

1. **Small Repository**: Only source code in git (~500KB)
2. **Easy Setup**: One command setup for new users
3. **Reproducible**: Same models downloaded every time
4. **Version Control**: Source code changes tracked properly
5. **No Large Files**: Avoids git LFS or large binary files
6. **Clean History**: No model updates cluttering git history

## Next Steps

1. **Commit and Push**: `git commit -m "Initial VERA implementation" && git push`
2. **Test Setup**: Try the setup script on a clean system
3. **Documentation**: Update repository URL in README
4. **CI/CD**: Consider adding automated testing

## Notes

- The setup script is designed for Linux systems
- Windows users may need to modify the script for their environment
- All model downloads happen automatically
- No manual intervention required after running `./setup.sh`

