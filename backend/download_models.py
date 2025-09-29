#!/usr/bin/env python3
"""
Model download script for VERA
Downloads Whisper, Piper, and Kokoro models needed for the application
"""

import os
import sys
import requests
import tarfile
import tempfile
import logging
import json
from pathlib import Path
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Piper voice download URLs from HuggingFace
PIPER_VOICES = {
    "en_US-amy-medium": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
        "description": "US English, female voice (Amy), medium quality"
    },
    "en_US-lessac-medium": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        "description": "US English, female voice (Lessac), medium quality"
    },
    "en_US-ryan-high": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx.json",
        "description": "US English, male voice (Ryan), high quality"
    }
}

# Kokoro ONNX model and voice download URLs from HuggingFace
KOKORO_MODEL = {
    "model_url": "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/onnx/model.onnx",
    "model_fp16_url": "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/onnx/model_fp16.onnx",
    "model_quantized_url": "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/onnx/model_quantized.onnx",
    "description": "Kokoro-82M ONNX TTS model"
}

KOKORO_VOICES = {
    "af": {
        "url": "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/voices/af.bin",
        "description": "Default American Female voice"
    },
    "af_bella": {
        "url": "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/voices/af_bella.bin",
        "description": "Bella - American Female voice"
    },
    "af_nicole": {
        "url": "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/voices/af_nicole.bin",
        "description": "Nicole - American Female voice"
    },
    "af_sarah": {
        "url": "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/voices/af_sarah.bin",
        "description": "Sarah - American Female voice"
    },
    "af_sky": {
        "url": "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/voices/af_sky.bin",
        "description": "Sky - American Female voice"
    },
    "am_adam": {
        "url": "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/voices/am_adam.bin",
        "description": "Adam - American Male voice"
    },
    "am_michael": {
        "url": "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/voices/am_michael.bin",
        "description": "Michael - American Male voice"
    },
    "bf_emma": {
        "url": "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/voices/bf_emma.bin",
        "description": "Emma - British Female voice"
    },
    "bf_isabella": {
        "url": "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/voices/bf_isabella.bin",
        "description": "Isabella - British Female voice"
    },
    "bm_george": {
        "url": "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/voices/bm_george.bin",
        "description": "George - British Male voice"
    },
    "bm_lewis": {
        "url": "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/voices/bm_lewis.bin",
        "description": "Lewis - British Male voice"
    }
}

def download_file(url: str, dest_path: Path) -> bool:
    """Download a file from URL to destination path"""
    try:
        logger.info(f"Downloading {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        print()  # New line after progress
        logger.info(f"Downloaded {dest_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def extract_tar_gz(tar_path: Path, extract_to: Path) -> bool:
    """Extract tar.gz file to destination directory"""
    try:
        logger.info(f"Extracting {tar_path.name}")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_to)
        logger.info(f"Extracted to {extract_to}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract {tar_path}: {e}")
        return False

def download_piper_voice(voice_name: str, voice_info: Dict, models_dir: Path) -> bool:
    """Download and install a Piper voice"""
    logger.info(f"Downloading Piper voice: {voice_name}")
    logger.info(f"Description: {voice_info['description']}")
    
    model_path = models_dir / f"{voice_name}.onnx"
    config_path = models_dir / f"{voice_name}.onnx.json"
    
    try:
        # Download the model file
        if not download_file(voice_info['url'], model_path):
            return False
        
        # Download the config file
        if not download_file(voice_info['config_url'], config_path):
            return False
        
        logger.info(f"Successfully installed voice: {voice_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download voice {voice_name}: {e}")
        # Clean up partial downloads
        if model_path.exists():
            model_path.unlink()
        if config_path.exists():
            config_path.unlink()
        return False

def download_kokoro_model(models_dir: Path) -> bool:
    """Download and install Kokoro ONNX model"""
    logger.info("Downloading Kokoro ONNX model")
    logger.info(f"Description: {KOKORO_MODEL['description']}")
    
    model_dir = models_dir / "kokoro" / "onnx"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the main model (fp32 version)
    model_path = model_dir / "model.onnx"
    
    try:
        if not download_file(KOKORO_MODEL["model_url"], model_path):
            return False
        
        # Also download the quantized version for better performance
        quantized_path = model_dir / "model_quantized.onnx"
        if not download_file(KOKORO_MODEL["model_quantized_url"], quantized_path):
            logger.warning("Failed to download quantized model, but main model is available")
        
        logger.info("Successfully installed Kokoro ONNX model")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download Kokoro model: {e}")
        # Clean up partial downloads
        if model_path.exists():
            model_path.unlink()
        return False

def download_kokoro_voice(voice_name: str, voice_info: Dict, models_dir: Path) -> bool:
    """Download and install a Kokoro voice"""
    logger.info(f"Downloading Kokoro voice: {voice_name}")
    logger.info(f"Description: {voice_info['description']}")
    
    voice_dir = models_dir / "kokoro" / "voices"
    voice_dir.mkdir(parents=True, exist_ok=True)
    
    voice_path = voice_dir / f"{voice_name}.bin"
    
    try:
        # Download the voice file
        if not download_file(voice_info['url'], voice_path):
            return False
        
        logger.info(f"Successfully installed voice: {voice_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download voice {voice_name}: {e}")
        # Clean up partial downloads
        if voice_path.exists():
            voice_path.unlink()
        return False

def check_whisper_model(model_name: str = "large-v3") -> bool:
    """Check if Whisper model is available (will download on first use)"""
    try:
        from faster_whisper import WhisperModel
        logger.info(f"Checking Whisper model: {model_name}")
        
        # This will download the model if it doesn't exist
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        logger.info(f"Whisper model {model_name} is available")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load Whisper model {model_name}: {e}")
        return False

def list_installed_voices(models_dir: Path) -> List[str]:
    """List installed Piper voices"""
    voices = []
    if models_dir.exists():
        for onnx_file in models_dir.glob("*.onnx"):
            voices.append(onnx_file.stem)
    return voices

def list_installed_kokoro_voices(models_dir: Path) -> List[str]:
    """List installed Kokoro voices"""
    voices = []
    voice_dir = models_dir / "kokoro" / "voices"
    if voice_dir.exists():
        for voice_file in voice_dir.glob("*.bin"):
            voices.append(voice_file.stem)
    return voices

def check_kokoro_model(models_dir: Path) -> bool:
    """Check if Kokoro model is installed"""
    model_path = models_dir / "kokoro" / "onnx" / "model.onnx"
    return model_path.exists()

def main():
    """Main function"""
    # Get script directory
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models"
    
    # Create models directory
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("VERA Model Download Script")
    logger.info("=" * 40)
    
    # Check what's already installed
    piper_dir = models_dir / "piper"
    piper_dir.mkdir(parents=True, exist_ok=True)
    installed_piper_voices = list_installed_voices(piper_dir)
    logger.info(f"Currently installed Piper voices: {installed_piper_voices}")
    
    installed_kokoro_voices = list_installed_kokoro_voices(models_dir)
    logger.info(f"Currently installed Kokoro voices: {installed_kokoro_voices}")
    
    kokoro_model_installed = check_kokoro_model(models_dir)
    logger.info(f"Kokoro model installed: {kokoro_model_installed}")
    
    # Download Piper voices
    logger.info("Downloading Piper voices...")
    piper_success_count = 0
    
    for voice_name, voice_info in PIPER_VOICES.items():
        if voice_name in installed_piper_voices:
            logger.info(f"Piper voice {voice_name} already installed, skipping")
            piper_success_count += 1
            continue
        
        if download_piper_voice(voice_name, voice_info, piper_dir):
            piper_success_count += 1
        else:
            logger.error(f"Failed to download Piper voice: {voice_name}")
    
    logger.info(f"Successfully installed {piper_success_count}/{len(PIPER_VOICES)} Piper voices")
    
    # Download Kokoro model
    logger.info("Downloading Kokoro model...")
    kokoro_model_success = False
    if kokoro_model_installed:
        logger.info("Kokoro model already installed, skipping")
        kokoro_model_success = True
    else:
        kokoro_model_success = download_kokoro_model(models_dir)
    
    # Download Kokoro voices
    logger.info("Downloading Kokoro voices...")
    kokoro_voice_success_count = 0
    
    for voice_name, voice_info in KOKORO_VOICES.items():
        if voice_name in installed_kokoro_voices:
            logger.info(f"Kokoro voice {voice_name} already installed, skipping")
            kokoro_voice_success_count += 1
            continue
        
        if download_kokoro_voice(voice_name, voice_info, models_dir):
            kokoro_voice_success_count += 1
        else:
            logger.error(f"Failed to download Kokoro voice: {voice_name}")
    
    logger.info(f"Successfully installed {kokoro_voice_success_count}/{len(KOKORO_VOICES)} Kokoro voices")
    
    # Check Whisper model
    logger.info("Checking Whisper model...")
    if check_whisper_model("large-v3"):
        logger.info("Whisper model is ready")
    else:
        logger.error("Whisper model check failed")
    
    # Final status
    final_piper_voices = list_installed_voices(piper_dir)
    final_kokoro_voices = list_installed_kokoro_voices(models_dir)
    final_kokoro_model = check_kokoro_model(models_dir)
    
    logger.info("=" * 40)
    logger.info("Model download complete!")
    logger.info(f"Installed Piper voices: {final_piper_voices}")
    logger.info(f"Installed Kokoro voices: {final_kokoro_voices}")
    logger.info(f"Kokoro model installed: {final_kokoro_model}")
    
    if len(final_piper_voices) == 0 and not final_kokoro_model:
        logger.error("No TTS models installed! The application may not work correctly.")
        return 1
    else:
        logger.info("Ready to run VERA!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
