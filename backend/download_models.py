#!/usr/bin/env python3
"""
Model download script for VERA
Downloads Whisper and Piper models needed for the application
"""

import os
import sys
import requests
import tarfile
import tempfile
import logging
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

def main():
    """Main function"""
    # Get script directory
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models" / "piper"
    
    # Create models directory
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("VERA Model Download Script")
    logger.info("=" * 40)
    
    # Check what's already installed
    installed_voices = list_installed_voices(models_dir)
    logger.info(f"Currently installed voices: {installed_voices}")
    
    # Download Piper voices
    logger.info("Downloading Piper voices...")
    success_count = 0
    
    for voice_name, voice_info in PIPER_VOICES.items():
        if voice_name in installed_voices:
            logger.info(f"Voice {voice_name} already installed, skipping")
            success_count += 1
            continue
        
        if download_piper_voice(voice_name, voice_info, models_dir):
            success_count += 1
        else:
            logger.error(f"Failed to download voice: {voice_name}")
    
    logger.info(f"Successfully installed {success_count}/{len(PIPER_VOICES)} Piper voices")
    
    # Check Whisper model
    logger.info("Checking Whisper model...")
    if check_whisper_model("large-v3"):
        logger.info("Whisper model is ready")
    else:
        logger.error("Whisper model check failed")
    
    # Final status
    final_voices = list_installed_voices(models_dir)
    logger.info("=" * 40)
    logger.info("Model download complete!")
    logger.info(f"Installed Piper voices: {final_voices}")
    
    if len(final_voices) == 0:
        logger.error("No Piper voices installed! The application may not work correctly.")
        return 1
    else:
        logger.info("Ready to run VERA!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
