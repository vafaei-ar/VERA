import logging
import struct
import requests
from typing import Dict, List, Optional
import numpy as np

from .base import TTSBackend

logger = logging.getLogger(__name__)


class ChatterboxBackend(TTSBackend):
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None) -> None:
        super().__init__()  # Initialize base class metrics
        self._ready = False
        self._voices: List[Dict] = []
        self._api_key = api_key
        self._api_url = api_url or "https://api.resemble.ai/v2/projects"
        
        # Check if API key is provided
        if not self._api_key:
            logger.warning("ResembleAI Chatterbox API key not provided. Set RESEMBLE_API_KEY environment variable.")
            return
            
        try:
            # Test API connectivity
            self._test_api_connection()
            
            # Initialize with default voices
            self._voices = [
                {"id": "default", "name": "Default Voice", "language": "en"},
                {"id": "chatterbox", "name": "Chatterbox Voice", "language": "en"},
            ]
            
            self._ready = True
            logger.info("ChatterboxBackend initialized")
        except Exception as e:
            logger.warning("Chatterbox API not available: %s", e)

    def _test_api_connection(self) -> None:
        """Test connection to ResembleAI API."""
        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(self._api_url, headers=headers, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"API connection failed: {response.status_code}")

    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        if not self._ready:
            raise RuntimeError("Chatterbox is not available. Check API key and connection.")
        if not text or not text.strip():
            return b""
            
        try:
            # Use ResembleAI API for synthesis
            headers = {
                "Authorization": f"Token {self._api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare synthesis request
            data = {
                "text": text,
                "voice_uuid": voice or "default",
                "output_format": "wav",
                "sample_rate": 16000
            }
            
            # Make API request
            response = requests.post(
                f"{self._api_url}/synthesize",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Synthesis failed: {response.status_code} - {response.text}")
            
            # Get audio data
            audio_data = response.content
            
            # Convert to 16kHz mono WAV if needed
            return self._ensure_wav_format(audio_data)
            
        except Exception as e:
            logger.error("Chatterbox synthesis failed: %s", e)
            raise

    def _ensure_wav_format(self, audio_data: bytes) -> bytes:
        """Ensure audio data is in 16kHz mono WAV format."""
        # Check if it's already a valid WAV file
        if len(audio_data) >= 44 and audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
            # Parse existing WAV header
            try:
                sample_rate = struct.unpack('<I', audio_data[24:28])[0]
                channels = struct.unpack('<H', audio_data[22:24])[0]
                
                # If already 16kHz mono, return as-is
                if sample_rate == 16000 and channels == 1:
                    return audio_data
                    
                # Otherwise, we'd need to resample (simplified for now)
                logger.warning("Audio format conversion not implemented. Returning original audio.")
                return audio_data
            except Exception:
                pass
        
        # If not a valid WAV, create a simple WAV header
        # This is a simplified approach - in production, you'd want proper audio processing
        logger.warning("Non-WAV audio received, creating basic WAV header")
        return self._create_simple_wav(audio_data)

    def _create_simple_wav(self, audio_data: bytes) -> bytes:
        """Create a simple WAV header for raw audio data."""
        # Assume the audio data is 16-bit PCM at 16kHz
        data_length = len(audio_data)
        sample_rate = 16000
        channels = 1
        bits_per_sample = 16
        
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_length,  # File size - 8
            b'WAVE',
            b'fmt ',
            16,  # Format chunk size
            1,   # Audio format (PCM)
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b'data',
            data_length  # Data size
        )
        return header + audio_data

    def estimate_duration(self, text: str, voice: Optional[str] = None) -> float:
        """Estimate duration based on text length (rough approximation)."""
        # Rough estimate: ~150 words per minute, ~2.5 characters per word
        words = len(text.split())
        return max(0.5, words / 2.5)  # Minimum 0.5 seconds

    def get_audio_duration(self, audio_data: bytes) -> float:
        """Get the actual duration of WAV audio data in seconds."""
        if len(audio_data) < 44:  # WAV header is 44 bytes
            return 0.0
            
        try:
            # Parse WAV header to get sample rate and data size
            sample_rate = struct.unpack('<I', audio_data[24:28])[0]
            data_size = struct.unpack('<I', audio_data[40:44])[0]
            channels = struct.unpack('<H', audio_data[22:24])[0]
            bits_per_sample = struct.unpack('<H', audio_data[34:36])[0]
            
            # Calculate duration
            bytes_per_sample = bits_per_sample // 8
            samples = data_size // (channels * bytes_per_sample)
            duration = samples / sample_rate
            
            return duration
        except Exception as e:
            logger.warning("Failed to parse WAV header: %s", e)
            return 0.0

    def list_voices(self) -> List[Dict]:
        if not self._ready:
            return []
            
        try:
            # Try to fetch available voices from API
            headers = {
                "Authorization": f"Token {self._api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self._api_url}/voices",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                voices_data = response.json()
                voices = []
                for voice in voices_data.get("items", []):
                    voices.append({
                        "id": voice.get("uuid", voice.get("id", "unknown")),
                        "name": voice.get("name", "Unknown Voice"),
                        "language": voice.get("language", "en")
                    })
                return voices
        except Exception as e:
            logger.warning("Failed to fetch voices from API: %s", e)
        
        # Fallback to default voices
        return self._voices

    def health(self) -> Dict:
        if not self._ready:
            return {"ok": False, "message": "Chatterbox API key not provided or connection failed"}
        
        try:
            # Test API connectivity
            self._test_api_connection()
            voices = self.list_voices()
            return {"ok": True, "message": f"Chatterbox ready with {len(voices)} voices"}
        except Exception as e:
            return {"ok": False, "message": f"Chatterbox API connection failed: {e}"}
