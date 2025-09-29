import logging
import io
import struct
import tempfile
from typing import Dict, List, Optional, Union
import numpy as np

from .base import TTSBackend

logger = logging.getLogger(__name__)


class SpeechT5Backend(TTSBackend):
    def __init__(self, model: Optional[str] = None) -> None:
        super().__init__()  # Initialize base class metrics
        self._ready = False
        self._voices: List[Dict] = []
        self._model = model or "microsoft/speecht5_tts"
        
        try:
            # Try to import transformers and torch for model loading
            import torch
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            
            self._torch = torch
            self._processor = None
            self._model_obj = None
            self._vocoder = None
            
            # Initialize with default voices
            self._voices = [
                {"id": "default", "name": "Default Voice", "language": "en"},
                {"id": "female", "name": "Female Voice", "language": "en"},
                {"id": "male", "name": "Male Voice", "language": "en"},
            ]
            
            self._ready = True
            logger.info("SpeechT5Backend initialized (lazy loading enabled)")
        except ImportError as e:
            self._torch = None
            self._processor = None
            self._model_obj = None
            self._vocoder = None
            logger.warning("SpeechT5 dependencies not installed: %s", e)

    def _ensure_model_loaded(self) -> None:
        """Lazy load the model when first needed."""
        if self._model_obj is not None:
            return
            
        if not self._ready or self._torch is None:
            raise RuntimeError("SpeechT5 dependencies not available. Install torch and transformers.")
            
        try:
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            
            logger.info("Loading SpeechT5 model: %s", self._model)
            self._processor = SpeechT5Processor.from_pretrained(self._model)
            self._model_obj = SpeechT5ForTextToSpeech.from_pretrained(self._model)
            self._vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            # Move to GPU if available
            device = "cuda" if self._torch.cuda.is_available() else "cpu"
            self._model_obj = self._model_obj.to(device)
            self._vocoder = self._vocoder.to(device)
            
            logger.info("SpeechT5 model loaded successfully on %s", device)
        except Exception as e:
            logger.error("Failed to load SpeechT5 model: %s", e)
            raise RuntimeError(f"Failed to load SpeechT5 model: {e}")

    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        if not self._ready:
            raise RuntimeError("SpeechT5 is not available. Install torch and transformers.")
        if not text or not text.strip():
            return b""
            
        self._ensure_model_loaded()
        
        try:
            # Prepare inputs
            inputs = self._processor(text=text, return_tensors="pt")
            
            # Move to same device as model
            device = next(self._model_obj.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate speech
            with self._torch.no_grad():
                speech = self._model_obj.generate_speech(
                    inputs["input_ids"], 
                    self._model_obj.speaker_embeddings[0],  # Use first speaker
                    vocoder=self._vocoder
                )
            
            # Convert to numpy and normalize
            audio_array = speech.cpu().numpy()
            
            # Convert to 16kHz mono WAV format
            return self._convert_to_wav(audio_array, sample_rate=16000)
            
        except Exception as e:
            logger.error("SpeechT5 synthesis failed: %s", e)
            raise

    def _convert_to_wav(self, audio_array: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Convert numpy audio array to 16kHz mono WAV bytes."""
        # Ensure audio is in the right format
        if audio_array.dtype != np.float32:
            # Normalize to [-1, 1] range if needed
            if audio_array.dtype in [np.int16, np.int32]:
                audio_array = audio_array.astype(np.float32) / (2**15 if audio_array.dtype == np.int16 else 2**31)
            else:
                audio_array = audio_array.astype(np.float32)
        
        # Clamp to valid range
        audio_array = np.clip(audio_array, -1.0, 1.0)
        
        # Convert to 16-bit PCM
        audio_16bit = (audio_array * 32767).astype(np.int16)
        
        # Create WAV header
        wav_header = self._create_wav_header(len(audio_16bit), sample_rate, channels=1, bits_per_sample=16)
        
        # Combine header and audio data
        return wav_header + audio_16bit.tobytes()

    def _create_wav_header(self, data_length: int, sample_rate: int, channels: int = 1, bits_per_sample: int = 16) -> bytes:
        """Create a WAV file header."""
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_length * 2,  # File size - 8
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
            data_length * 2  # Data size
        )
        return header

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
        return self._voices

    def health(self) -> Dict:
        if not self._ready:
            return {"ok": False, "message": "SpeechT5 dependencies not installed"}
        
        try:
            self._ensure_model_loaded()
            return {"ok": True, "message": f"SpeechT5 ready with {len(self._voices)} voices"}
        except Exception as e:
            return {"ok": False, "message": f"SpeechT5 model loading failed: {e}"}
