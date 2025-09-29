import logging
import struct
import tempfile
from typing import Dict, List, Optional
import numpy as np

from .base import TTSBackend

logger = logging.getLogger(__name__)


class VibeVoiceBackend(TTSBackend):
    def __init__(self, model: Optional[str] = None) -> None:
        super().__init__()  # Initialize base class metrics
        self._ready = False
        self._voices: List[Dict] = []
        self._model = model or "microsoft/VibeVoice-1.5B"
        
        try:
            # Try to import transformers and torch for model loading
            import torch
            from transformers import AutoProcessor, AutoModel
            
            self._torch = torch
            self._processor = None
            self._model_obj = None
            
            # Initialize with default voices (VibeVoice supports multiple speakers)
            self._voices = [
                {"id": "default", "name": "Default Voice", "language": "en"},
                {"id": "speaker_0", "name": "Speaker 0", "language": "en"},
                {"id": "speaker_1", "name": "Speaker 1", "language": "en"},
                {"id": "speaker_2", "name": "Speaker 2", "language": "en"},
                {"id": "speaker_3", "name": "Speaker 3", "language": "en"},
            ]
            
            self._ready = True
            logger.info("VibeVoiceBackend initialized (lazy loading enabled)")
        except ImportError as e:
            self._torch = None
            self._processor = None
            self._model_obj = None
            logger.warning("VibeVoice dependencies not installed: %s", e)

    def _ensure_model_loaded(self) -> None:
        """Lazy load the model when first needed."""
        if self._model_obj is not None:
            return
            
        if not self._ready or self._torch is None:
            raise RuntimeError("VibeVoice dependencies not available. Install torch and transformers.")
            
        try:
            from transformers import AutoProcessor, AutoModel
            
            logger.info("Loading VibeVoice model: %s", self._model)
            logger.warning("VibeVoice requires significant VRAM (8GB+ recommended) and may take time to load")
            
            self._processor = AutoProcessor.from_pretrained(self._model)
            self._model_obj = AutoModel.from_pretrained(
                self._model,
                torch_dtype=self._torch.float16 if self._torch.cuda.is_available() else self._torch.float32,
                device_map="auto" if self._torch.cuda.is_available() else "cpu"
            )
            logger.info("VibeVoice model loaded successfully")
        except Exception as e:
            logger.error("Failed to load VibeVoice model: %s", e)
            raise RuntimeError(f"Failed to load VibeVoice model: {e}")

    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        if not self._ready:
            raise RuntimeError("VibeVoice is not available. Install torch and transformers.")
        if not text or not text.strip():
            return b""
            
        # VibeVoice constraints: English and Chinese only
        if not self._is_supported_language(text):
            logger.warning("VibeVoice only supports English and Chinese text. Text may not synthesize correctly.")
            
        self._ensure_model_loaded()
        
        try:
            # Process text with the model
            inputs = self._processor(text=text, return_tensors="pt")
            
            # Generate speech
            with self._torch.no_grad():
                speech = self._model_obj.generate(**inputs)
            
            # Convert to numpy array and ensure it's the right format
            if hasattr(speech, 'cpu'):
                audio_array = speech.cpu().numpy()
            else:
                audio_array = np.array(speech)
            
            # Handle different output shapes
            if audio_array.ndim > 1:
                audio_array = audio_array.squeeze()
            
            # Convert to 16kHz mono WAV format
            return self._convert_to_wav(audio_array, sample_rate=16000)
            
        except Exception as e:
            logger.error("VibeVoice synthesis failed: %s", e)
            raise

    def _is_supported_language(self, text: str) -> bool:
        """Check if text is in supported languages (English/Chinese)."""
        # Simple heuristic: check for common English/Chinese characters
        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return True  # No alphabetic characters, probably fine
        
        # If more than 80% of alphabetic characters are English or Chinese, consider supported
        return (english_chars + chinese_chars) / total_chars > 0.8

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
            return {"ok": False, "message": "VibeVoice dependencies not installed"}
        
        try:
            self._ensure_model_loaded()
            return {"ok": True, "message": f"VibeVoice ready with {len(self._voices)} voices (VRAM intensive)"}
        except Exception as e:
            return {"ok": False, "message": f"VibeVoice model loading failed: {e}"}
