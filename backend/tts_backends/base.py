from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TTSBackend(ABC):
    """Abstract interface for all TTS backends.

    Every backend must return WAV bytes at 16kHz mono in synthesize().
    """

    def __init__(self):
        self._synthesis_count = 0
        self._total_synthesis_time = 0.0
        self._last_synthesis_time = 0.0

    @abstractmethod
    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        """Synthesize speech from text and return WAV bytes (16kHz mono)."""
        raise NotImplementedError

    def synthesize_with_metrics(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        """Synthesize speech with timing metrics."""
        start_time = time.time()
        try:
            result = self.synthesize(text, voice, **kwargs)
            synthesis_time = time.time() - start_time
            
            # Update metrics
            self._synthesis_count += 1
            self._total_synthesis_time += synthesis_time
            self._last_synthesis_time = synthesis_time
            
            # Log metrics
            logger.info(f"TTS synthesis completed in {synthesis_time:.3f}s (text length: {len(text)}, audio size: {len(result)} bytes)")
            
            return result
        except Exception as e:
            synthesis_time = time.time() - start_time
            logger.error(f"TTS synthesis failed after {synthesis_time:.3f}s: {e}")
            raise

    def estimate_duration(self, text: str, voice: Optional[str] = None) -> float:
        """Optional: rough duration estimate in seconds."""
        if not text:
            return 0.0
        # default heuristic ~150 wpm -> ~12.5 chars/sec
        chars_per_second = (150 * 5) / 60
        return len(text) / chars_per_second

    def get_audio_duration(self, audio_data: bytes) -> float:
        """Best-effort duration extraction from WAV bytes."""
        try:
            import struct
            if len(audio_data) < 44:
                return 0.0
            if not (audio_data.startswith(b"RIFF") and b"WAVE" in audio_data[:12]):
                return 0.0
            sample_rate = struct.unpack('<I', audio_data[24:28])[0]
            data_size = struct.unpack('<I', audio_data[40:44])[0]
            # assume 16-bit mono (2 bytes/sample)
            samples = data_size // 2
            return samples / float(sample_rate or 16000)
        except Exception:
            return 0.0

    def list_voices(self) -> List[Dict]:
        """Optional: list available voices metadata.
        Returns list of {id, name, language} dicts.
        """
        return []

    def health(self) -> Dict:
        """Optional: health information for this backend."""
        return {"ok": True, "message": "ready"}

    def get_metrics(self) -> Dict:
        """Get synthesis metrics for this backend."""
        avg_time = self._total_synthesis_time / self._synthesis_count if self._synthesis_count > 0 else 0.0
        return {
            "synthesis_count": self._synthesis_count,
            "total_synthesis_time": self._total_synthesis_time,
            "average_synthesis_time": avg_time,
            "last_synthesis_time": self._last_synthesis_time
        }
