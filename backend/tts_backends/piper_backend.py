import os
import subprocess
import tempfile
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

from .base import TTSBackend

logger = logging.getLogger(__name__)


class PiperBackend(TTSBackend):
    def __init__(
        self,
        binary_path: str = "./piper/piper",
        voice_dir: str = "./models/piper",
        default_voice: str = "en_US-amy-medium",
        speaking_rate: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> None:
        super().__init__()  # Initialize base class metrics
        self.binary_path = Path(binary_path)
        self.voice_dir = Path(voice_dir)
        self.default_voice = default_voice
        self.speaking_rate = speaking_rate
        self.noise_scale = noise_scale
        self.noise_w = noise_w

        if not self.binary_path.exists():
            raise FileNotFoundError(f"Piper binary not found at {self.binary_path}")
        self.voice_dir.mkdir(parents=True, exist_ok=True)

        self._voice_cache: Dict[str, Dict] = {}
        self._scan_voices()

        logger.info("PiperBackend ready. Voices: %s", list(self._voice_cache.keys()))

    def _scan_voices(self) -> None:
        self._voice_cache.clear()
        for onnx_file in self.voice_dir.glob("*.onnx"):
            voice_name = onnx_file.stem
            cfg = onnx_file.with_suffix(".onnx.json")
            if not cfg.exists():
                continue
            try:
                with open(cfg, "r") as f:
                    meta = json.load(f)
                self._voice_cache[voice_name] = {
                    "model_path": str(onnx_file),
                    "config_path": str(cfg),
                    "language": meta.get("language", {}).get("code", "unknown"),
                    "sample_rate": meta.get("audio", {}).get("sample_rate", 22050),
                }
            except Exception as e:
                logger.warning("Failed to parse voice config for %s: %s", voice_name, e)

    # TTSBackend
    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        if not text or not text.strip():
            return b""

        voice_name = voice or self.default_voice
        if voice_name not in self._voice_cache:
            logger.warning("Voice %s not found; falling back to default", voice_name)
            voice_name = self.default_voice if self.default_voice in self._voice_cache else next(iter(self._voice_cache), None)
            if voice_name is None:
                raise RuntimeError("No Piper voices available")

        vinfo = self._voice_cache[voice_name]
        length_scale = max(0.1, min(3.0, float(kwargs.get("speaking_rate", self.speaking_rate))))
        noise_scale = max(0.0, min(2.0, float(kwargs.get("noise_scale", self.noise_scale))))
        noise_w = max(0.0, min(2.0, float(kwargs.get("noise_w", self.noise_w))))

        temp_wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_wav_path = tmp.name

            cmd = [
                str(self.binary_path),
                "-m", vinfo["model_path"],
                "-c", vinfo["config_path"],
                "-f", temp_wav_path,
                "--length_scale", str(length_scale),
                "--noise_scale", str(noise_scale),
                "--noise_w", str(noise_w),
            ]
            speaker_id = kwargs.get("speaker_id")
            if speaker_id is not None:
                cmd.extend(["--speaker", str(int(speaker_id))])

            logger.debug("Running Piper command: %s", " ".join(cmd))
            result = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))

            with open(temp_wav_path, "rb") as f:
                audio = f.read()
            return audio
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                except Exception:
                    pass

    def list_voices(self) -> List[Dict]:
        items: List[Dict] = []
        for name, meta in self._voice_cache.items():
            items.append({
                "id": name,
                "name": name,
                "language": meta.get("language", "unknown"),
            })
        return items

    def health(self) -> Dict:
        return {
            "ok": self.binary_path.exists() and bool(self._voice_cache),
            "message": f"piper={'ok' if self.binary_path.exists() else 'missing'} voices={len(self._voice_cache)}",
        }
