from __future__ import annotations

from typing import Any, Dict
import logging

from tts_backends.base import TTSBackend
from tts_backends.piper_backend import PiperBackend
from tts_backends.kokoro_backend import KokoroBackend
from tts_backends.vibevoice_backend import VibeVoiceBackend
from tts_backends.chatterbox_backend import ChatterboxBackend

logger = logging.getLogger(__name__)


def create_tts_backend(config: Dict[str, Any]) -> TTSBackend:
    backend_name = config.get("tts", {}).get("backend", "piper").lower()

    if backend_name == "piper":
        tcfg = config.get("models", {}).get("piper", {})
        return PiperBackend(
            binary_path=tcfg.get("binary_path", "./piper/piper"),
            voice_dir=tcfg.get("voice_dir", "./models/piper"),
            default_voice=tcfg.get("default_voice", "en_US-amy-medium"),
            speaking_rate=tcfg.get("speaking_rate", 1.0),
            noise_scale=tcfg.get("noise_scale", 0.667),
            noise_w=tcfg.get("noise_w", 0.8),
        )

    if backend_name == "kokoro":
        kokoro_config = config.get("models", {}).get("kokoro", {})
        return KokoroBackend(
            model=kokoro_config.get("model", "onnx-community/Kokoro-82M-ONNX"),
            default_voice=kokoro_config.get("default_voice", "af")
        )

    if backend_name == "vibevoice":
        vibevoice_config = config.get("models", {}).get("vibevoice", {})
        return VibeVoiceBackend(model=vibevoice_config.get("model", "microsoft/VibeVoice-1.5B"))

    if backend_name == "chatterbox":
        chatterbox_config = config.get("models", {}).get("chatterbox", {})
        import os
        api_key = chatterbox_config.get("api_key") or os.getenv("RESEMBLE_API_KEY")
        return ChatterboxBackend(
            api_key=api_key,
            api_url=chatterbox_config.get("api_url")
        )

    logger.warning("Unknown TTS backend '%s', falling back to piper", backend_name)
    tcfg = config.get("models", {}).get("piper", {})
    return PiperBackend(
        binary_path=tcfg.get("binary_path", "./piper/piper"),
        voice_dir=tcfg.get("voice_dir", "./models/piper"),
        default_voice=tcfg.get("default_voice", "en_US-amy-medium"),
        speaking_rate=tcfg.get("speaking_rate", 1.0),
        noise_scale=tcfg.get("noise_scale", 0.667),
        noise_w=tcfg.get("noise_w", 0.8),
    )
