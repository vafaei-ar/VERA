import logging
import io
import struct
import tempfile
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import numpy as np

from .base import TTSBackend

logger = logging.getLogger(__name__)


class KokoroBackend(TTSBackend):
    def __init__(self, model: Optional[str] = None, default_voice: Optional[str] = None) -> None:
        super().__init__()  # Initialize base class metrics
        self._ready = False
        self._voices: List[Dict] = []
        self._model = model or "onnx-community/Kokoro-82M-ONNX"
        self._default_voice = default_voice or "af"
        self._token_map: Optional[Dict[str, int]] = None
        self._bos_id: int = 0
        self._eos_id: int = 0
        
        try:
            # Try to import ONNX Runtime for model loading
            import onnxruntime as ort
            import numpy as np
            
            self._ort = ort
            self._np = np
            self._session = None
            
            # Initialize with available voices from the ONNX model
            self._voices = [
                {"id": "af", "name": "Default (af)", "language": "en", "gender": "female", "nationality": "american"},
                {"id": "af_bella", "name": "Bella (af_bella)", "language": "en", "gender": "female", "nationality": "american"},
                {"id": "af_nicole", "name": "Nicole (af_nicole)", "language": "en", "gender": "female", "nationality": "american"},
                {"id": "af_sarah", "name": "Sarah (af_sarah)", "language": "en", "gender": "female", "nationality": "american"},
                {"id": "af_sky", "name": "Sky (af_sky)", "language": "en", "gender": "female", "nationality": "american"},
                {"id": "am_adam", "name": "Adam (am_adam)", "language": "en", "gender": "male", "nationality": "american"},
                {"id": "am_michael", "name": "Michael (am_michael)", "language": "en", "gender": "male", "nationality": "american"},
                {"id": "bf_emma", "name": "Emma (bf_emma)", "language": "en", "gender": "female", "nationality": "british"},
                {"id": "bf_isabella", "name": "Isabella (bf_isabella)", "language": "en", "gender": "female", "nationality": "british"},
                {"id": "bm_george", "name": "George (bm_george)", "language": "en", "gender": "male", "nationality": "british"},
                {"id": "bm_lewis", "name": "Lewis (bm_lewis)", "language": "en", "gender": "male", "nationality": "british"},
            ]
            
            # Attempt to load tokenizer tokens.json if available
            try:
                script_dir = Path(__file__).parent.parent
                tokens_path = script_dir / "models" / "kokoro" / "onnx" / "tokens.json"
                if tokens_path.exists():
                    with open(tokens_path, "r", encoding="utf-8") as f:
                        tokens = json.load(f)
                    vocab = tokens.get("model", {}).get("vocab") or tokens.get("vocab")
                    if isinstance(vocab, dict):
                        self._token_map = {str(k): int(v) for k, v in vocab.items()}
                        # Special token "$" maps to 0 in this tokenizer
                        self._bos_id = int(self._token_map.get("$", 0))
                        self._eos_id = int(self._token_map.get("$", 0))
                        logger.info("Kokoro tokens loaded: %d symbols", len(self._token_map))
                    else:
                        logger.warning("Kokoro tokens.json present but no vocab found")
                else:
                    logger.warning("Kokoro tokens.json not found at %s", tokens_path)
            except Exception as e:
                logger.warning("Failed to load Kokoro tokens.json: %s", e)

            self._ready = True
            logger.info("KokoroBackend initialized (ONNX version, lazy loading enabled)")
        except ImportError as e:
            self._ort = None
            self._np = None
            self._session = None
            logger.warning("Kokoro ONNX dependencies not installed: %s", e)

    def _ensure_model_loaded(self) -> None:
        """Lazy load the model when first needed."""
        if self._session is not None:
            return
            
        if not self._ready or self._ort is None:
            raise RuntimeError("Kokoro ONNX dependencies not available. Install onnxruntime.")
            
        try:
            # Load the ONNX model
            logger.info("Loading Kokoro ONNX model: %s", self._model)
            
            # Try to load from local models directory first
            import os
            from pathlib import Path
            
            # Look for models in the local models directory
            script_dir = Path(__file__).parent.parent
            models_dir = script_dir / "models" / "kokoro" / "onnx"
            
            # Prefer full precision for better quality, fall back to quantized
            model_path = models_dir / "model.onnx"
            quantized_path = models_dir / "model_quantized.onnx"
            
            if model_path.exists():
                logger.info("Loading full precision Kokoro model")
                self._session = self._ort.InferenceSession(str(model_path))
            elif quantized_path.exists():
                logger.info("Loading quantized Kokoro model (fallback)")
                self._session = self._ort.InferenceSession(str(quantized_path))
            else:
                logger.warning("Kokoro model files not found in %s", models_dir)
                logger.warning("Using mock implementation - will generate silence audio")
                logger.warning("Run 'python download_models.py' to download Kokoro models")
                self._session = "mock"
                return
            
            logger.info("Kokoro ONNX model loaded successfully")
        except Exception as e:
            logger.error("Failed to load Kokoro model: %s", e)
            raise RuntimeError(f"Failed to load Kokoro model: {e}")

    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        if not self._ready:
            raise RuntimeError("Kokoro is not available. Install onnxruntime.")
        if not text or not text.strip():
            return b""
            
        self._ensure_model_loaded()
        
        try:
            # Check if we're using mock implementation
            if self._session == "mock":
                logger.warning("Kokoro synthesis using mock implementation (audible tone)")
                # Generate an audible placeholder tone so users can hear feedback
                # Tone frequency depends on voice id to provide slight variation
                duration = max(0.6, min(3.0, self.estimate_duration(text)))
                sample_rate = 16000
                t = self._np.arange(int(duration * sample_rate), dtype=self._np.float32) / sample_rate
                base_freq = 440.0
                # Modulate frequency slightly by selected voice
                voice_id = (voice or self._default_voice or "af")
                freq_offset = (sum(ord(c) for c in voice_id) % 80) - 40  # -40..+39 Hz
                freq = max(220.0, min(880.0, base_freq + float(freq_offset)))
                # Simple amplitude envelope to avoid clicks
                env = self._np.ones_like(t)
                attack = int(0.02 * sample_rate)
                release = int(0.04 * sample_rate)
                if attack > 0:
                    env[:attack] = self._np.linspace(0.0, 1.0, attack, dtype=self._np.float32)
                if release > 0:
                    env[-release:] = self._np.linspace(1.0, 0.0, release, dtype=self._np.float32)
                audio_array = (0.2 * self._np.sin(2.0 * self._np.pi * freq * t) * env).astype(self._np.float32)
                return self._convert_to_wav(audio_array, sample_rate=sample_rate)
            
            # Real ONNX implementation
            logger.info("Kokoro synthesis using ONNX model")
            
            # If tokens are available, run true inference path
            if self._token_map:
                try:
                    # Prepare inputs
                    target_voice = (voice or self._default_voice)
                    input_ids = self._text_to_ids(text)
                    # Safeguard: excessively long inputs can trigger ONNX shape issues
                    max_len = int(kwargs.get("max_tokens", 512))
                    if input_ids.shape[1] > max_len:
                        # Use chunked synthesis instead of naive truncation
                        logger.info(
                            "Kokoro: chunking input tokens=%d to <=%d for voice=%s (text_len=%d)",
                            int(input_ids.shape[1]), max_len, target_voice, len(text)
                        )
                        return self._synthesize_chunked(text, target_voice, max_len)

                    style_vec = self._load_voice_style(target_voice)
                    speed = self._np.array([1.0], dtype=self._np.float32)

                    # Retry policy: 1 quick retry on onnxruntime failure
                    last_err = None
                    for attempt in range(2):
                        try:
                            outputs = self._session.run(["waveform"], {
                                "input_ids": input_ids,
                                "style": style_vec,
                                "speed": speed,
                            })
                            waveform = outputs[0][0].astype(self._np.float32)
                            # Keep native 24kHz for better quality
                            return self._convert_to_wav(waveform, sample_rate=24000)
                        except Exception as run_err:
                            last_err = run_err
                            logger.error(
                                "Kokoro inference error (attempt %d/2) voice=%s tokens=%d msg=%s",
                                attempt + 1, target_voice, int(input_ids.shape[1]), str(run_err)
                            )
                            # brief input normalization before retry
                            if attempt == 0:
                                # Light normalization: collapse spaces
                                try:
                                    import re
                                    normalized = re.sub(r"\s+", " ", text).strip()
                                    input_ids = self._text_to_ids(normalized)
                                    if input_ids.shape[1] > max_len:
                                        # Try chunked synthesis on retry
                                        return self._synthesize_chunked(normalized, target_voice, max_len)
                                except Exception:
                                    pass
                    # If both attempts failed, fall through to placeholder
                    logger.error(
                        "Kokoro real inference failed after retries for voice=%s tokens=%d: %s",
                        target_voice, int(input_ids.shape[1]), str(last_err)
                    )
                    # Final fallback: chunked synthesis to increase robustness
                    try:
                        return self._synthesize_chunked(text, target_voice, max_len)
                    except Exception as chunk_err:
                        logger.error("Kokoro chunked fallback failed: %s", chunk_err)
                except Exception as e:
                    logger.error("Kokoro real inference setup failed: %s", e)

            # Fallback: short audible tone (placeholder)
            estimated_duration = self.estimate_duration(text)
            sample_rate = 24000  # Kokoro uses 24kHz internally
            t = self._np.arange(int(estimated_duration * sample_rate), dtype=self._np.float32) / sample_rate
            voice_id = (voice or self._default_voice or "af")
            base = 350.0 if voice_id.startswith("af") or voice_id.startswith("bf") else 220.0
            vibrato = 3.0 + (len(text) % 5)
            freq = base + 5.0 * self._np.sin(2.0 * self._np.pi * (vibrato) * t)
            phase = 2.0 * self._np.pi * self._np.cumsum(freq) / sample_rate
            env = self._np.minimum(1.0, self._np.linspace(0.0, 1.0, max(1, int(0.03 * sample_rate)), dtype=self._np.float32))
            env = self._np.pad(env, (0, max(0, t.size - env.size)), mode='edge')
            tail = self._np.linspace(1.0, 0.0, max(1, int(0.05 * sample_rate)), dtype=self._np.float32)
            env[-tail.size:] = tail
            audio_array = (0.18 * self._np.sin(phase) * env).astype(self._np.float32)
            return self._convert_to_wav(audio_array.astype(self._np.float32), sample_rate=16000)
            
        except Exception as e:
            logger.error("Kokoro synthesis failed: %s", e)
            raise

    def _synthesize_chunked(self, text: str, voice: str, max_tokens: int) -> bytes:
        """Split text into sentence-like chunks under token limit, synthesize, and concatenate."""
        # Split by sentence delimiters while keeping simple structure
        import re
        sentences = re.split(r"(?<=[\.!?])\s+", text.strip())
        chunks: list[str] = []
        current = ""
        for sent in sentences:
            candidate = (current + (" " if current else "") + sent).strip()
            if not candidate:
                continue
            if self._text_to_ids(candidate).shape[1] <= max_tokens:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If a single sentence is too long, hard-split by commas/spaces
                if self._text_to_ids(sent).shape[1] > max_tokens:
                    parts = re.split(r",\s+|\s+", sent)
                    buf = ""
                    for p in parts:
                        cand2 = (buf + (" " if buf else "") + p).strip()
                        if not cand2:
                            continue
                        if self._text_to_ids(cand2).shape[1] <= max_tokens:
                            buf = cand2
                        else:
                            if buf:
                                chunks.append(buf)
                            buf = p
                    if buf:
                        chunks.append(buf)
                    current = ""
                else:
                    current = sent
        if current:
            chunks.append(current)

        if not chunks:
            chunks = [text]

        logger.info("Kokoro: chunked synthesis %d chunks (max_tokens=%d) voice=%s", len(chunks), max_tokens, voice)

        waveforms: list[np.ndarray] = []
        style_vec = self._load_voice_style(voice)
        speed = self._np.array([1.0], dtype=self._np.float32)
        for idx, chunk in enumerate(chunks):
            ids = self._text_to_ids(chunk)
            if ids.shape[1] > max_tokens:
                ids = ids[:, :max_tokens]
            outputs = self._session.run(["waveform"], {
                "input_ids": ids,
                "style": style_vec,
                "speed": speed,
            })
            wf = outputs[0][0].astype(self._np.float32)
            waveforms.append(wf)

        if not waveforms:
            return self._convert_to_wav(self._np.zeros(1, dtype=self._np.float32), sample_rate=24000)

        combined = self._np.concatenate(waveforms)
        return self._convert_to_wav(combined, sample_rate=24000)

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
            return {"ok": False, "message": "Kokoro dependencies not installed"}
        
        try:
            self._ensure_model_loaded()
            if not self._token_map:
                return {"ok": False, "message": "Kokoro tokens.json missing - placeholder audio active"}
            return {"ok": True, "message": f"Kokoro ready with {len(self._voices)} voices"}
        except Exception as e:
            return {"ok": False, "message": f"Kokoro model loading failed: {e}"}

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _normalize_text(self, text: str) -> str:
        return text.strip()

    def _text_to_ids(self, text: str) -> np.ndarray:
        assert self._token_map is not None
        # Try to produce IPA-like symbols to match tokenizer vocab
        ipa = None
        try:
            from phonemizer import phonemize
            ipa = phonemize(
                text,
                language="en-us",
                backend="espeak",
                strip=True,
                with_stress=True,
                punctuation_marks=None,
            )
        except Exception:
            try:
                from gruut import sentences
                parts: List[str] = []
                for sent in sentences(text, lang="en-us"):
                    for word in sent:
                        if hasattr(word, "phonemes") and word.phonemes:
                            parts.extend([p.text for p in word.phonemes])
                        else:
                            parts.extend(list(str(word.text)))
                    parts.append(" ")
                ipa = "".join(parts)
            except Exception:
                ipa = self._normalize_text(text)

        ids: List[int] = [self._bos_id]
        for ch in ipa:
            ids.append(self._token_map.get(ch, self._token_map.get(' ', self._eos_id)))
        ids.append(self._eos_id)
        return np.array([ids], dtype=np.int64)

    def _load_voice_style(self, voice_id: str) -> np.ndarray:
        script_dir = Path(__file__).parent.parent
        voice_path = script_dir / "models" / "kokoro" / "voices" / f"{voice_id}.bin"
        if not voice_path.exists():
            # try default
            fallback = script_dir / "models" / "kokoro" / "voices" / f"{self._default_voice}.bin"
            if fallback.exists():
                voice_path = fallback
            else:
                raise RuntimeError(f"Kokoro voice file not found: {voice_path}")
        arr = np.fromfile(str(voice_path), dtype=np.float32)
        if arr.size < 256:
            raise RuntimeError(f"Invalid Kokoro voice embedding: {voice_path}")
        return arr[:256].reshape(1, 256).astype(np.float32)
