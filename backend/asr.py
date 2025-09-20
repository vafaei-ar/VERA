import io
import logging
import numpy as np
import soundfile as sf
import torch
from typing import Tuple, Optional, List, Dict
from faster_whisper import WhisperModel
from audio_io import pcm_bytes_to_numpy

logger = logging.getLogger(__name__)

class ASR:
    """Automatic Speech Recognition using faster-whisper"""
    
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        device_index: int = 0
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.device_index = device_index
        self.model = None
        self.sample_rate = 16000  # Whisper expects 16kHz
        
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            
            # Try CUDA first, fall back to CPU if needed
            device_str = self.device
            compute_type = self.compute_type
            
            if self.device == "cuda":
                try:
                    # Test CUDA availability for faster-whisper
                    # Use simple "cuda" to let ctranslate2 choose the best device
                    device_str = "cuda"
                    
                    # Try loading with CUDA
                    self.model = WhisperModel(
                        self.model_size,
                        device=device_str,
                        compute_type=compute_type,
                        download_root=None,
                        local_files_only=False
                    )
                    logger.info(f"Whisper model {self.model_size} loaded successfully on {device_str}")
                    
                except Exception as cuda_error:
                    logger.warning(f"CUDA failed ({cuda_error}), falling back to CPU")
                    device_str = "cpu"
                    compute_type = "int8"  # More efficient for CPU
                    
                    self.model = WhisperModel(
                        self.model_size,
                        device=device_str,
                        compute_type=compute_type,
                        download_root=None,
                        local_files_only=False
                    )
                    logger.info(f"Whisper model {self.model_size} loaded successfully on CPU")
            else:
                # Direct CPU loading
                self.model = WhisperModel(
                    self.model_size,
                    device=device_str,
                    compute_type=compute_type,
                    download_root=None,
                    local_files_only=False
                )
                logger.info(f"Whisper model {self.model_size} loaded successfully on {device_str}")
            
            # Update instance variables to reflect actual device used
            self.device = device_str
            self.compute_type = compute_type
            
            # Skip warmup for now to avoid cuDNN issues
            # self._warmup()
            logger.info("Skipping warmup to avoid cuDNN issues")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _warmup(self):
        """Warm up the model with a short audio clip"""
        try:
            # Generate 1 second of silence for warmup
            warmup_audio = np.zeros(self.sample_rate, dtype=np.float32)
            warmup_bytes = (warmup_audio * 32767).astype(np.int16).tobytes()
            
            logger.info("Warming up Whisper model...")
            text, confidence = self.transcribe(warmup_bytes)
            logger.info("Whisper model warmup completed")
            
        except Exception as e:
            logger.warning(f"Model warmup failed (non-critical): {e}")
    
    def transcribe(self, wav_bytes: bytes, language: str = "en") -> Tuple[str, float]:
        """
        Transcribe audio bytes to text
        
        Args:
            wav_bytes: PCM16 audio bytes (16kHz, mono)
            language: Language code for transcription
            
        Returns:
            Tuple of (transcribed_text, average_confidence)
        """
        if not wav_bytes or len(wav_bytes) < 2:
            return "", 0.0
        
        try:
            # Convert PCM bytes to numpy array
            pcm_array = np.frombuffer(wav_bytes, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_float = pcm_array.astype(np.float32) / 32768.0
            
            # Create a WAV file in memory for faster-whisper
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_float, self.sample_rate, format='WAV')
            wav_buffer.seek(0)
            
            # Transcribe with CUDA fallback
            try:
                segments, info = self.model.transcribe(
                    wav_buffer,
                    beam_size=1,  # Faster decoding
                    language=language,
                    condition_on_previous_text=False,  # Each utterance is independent
                    vad_filter=False,  # We already did VAD
                    vad_parameters=None
                )
            except Exception as cuda_error:
                if "cudnn" in str(cuda_error).lower() or "cuda" in str(cuda_error).lower():
                    logger.warning(f"CUDA/cuDNN error during transcription: {cuda_error}")
                    logger.info("Falling back to CPU for this transcription...")
                    
                    # Create a CPU model for fallback
                    cpu_model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8",
                        download_root=None,
                        local_files_only=False
                    )
                    
                    # Reset buffer position
                    wav_buffer.seek(0)
                    segments, info = cpu_model.transcribe(
                        wav_buffer,
                        beam_size=1,
                        language=language,
                        condition_on_previous_text=False,
                        vad_filter=False,
                        vad_parameters=None
                    )
                else:
                    raise
            
            # Collect segments
            text_segments = []
            confidences = []
            
            for segment in segments:
                text_segments.append(segment.text.strip())
                # Use average log probability as confidence proxy
                if hasattr(segment, 'avg_logprob'):
                    # Convert log prob to a 0-1 confidence score
                    confidence = max(0.0, min(1.0, (segment.avg_logprob + 1.0) / 2.0))
                    confidences.append(confidence)
            
            # Combine text and calculate average confidence
            final_text = " ".join(text_segments).strip()
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Log the transcription
            logger.info(f"Transcribed: '{final_text}' (confidence: {avg_confidence:.3f})")
            
            return final_text, float(avg_confidence)
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return "", 0.0
    
    def transcribe_with_timestamps(self, wav_bytes: bytes, language: str = "en") -> Dict:
        """
        Transcribe audio with word-level timestamps
        
        Returns:
            Dict with text, segments, and metadata
        """
        if not wav_bytes or len(wav_bytes) < 2:
            return {
                "text": "",
                "segments": [],
                "language": language,
                "confidence": 0.0
            }
        
        try:
            # Convert PCM bytes to numpy array
            pcm_array = np.frombuffer(wav_bytes, dtype=np.int16)
            audio_float = pcm_array.astype(np.float32) / 32768.0
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_float, self.sample_rate, format='WAV')
            wav_buffer.seek(0)
            
            # Transcribe with word timestamps
            segments, info = self.model.transcribe(
                wav_buffer,
                beam_size=1,
                language=language,
                condition_on_previous_text=False,
                vad_filter=False,
                word_timestamps=True
            )
            
            # Process segments
            result_segments = []
            all_text = []
            all_confidences = []
            
            for segment in segments:
                segment_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": []
                }
                
                # Add word-level timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_data = {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word,
                            "probability": getattr(word, 'probability', 0.0)
                        }
                        segment_data["words"].append(word_data)
                
                # Calculate segment confidence
                if hasattr(segment, 'avg_logprob'):
                    confidence = max(0.0, min(1.0, (segment.avg_logprob + 1.0) / 2.0))
                    segment_data["confidence"] = confidence
                    all_confidences.append(confidence)
                
                result_segments.append(segment_data)
                all_text.append(segment.text.strip())
            
            return {
                "text": " ".join(all_text).strip(),
                "segments": result_segments,
                "language": info.language if hasattr(info, 'language') else language,
                "confidence": np.mean(all_confidences) if all_confidences else 0.0,
                "duration": len(audio_float) / self.sample_rate
            }
            
        except Exception as e:
            logger.error(f"Detailed transcription failed: {e}")
            return {
                "text": "",
                "segments": [],
                "language": language,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        # Common Whisper language codes
        return [
            "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh",
            "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi"
        ]
    
    def detect_language(self, wav_bytes: bytes) -> Tuple[str, float]:
        """
        Detect the language of the audio
        
        Returns:
            Tuple of (language_code, confidence)
        """
        if not wav_bytes or len(wav_bytes) < 2:
            return "en", 0.0
        
        try:
            # Convert PCM bytes to numpy array
            pcm_array = np.frombuffer(wav_bytes, dtype=np.int16)
            audio_float = pcm_array.astype(np.float32) / 32768.0
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_float, self.sample_rate, format='WAV')
            wav_buffer.seek(0)
            
            # Detect language
            segments, info = self.model.transcribe(
                wav_buffer,
                beam_size=1,
                language=None,  # Auto-detect
                condition_on_previous_text=False,
                vad_filter=False
            )
            
            # Extract language info
            detected_language = getattr(info, 'language', 'en')
            language_probability = getattr(info, 'language_probability', 0.0)
            
            return detected_language, language_probability
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "en", 0.0
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "device_index": self.device_index,
            "sample_rate": self.sample_rate,
            "model_loaded": self.model is not None
        }
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for processing"""
        try:
            return torch.cuda.is_available() and self.device == "cuda"
        except:
            return False
    
    def get_gpu_memory_usage(self) -> Optional[Dict]:
        """Get GPU memory usage information"""
        if not self.is_gpu_available():
            return None
        
        try:
            device_idx = self.device_index if self.device_index is not None else 0
            
            allocated = torch.cuda.memory_allocated(device_idx)
            cached = torch.cuda.memory_reserved(device_idx)
            total = torch.cuda.get_device_properties(device_idx).total_memory
            
            return {
                "allocated_mb": allocated / (1024 * 1024),
                "cached_mb": cached / (1024 * 1024),
                "total_mb": total / (1024 * 1024),
                "utilization": allocated / total if total > 0 else 0.0
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return None
