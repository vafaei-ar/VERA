import numpy as np
import torch
import logging
from typing import Optional, Tuple, Literal
from audio_io import pcm_bytes_to_numpy

logger = logging.getLogger(__name__)

class StreamingVAD:
    """Streaming Voice Activity Detection with multiple backends"""
    
    def __init__(
        self, 
        sample_rate: int = 16000,
        frame_length_ms: int = 30,
        energy_threshold: float = 200.0,
        silence_duration_ms: int = 900,
        vad_type: str = "energy",
        max_speech_duration_ms: int = 15000,
    ):
        self.sample_rate = sample_rate
        self.frame_length_ms = frame_length_ms
        self.frame_length_samples = int(sample_rate * frame_length_ms / 1000)
        self.energy_threshold = energy_threshold
        self.silence_duration_ms = silence_duration_ms
        self.vad_type = vad_type
        self.max_speech_duration_ms = max_speech_duration_ms
        
        # State tracking
        self.audio_buffer = bytearray()
        self.silence_duration = 0
        self.speech_duration = 0
        self.in_speech = False
        self.speech_started = False
        self.total_frames = 0
        
        # Silero VAD setup
        self.silero_model = None
        if vad_type == "silero":
            self._init_silero_vad()
    
    def _init_silero_vad(self):
        """Initialize Silero VAD model"""
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.silero_model = model
            self.get_speech_timestamps = utils[0]
            logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Silero VAD, falling back to energy-based: {e}")
            self.vad_type = "energy"
            self.silero_model = None
    
    def accept_frame(self, pcm_bytes: bytes) -> str:
        """
        Process an audio frame and return speech state
        
        Returns:
            "continue" - keep collecting audio
            "finalize_utterance" - end of utterance detected
        """
        if not pcm_bytes:
            return "continue"
        
        self.audio_buffer.extend(pcm_bytes)
        self.total_frames += 1
        
        # Convert to numpy for analysis (int16 for energy calc)
        int16_frame = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_np = int16_frame.astype(np.float32) / 32768.0
        
        # Determine if this frame contains speech
        has_speech = self._detect_speech_in_frame(audio_np)
        
        # Update state machine
        if has_speech:
            if not self.in_speech:
                # Speech started
                self.in_speech = True
                self.speech_started = True
                logger.info("Speech detected - starting utterance")
            
            # Reset silence counter
            self.silence_duration = 0
            self.speech_duration += self.frame_length_ms
            # Force finalize if speech is too long (safety net)
            if self.speech_duration >= self.max_speech_duration_ms:
                logger.info(f"Finalizing utterance due to max duration: {self.speech_duration}ms")
                self.in_speech = False
                self.silence_duration = 0
                self.speech_duration = 0
                self.speech_started = False
                return "finalize_utterance"
        else:
            if self.in_speech:
                # We're in speech but this frame is silent
                self.silence_duration += self.frame_length_ms
                
                if self.silence_duration >= self.silence_duration_ms:
                    # End of utterance
                    logger.info(f"Finalizing utterance due to silence: {self.silence_duration}ms")
                    silence_dur = self.silence_duration
                    was_speaking = self.speech_started
                    self.in_speech = False
                    self._reset_state()
                    
                    if was_speaking:
                        logger.info(f"End of utterance detected after {silence_dur}ms silence")
                        return "finalize_utterance"
        
        return "continue"
    
    def _detect_speech_in_frame(self, audio_np: np.ndarray) -> bool:
        """Detect if the current frame contains speech"""
        if self.vad_type == "silero" and self.silero_model is not None:
            return self._silero_detect_speech(audio_np)
        else:
            return self._energy_detect_speech(audio_np)
    
    def _energy_detect_speech(self, audio_np: np.ndarray) -> bool:
        """Energy-based speech detection (mean absolute on int16 scale)"""
        if len(audio_np) == 0:
            return False
        # Compute mean absolute energy on int16 scale for robustness
        energy_int16 = np.mean(np.abs((audio_np * 32768.0).astype(np.int16))).item()
        
        # Debug logging every 50 frames to see what's happening
        if self.total_frames % 50 == 0:
            logger.info(f"VAD energy: {energy_int16:.1f} (threshold: {self.energy_threshold}) - Speech: {energy_int16 > self.energy_threshold}")
        
        return energy_int16 > self.energy_threshold
    
    def _silero_detect_speech(self, audio_np: np.ndarray) -> bool:
        """Silero VAD-based speech detection"""
        if len(audio_np) < 512:  # Silero needs minimum audio length
            return False
        
        try:
            # Silero expects 16kHz audio
            if len(audio_np) < 512:
                # Pad if too short
                padded = np.zeros(512, dtype=np.float32)
                padded[:len(audio_np)] = audio_np
                audio_np = padded
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
            
            # Get speech probability
            speech_prob = self.silero_model(audio_tensor, self.sample_rate).item()
            
            # Threshold for speech detection (0.5 is typical)
            return speech_prob > 0.5
            
        except Exception as e:
            logger.warning(f"Silero VAD error, falling back to energy: {e}")
            return self._energy_detect_speech(audio_np)
    
    def _reset_state(self):
        """Reset VAD state for next utterance"""
        self.silence_duration = 0
        self.speech_started = False
        self.speech_duration = 0
        # Note: we don't clear audio_buffer here as it's managed externally
    
    def get_collected_audio(self) -> bytes:
        """Get all collected audio data"""
        return bytes(self.audio_buffer)
    
    def clear_buffer(self):
        """Clear the internal audio buffer"""
        self.audio_buffer.clear()
    
    def get_stats(self) -> dict:
        """Get VAD statistics"""
        return {
            "total_frames": self.total_frames,
            "buffer_size_bytes": len(self.audio_buffer),
            "buffer_duration_ms": len(self.audio_buffer) / (self.sample_rate * 2) * 1000,  # 2 bytes per sample
            "in_speech": self.in_speech,
            "silence_duration_ms": self.silence_duration,
            "vad_type": self.vad_type
        }

class BatchVAD:
    """Batch VAD for processing complete audio segments"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.silero_model = None
        self._init_silero_vad()
    
    def _init_silero_vad(self):
        """Initialize Silero VAD model"""
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.silero_model = model
            self.get_speech_timestamps = utils[0]
            logger.info("Batch Silero VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD for batch processing: {e}")
    
    def get_speech_segments(self, audio_data: np.ndarray) -> list:
        """
        Get speech segments from audio data
        
        Returns:
            List of dicts with 'start' and 'end' timestamps in seconds
        """
        if self.silero_model is None:
            return self._energy_based_segments(audio_data)
        
        try:
            # Convert to torch tensor
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # Convert to mono if stereo
            
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.silero_model,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=250,
                max_speech_duration_s=30
            )
            
            # Convert to seconds
            segments = []
            for ts in speech_timestamps:
                segments.append({
                    'start': ts['start'] / self.sample_rate,
                    'end': ts['end'] / self.sample_rate
                })
            
            return segments
            
        except Exception as e:
            logger.warning(f"Silero batch VAD failed, using energy-based: {e}")
            return self._energy_based_segments(audio_data)
    
    def _energy_based_segments(self, audio_data: np.ndarray, 
                              energy_threshold: float = 0.01,
                              min_segment_duration: float = 0.5) -> list:
        """Fallback energy-based segmentation"""
        # Simple energy-based segmentation
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        segments = []
        in_speech = False
        speech_start = 0
        
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i + frame_length]
            energy = np.sqrt(np.mean(frame ** 2))
            
            if energy > energy_threshold:
                if not in_speech:
                    speech_start = i / self.sample_rate
                    in_speech = True
            else:
                if in_speech:
                    speech_end = i / self.sample_rate
                    duration = speech_end - speech_start
                    
                    if duration >= min_segment_duration:
                        segments.append({
                            'start': speech_start,
                            'end': speech_end
                        })
                    
                    in_speech = False
        
        # Handle case where speech continues to end of audio
        if in_speech:
            speech_end = len(audio_data) / self.sample_rate
            duration = speech_end - speech_start
            if duration >= min_segment_duration:
                segments.append({
                    'start': speech_start,
                    'end': speech_end
                })
        
        return segments
