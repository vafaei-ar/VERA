import wave
import os
import numpy as np
from pathlib import Path
from typing import Union

class WavAppendWriter:
    """WAV file writer that supports appending PCM16 audio data"""
    
    def __init__(self, path: Union[str, Path], sample_rate: int = 16000, channels: int = 1):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = 2  # 16-bit = 2 bytes
        self._ensure_header()
    
    def _ensure_header(self):
        """Create WAV file with proper header if it doesn't exist"""
        if not self.path.exists():
            # Create parent directory if needed
            self.path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create empty WAV file with proper header
            with wave.open(str(self.path), 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.sample_width)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(b'')  # Empty audio data
    
    def append(self, pcm_bytes: bytes):
        """Append PCM16 audio data to the WAV file"""
        if not pcm_bytes:
            return
        
        # Read existing audio data
        with wave.open(str(self.path), 'rb') as wav_file:
            params = wav_file.getparams()
            existing_frames = wav_file.readframes(wav_file.getnframes())
        
        # Write back with appended data
        with wave.open(str(self.path), 'wb') as wav_file:
            wav_file.setparams(params)
            wav_file.writeframes(existing_frames + pcm_bytes)
    
    def close(self):
        """Close the writer (no-op for this implementation)"""
        pass
    
    def get_duration(self) -> float:
        """Get duration of recorded audio in seconds"""
        if not self.path.exists():
            return 0.0
        
        try:
            with wave.open(str(self.path), 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                return frames / float(rate)
        except:
            return 0.0
    
    def get_file_size(self) -> int:
        """Get file size in bytes"""
        if not self.path.exists():
            return 0
        return self.path.stat().st_size

def pcm_bytes_to_numpy(pcm_bytes: bytes, dtype=np.float32) -> np.ndarray:
    """Convert PCM16 bytes to numpy array"""
    # Convert bytes to int16 array
    int16_array = np.frombuffer(pcm_bytes, dtype=np.int16)
    
    # Convert to float32 and normalize
    if dtype == np.float32:
        return int16_array.astype(np.float32) / 32768.0
    elif dtype == np.int16:
        return int16_array
    else:
        return int16_array.astype(dtype)

def numpy_to_pcm_bytes(audio_array: np.ndarray) -> bytes:
    """Convert numpy array to PCM16 bytes"""
    if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
        # Normalize float audio to int16 range
        audio_array = np.clip(audio_array, -1.0, 1.0)
        int16_array = (audio_array * 32767).astype(np.int16)
    elif audio_array.dtype == np.int16:
        int16_array = audio_array
    else:
        int16_array = audio_array.astype(np.int16)
    
    return int16_array.tobytes()

def resample_audio(audio_array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple resampling using numpy (for basic use cases)"""
    if orig_sr == target_sr:
        return audio_array
    
    # Simple linear interpolation resampling
    # For production, consider using librosa.resample or scipy.signal.resample
    duration = len(audio_array) / orig_sr
    target_length = int(duration * target_sr)
    
    # Create new time indices
    orig_indices = np.linspace(0, len(audio_array) - 1, len(audio_array))
    target_indices = np.linspace(0, len(audio_array) - 1, target_length)
    
    # Interpolate
    resampled = np.interp(target_indices, orig_indices, audio_array)
    return resampled

class AudioBuffer:
    """Circular buffer for streaming audio processing"""
    
    def __init__(self, max_duration_seconds: float = 30.0, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.samples_written = 0
    
    def write(self, audio_data: np.ndarray):
        """Write audio data to the circular buffer"""
        data_len = len(audio_data)
        
        if data_len > self.max_samples:
            # If data is larger than buffer, take the last portion
            audio_data = audio_data[-self.max_samples:]
            data_len = len(audio_data)
        
        # Handle wrap-around
        if self.write_pos + data_len <= self.max_samples:
            self.buffer[self.write_pos:self.write_pos + data_len] = audio_data
        else:
            # Split write across wrap-around
            first_part = self.max_samples - self.write_pos
            self.buffer[self.write_pos:] = audio_data[:first_part]
            self.buffer[:data_len - first_part] = audio_data[first_part:]
        
        self.write_pos = (self.write_pos + data_len) % self.max_samples
        self.samples_written += data_len
    
    def get_recent_audio(self, duration_seconds: float) -> np.ndarray:
        """Get the most recent audio data"""
        num_samples = min(int(duration_seconds * self.sample_rate), 
                         min(self.samples_written, self.max_samples))
        
        if num_samples == 0:
            return np.array([], dtype=np.float32)
        
        if self.samples_written < self.max_samples:
            # Buffer not yet full
            start_pos = max(0, self.write_pos - num_samples)
            return self.buffer[start_pos:self.write_pos].copy()
        else:
            # Buffer is full, need to handle circular read
            start_pos = (self.write_pos - num_samples) % self.max_samples
            
            if start_pos + num_samples <= self.max_samples:
                return self.buffer[start_pos:start_pos + num_samples].copy()
            else:
                # Wrap-around read
                first_part = self.max_samples - start_pos
                result = np.zeros(num_samples, dtype=np.float32)
                result[:first_part] = self.buffer[start_pos:]
                result[first_part:] = self.buffer[:num_samples - first_part]
                return result
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.fill(0.0)
        self.write_pos = 0
        self.samples_written = 0

