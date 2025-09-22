import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

logger = logging.getLogger(__name__)

class TTS:
    """Text-to-Speech using Piper TTS"""
    
    def __init__(
        self,
        binary_path: str = "./piper/piper",
        voice_dir: str = "./models/piper",
        default_voice: str = "en_US-amy-medium",
        speaking_rate: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8
    ):
        self.binary_path = Path(binary_path)
        self.voice_dir = Path(voice_dir)
        self.default_voice = default_voice
        self.speaking_rate = speaking_rate  # length_scale in piper
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        
        # Validate binary
        if not self.binary_path.exists():
            raise FileNotFoundError(f"Piper binary not found at {self.binary_path}")
        
        # Create voice directory if it doesn't exist
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for voice info
        self._voice_cache = {}
        self._scan_available_voices()
        
        logger.info(f"TTS initialized with binary: {self.binary_path}")
        logger.info(f"Available voices: {list(self._voice_cache.keys())}")
    
    def _scan_available_voices(self):
        """Scan for available voice models in the voice directory"""
        self._voice_cache.clear()
        
        for onnx_file in self.voice_dir.glob("*.onnx"):
            voice_name = onnx_file.stem
            config_file = onnx_file.with_suffix(".onnx.json")
            
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    self._voice_cache[voice_name] = {
                        "model_path": str(onnx_file),
                        "config_path": str(config_file),
                        "config": config,
                        "language": config.get("language", {}).get("code", "unknown"),
                        "dataset": config.get("dataset", "unknown"),
                        "sample_rate": config.get("audio", {}).get("sample_rate", 22050)
                    }
                except Exception as e:
                    logger.warning(f"Failed to load config for {voice_name}: {e}")
        
        if not self._voice_cache:
            logger.warning("No Piper voices found in voice directory")
    
    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            voice: Voice name to use (defaults to default_voice)
            **kwargs: Additional parameters (speaking_rate, noise_scale, etc.)
            
        Returns:
            WAV audio data as bytes
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to TTS synthesis")
            return b""
        
        voice_name = voice or self.default_voice
        
        # Get voice info with better error handling
        if voice_name not in self._voice_cache:
            logger.warning(f"Voice {voice_name} not found, using default")
            voice_name = self.default_voice
            
            if voice_name not in self._voice_cache:
                available = list(self._voice_cache.keys())
                if available:
                    voice_name = available[0]
                    logger.warning(f"Default voice not found, using {voice_name}")
                else:
                    raise RuntimeError("No voices available for TTS synthesis")
        
        voice_info = self._voice_cache[voice_name]
        
        # Validate voice files exist
        if not os.path.exists(voice_info["model_path"]):
            raise RuntimeError(f"Voice model file not found: {voice_info['model_path']}")
        if not os.path.exists(voice_info["config_path"]):
            raise RuntimeError(f"Voice config file not found: {voice_info['config_path']}")
        
        # Get synthesis parameters with validation
        length_scale = max(0.1, min(3.0, kwargs.get("speaking_rate", self.speaking_rate)))
        noise_scale = max(0.0, min(2.0, kwargs.get("noise_scale", self.noise_scale)))
        noise_w = max(0.0, min(2.0, kwargs.get("noise_w", self.noise_w)))
        
        temp_wav_path = None
        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            
            # Build piper command
            cmd = [
                str(self.binary_path),
                "-m", voice_info["model_path"],
                "-c", voice_info["config_path"],
                "-f", temp_wav_path,
                "--length_scale", str(length_scale),
                "--noise_scale", str(noise_scale),
                "--noise_w", str(noise_w)
            ]
            
            # Add speaker ID if specified
            speaker_id = kwargs.get("speaker_id", 0)
            if speaker_id:
                cmd.extend(["--speaker", str(speaker_id)])
            
            logger.debug(f"Running TTS command: {' '.join(cmd)}")
            logger.info(f"Synthesizing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Run piper with better error handling
            result = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                capture_output=True,
                timeout=60  # Increased timeout to 60 seconds
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                stdout_msg = result.stdout.decode('utf-8', errors='ignore')
                logger.error(f"Piper TTS failed with return code {result.returncode}")
                logger.error(f"STDERR: {error_msg}")
                logger.error(f"STDOUT: {stdout_msg}")
                raise RuntimeError(f"TTS synthesis failed: {error_msg or 'Unknown error'}")
            
            # Validate output file was created and has content
            if not os.path.exists(temp_wav_path):
                raise RuntimeError("TTS synthesis failed: No output file generated")
            
            file_size = os.path.getsize(temp_wav_path)
            if file_size == 0:
                raise RuntimeError("TTS synthesis failed: Empty output file generated")
            
            # Read the generated audio
            with open(temp_wav_path, 'rb') as f:
                audio_data = f.read()
            
            # Validate audio data
            if len(audio_data) == 0:
                raise RuntimeError("TTS synthesis failed: No audio data generated")
            
            # Comprehensive WAV header validation
            if not audio_data.startswith(b'RIFF') or b'WAVE' not in audio_data[:12]:
                logger.warning("Generated audio may not be valid WAV format")
            
            # Validate WAV file structure
            try:
                import struct
                if len(audio_data) >= 44:  # Minimum WAV header size
                    # Check file size matches header
                    file_size = struct.unpack('<I', audio_data[4:8])[0]
                    if file_size + 8 != len(audio_data):
                        logger.warning(f"WAV file size mismatch: header says {file_size + 8}, actual {len(audio_data)}")
                    
                    # Check sample rate
                    sample_rate = struct.unpack('<I', audio_data[24:28])[0]
                    if sample_rate not in [16000, 22050, 44100, 48000]:
                        logger.warning(f"Unexpected sample rate: {sample_rate}")
                    
                    # Check data chunk
                    data_size = struct.unpack('<I', audio_data[40:44])[0]
                    if data_size == 0:
                        logger.warning("WAV file has no audio data")
                    
            except Exception as e:
                logger.warning(f"WAV validation error: {e}")
            
            logger.info(f"Successfully synthesized {len(text)} characters to {len(audio_data)} bytes")
            return audio_data
            
        except subprocess.TimeoutExpired:
            logger.error(f"TTS synthesis timed out after 60 seconds for text: '{text[:50]}...'")
            raise RuntimeError("TTS synthesis timed out - text may be too long or system overloaded")
        except FileNotFoundError as e:
            logger.error(f"Required file not found during TTS synthesis: {e}")
            raise RuntimeError(f"TTS synthesis failed: Missing required file - {e}")
        except PermissionError as e:
            logger.error(f"Permission denied during TTS synthesis: {e}")
            raise RuntimeError(f"TTS synthesis failed: Permission denied - {e}")
        except Exception as e:
            logger.error(f"Unexpected TTS synthesis error: {e}")
            raise RuntimeError(f"TTS synthesis failed: {str(e)}")
        finally:
            # Clean up temporary file
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary file {temp_wav_path}: {cleanup_error}")
    
    def synthesize_to_file(self, text: str, output_path: Union[str, Path], 
                          voice: Optional[str] = None, **kwargs) -> bool:
        """
        Synthesize speech directly to file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            audio_data = self.synthesize(text, voice, **kwargs)
            
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            
            logger.info(f"Synthesized audio saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to synthesize to file: {e}")
            return False
    
    def get_available_voices(self) -> Dict[str, Dict]:
        """Get information about available voices"""
        return self._voice_cache.copy()
    
    def get_voice_info(self, voice_name: str) -> Optional[Dict]:
        """Get detailed information about a specific voice"""
        return self._voice_cache.get(voice_name)
    
    def is_voice_available(self, voice_name: str) -> bool:
        """Check if a voice is available"""
        return voice_name in self._voice_cache
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages from available voices"""
        languages = set()
        for voice_info in self._voice_cache.values():
            languages.add(voice_info["language"])
        return sorted(list(languages))
    
    def get_voices_by_language(self, language: str) -> List[str]:
        """Get voices available for a specific language"""
        voices = []
        for voice_name, voice_info in self._voice_cache.items():
            if voice_info["language"] == language:
                voices.append(voice_name)
        return voices
    
    def test_voice(self, voice_name: str, test_text: str = "Hello, this is a test.") -> bool:
        """
        Test if a voice works by synthesizing a short phrase
        
        Returns:
            True if voice works, False otherwise
        """
        try:
            audio_data = self.synthesize(test_text, voice=voice_name)
            return len(audio_data) > 0
        except Exception as e:
            logger.error(f"Voice test failed for {voice_name}: {e}")
            return False
    
    def estimate_duration(self, text: str, voice: Optional[str] = None) -> float:
        """
        Estimate the duration of synthesized speech in seconds
        
        This is a rough estimate based on character count and speaking rate
        """
        if not text:
            return 0.0
        
        # Rough estimation: ~150 words per minute, ~5 characters per word
        # Adjust for speaking rate
        chars_per_second = (150 * 5) / 60  # ~12.5 chars/sec at normal rate
        chars_per_second /= self.speaking_rate  # Adjust for rate
        
        return len(text) / chars_per_second
    
    def get_audio_duration(self, audio_data: bytes) -> float:
        """
        Get the actual duration of WAV audio data in seconds
        
        Args:
            audio_data: WAV audio data as bytes
            
        Returns:
            Duration in seconds, or 0.0 if unable to determine
        """
        try:
            import struct
            
            if len(audio_data) < 44:  # Minimum WAV header size
                return 0.0
            
            # Check for WAV header
            if not (audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:12]):
                return 0.0
            
            # Extract sample rate and number of samples from WAV header
            # Sample rate is at offset 24-27 (little endian)
            sample_rate = struct.unpack('<I', audio_data[24:28])[0]
            
            # Data size is at offset 40-43 (little endian)
            data_size = struct.unpack('<I', audio_data[40:44])[0]
            
            # Assuming 16-bit mono audio (2 bytes per sample)
            num_samples = data_size // 2
            duration = num_samples / sample_rate
            
            return duration
            
        except Exception as e:
            logger.warning(f"Failed to calculate audio duration: {e}")
            return 0.0
    
    def get_system_info(self) -> Dict:
        """Get TTS system information"""
        return {
            "binary_path": str(self.binary_path),
            "binary_exists": self.binary_path.exists(),
            "voice_dir": str(self.voice_dir),
            "default_voice": self.default_voice,
            "available_voices": list(self._voice_cache.keys()),
            "speaking_rate": self.speaking_rate,
            "noise_scale": self.noise_scale,
            "noise_w": self.noise_w
        }
    
    def refresh_voices(self):
        """Rescan the voice directory for new voices"""
        logger.info("Rescanning voice directory...")
        self._scan_available_voices()
        logger.info(f"Found {len(self._voice_cache)} voices after rescan")

class TTSVoiceManager:
    """Helper class for managing TTS voices and downloads"""
    
    def __init__(self, voice_dir: Union[str, Path]):
        self.voice_dir = Path(voice_dir)
        self.voice_dir.mkdir(parents=True, exist_ok=True)
    
    def download_voice(self, voice_url: str, voice_name: str) -> bool:
        """
        Download a voice model from URL
        
        Args:
            voice_url: URL to the voice model (.tar.gz)
            voice_name: Name for the voice
            
        Returns:
            True if successful
        """
        try:
            import requests
            import tarfile
            
            logger.info(f"Downloading voice {voice_name} from {voice_url}")
            
            response = requests.get(voice_url, stream=True)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_path = temp_file.name
            
            # Extract to voice directory
            with tarfile.open(temp_path, 'r:gz') as tar:
                tar.extractall(self.voice_dir)
            
            # Clean up
            os.unlink(temp_path)
            
            logger.info(f"Successfully downloaded voice {voice_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download voice {voice_name}: {e}")
            return False
    
    def list_downloaded_voices(self) -> List[str]:
        """List all downloaded voice models"""
        voices = []
        for onnx_file in self.voice_dir.glob("*.onnx"):
            voices.append(onnx_file.stem)
        return voices
    
    def remove_voice(self, voice_name: str) -> bool:
        """Remove a voice model"""
        try:
            model_file = self.voice_dir / f"{voice_name}.onnx"
            config_file = self.voice_dir / f"{voice_name}.onnx.json"
            
            if model_file.exists():
                model_file.unlink()
            if config_file.exists():
                config_file.unlink()
            
            logger.info(f"Removed voice {voice_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove voice {voice_name}: {e}")
            return False

