import numpy as np
from collections import deque


class VoiceProcessor:
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = 1
        self.is_recording = False
        self.audio_data = []

    def start_recording(self, duration=5):
        """Start recording audio for specified duration"""
        try:
            import pyaudio
            import threading

            self.is_recording = True
            self.audio_data = []

            def record_audio():
                try:
                    p = pyaudio.PyAudio()
                    stream = p.open(
                        format=pyaudio.paInt16,
                        channels=self.channels,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk_size
                    )

                    frames_to_record = int(self.sample_rate / self.chunk_size * duration)

                    for i in range(frames_to_record):
                        if not self.is_recording:
                            break
                        data = stream.read(self.chunk_size, exception_on_overflow=False)
                        self.audio_data.append(data)

                    stream.stop_stream()
                    stream.close()
                    p.terminate()

                except Exception as e:
                    print(f"Audio recording error: {e}")
                    self.is_recording = False

            recording_thread = threading.Thread(target=record_audio)
            recording_thread.daemon = True
            recording_thread.start()
            return recording_thread

        except ImportError:
            print("PyAudio not available - audio recording disabled")
            self.is_recording = False
            return None
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
            return None

    def stop_recording(self):
        """Stop recording and return audio data"""
        self.is_recording = False
        try:
            return b''.join(self.audio_data)
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return b''

    def process_audio_chunk(self, audio_chunk):
        """Process single audio chunk for real-time analysis"""
        try:
            if audio_chunk is None or len(audio_chunk) == 0:
                return {}

            # Convert to numpy array
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

            # Basic audio features for real-time processing
            features = {
                'rms_energy': float(self._calculate_rms(audio_array)),
                'zero_crossing_rate': float(self._calculate_zcr(audio_array)),
                'pitch_estimate': float(self._estimate_pitch(audio_array))
            }

            return features

        except Exception as e:
            print(f"Audio chunk processing error: {e}")
            return {}

    def _calculate_rms(self, audio_data):
        """Calculate RMS energy"""
        try:
            if len(audio_data) == 0:
                return 0.0
            return np.sqrt(np.mean(audio_data ** 2))
        except Exception as e:
            print(f"RMS calculation error: {e}")
            return 0.0

    def _calculate_zcr(self, audio_data):
        """Calculate Zero Crossing Rate"""
        try:
            if len(audio_data) < 2:
                return 0.0
            zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
            return zero_crossings / len(audio_data)
        except Exception as e:
            print(f"ZCR calculation error: {e}")
            return 0.0

    def _estimate_pitch(self, audio_data):
        """Basic pitch estimation using autocorrelation"""
        try:
            if len(audio_data) < 2:
                return 0.0

            # Simple autocorrelation for pitch detection
            correlation = np.correlate(audio_data, audio_data, mode='full')
            correlation = correlation[len(correlation) // 2:]

            # Find first peak after zero lag (skip first few samples to avoid noise)
            start_index = 10  # Skip very short lags
            if start_index >= len(correlation):
                return 0.0

            peak_index = np.argmax(correlation[start_index:]) + start_index

            if peak_index > 0:
                pitch = self.sample_rate / peak_index
                # Limit pitch to human voice range (50-500 Hz)
                return max(50.0, min(500.0, pitch))
            return 0.0

        except Exception as e:
            print(f"Pitch estimation error: {e}")
            return 0.0

    def get_audio_duration(self, audio_data):
        """Calculate duration of audio data"""
        try:
            if audio_data is None:
                return 0.0
            return len(audio_data) / (self.sample_rate * 2)  # 2 bytes per sample for int16
        except Exception as e:
            print(f"Duration calculation error: {e}")
            return 0.0

    def is_recording_active(self):
        """Check if recording is currently active"""
        return self.is_recording

    def get_recorded_duration(self):
        """Get duration of currently recorded audio"""
        try:
            total_bytes = sum(len(chunk) for chunk in self.audio_data)
            return total_bytes / (self.sample_rate * 2)  # 2 bytes per sample
        except Exception as e:
            print(f"Recorded duration calculation error: {e}")
            return 0.0