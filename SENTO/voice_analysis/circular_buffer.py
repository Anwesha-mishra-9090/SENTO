import numpy as np
from collections import deque


class CircularBuffer:
    def __init__(self, size, dtype=np.float32):
        self.size = size
        self.buffer = deque(maxlen=size)
        self.dtype = dtype

    def add(self, data):
        """Add data to buffer"""
        try:
            if isinstance(data, (list, np.ndarray)):
                self.buffer.extend(data)
            else:
                self.buffer.append(data)
            return True
        except Exception as e:
            print(f"Error adding data to circular buffer: {e}")
            return False

    def get(self, n=None):
        """Get data from buffer"""
        try:
            if n is None:
                return list(self.buffer)
            else:
                return list(self.buffer)[-n:]
        except Exception as e:
            print(f"Error getting data from circular buffer: {e}")
            return []

    def clear(self):
        """Clear buffer"""
        try:
            self.buffer.clear()
            return True
        except Exception as e:
            print(f"Error clearing circular buffer: {e}")
            return False

    def is_full(self):
        """Check if buffer is full"""
        return len(self.buffer) >= self.size

    def get_mean(self):
        """Calculate mean of buffer contents"""
        try:
            if len(self.buffer) == 0:
                return 0.0
            return float(np.mean(list(self.buffer)))
        except Exception as e:
            print(f"Error calculating mean: {e}")
            return 0.0

    def get_std(self):
        """Calculate standard deviation of buffer contents"""
        try:
            if len(self.buffer) == 0:
                return 0.0
            return float(np.std(list(self.buffer)))
        except Exception as e:
            print(f"Error calculating std: {e}")
            return 0.0

    def get_recent(self, n=1):
        """Get most recent n elements"""
        try:
            if n <= 0:
                return []
            return list(self.buffer)[-n:]
        except Exception as e:
            print(f"Error getting recent elements: {e}")
            return []


class AudioCircularBuffer:
    def __init__(self, sample_rate=16000, buffer_duration=5):
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.buffer_size = sample_rate * buffer_duration
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_index = 0
        self.is_buffer_full = False

    def add_audio_chunk(self, audio_chunk):
        """Add audio chunk to circular buffer"""
        try:
            if audio_chunk is None or len(audio_chunk) == 0:
                return False

            chunk_length = len(audio_chunk)

            if chunk_length >= self.buffer_size:
                # If chunk is larger than buffer, take the last part
                audio_chunk = audio_chunk[-self.buffer_size:]
                chunk_length = self.buffer_size

            # Calculate indices for circular write
            end_index = self.write_index + chunk_length

            if end_index <= self.buffer_size:
                # Normal write without wrap-around
                self.buffer[self.write_index:end_index] = audio_chunk
            else:
                # Write with wrap-around
                first_part = self.buffer_size - self.write_index
                self.buffer[self.write_index:] = audio_chunk[:first_part]
                self.buffer[:chunk_length - first_part] = audio_chunk[first_part:]

            # Update write index
            self.write_index = (self.write_index + chunk_length) % self.buffer_size

            # Mark buffer as full after first complete cycle
            if not self.is_buffer_full and self.write_index == 0 and chunk_length > 0:
                self.is_buffer_full = True

            return True

        except Exception as e:
            print(f"Error adding audio chunk to circular buffer: {e}")
            return False

    def get_audio_data(self, duration=None):
        """Get audio data from buffer"""
        try:
            if duration is None:
                duration = self.buffer_duration

            samples_needed = int(self.sample_rate * duration)
            samples_needed = min(samples_needed, self.buffer_size)  # Ensure we don't exceed buffer size

            if not self.is_buffer_full:
                # Buffer not full yet, return what we have
                available_samples = min(self.write_index, samples_needed)
                return self.buffer[:available_samples]
            else:
                # Buffer is full, return last 'duration' seconds
                start_index = (self.write_index - samples_needed) % self.buffer_size
                if start_index + samples_needed <= self.buffer_size:
                    return self.buffer[start_index:start_index + samples_needed]
                else:
                    # Wrap-around case
                    first_part = self.buffer_size - start_index
                    result = np.zeros(samples_needed, dtype=np.float32)
                    result[:first_part] = self.buffer[start_index:]
                    result[first_part:] = self.buffer[:samples_needed - first_part]
                    return result

        except Exception as e:
            print(f"Error getting audio data from circular buffer: {e}")
            return np.array([], dtype=np.float32)

    def get_current_buffer_size(self):
        """Get current number of samples in buffer"""
        if self.is_buffer_full:
            return self.buffer_size
        else:
            return self.write_index

    def clear(self):
        """Clear the buffer"""
        try:
            self.buffer.fill(0)
            self.write_index = 0
            self.is_buffer_full = False
            return True
        except Exception as e:
            print(f"Error clearing audio circular buffer: {e}")
            return False

    def is_full(self):
        """Check if buffer is full"""
        return self.is_buffer_full