import pyaudio
import numpy as np
import threading
import queue
import time
from datetime import datetime


class VoiceInterface:
    def __init__(self, emotion_orchestrator, feature_extractor, text_to_speech=None):
        self.emotion_orchestrator = emotion_orchestrator
        self.feature_extractor = feature_extractor
        self.text_to_speech = text_to_speech

        # Audio configuration
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        self.audio_format = pyaudio.paInt16

        # Voice activity detection
        self.silence_threshold = 500
        self.silence_duration = 1.0  # seconds
        self.min_utterance_duration = 0.5  # seconds

        # State management
        self.is_listening = False
        self.is_processing = False
        self.audio_queue = queue.Queue()
        self.current_audio = []

        # Initialize audio
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def start_listening(self):
        """Start listening for voice input"""
        if self.is_listening:
            print("Already listening...")
            return

        try:
            self.is_listening = True
            self.current_audio = []

            # Start audio stream
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )

            print("Voice interface activated - Listening...")

            # Start processing thread
            processing_thread = threading.Thread(target=self._process_audio)
            processing_thread.daemon = True
            processing_thread.start()

            self.stream.start_stream()

        except Exception as e:
            print(f"Error starting voice interface: {e}")
            self.is_listening = False

    def stop_listening(self):
        """Stop listening for voice input"""
        self.is_listening = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        print("Voice interface deactivated")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for real-time processing"""
        if self.is_listening:
            self.audio_queue.put(in_data)

        return (in_data, pyaudio.paContinue)

    def _process_audio(self):
        """Process audio data for voice activity and emotion detection"""
        silence_frames = 0
        max_silence_frames = int(self.silence_duration * self.sample_rate / self.chunk_size)
        min_utterance_frames = int(self.min_utterance_duration * self.sample_rate / self.chunk_size)

        utterance_frames = 0
        is_speaking = False

        while self.is_listening:
            try:
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get(timeout=1.0)

                # Convert to numpy array
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)

                # Voice activity detection
                rms_energy = np.sqrt(np.mean(audio_data ** 2))

                if rms_energy > self.silence_threshold:
                    # Voice detected
                    silence_frames = 0
                    is_speaking = True
                    utterance_frames += 1
                    self.current_audio.append(audio_chunk)

                else:
                    # Silence detected
                    silence_frames += 1

                    if is_speaking and silence_frames >= max_silence_frames:
                        # End of utterance detected
                        if utterance_frames >= min_utterance_frames:
                            # Process the complete utterance
                            self._process_utterance(b''.join(self.current_audio))

                        # Reset for next utterance
                        self.current_audio = []
                        is_speaking = False
                        utterance_frames = 0
                        silence_frames = 0

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio processing error: {e}")

    def _process_utterance(self, audio_data):
        """Process a complete utterance for emotion analysis"""
        if self.is_processing:
            return

        self.is_processing = True

        try:
            print("Processing utterance...")

            # Extract audio features
            audio_features = self.feature_extractor.extract_audio_features(audio_data)

            if audio_features:
                # Analyze emotion from voice
                emotion_result = self.emotion_orchestrator.analyze_emotion(
                    audio_features,
                    input_type='voice'
                )

                # Display results
                self._display_emotion_results(emotion_result)

                # Generate voice response if TTS is available
                if self.text_to_speech:
                    self._generate_voice_response(emotion_result)

        except Exception as e:
            print(f"Utterance processing error: {e}")
        finally:
            self.is_processing = False

    def _display_emotion_results(self, emotion_result):
        """Display emotion analysis results"""
        emotion = emotion_result.get('emotion', 'unknown')
        confidence = emotion_result.get('confidence', 0)

        print(f"\nEmotion Detected: {emotion.upper()}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")

        # Display emotion-specific feedback
        feedback = self._get_emotion_feedback(emotion, confidence)
        if feedback:
            print(f"Feedback: {feedback}")

    def _get_emotion_feedback(self, emotion, confidence):
        """Get appropriate feedback for detected emotion"""
        feedback_map = {
            'happy': "It's wonderful to hear happiness in your voice!",
            'sad': "I hear the sadness. Remember, difficult emotions are temporary.",
            'angry': "I sense frustration. Taking a deep breath can help.",
            'fear': "It sounds like you're feeling anxious. You're safe here.",
            'surprise': "Surprise detected! Life is full of unexpected moments.",
            'neutral': "Your voice sounds calm and balanced."
        }

        base_feedback = feedback_map.get(emotion, "Thank you for sharing how you feel.")

        if confidence > 0.8:
            confidence_note = " I'm quite confident about this analysis."
        elif confidence > 0.6:
            confidence_note = " This is my best interpretation."
        else:
            confidence_note = " The emotional signal wasn't very clear."

        return base_feedback + confidence_note

    def _generate_voice_response(self, emotion_result):
        """Generate voice response using text-to-speech"""
        if not self.text_to_speech:
            return

        emotion = emotion_result.get('emotion', 'neutral')

        # Simple response mapping
        response_map = {
            'happy': "It's wonderful to hear the happiness in your voice!",
            'sad': "I hear that you're feeling down. Remember, I'm here to support you.",
            'angry': "I sense some frustration. Would you like to talk about what's bothering you?",
            'fear': "It sounds like you're feeling anxious. Let's work through this together.",
            'neutral': "Thanks for checking in. How are you really feeling today?"
        }

        response = response_map.get(emotion, "Thank you for sharing how you feel.")

        try:
            self.text_to_speech.speak(response)
        except Exception as e:
            print(f"TTS error: {e}")

    def record_audio_sample(self, duration=5):
        """Record a fixed-duration audio sample"""
        print(f"Recording {duration} second audio sample...")

        frames = []
        stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        try:
            for i in range(0, int(self.sample_rate / self.chunk_size * duration)):
                data = stream.read(self.chunk_size)
                frames.append(data)

            audio_data = b''.join(frames)

            # Process the recording
            self._process_utterance(audio_data)

        finally:
            stream.stop_stream()
            stream.close()

    def get_voice_interface_status(self):
        """Get voice interface status"""
        return {
            'is_listening': self.is_listening,
            'is_processing': self.is_processing,
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size,
            'audio_queue_size': self.audio_queue.qsize(),
            'current_audio_length': len(self.current_audio)
        }

    def calibrate_silence_threshold(self, calibration_duration=3):
        """Calibrate silence threshold based on ambient noise"""
        print("Calibrating silence threshold...")

        frames = []
        stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        try:
            for i in range(0, int(self.sample_rate / self.chunk_size * calibration_duration)):
                data = stream.read(self.chunk_size)
                frames.append(data)

            # Calculate ambient noise level
            all_audio = b''.join(frames)
            audio_array = np.frombuffer(all_audio, dtype=np.int16)
            ambient_rms = np.sqrt(np.mean(audio_array ** 2))

            # Set threshold slightly above ambient noise
            self.silence_threshold = ambient_rms * 1.5

            print(f"Silence threshold calibrated: {self.silence_threshold:.2f}")

        finally:
            stream.stop_stream()
            stream.close()


class TextToSpeech:
    """Simple text-to-speech interface (placeholder implementation)"""

    def __init__(self):
        self.is_speaking = False

    def speak(self, text):
        """Convert text to speech"""
        # In a real implementation, this would use a TTS engine like pyttsx3 or gTTS
        # For now, just print the text
        print(f"SENTIO: {text}")
        self.is_speaking = True

        # Simulate speaking time
        speaking_time = len(text.split()) * 0.1  # Rough estimate
        time.sleep(min(speaking_time, 3))

        self.is_speaking = False

    def stop_speaking(self):
        """Stop current speech"""
        self.is_speaking = False


# Example usage
class VoiceEmotionAssistant:
    def __init__(self, emotion_orchestrator, feature_extractor):
        self.voice_interface = VoiceInterface(
            emotion_orchestrator,
            feature_extractor,
            text_to_speech=TextToSpeech()
        )

    def start_assistant(self):
        """Start the voice emotion assistant"""
        print("Starting SENTIO Voice Emotion Assistant")
        print("Commands:")
        print("  'start' - Begin listening")
        print("  'stop'  - Stop listening")
        print("  'record' - Record a sample")
        print("  'calibrate' - Calibrate microphone")
        print("  'status' - Check status")
        print("  'quit' - Exit assistant")

        while True:
            try:
                command = input("\nSENTIO Voice Assistant > ").strip().lower()

                if command == 'start':
                    self.voice_interface.start_listening()
                elif command == 'stop':
                    self.voice_interface.stop_listening()
                elif command == 'record':
                    self.voice_interface.record_audio_sample()
                elif command == 'calibrate':
                    self.voice_interface.calibrate_silence_threshold()
                elif command == 'status':
                    status = self.voice_interface.get_voice_interface_status()
                    print(f"Status: {status}")
                elif command in ['quit', 'exit']:
                    self.voice_interface.stop_listening()
                    print("Goodbye!")
                    break
                else:
                    print("Unknown command")

            except KeyboardInterrupt:
                self.voice_interface.stop_listening()
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")