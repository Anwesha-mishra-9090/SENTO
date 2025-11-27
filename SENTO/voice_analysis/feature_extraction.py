import numpy as np


class FeatureExtractor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.feature_config = {
            'mfcc': True,
            'spectral_contrast': True,
            'chroma': True,
            'tonnetz': False
        }

    def extract_features(self, audio_data):
        """Extract comprehensive audio features"""
        try:
            # Convert bytes to numpy array
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0
            elif isinstance(audio_data, np.ndarray):
                if audio_data.dtype == np.int16:
                    audio_float = audio_data.astype(np.float32) / 32768.0
                else:
                    audio_float = audio_data.astype(np.float32)
            else:
                print(f"Unsupported audio data type: {type(audio_data)}")
                return {}

            # Check if audio data is valid
            if len(audio_float) == 0:
                print("Empty audio data")
                return {}

            features = {}

            # Extract MFCC features
            if self.feature_config['mfcc']:
                mfcc_features = self._extract_mfcc_features(audio_float)
                if mfcc_features:
                    features.update(mfcc_features)

            # Extract spectral features
            if self.feature_config['spectral_contrast']:
                spectral_features = self._extract_spectral_features(audio_float)
                if spectral_features:
                    features.update(spectral_features)

            # Extract chroma features
            if self.feature_config['chroma']:
                chroma_features = self._extract_chroma_features(audio_float)
                if chroma_features:
                    features.update(chroma_features)

            # Basic statistical features
            statistical_features = self._extract_statistical_features(audio_float)
            if statistical_features:
                features.update(statistical_features)

            return features

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return {}

    def _extract_mfcc_features(self, audio_data):
        """Extract MFCC features"""
        try:
            import librosa

            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=13
            )
            return {
                'mfcc_mean': np.mean(mfccs, axis=1).tolist(),
                'mfcc_std': np.std(mfccs, axis=1).tolist()
            }
        except Exception as e:
            print(f"MFCC extraction error: {e}")
            return {}

    def _extract_spectral_features(self, audio_data):
        """Extract spectral features"""
        try:
            import librosa

            spectral_contrast = librosa.feature.spectral_contrast(
                y=audio_data,
                sr=self.sample_rate
            )
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data,
                sr=self.sample_rate
            )
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data,
                sr=self.sample_rate
            )

            return {
                'spectral_contrast': np.mean(spectral_contrast, axis=1).tolist(),
                'spectral_centroid': float(np.mean(spectral_centroid)),
                'spectral_rolloff': float(np.mean(spectral_rolloff))
            }
        except Exception as e:
            print(f"Spectral feature extraction error: {e}")
            return {}

    def _extract_chroma_features(self, audio_data):
        """Extract chroma features"""
        try:
            import librosa

            chroma = librosa.feature.chroma_stft(
                y=audio_data,
                sr=self.sample_rate
            )
            return {
                'chroma': np.mean(chroma, axis=1).tolist()
            }
        except Exception as e:
            print(f"Chroma feature extraction error: {e}")
            return {}

    def _extract_statistical_features(self, audio_data):
        """Extract basic statistical features"""
        try:
            features = {}

            # Energy features
            features['rms_energy'] = float(np.sqrt(np.mean(audio_data ** 2)))
            features['energy_entropy'] = float(self._calculate_energy_entropy(audio_data))

            # Temporal features
            features['zero_crossing_rate'] = float(self._calculate_zcr(audio_data))

            # Statistical features
            features['amplitude_mean'] = float(np.mean(audio_data))
            features['amplitude_std'] = float(np.std(audio_data))
            features['amplitude_max'] = float(np.max(np.abs(audio_data)))

            return features
        except Exception as e:
            print(f"Statistical feature extraction error: {e}")
            return {}

    def _calculate_energy_entropy(self, audio_data, num_frames=10):
        """Calculate energy entropy of audio signal"""
        try:
            if len(audio_data) == 0:
                return 0.0

            frame_length = len(audio_data) // num_frames
            if frame_length == 0:
                return 0.0

            energies = []
            for i in range(num_frames):
                start = i * frame_length
                end = min((i + 1) * frame_length, len(audio_data))
                if start >= end:
                    continue
                frame_energy = np.sum(audio_data[start:end] ** 2)
                energies.append(frame_energy)

            if not energies:
                return 0.0

            total_energy = np.sum(energies)
            if total_energy == 0:
                return 0.0

            probabilities = np.array(energies) / total_energy
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

            return float(entropy)
        except Exception as e:
            print(f"Energy entropy calculation error: {e}")
            return 0.0

    def _calculate_zcr(self, audio_data):
        """Calculate Zero Crossing Rate"""
        try:
            if len(audio_data) < 2:
                return 0.0
            zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
            return float(zero_crossings / len(audio_data))
        except Exception as e:
            print(f"ZCR calculation error: {e}")
            return 0.0

    def normalize_features(self, features):
        """Normalize features to 0-1 range"""
        try:
            if not features:
                return {}

            normalized = {}
            for key, value in features.items():
                try:
                    if isinstance(value, (list, np.ndarray)):
                        # Normalize each element in array
                        arr = np.array(value)
                        min_val = np.min(arr)
                        max_val = np.max(arr)
                        if max_val - min_val > 1e-10:
                            norm_value = (arr - min_val) / (max_val - min_val)
                        else:
                            norm_value = np.zeros_like(arr)
                        normalized[key] = norm_value.tolist()
                    else:
                        # For single values, use simple scaling
                        normalized[key] = max(0.0, min(1.0, float(value)))
                except Exception as e:
                    print(f"Error normalizing feature {key}: {e}")
                    normalized[key] = value  # Keep original value if normalization fails

            return normalized
        except Exception as e:
            print(f"Feature normalization error: {e}")
            return features  # Return original features if normalization fails