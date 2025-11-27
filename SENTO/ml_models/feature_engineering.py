import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import librosa
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_configs = self._initialize_feature_configs()

    def _initialize_feature_configs(self):
        """Initialize feature engineering configurations"""
        return {
            'audio_features': {
                'mfcc': {'n_mfcc': 13, 'use_delta': True},
                'spectral': {'include_contrast': True, 'include_centroid': True},
                'chroma': {'n_chroma': 12},
                'tonnetz': {'use_tonnetz': True}
            },
            'text_features': {
                'sentiment': {'use_vader': True, 'use_textblob': True},
                'emotional_lexicon': {'custom_lexicon': True},
                'linguistic': {'pos_tags': True, 'readability': True}
            },
            'temporal_features': {
                'rolling_stats': {'window_size': 5},
                'trend_features': {'include_derivatives': True}
            }
        }

    def extract_audio_features(self, audio_data, sample_rate=16000):
        """Extract comprehensive audio features for emotion analysis"""
        features = {}

        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data

            # Basic audio features
            features.update(self._extract_basic_audio_features(audio_float, sample_rate))

            # MFCC features
            features.update(self._extract_mfcc_features(audio_float, sample_rate))

            # Spectral features
            features.update(self._extract_spectral_features(audio_float, sample_rate))

            # Chroma features
            features.update(self._extract_chroma_features(audio_float, sample_rate))

            # Statistical features
            features.update(self._extract_statistical_features(audio_float))

        except Exception as e:
            print(f"Audio feature extraction error: {e}")
            # Return fallback features
            features = self._get_fallback_audio_features()

        return features

    def _extract_basic_audio_features(self, audio_data, sample_rate):
        """Extract basic audio features"""
        features = {}

        # RMS energy
        features['rms_energy'] = np.sqrt(np.mean(audio_data ** 2))

        # Zero crossing rate
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))

        # Pitch features (basic)
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(audio_data,
                                                         fmin=50, fmax=500,
                                                         sr=sample_rate)
            f0 = f0[~np.isnan(f0)]
            if len(f0) > 0:
                features['pitch_mean'] = np.mean(f0)
                features['pitch_std'] = np.std(f0)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
        except:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0

        return features

    def _extract_mfcc_features(self, audio_data, sample_rate):
        """Extract MFCC features"""
        features = {}

        try:
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate,
                                         n_mfcc=self.feature_configs['audio_features']['mfcc']['n_mfcc'])

            # MFCC statistics
            for i in range(len(mfccs)):
                features[f'mfcc_{i + 1}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i + 1}_std'] = np.std(mfccs[i])

            # Delta MFCCs
            if self.feature_configs['audio_features']['mfcc']['use_delta']:
                delta_mfccs = librosa.feature.delta(mfccs)
                for i in range(len(delta_mfccs)):
                    features[f'mfcc_delta_{i + 1}_mean'] = np.mean(delta_mfccs[i])

        except Exception as e:
            print(f"MFCC extraction error: {e}")

        return features

    def _extract_spectral_features(self, audio_data, sample_rate):
        """Extract spectral features"""
        features = {}

        try:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)

            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)

            # Spectral contrast
            if self.feature_configs['audio_features']['spectral']['include_contrast']:
                spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
                for i in range(len(spectral_contrast)):
                    features[f'spectral_contrast_{i + 1}'] = np.mean(spectral_contrast[i])

        except Exception as e:
            print(f"Spectral feature extraction error: {e}")

        return features

    def _extract_chroma_features(self, audio_data, sample_rate):
        """Extract chroma features"""
        features = {}

        try:
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
            chroma_mean = np.mean(chroma, axis=1)

            for i in range(len(chroma_mean)):
                features[f'chroma_{i + 1}'] = chroma_mean[i]

        except Exception as e:
            print(f"Chroma feature extraction error: {e}")

        return features

    def _extract_statistical_features(self, audio_data):
        """Extract statistical features"""
        features = {}

        features['amplitude_mean'] = np.mean(audio_data)
        features['amplitude_std'] = np.std(audio_data)
        features['amplitude_skew'] = pd.Series(audio_data).skew()
        features['amplitude_kurtosis'] = pd.Series(audio_data).kurtosis()

        return features

    def _get_fallback_audio_features(self):
        """Get fallback audio features when extraction fails"""
        return {
            'rms_energy': 0.1,
            'zero_crossing_rate': 0.05,
            'pitch_mean': 150,
            'pitch_std': 20,
            'spectral_centroid_mean': 1000,
            'amplitude_mean': 0.0,
            'amplitude_std': 0.1
        }

    def extract_text_features(self, text_data, emotional_lexicon=None):
        """Extract text features for sentiment analysis"""
        features = {}

        if not text_data or not isinstance(text_data, str):
            return self._get_fallback_text_features()

        try:
            # Basic text statistics
            features.update(self._extract_text_statistics(text_data))

            # Sentiment features
            features.update(self._extract_sentiment_features(text_data))

            # Emotional lexicon features
            if emotional_lexicon:
                features.update(self._extract_lexicon_features(text_data, emotional_lexicon))

            # Readability features
            features.update(self._extract_readability_features(text_data))

        except Exception as e:
            print(f"Text feature extraction error: {e}")
            features = self._get_fallback_text_features()

        return features

    def _extract_text_statistics(self, text):
        """Extract basic text statistics"""
        features = {}

        words = text.split()
        sentences = text.split('. ')

        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
        features['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0

        # Character-level features
        features['char_count'] = len(text)
        features['digit_count'] = sum(c.isdigit() for c in text)
        features['uppercase_ratio'] = sum(c.isupper() for c in text) / len(text) if text else 0

        return features

    def _extract_sentiment_features(self, text):
        """Extract sentiment features using multiple approaches"""
        features = {}

        try:
            # VADER sentiment
            from nltk.sentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            vader_scores = sia.polarity_scores(text)

            features['vader_compound'] = vader_scores['compound']
            features['vader_positive'] = vader_scores['pos']
            features['vader_negative'] = vader_scores['neg']
            features['vader_neutral'] = vader_scores['neu']

            # TextBlob sentiment
            from textblob import TextBlob
            blob = TextBlob(text)
            features['textblob_polarity'] = blob.sentiment.polarity
            features['textblob_subjectivity'] = blob.sentiment.subjectivity

        except Exception as e:
            print(f"Sentiment feature extraction error: {e}")
            # Set default values
            features.update({
                'vader_compound': 0.0,
                'vader_positive': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 1.0,
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0
            })

        return features

    def _extract_lexicon_features(self, text, emotional_lexicon):
        """Extract emotional lexicon features"""
        features = {}

        text_lower = text.lower()
        words = text_lower.split()

        for emotion, emotion_words in emotional_lexicon.items():
            count = sum(1 for word in words if word in emotion_words)
            features[f'lexicon_{emotion}_count'] = count
            features[f'lexicon_{emotion}_ratio'] = count / len(words) if words else 0

        return features

    def _extract_readability_features(self, text):
        """Extract readability features"""
        features = {}

        sentences = text.split('. ')
        words = text.split()

        if len(sentences) > 0 and len(words) > 0:
            # Simple readability score (higher = more complex)
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            features['readability_score'] = (avg_sentence_length + avg_word_length) / 2
        else:
            features['readability_score'] = 0

        return features

    def _get_fallback_text_features(self):
        """Get fallback text features"""
        return {
            'word_count': 0,
            'sentence_count': 0,
            'vader_compound': 0.0,
            'textblob_polarity': 0.0,
            'readability_score': 0.0
        }

    def create_temporal_features(self, emotion_sequence, window_size=5):
        """Create temporal features from emotion sequence"""
        features = {}

        if len(emotion_sequence) < window_size:
            return self._get_fallback_temporal_features()

        try:
            # Convert emotions to numerical values
            emotion_mapping = {'happy': 1, 'sad': -1, 'angry': -0.5,
                               'fear': -0.8, 'surprise': 0.3, 'neutral': 0}
            emotion_values = [emotion_mapping.get(emotion, 0) for emotion in emotion_sequence]

            # Rolling statistics
            emotion_series = pd.Series(emotion_values)
            rolling_mean = emotion_series.rolling(window=window_size).mean()
            rolling_std = emotion_series.rolling(window=window_size).std()

            features['temporal_mean'] = rolling_mean.iloc[-1] if not rolling_mean.empty else 0
            features['temporal_std'] = rolling_std.iloc[-1] if not rolling_std.empty else 0
            features['temporal_trend'] = self._calculate_trend(emotion_values)
            features['emotional_volatility'] = np.std(emotion_values)

            # Transition features
            features['transition_count'] = sum(1 for i in range(1, len(emotion_sequence))
                                               if emotion_sequence[i] != emotion_sequence[i - 1])
            features['transition_frequency'] = features['transition_count'] / len(emotion_sequence)

        except Exception as e:
            print(f"Temporal feature extraction error: {e}")
            features = self._get_fallback_temporal_features()

        return features

    def _calculate_trend(self, values):
        """Calculate trend of values"""
        if len(values) < 2:
            return 0

        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope

    def _get_fallback_temporal_features(self):
        """Get fallback temporal features"""
        return {
            'temporal_mean': 0.0,
            'temporal_std': 0.1,
            'temporal_trend': 0.0,
            'emotional_volatility': 0.1,
            'transition_frequency': 0.0
        }

    def normalize_features(self, features, feature_type='audio'):
        """Normalize features to consistent scale"""
        normalized = {}

        for key, value in features.items():
            if isinstance(value, (int, float)):
                # Apply feature-specific normalization
                if 'mfcc' in key:
                    normalized[key] = self._normalize_mfcc(value)
                elif 'pitch' in key:
                    normalized[key] = self._normalize_pitch(value)
                elif 'vader' in key or 'textblob' in key:
                    normalized[key] = self._normalize_sentiment(value)
                else:
                    normalized[key] = self._normalize_general(value)
            else:
                normalized[key] = value

        return normalized

    def _normalize_mfcc(self, value):
        """Normalize MFCC values"""
        return (value + 100) / 200  # Rough normalization for MFCC range

    def _normalize_pitch(self, value):
        """Normalize pitch values"""
        return min(max(value / 500, 0), 1)  # Assume pitch range 0-500 Hz

    def _normalize_sentiment(self, value):
        """Normalize sentiment values"""
        return (value + 1) / 2  # Convert from [-1,1] to [0,1]

    def _normalize_general(self, value):
        """General normalization for unknown features"""
        return min(max(value, 0), 1)  # Clip to [0,1] range

    def create_feature_vector(self, audio_features=None, text_features=None,
                              temporal_features=None, context_features=None):
        """Create combined feature vector from all feature types"""
        feature_vector = {}

        # Combine all features
        if audio_features:
            feature_vector.update(audio_features)
        if text_features:
            feature_vector.update(text_features)
        if temporal_features:
            feature_vector.update(temporal_features)
        if context_features:
            feature_vector.update(context_features)

        # Convert to numpy array
        feature_names = sorted(feature_vector.keys())
        feature_values = [feature_vector[name] for name in feature_names]

        return np.array(feature_values), feature_names

    def select_important_features(self, features, labels, k=20):
        """Select most important features using statistical tests"""
        if len(features) < 2 or len(np.unique(labels)) < 2:
            return features, list(range(features.shape[1]))

        try:
            # Use ANOVA F-value for feature selection
            selector = SelectKBest(score_func=f_classif, k=min(k, features.shape[1]))
            selected_features = selector.fit_transform(features, labels)

            # Get selected feature indices
            feature_indices = selector.get_support(indices=True)

            return selected_features, feature_indices

        except Exception as e:
            print(f"Feature selection error: {e}")
            return features, list(range(features.shape[1]))