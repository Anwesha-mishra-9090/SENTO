import numpy as np
import joblib


class EmotionClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.emotion_labels = ['angry', 'happy', 'sad', 'neutral', 'fear', 'surprise']
        self.is_trained = False

    def train_model(self, features, labels):
        """Train emotion classification model"""
        try:
            # Check if we have enough data
            if len(features) == 0 or len(labels) == 0:
                print("No training data provided")
                return False

            # Convert to numpy arrays if needed
            features_array = np.array(features)
            labels_array = np.array(labels)

            # Check feature dimensions
            if len(features_array.shape) != 2:
                print("Invalid feature dimensions")
                return False

            # Try to import sklearn (might not be available)
            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import StandardScaler
            except ImportError as e:
                print(f"scikit-learn not available: {e}")
                return False

            # Scale features
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features_array)

            # Train Random Forest classifier
            self.model = RandomForestClassifier(
                n_estimators=50,  # Reduced for faster training
                max_depth=8,
                random_state=42
            )

            self.model.fit(features_scaled, labels_array)
            self.is_trained = True

            # Calculate training accuracy
            train_accuracy = self.model.score(features_scaled, labels_array)
            print(f"Model trained with accuracy: {train_accuracy:.3f}")

            return True

        except Exception as e:
            print(f"Training error: {e}")
            self.is_trained = False
            return False

    def predict_emotion(self, features):
        """Predict emotion from audio features"""
        if not self.is_trained or self.model is None:
            return self._fallback_prediction(features)

        try:
            # Ensure features are in correct format
            feature_vector = self._prepare_feature_vector(features)
            if feature_vector is None:
                return self._fallback_prediction(features)

            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])

            # Predict emotion
            prediction = self.model.predict(feature_vector_scaled)[0]
            probabilities = self.model.predict_proba(feature_vector_scaled)[0]

            confidence = float(np.max(probabilities))

            # Convert probabilities to dictionary with string keys
            prob_dict = {}
            for i, emotion in enumerate(self.model.classes_):
                prob_dict[str(emotion)] = float(probabilities[i])

            return {
                'emotion': str(prediction),
                'confidence': confidence,
                'probabilities': prob_dict
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_prediction(features)

    def _prepare_feature_vector(self, features):
        """Prepare feature vector for prediction"""
        try:
            if features is None or not features:
                return None

            feature_vector = []

            for key in sorted(features.keys()):
                value = features[key]
                if isinstance(value, (list, np.ndarray)):
                    # Flatten arrays
                    feature_vector.extend(np.array(value).flatten())
                else:
                    # Convert single values to float
                    feature_vector.append(float(value))

            return np.array(feature_vector)

        except Exception as e:
            print(f"Feature preparation error: {e}")
            return None

    def _fallback_prediction(self, features):
        """Fallback emotion prediction using rule-based approach"""
        try:
            # Simple rule-based emotion detection based on audio features
            energy = features.get('rms_energy', 0)
            zcr = features.get('zero_crossing_rate', 0)
            spectral_centroid = features.get('spectral_centroid', 0)

            # Convert to floats
            energy = float(energy)
            zcr = float(zcr)
            spectral_centroid = float(spectral_centroid)

            if energy > 0.1 and zcr > 0.1:
                emotion = 'happy' if spectral_centroid > 0.05 else 'angry'
            elif energy < 0.05:
                emotion = 'sad'
            else:
                emotion = 'neutral'

            # Create basic probability distribution
            probabilities = {emotion: 0.6}
            for other_emotion in self.emotion_labels:
                if other_emotion != emotion:
                    probabilities[other_emotion] = 0.4 / (len(self.emotion_labels) - 1)

            return {
                'emotion': emotion,
                'confidence': 0.6,  # Lower confidence for fallback
                'probabilities': probabilities
            }

        except Exception as e:
            print(f"Fallback prediction error: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 0.5,
                'probabilities': {'neutral': 1.0}
            }

    def save_model(self, filepath):
        """Save trained model to file"""
        try:
            if self.is_trained and self.model is not None:
                model_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'labels': self.emotion_labels
                }
                joblib.dump(model_data, filepath)
                print(f"Model saved to {filepath}")
                return True
            else:
                print("No trained model to save")
                return False
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_model(self, filepath):
        """Load trained model from file"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.emotion_labels = model_data['labels']
            self.is_trained = True
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_trained = False
            return False

    def get_model_info(self):
        """Get information about the current model"""
        return {
            'is_trained': self.is_trained,
            'emotion_labels': self.emotion_labels,
            'model_type': type(self.model).__name__ if self.model else 'None'
        }