import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


class StressPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'negative_emotion_ratio', 'emotional_volatility', 'avg_intensity',
            'high_intensity_count', 'stress_emotion_frequency', 'recovery_time'
        ]

    def prepare_training_data(self, emotion_data, lookback_days=7, forecast_days=1):
        """Prepare data for stress prediction model"""
        if len(emotion_data) < lookback_days + forecast_days:
            return None, None

        # Convert to DataFrame
        df = self._emotion_data_to_dataframe(emotion_data)

        features = []
        targets = []

        for i in range(lookback_days, len(df) - forecast_days):
            # Lookback window features
            lookback_data = df.iloc[i - lookback_days:i]

            # Extract features from lookback window
            feature_vector = self._extract_features(lookback_data)

            # Target: stress level in forecast period
            forecast_data = df.iloc[i:i + forecast_days]
            target_stress = self._calculate_stress_level(forecast_data)

            features.append(feature_vector)
            targets.append(target_stress)

        return np.array(features), np.array(targets)

    def _emotion_data_to_dataframe(self, emotion_data):
        """Convert emotion data to DataFrame"""
        records = []
        for entry in emotion_data:
            if 'timestamp' in entry:
                records.append({
                    'timestamp': pd.to_datetime(entry['timestamp']),
                    'emotion': entry['emotion'],
                    'intensity': entry.get('intensity', 1.0),
                    'valence': entry.get('valence', 0.0)
                })

        df = pd.DataFrame(records)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def _extract_features(self, data_window):
        """Extract features from emotion data window"""
        features = []

        # 1. Negative emotion ratio
        negative_emotions = ['sad', 'angry', 'fear']
        negative_count = sum(1 for emotion in data_window['emotion'] if emotion in negative_emotions)
        negative_ratio = negative_count / len(data_window)
        features.append(negative_ratio)

        # 2. Emotional volatility
        valence_changes = data_window['valence'].diff().abs().mean()
        features.append(valence_changes if not np.isnan(valence_changes) else 0.0)

        # 3. Average intensity
        avg_intensity = data_window['intensity'].mean()
        features.append(avg_intensity)

        # 4. High intensity count
        high_intensity_threshold = data_window['intensity'].quantile(0.75)
        high_intensity_count = sum(data_window['intensity'] > high_intensity_threshold)
        features.append(high_intensity_count)

        # 5. Stress emotion frequency
        stress_emotions = ['angry', 'fear']
        stress_count = sum(1 for emotion in data_window['emotion'] if emotion in stress_emotions)
        stress_frequency = stress_count / len(data_window)
        features.append(stress_frequency)

        # 6. Recovery time (simplified - time from negative to positive)
        recovery_time = self._estimate_recovery_time(data_window)
        features.append(recovery_time)

        return features

    def _estimate_recovery_time(self, data_window):
        """Estimate emotional recovery time from negative states"""
        negative_emotions = ['sad', 'angry', 'fear']
        recovery_sequences = []

        in_negative_phase = False
        negative_start = None

        for i, (_, row) in enumerate(data_window.iterrows()):
            if row['emotion'] in negative_emotions and not in_negative_phase:
                in_negative_phase = True
                negative_start = i
            elif row['emotion'] not in negative_emotions and in_negative_phase:
                in_negative_phase = False
                if negative_start is not None:
                    recovery_time = i - negative_start
                    recovery_sequences.append(recovery_time)

        if recovery_sequences:
            return np.mean(recovery_sequences)
        else:
            return 0.0

    def _calculate_stress_level(self, data_window):
        """Calculate stress level from emotion data"""
        if data_window.empty:
            return 0.0

        stress_indicators = []

        # Negative emotion presence
        negative_emotions = ['sad', 'angry', 'fear']
        negative_ratio = sum(1 for emotion in data_window['emotion'] if emotion in negative_emotions) / len(data_window)
        stress_indicators.append(negative_ratio)

        # High intensity
        high_intensity_ratio = sum(data_window['intensity'] > 1.5) / len(data_window)
        stress_indicators.append(high_intensity_ratio)

        # Low valence
        avg_valence = data_window['valence'].mean()
        valence_stress = max(0, -avg_valence)  # Only negative valence contributes
        stress_indicators.append(valence_stress)

        return np.mean(stress_indicators)

    def train_model(self, emotion_data, lookback_days=7, forecast_days=1):
        """Train the stress prediction model"""
        try:
            features, targets = self.prepare_training_data(emotion_data, lookback_days, forecast_days)

            if features is None or len(features) == 0:
                print("Insufficient data for training")
                return False

            # Scale features
            features_scaled = self.scaler.fit_transform(features)

            # Train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                min_samples_split=5,
                min_samples_leaf=2
            )

            self.model.fit(features_scaled, targets)
            self.is_trained = True

            # Calculate training performance
            train_predictions = self.model.predict(features_scaled)
            mse = np.mean((train_predictions - targets) ** 2)
            print(f"Model trained successfully. MSE: {mse:.4f}")

            return True

        except Exception as e:
            print(f"Training error: {e}")
            return False

    def predict_stress(self, recent_emotion_data, forecast_horizon=24):
        """Predict stress levels for future period"""
        if not self.is_trained or self.model is None:
            return self._fallback_prediction(recent_emotion_data)

        try:
            # Convert to DataFrame and get recent data
            df = self._emotion_data_to_dataframe(recent_emotion_data)

            if len(df) < 7:  # Minimum lookback period
                return self._fallback_prediction(recent_emotion_data)

            # Use most recent data for prediction
            lookback_data = df.tail(7)

            # Extract features
            features = self._extract_features(lookback_data)
            features_scaled = self.scaler.transform([features])

            # Predict stress level
            predicted_stress = self.model.predict(features_scaled)[0]
            confidence = self._calculate_prediction_confidence(features)

            # Generate prediction insights
            insights = self._generate_stress_insights(features, predicted_stress)

            return {
                'predicted_stress': float(predicted_stress),
                'confidence': float(confidence),
                'risk_level': self._classify_risk_level(predicted_stress),
                'forecast_horizon_hours': forecast_horizon,
                'key_risk_factors': self._identify_risk_factors(features),
                'recommendations': self._generate_stress_recommendations(predicted_stress, features),
                'insights': insights
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_prediction(recent_emotion_data)

    def _calculate_prediction_confidence(self, features):
        """Calculate prediction confidence based on feature quality"""
        # Simple confidence calculation based on data quality indicators
        confidence_indicators = []

        # Negative emotion ratio confidence
        neg_ratio = features[0]
        if 0.1 <= neg_ratio <= 0.9:  # Neither too low nor too high
            confidence_indicators.append(0.8)
        else:
            confidence_indicators.append(0.5)

        # Volatility confidence
        volatility = features[1]
        if volatility > 0:  # Some variability present
            confidence_indicators.append(0.7)
        else:
            confidence_indicators.append(0.3)

        return np.mean(confidence_indicators) if confidence_indicators else 0.5

    def _classify_risk_level(self, stress_score):
        """Classify stress risk level"""
        if stress_score < 0.3:
            return 'low'
        elif stress_score < 0.6:
            return 'moderate'
        else:
            return 'high'

    def _identify_risk_factors(self, features):
        """Identify key risk factors from features"""
        risk_factors = []

        feature_values = {
            'negative_emotion_ratio': features[0],
            'emotional_volatility': features[1],
            'avg_intensity': features[2],
            'high_intensity_count': features[3],
            'stress_emotion_frequency': features[4],
            'recovery_time': features[5]
        }

        # Identify high-risk factors
        if feature_values['negative_emotion_ratio'] > 0.5:
            risk_factors.append("High frequency of negative emotions")

        if feature_values['emotional_volatility'] > 0.4:
            risk_factors.append("High emotional variability")

        if feature_values['stress_emotion_frequency'] > 0.3:
            risk_factors.append("Frequent stress-related emotions")

        if feature_values['recovery_time'] > 3:
            risk_factors.append("Slow emotional recovery")

        return risk_factors if risk_factors else ["No significant risk factors identified"]

    def _generate_stress_insights(self, features, predicted_stress):
        """Generate insights from stress prediction"""
        insights = []

        if predicted_stress > 0.6:
            insights.append("High stress risk detected in upcoming period")
        elif predicted_stress > 0.3:
            insights.append("Moderate stress levels expected")
        else:
            insights.append("Low stress levels anticipated")

        if features[1] > 0.5:  # High volatility
            insights.append("Emotional instability may contribute to stress")

        if features[4] > 0.4:  # High stress emotion frequency
            insights.append("Recent stress-related emotions indicate elevated risk")

        return insights

    def _generate_stress_recommendations(self, predicted_stress, features):
        """Generate stress management recommendations"""
        recommendations = []

        if predicted_stress > 0.6:
            recommendations.append("Consider practicing deep breathing exercises")
            recommendations.append("Schedule breaks throughout the day")
            recommendations.append("Reach out to support network if needed")
        elif predicted_stress > 0.3:
            recommendations.append("Practice mindfulness meditation")
            recommendations.append("Maintain regular sleep schedule")
            recommendations.append("Engage in light physical activity")
        else:
            recommendations.append("Continue current stress management practices")
            recommendations.append("Maintain healthy routines")

        # Feature-specific recommendations
        if features[0] > 0.5:  # High negative emotion ratio
            recommendations.append("Focus on positive activity engagement")

        if features[1] > 0.4:  # High volatility
            recommendations.append("Establish consistent daily routines")

        return recommendations

    def _fallback_prediction(self, emotion_data):
        """Fallback prediction when model is not available"""
        if not emotion_data:
            return {
                'predicted_stress': 0.3,
                'confidence': 0.3,
                'risk_level': 'unknown',
                'forecast_horizon_hours': 24,
                'key_risk_factors': ['Insufficient data for accurate prediction'],
                'recommendations': ['Continue tracking emotions for better predictions'],
                'insights': ['Prediction model requires more training data']
            }

        # Simple rule-based fallback
        recent_emotions = emotion_data[-5:]  # Last 5 entries
        negative_count = sum(1 for entry in recent_emotions
                             if entry['emotion'] in ['sad', 'angry', 'fear'])

        stress_level = negative_count / len(recent_emotions)

        return {
            'predicted_stress': float(stress_level),
            'confidence': 0.4,
            'risk_level': 'moderate' if stress_level > 0.3 else 'low',
            'forecast_horizon_hours': 24,
            'key_risk_factors': ['Using basic pattern analysis'],
            'recommendations': ['Model training will improve prediction accuracy'],
            'insights': ['Basic analysis based on recent emotional patterns']
        }

    def predict_burnout_risk(self, emotion_data, time_period="30d"):
        """Predict burnout risk based on emotional patterns"""
        if len(emotion_data) < 10:
            return self._fallback_burnout_prediction()

        # Analyze long-term patterns for burnout risk
        df = self._emotion_data_to_dataframe(emotion_data)

        burnout_indicators = {
            'chronic_stress': self._assess_chronic_stress(df),
            'emotional_exhaustion': self._assess_emotional_exhaustion(df),
            'reduced_accomplishment': self._assess_reduced_accomplishment(df),
            'depersonalization': self._assess_depersonalization(df)
        }

        burnout_score = np.mean(list(burnout_indicators.values()))

        return {
            'burnout_risk_score': float(burnout_score),
            'risk_level': self._classify_burnout_risk(burnout_score),
            'indicators': burnout_indicators,
            'timeframe': 'next 1-2 months',
            'recommendations': self._generate_burnout_recommendations(burnout_score, burnout_indicators)
        }

    def _assess_chronic_stress(self, df):
        """Assess chronic stress indicator"""
        if len(df) < 7:
            return 0.0

        # High negative emotion persistence
        negative_emotions = ['sad', 'angry', 'fear']
        recent_data = df.tail(7)
        negative_ratio = sum(1 for emotion in recent_data['emotion'] if emotion in negative_emotions) / len(recent_data)

        return min(negative_ratio * 1.5, 1.0)

    def _assess_emotional_exhaustion(self, df):
        """Assess emotional exhaustion indicator"""
        if len(df) < 10:
            return 0.0

        # Low emotional variability and intensity
        intensity_variation = df['intensity'].std()
        emotion_variation = len(df['emotion'].unique()) / len(df)

        # Low variation suggests exhaustion
        exhaustion_score = (1 - intensity_variation) * 0.5 + (1 - emotion_variation) * 0.5
        return min(exhaustion_score, 1.0)

    def _assess_reduced_accomplishment(self, df):
        """Assess reduced accomplishment indicator"""
        # This would typically use productivity data
        # For now, use emotional patterns as proxy
        if len(df) < 5:
            return 0.0

        # High neutral/low arousal emotions might indicate reduced engagement
        low_arousal_emotions = ['sad', 'neutral']
        low_arousal_ratio = sum(1 for emotion in df['emotion'] if emotion in low_arousal_emotions) / len(df)

        return min(low_arousal_ratio * 1.2, 1.0)

    def _assess_depersonalization(self, df):
        """Assess depersonalization indicator"""
        # Difficult to assess from emotion data alone
        # Placeholder implementation
        return 0.3

    def _classify_burnout_risk(self, burnout_score):
        """Classify burnout risk level"""
        if burnout_score < 0.3:
            return 'low'
        elif burnout_score < 0.6:
            return 'moderate'
        else:
            return 'high'

    def _generate_burnout_recommendations(self, burnout_score, indicators):
        """Generate burnout prevention recommendations"""
        recommendations = []

        if burnout_score > 0.6:
            recommendations.append("Consider professional mental health support")
            recommendations.append("Implement strict work-life boundaries")
            recommendations.append("Prioritize rest and recovery")
        elif burnout_score > 0.3:
            recommendations.append("Schedule regular breaks and vacations")
            recommendations.append("Practice stress management techniques")
            recommendations.append("Maintain social connections")

        # Indicator-specific recommendations
        if indicators['chronic_stress'] > 0.5:
            recommendations.append("Develop daily relaxation practices")

        if indicators['emotional_exhaustion'] > 0.5:
            recommendations.append("Focus on activities that bring joy and fulfillment")

        return recommendations if recommendations else ["Maintain current wellness practices"]

    def _fallback_burnout_prediction(self):
        """Fallback burnout prediction"""
        return {
            'burnout_risk_score': 0.3,
            'risk_level': 'unknown',
            'indicators': {'chronic_stress': 0.3, 'emotional_exhaustion': 0.3,
                           'reduced_accomplishment': 0.3, 'depersonalization': 0.3},
            'timeframe': 'unknown',
            'recommendations': ['More emotional data needed for accurate burnout assessment']
        }

    def save_model(self, filepath):
        """Save trained model to file"""
        if self.is_trained:
            import joblib
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, filepath)
            return True
        return False

    def load_model(self, filepath):
        """Load trained model from file"""
        try:
            import joblib
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            return True
        except:
            return False