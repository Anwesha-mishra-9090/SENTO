import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import json


class ModelServer:
    def __init__(self, model_manager, cache_manager=None):
        self.model_manager = model_manager
        self.cache_manager = cache_manager
        self.loaded_models = {}
        self.model_metrics = {}
        self.prediction_queue = []
        self.is_serving = False
        self._initialize_serving()

    def _initialize_serving(self):
        """Initialize model serving environment"""
        # Load essential models
        essential_models = [
            'voice_emotion_classifier',
            'text_sentiment_analyzer',
            'stress_predictor'
        ]

        for model_name in essential_models:
            try:
                model = self.model_manager.load_model(model_name)
                self.loaded_models[model_name] = model
                self.model_metrics[model_name] = {
                    'load_time': datetime.now(),
                    'prediction_count': 0,
                    'average_inference_time': 0,
                    'error_count': 0
                }
                print(f"[SUCCESS] Loaded model: {model_name}")
            except Exception as e:
                print(f"[ERROR] Failed to load model {model_name}: {e}")
                # Create fallback model entry
                self.loaded_models[model_name] = None

    def predict_voice_emotion(self, audio_features, use_cache=True):
        """Predict emotion from voice features"""
        model_name = 'voice_emotion_classifier'

        if use_cache and self.cache_manager:
            cache_key = f"voice_emotion_{hash(str(audio_features))}"
            cached_result = self.cache_manager.get_cached_emotion_analysis(cache_key)
            if cached_result:
                return cached_result

        start_time = time.time()

        try:
            if model_name not in self.loaded_models or self.loaded_models[model_name] is None:
                try:
                    self.loaded_models[model_name] = self.model_manager.load_model(model_name)
                except:
                    # Use fallback if model not available
                    return self._get_fallback_emotion_prediction()

            model = self.loaded_models[model_name]

            # Prepare features for prediction
            if isinstance(audio_features, dict):
                feature_vector = np.array(list(audio_features.values())).reshape(1, -1)
            else:
                feature_vector = audio_features.reshape(1, -1)

            # Make prediction
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(feature_vector)[0]
                prediction = model.predict(feature_vector)[0]

                result = {
                    'emotion': prediction,
                    'confidence': np.max(probabilities),
                    'probabilities': dict(zip(model.classes_, probabilities)),
                    'model_used': model_name,
                    'inference_time': time.time() - start_time
                }
            else:
                prediction = model.predict(feature_vector)[0]
                result = {
                    'emotion': prediction,
                    'confidence': 0.8,  # Default confidence
                    'model_used': model_name,
                    'inference_time': time.time() - start_time
                }

            # Update metrics
            self._update_model_metrics(model_name, time.time() - start_time)

            # Cache result
            if use_cache and self.cache_manager:
                self.cache_manager.cache_emotion_analysis(str(audio_features), result)

            return result

        except Exception as e:
            self._update_model_metrics(model_name, 0, error=True)
            print(f"Voice emotion prediction error: {e}")
            return self._get_fallback_emotion_prediction()

    def predict_text_sentiment(self, text_features, use_cache=True):
        """Predict sentiment from text features"""
        model_name = 'text_sentiment_analyzer'

        if use_cache and self.cache_manager:
            cache_key = f"text_sentiment_{hash(str(text_features))}"
            cached_result = self.cache_manager.get_cached_emotion_analysis(cache_key)
            if cached_result:
                return cached_result

        start_time = time.time()

        try:
            if model_name not in self.loaded_models or self.loaded_models[model_name] is None:
                try:
                    self.loaded_models[model_name] = self.model_manager.load_model(model_name)
                except:
                    # Use fallback if model not available
                    return self._get_fallback_sentiment_prediction()

            model = self.loaded_models[model_name]

            # Prepare features
            if isinstance(text_features, dict):
                feature_vector = np.array(list(text_features.values())).reshape(1, -1)
            else:
                feature_vector = text_features.reshape(1, -1)

            # Make prediction
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(feature_vector)[0]
                prediction = model.predict(feature_vector)[0]

                result = {
                    'sentiment': prediction,
                    'confidence': np.max(probabilities),
                    'probabilities': dict(zip(model.classes_, probabilities)),
                    'model_used': model_name,
                    'inference_time': time.time() - start_time
                }
            else:
                prediction = model.predict(feature_vector)[0]
                result = {
                    'sentiment': prediction,
                    'confidence': 0.7,
                    'model_used': model_name,
                    'inference_time': time.time() - start_time
                }

            # Update metrics
            self._update_model_metrics(model_name, time.time() - start_time)

            # Cache result
            if use_cache and self.cache_manager:
                self.cache_manager.cache_emotion_analysis(str(text_features), result)

            return result

        except Exception as e:
            self._update_model_metrics(model_name, 0, error=True)
            print(f"Text sentiment prediction error: {e}")
            return self._get_fallback_sentiment_prediction()

    def predict_stress_level(self, temporal_features, emotional_context, use_cache=True):
        """Predict stress level from temporal and emotional features"""
        model_name = 'stress_predictor'

        if use_cache and self.cache_manager:
            user_id = emotional_context.get('user_id', 'default')
            cached_result = self.cache_manager.get_cached_stress_prediction(user_id)
            if cached_result:
                return cached_result

        start_time = time.time()

        try:
            if model_name not in self.loaded_models or self.loaded_models[model_name] is None:
                try:
                    self.loaded_models[model_name] = self.model_manager.load_model(model_name)
                except:
                    # Use fallback if model not available
                    return self._get_fallback_stress_prediction()

            model = self.loaded_models[model_name]

            # Combine features for stress prediction
            combined_features = self._combine_stress_features(temporal_features, emotional_context)
            feature_vector = np.array(list(combined_features.values())).reshape(1, -1)

            # Make prediction
            prediction = model.predict(feature_vector)[0]

            result = {
                'stress_level': float(prediction),
                'risk_category': self._classify_stress_risk(prediction),
                'feature_importance': self._get_stress_feature_importance(combined_features),
                'model_used': model_name,
                'inference_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

            # Update metrics
            self._update_model_metrics(model_name, time.time() - start_time)

            # Cache result
            if use_cache and self.cache_manager:
                user_id = emotional_context.get('user_id', 'default')
                self.cache_manager.cache_stress_prediction(user_id, result)

            return result

        except Exception as e:
            self._update_model_metrics(model_name, 0, error=True)
            print(f"Stress prediction error: {e}")
            return self._get_fallback_stress_prediction()

    def _combine_stress_features(self, temporal_features, emotional_context):
        """Combine features for stress prediction"""
        combined = {}

        # Add temporal features
        if temporal_features:
            combined.update(temporal_features)

        # Add emotional context features
        if emotional_context:
            emotion = emotional_context.get('current_emotion', 'neutral')
            intensity = emotional_context.get('intensity', 1.0)
            valence = emotional_context.get('valence', 0.0)

            # Convert emotion to numerical features
            emotion_mapping = {'happy': 1, 'sad': -1, 'angry': -0.8,
                               'fear': -0.9, 'surprise': 0.2, 'neutral': 0}
            combined['emotion_value'] = emotion_mapping.get(emotion, 0)
            combined['emotional_intensity'] = intensity
            combined['emotional_valence'] = valence

        return combined

    def _classify_stress_risk(self, stress_level):
        """Classify stress risk category"""
        if stress_level < 0.3:
            return 'low'
        elif stress_level < 0.6:
            return 'moderate'
        else:
            return 'high'

    def _get_stress_feature_importance(self, features):
        """Get feature importance for stress prediction (simplified)"""
        importance = {}

        # Simplified feature importance based on domain knowledge
        high_importance_features = ['emotional_intensity', 'emotion_value', 'temporal_std']
        medium_importance_features = ['emotional_valence', 'temporal_mean', 'transition_frequency']

        for feature in features.keys():
            if feature in high_importance_features:
                importance[feature] = 'high'
            elif feature in medium_importance_features:
                importance[feature] = 'medium'
            else:
                importance[feature] = 'low'

        return importance

    def batch_predict(self, prediction_requests):
        """Process multiple prediction requests in batch"""
        results = []

        for request in prediction_requests:
            try:
                if request['type'] == 'voice_emotion':
                    result = self.predict_voice_emotion(request['features'], use_cache=False)
                elif request['type'] == 'text_sentiment':
                    result = self.predict_text_sentiment(request['features'], use_cache=False)
                elif request['type'] == 'stress_level':
                    result = self.predict_stress_level(
                        request.get('temporal_features', {}),
                        request.get('emotional_context', {}),
                        use_cache=False
                    )
                else:
                    result = {'error': f"Unknown prediction type: {request['type']}"}

                results.append({
                    'request_id': request.get('id', 'unknown'),
                    'result': result
                })

            except Exception as e:
                results.append({
                    'request_id': request.get('id', 'unknown'),
                    'error': str(e)
                })

        return results

    def _update_model_metrics(self, model_name, inference_time, error=False):
        """Update model performance metrics - FIXED"""
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = {
                'load_time': datetime.now(),
                'prediction_count': 0,
                'total_inference_time': 0.0,  # Initialize as float
                'error_count': 0,
                'average_inference_time': 0.0
            }

        metrics = self.model_metrics[model_name]
        metrics['prediction_count'] += 1

        if error:
            metrics['error_count'] += 1
        else:
            # FIX: Ensure total_inference_time exists and is numeric
            if 'total_inference_time' not in metrics:
                metrics['total_inference_time'] = 0.0
            metrics['total_inference_time'] += inference_time

            # FIX: Safe division
            successful_predictions = metrics['prediction_count'] - metrics['error_count']
            if successful_predictions > 0:
                metrics['average_inference_time'] = metrics['total_inference_time'] / successful_predictions

    def get_serving_metrics(self):
        """Get comprehensive serving metrics"""
        metrics = {
            'total_models_loaded': len([m for m in self.loaded_models.values() if m is not None]),
            'models_loaded': [name for name, model in self.loaded_models.items() if model is not None],
            'model_performance': {},
            'cache_statistics': {},
            'system_health': self._check_system_health()
        }

        # Add model-specific metrics
        for model_name, model_metrics in self.model_metrics.items():
            metrics['model_performance'][model_name] = {
                'prediction_count': model_metrics['prediction_count'],
                'error_count': model_metrics['error_count'],
                'error_rate': model_metrics['error_count'] / max(1, model_metrics['prediction_count']),
                'average_inference_time': model_metrics.get('average_inference_time', 0),
                'uptime_hours': (datetime.now() - model_metrics['load_time']).total_seconds() / 3600
            }

        # Add cache statistics if available
        if self.cache_manager:
            metrics['cache_statistics'] = self.cache_manager.get_all_stats()

        return metrics

    def _check_system_health(self):
        """Check overall system health"""
        health = {
            'status': 'healthy',
            'issues': [],
            'timestamp': datetime.now().isoformat()
        }

        # Check if essential models are loaded
        essential_models = ['voice_emotion_classifier', 'text_sentiment_analyzer']
        for model_name in essential_models:
            if model_name not in self.loaded_models or self.loaded_models[model_name] is None:
                health['status'] = 'degraded'
                health['issues'].append(f"Essential model not loaded: {model_name}")

        # Check model error rates
        for model_name, metrics in self.model_metrics.items():
            if metrics['prediction_count'] > 0:
                error_rate = metrics['error_count'] / metrics['prediction_count']
                if error_rate > 0.1:  # 10% error rate threshold
                    health['status'] = 'degraded'
                    health['issues'].append(f"High error rate for {model_name}: {error_rate:.2f}")

        return health

    def _get_fallback_emotion_prediction(self):
        """Get fallback emotion prediction"""
        return {
            'emotion': 'neutral',
            'confidence': 0.5,
            'probabilities': {'neutral': 0.5, 'happy': 0.2, 'sad': 0.2, 'angry': 0.1},
            'model_used': 'fallback',
            'inference_time': 0.001,
            'fallback_reason': 'model_error'
        }

    def _get_fallback_sentiment_prediction(self):
        """Get fallback sentiment prediction"""
        return {
            'sentiment': 'neutral',
            'confidence': 0.5,
            'model_used': 'fallback',
            'inference_time': 0.001,
            'fallback_reason': 'model_error'
        }

    def _get_fallback_stress_prediction(self):
        """Get fallback stress prediction"""
        return {
            'stress_level': 0.3,
            'risk_category': 'low',
            'feature_importance': {},
            'model_used': 'fallback',
            'inference_time': 0.001,
            'fallback_reason': 'model_error',
            'timestamp': datetime.now().isoformat()
        }

    def warmup_models(self, warmup_data=None):
        """Warm up models with sample data"""
        if warmup_data is None:
            warmup_data = self._generate_warmup_data()

        print("Warming up models...")

        for model_name in self.loaded_models.keys():
            try:
                if self.loaded_models[model_name] is None:
                    continue

                if model_name == 'voice_emotion_classifier' and 'voice' in warmup_data:
                    self.predict_voice_emotion(warmup_data['voice'], use_cache=False)
                elif model_name == 'text_sentiment_analyzer' and 'text' in warmup_data:
                    self.predict_text_sentiment(warmup_data['text'], use_cache=False)
                elif model_name == 'stress_predictor' and 'stress' in warmup_data:
                    self.predict_stress_level(
                        warmup_data['stress']['temporal_features'],
                        warmup_data['stress']['emotional_context'],
                        use_cache=False
                    )

                print(f"[SUCCESS] Warmed up: {model_name}")

            except Exception as e:
                print(f"[ERROR] Warmup failed for {model_name}: {e}")

    def _generate_warmup_data(self):
        """Generate sample data for model warmup"""
        return {
            'voice': {
                'rms_energy': 0.1,
                'zero_crossing_rate': 0.05,
                'pitch_mean': 150,
                'mfcc_1_mean': -100,
                'spectral_centroid_mean': 1000
            },
            'text': {
                'word_count': 10,
                'vader_compound': 0.1,
                'textblob_polarity': 0.1,
                'readability_score': 5.0
            },
            'stress': {
                'temporal_features': {
                    'temporal_mean': 0.1,
                    'temporal_std': 0.2,
                    'emotional_volatility': 0.3
                },
                'emotional_context': {
                    'current_emotion': 'neutral',
                    'intensity': 1.0,
                    'valence': 0.0
                }
            }
        }