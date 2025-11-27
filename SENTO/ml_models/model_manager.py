import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json


class ModelManager:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.model_metadata = {}
        self._ensure_models_directory()

    def _ensure_models_directory(self):
        """Ensure models directory exists"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def save_model(self, model, model_name, metadata=None):
        """Save a trained model"""
        if metadata is None:
            metadata = {}

        # Add save timestamp
        metadata['saved_at'] = datetime.now().isoformat()
        metadata['model_name'] = model_name

        # Save model file
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)

        # Save metadata
        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update in-memory storage
        self.loaded_models[model_name] = model
        self.model_metadata[model_name] = metadata

        return model_path

    def load_model(self, model_name):
        """Load a trained model"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")

        try:
            # Load model
            model = joblib.load(model_path)
            self.loaded_models[model_name] = model

            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.model_metadata[model_name] = metadata

            return model

        except Exception as e:
            raise Exception(f"Error loading model {model_name}: {str(e)}")

    def get_model_metadata(self, model_name):
        """Get metadata for a model"""
        if model_name in self.model_metadata:
            return self.model_metadata[model_name]

        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)

        return {}

    def list_available_models(self):
        """List all available models"""
        models = []

        for filename in os.listdir(self.models_dir):
            if filename.endswith('.joblib'):
                model_name = filename.replace('.joblib', '')
                metadata = self.get_model_metadata(model_name)
                models.append({
                    'name': model_name,
                    'metadata': metadata
                })

        return models

    def delete_model(self, model_name):
        """Delete a model"""
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")

        # Remove from memory
        self.loaded_models.pop(model_name, None)
        self.model_metadata.pop(model_name, None)

        # Remove files
        deleted_files = []
        if os.path.exists(model_path):
            os.remove(model_path)
            deleted_files.append(model_path)

        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            deleted_files.append(metadata_path)

        return deleted_files

    def update_model_metadata(self, model_name, new_metadata):
        """Update model metadata"""
        if model_name not in self.model_metadata:
            # Try to load existing metadata
            existing_metadata = self.get_model_metadata(model_name)
            self.model_metadata[model_name] = existing_metadata

        # Update metadata
        self.model_metadata[model_name].update(new_metadata)
        self.model_metadata[model_name]['updated_at'] = datetime.now().isoformat()

        # Save to file
        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata[model_name], f, indent=2)

        return True

    def get_model_performance(self, model_name):
        """Get performance metrics for a model"""
        metadata = self.get_model_metadata(model_name)
        return metadata.get('performance_metrics', {})

    def validate_model(self, model_name, validation_data, validation_labels):
        """Validate model performance"""
        model = self.load_model(model_name)

        try:
            predictions = model.predict(validation_data)

            # Calculate metrics based on model type
            if hasattr(model, 'predict_proba'):
                # Classification model
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                accuracy = accuracy_score(validation_labels, predictions)
                precision = precision_score(validation_labels, predictions, average='weighted')
                recall = recall_score(validation_labels, predictions, average='weighted')
                f1 = f1_score(validation_labels, predictions, average='weighted')

                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
            else:
                # Regression model
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

                mse = mean_squared_error(validation_labels, predictions)
                mae = mean_absolute_error(validation_labels, predictions)
                r2 = r2_score(validation_labels, predictions)

                metrics = {
                    'mse': mse,
                    'mae': mae,
                    'r2_score': r2
                }

            # Update metadata
            self.update_model_metadata(model_name, {
                'last_validated': datetime.now().isoformat(),
                'performance_metrics': metrics
            })

            return metrics

        except Exception as e:
            raise Exception(f"Model validation failed: {str(e)}")


class EmotionalModelManager(ModelManager):
    def __init__(self, models_dir="emotional_models"):
        super().__init__(models_dir)
        self.required_models = [
            'voice_emotion_classifier',
            'text_sentiment_analyzer',
            'stress_predictor',
            'emotional_pattern_detector'
        ]

    def initialize_default_models(self):
        """Initialize default models for SENTIO"""
        default_models = {}

        for model_name in self.required_models:
            try:
                model = self.load_model(model_name)
                default_models[model_name] = model
            except FileNotFoundError:
                print(f"Warning: Default model {model_name} not found")
                default_models[model_name] = None

        return default_models

    def get_voice_emotion_model(self):
        """Get voice emotion classification model"""
        try:
            return self.load_model('voice_emotion_classifier')
        except FileNotFoundError:
            print("Voice emotion model not found. Using fallback.")
            return None

    def get_text_sentiment_model(self):
        """Get text sentiment analysis model"""
        try:
            return self.load_model('text_sentiment_analyzer')
        except FileNotFoundError:
            print("Text sentiment model not found. Using fallback.")
            return None

    def get_stress_prediction_model(self):
        """Get stress prediction model"""
        try:
            return self.load_model('stress_predictor')
        except FileNotFoundError:
            print("Stress prediction model not found. Using fallback.")
            return None

    def check_model_health(self):
        """Check health of all required models"""
        health_report = {}

        for model_name in self.required_models:
            try:
                model = self.load_model(model_name)
                metadata = self.get_model_metadata(model_name)

                health_report[model_name] = {
                    'status': 'healthy',
                    'loaded': True,
                    'metadata': metadata,
                    'last_trained': metadata.get('saved_at', 'unknown'),
                    'performance': metadata.get('performance_metrics', {})
                }

            except Exception as e:
                health_report[model_name] = {
                    'status': 'unhealthy',
                    'loaded': False,
                    'error': str(e),
                    'last_trained': 'unknown'
                }

        return health_report

    def retrain_model(self, model_name, training_data, training_labels, model_class, **kwargs):
        """Retrain a model with new data"""
        try:
            # Initialize new model
            model = model_class(**kwargs)

            # Train model
            model.fit(training_data, training_labels)

            # Create metadata
            metadata = {
                'retrained_at': datetime.now().isoformat(),
                'training_samples': len(training_data),
                'model_class': model_class.__name__,
                'training_parameters': kwargs
            }

            # Save model
            self.save_model(model, model_name, metadata)

            return True

        except Exception as e:
            print(f"Error retraining model {model_name}: {e}")
            return False

    def create_model_backup(self, backup_dir="model_backups"):
        """Create backup of all models"""
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"models_backup_{backup_timestamp}")
        os.makedirs(backup_path)

        # Copy all model files
        import shutil
        for filename in os.listdir(self.models_dir):
            src_path = os.path.join(self.models_dir, filename)
            dst_path = os.path.join(backup_path, filename)
            shutil.copy2(src_path, dst_path)

        # Create backup manifest
        manifest = {
            'backup_created': datetime.now().isoformat(),
            'models_backed_up': len([f for f in os.listdir(self.models_dir) if f.endswith('.joblib')]),
            'source_directory': self.models_dir,
            'backup_directory': backup_path
        }

        manifest_path = os.path.join(backup_path, 'backup_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        return backup_path