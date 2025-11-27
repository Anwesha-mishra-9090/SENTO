from datetime import datetime


class DataPipeline:
    def __init__(self):
        self.processors = []
        self.validators = []

    def add_processor(self, processor):
        """Add data processor to pipeline"""
        self.processors.append(processor)

    def add_validator(self, validator):
        """Add data validator to pipeline"""
        self.validators.append(validator)

    def process_emotional_data(self, raw_data):
        """Process emotional data through the pipeline"""
        try:
            # Validate input
            if not self._validate_data(raw_data):
                raise ValueError("Invalid emotional data")

            # Process through pipeline
            processed_data = raw_data.copy()
            for processor in self.processors:
                processed_data = processor(processed_data)

            return processed_data

        except Exception as e:
            print(f"Data pipeline error: {e}")
            return None

    def _validate_data(self, data):
        """Validate emotional data structure"""
        required_fields = ['timestamp', 'emotion']
        return all(field in data for field in required_fields)

    def create_emotion_batch(self, emotion_data_list):
        """Create batch processing for multiple emotion entries"""
        processed_batch = []
        for data in emotion_data_list:
            processed = self.process_emotional_data(data)
            if processed:
                processed_batch.append(processed)
        return processed_batch


# Example processors
def timestamp_processor(data):
    """Ensure timestamp is in correct format"""
    if 'timestamp' in data:
        try:
            # Convert to standard format
            dt = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            data['timestamp'] = dt.isoformat()
        except:
            data['timestamp'] = datetime.now().isoformat()
    return data


def confidence_normalizer(data):
    """Normalize confidence scores"""
    if 'confidence' in data:
        data['confidence'] = max(0.0, min(1.0, data['confidence']))
    return data