class MultiModalFusion:
    def __init__(self):
        self.fusion_strategies = {
            'voice_text': self._fuse_voice_text,
            'confidence_weighted': self._confidence_weighted_fusion
        }

    def fuse_modalities(self, voice_data, text_data, strategy='confidence_weighted'):
        """Fuse emotion data from multiple modalities"""
        if voice_data is None and text_data is None:
            return {"emotion": "neutral", "confidence": 0.0}

        fusion_strategy = self.fusion_strategies.get(strategy, self._confidence_weighted_fusion)
        return fusion_strategy(voice_data, text_data)

    def _fuse_voice_text(self, voice_data, text_data):
        """Basic voice-text fusion"""
        if voice_data is None:
            return text_data
        if text_data is None:
            return voice_data

        # Simple average fusion
        fused_confidence = (voice_data.get('confidence', 0) + text_data.get('confidence', 0)) / 2

        return {
            'emotion': text_data.get('emotion', 'neutral'),  # Prefer text for now
            'confidence': fused_confidence,
            'source': 'multi_modal'
        }

    def _confidence_weighted_fusion(self, voice_data, text_data):
        """Weighted fusion based on confidence scores"""
        if voice_data is None:
            return text_data
        if text_data is None:
            return voice_data

        voice_conf = voice_data.get('confidence', 0)
        text_conf = text_data.get('confidence', 0)

        # Use higher confidence source
        if voice_conf > text_conf:
            return voice_data
        else:
            return text_data

    def calculate_emotional_coherence(self, voice_emotion, text_emotion):
        """Calculate coherence between voice and text emotions"""
        if voice_emotion is None or text_emotion is None:
            return 0.0

        emotion_mapping = {
            'happy': 1, 'sad': -1, 'angry': -0.5, 'fear': -0.8,
            'surprise': 0.3, 'neutral': 0
        }

        voice_val = emotion_mapping.get(voice_emotion.get('emotion', 'neutral'), 0)
        text_val = emotion_mapping.get(text_emotion.get('emotion', 'neutral'), 0)

        coherence = 1 - abs(voice_val - text_val) / 2
        return max(0.0, coherence)