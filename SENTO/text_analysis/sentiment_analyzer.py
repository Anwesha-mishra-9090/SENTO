import re
import numpy as np
from textblob import TextBlob
from collections import Counter


class SentimentAnalyzer:
    def __init__(self):
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            self.sia = SentimentIntensityAnalyzer()
        except Exception as e:
            print(f"VADER sentiment analyzer not available: {e}")
            self.sia = None

        self.emotion_lexicon = self._load_emotion_lexicon()
        self.intensity_modifiers = {
            'very': 1.5, 'extremely': 2.0, 'really': 1.3, 'so': 1.2,
            'slightly': 0.7, 'somewhat': 0.8, 'quite': 1.1
        }

    def _load_emotion_lexicon(self):
        """Load emotion word lexicon"""
        return {
            'happy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'good', 'fantastic'],
            'sad': ['sad', 'depressed', 'unhappy', 'miserable', 'terrible', 'awful', 'bad', 'horrible'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated', 'upset'],
            'fear': ['scared', 'afraid', 'fear', 'worried', 'anxious', 'nervous', 'terrified'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
            'neutral': ['okay', 'fine', 'alright', 'normal', 'usual', 'ordinary']
        }

    def analyze_sentiment(self, text):
        """Comprehensive sentiment and emotion analysis"""
        if not text or not text.strip():
            return self._get_neutral_response()

        # Clean text
        cleaned_text = self._clean_text(text)

        # Multiple analysis approaches
        vader_scores = self._vader_analysis(cleaned_text)
        textblob_scores = self._textblob_analysis(cleaned_text)
        lexicon_emotion = self._lexicon_based_analysis(cleaned_text)
        intensity_score = self._calculate_intensity(cleaned_text)

        # Fuse results
        final_emotion, confidence = self._fuse_analyses(
            vader_scores, textblob_scores, lexicon_emotion, intensity_score
        )

        return {
            'emotion': final_emotion,
            'confidence': confidence,
            'valence': vader_scores.get('compound', 0.0),
            'intensity': intensity_score,
            'analysis_breakdown': {
                'vader': vader_scores,
                'textblob': textblob_scores,
                'lexicon': lexicon_emotion
            },
            'key_phrases': self._extract_key_phrases(cleaned_text)
        }

    def _clean_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep emotional punctuation
        text = re.sub(r'[^\w\s!?]', '', text)
        return text.lower()

    def _vader_analysis(self, text):
        """VADER sentiment analysis"""
        if self.sia is None:
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}

        try:
            scores = self.sia.polarity_scores(text)
            return scores
        except Exception as e:
            print(f"VADER analysis error: {e}")
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}

    def _textblob_analysis(self, text):
        """TextBlob sentiment analysis"""
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            print(f"TextBlob analysis error: {e}")
            return {
                'polarity': 0.0,
                'subjectivity': 0.0
            }

    def _lexicon_based_analysis(self, text):
        """Emotion analysis based on emotion lexicon"""
        if not text:
            return {'emotion': 'neutral', 'count': 0, 'all_counts': {}}

        words = text.split()
        emotion_counts = {emotion: 0 for emotion in self.emotion_lexicon.keys()}

        for word in words:
            for emotion, emotion_words in self.emotion_lexicon.items():
                if word in emotion_words:
                    emotion_counts[emotion] += 1

        # Find dominant emotion
        total_emotion_words = sum(emotion_counts.values())
        if total_emotion_words == 0:
            return {'emotion': 'neutral', 'count': 0, 'all_counts': emotion_counts}

        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        return {
            'emotion': dominant_emotion[0],
            'count': dominant_emotion[1],
            'all_counts': emotion_counts
        }

    def _calculate_intensity(self, text):
        """Calculate emotional intensity from text"""
        if not text:
            return 1.0

        words = text.split()
        intensity = 1.0  # Base intensity

        # Check for intensity modifiers
        for i, word in enumerate(words):
            if word in self.intensity_modifiers:
                intensity *= self.intensity_modifiers[word]

        # Check for exclamation marks and question marks
        excl_count = text.count('!')
        ques_count = text.count('?')
        intensity *= (1 + excl_count * 0.2 + ques_count * 0.1)

        # Check for capitalization (if present in original text)
        if any(word.isupper() for word in text.split()):
            intensity *= 1.3

        return min(intensity, 3.0)  # Cap at 3.0

    def _fuse_analyses(self, vader_scores, textblob_scores, lexicon_emotion, intensity_score):
        """Fuse multiple analysis results"""
        # VADER-based emotion
        vader_compound = vader_scores.get('compound', 0.0)
        if vader_compound >= 0.05:
            vader_emotion = 'happy'
        elif vader_compound <= -0.05:
            vader_emotion = 'sad'
        else:
            vader_emotion = 'neutral'

        # TextBlob-based emotion
        textblob_polarity = textblob_scores.get('polarity', 0.0)
        if textblob_polarity > 0:
            textblob_emotion = 'happy'
        elif textblob_polarity < 0:
            textblob_emotion = 'sad'
        else:
            textblob_emotion = 'neutral'

        # Lexicon-based emotion
        lexicon_primary = lexicon_emotion.get('emotion', 'neutral')

        # Voting system
        emotions = [vader_emotion, textblob_emotion, lexicon_primary]
        emotion_counts = Counter(emotions)

        # Get most common emotion
        if emotion_counts:
            final_emotion, count = emotion_counts.most_common(1)[0]
            # Calculate confidence
            confidence = count / len(emotions)
        else:
            final_emotion = 'neutral'
            confidence = 0.5

        # Adjust confidence based on intensity and agreement
        if intensity_score > 1.5:
            confidence *= 1.2
        if len(set(emotions)) == 1:  # All agree
            confidence *= 1.3

        return final_emotion, min(confidence, 1.0)

    def _extract_key_phrases(self, text):
        """Extract key emotional phrases"""
        if not text:
            return []

        phrases = []
        words = text.split()

        # Look for emotion words with modifiers
        for i, word in enumerate(words):
            if any(word in emotion_words for emotion_words in self.emotion_lexicon.values()):
                # Get context around emotion word
                start = max(0, i - 2)
                end = min(len(words), i + 3)
                phrase = ' '.join(words[start:end])
                phrases.append(phrase)

        return phrases[:3]  # Return top 3 phrases

    def _get_neutral_response(self):
        """Return neutral response for empty text"""
        return {
            'emotion': 'neutral',
            'confidence': 0.5,
            'valence': 0.0,
            'intensity': 1.0,
            'analysis_breakdown': {
                'vader': {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0},
                'textblob': {'polarity': 0.0, 'subjectivity': 0.0},
                'lexicon': {'emotion': 'neutral', 'count': 0, 'all_counts': {}}
            },
            'key_phrases': []
        }

    def analyze_emotional_context(self, text, previous_context=None):
        """Analyze emotion with context from previous interactions"""
        current_analysis = self.analyze_sentiment(text)

        if previous_context:
            # Adjust based on previous emotional state
            context_adjustment = self._calculate_context_adjustment(
                current_analysis, previous_context
            )
            current_analysis['context_adjusted'] = context_adjustment

        return current_analysis

    def _calculate_context_adjustment(self, current_analysis, previous_context):
        """Calculate emotional context adjustment"""
        # Simple context adjustment - can be enhanced
        previous_emotion = previous_context.get('emotion', 'neutral')
        current_emotion = current_analysis['emotion']

        if previous_emotion == current_emotion:
            return {'adjustment': 'consistent', 'impact': 0.1}
        else:
            return {'adjustment': 'changed', 'impact': -0.1}