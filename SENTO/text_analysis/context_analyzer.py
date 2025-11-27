import json
import re
import numpy as np
from datetime import datetime
from collections import defaultdict, deque


class ContextAnalyzer:
    def __init__(self, context_window=10):
        self.context_window = context_window
        self.conversation_history = deque(maxlen=context_window)
        self.user_profile = {}
        self.emotional_baseline = {}

    def add_conversation_turn(self, user_input, emotion_analysis, response=None):
        """Add a conversation turn to context"""
        turn = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'emotion_analysis': emotion_analysis,
            'response': response,
            'context_tags': self._extract_context_tags(user_input, emotion_analysis)
        }

        self.conversation_history.append(turn)
        self._update_user_profile(turn)
        self._update_emotional_baseline(emotion_analysis)

    def _extract_context_tags(self, user_input, emotion_analysis):
        """Extract context tags from user input and emotion"""
        tags = set()

        # Time-based tags
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            tags.add('morning')
        elif 12 <= current_hour < 17:
            tags.add('afternoon')
        elif 17 <= current_hour < 22:
            tags.add('evening')
        else:
            tags.add('night')

        # Emotion-based tags
        emotion = emotion_analysis.get('emotion', 'neutral')
        tags.add(f'emotion_{emotion}')

        # Intensity-based tags
        intensity = emotion_analysis.get('intensity', 1.0)
        if intensity > 1.5:
            tags.add('high_intensity')
        elif intensity < 0.7:
            tags.add('low_intensity')

        # Content-based tags (simple keyword matching)
        content_keywords = {
            'work': ['work', 'job', 'office', 'meeting', 'deadline'],
            'family': ['family', 'mom', 'dad', 'parents', 'children'],
            'friends': ['friend', 'friends', 'buddy', 'pal'],
            'health': ['health', 'sick', 'tired', 'sleep', 'exercise'],
            'hobbies': ['hobby', 'game', 'movie', 'music', 'book']
        }

        user_input_lower = user_input.lower()
        for category, keywords in content_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                tags.add(category)

        return list(tags)

    def _update_user_profile(self, conversation_turn):
        """Update user profile based on conversation history"""
        emotion = conversation_turn['emotion_analysis'].get('emotion', 'neutral')

        # Update emotion frequency
        if 'emotion_frequency' not in self.user_profile:
            self.user_profile['emotion_frequency'] = defaultdict(int)

        self.user_profile['emotion_frequency'][emotion] += 1

        # Update common topics
        tags = conversation_turn.get('context_tags', [])
        for tag in tags:
            if tag.startswith('emotion_'):
                continue
            if 'common_topics' not in self.user_profile:
                self.user_profile['common_topics'] = defaultdict(int)
            self.user_profile['common_topics'][tag] += 1

        # Update conversation patterns
        self._update_conversation_patterns(conversation_turn)

    def _update_conversation_patterns(self, conversation_turn):
        """Update conversation pattern analysis"""
        if 'conversation_patterns' not in self.user_profile:
            self.user_profile['conversation_patterns'] = {
                'emotional_shifts': [],
                'response_times': [],
                'topic_duration': defaultdict(list)
            }

        # Analyze emotional shifts if we have previous turns
        if len(self.conversation_history) > 1:
            prev_turn = self.conversation_history[-2]
            prev_emotion = prev_turn['emotion_analysis'].get('emotion')
            current_emotion = conversation_turn['emotion_analysis'].get('emotion')

            if prev_emotion != current_emotion:
                self.user_profile['conversation_patterns']['emotional_shifts'].append({
                    'from': prev_emotion,
                    'to': current_emotion,
                    'timestamp': conversation_turn['timestamp']
                })

    def _update_emotional_baseline(self, emotion_analysis):
        """Update emotional baseline for the user"""
        emotion = emotion_analysis.get('emotion', 'neutral')
        intensity = emotion_analysis.get('intensity', 1.0)
        valence = emotion_analysis.get('valence', 0.0)

        if 'emotional_baseline' not in self.user_profile:
            self.user_profile['emotional_baseline'] = {
                'emotion_counts': defaultdict(int),
                'intensity_values': [],
                'valence_values': [],
                'total_interactions': 0
            }

        baseline = self.user_profile['emotional_baseline']
        baseline['emotion_counts'][emotion] += 1
        baseline['intensity_values'].append(intensity)
        baseline['valence_values'].append(valence)
        baseline['total_interactions'] += 1

        # Keep only recent values for baseline calculation
        if len(baseline['intensity_values']) > 50:
            baseline['intensity_values'] = baseline['intensity_values'][-50:]
            baseline['valence_values'] = baseline['valence_values'][-50:]

    def get_current_context(self):
        """Get current context analysis"""
        if not self.conversation_history:
            return {}

        recent_turns = list(self.conversation_history)[-3:]  # Last 3 turns

        current_emotion = recent_turns[-1]['emotion_analysis'].get('emotion', 'neutral')
        emotional_trend = self._calculate_emotional_trend(recent_turns)
        dominant_topics = self._get_dominant_topics(recent_turns)

        return {
            'current_emotion': current_emotion,
            'emotional_trend': emotional_trend,
            'dominant_topics': dominant_topics,
            'conversation_length': len(self.conversation_history),
            'user_engagement': self._calculate_engagement_level(),
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_emotional_trend(self, recent_turns):
        """Calculate emotional trend from recent turns"""
        if len(recent_turns) < 2:
            return 'stable'

        emotions = [turn['emotion_analysis'].get('emotion', 'neutral') for turn in recent_turns]

        # Simple trend analysis
        if all(emotion == emotions[0] for emotion in emotions):
            return 'stable'
        elif emotions[-1] in ['happy', 'excited'] and emotions[0] in ['sad', 'angry']:
            return 'improving'
        elif emotions[-1] in ['sad', 'angry'] and emotions[0] in ['happy', 'neutral']:
            return 'declining'
        else:
            return 'fluctuating'

    def _get_dominant_topics(self, recent_turns):
        """Get dominant topics from recent conversation"""
        topic_counts = defaultdict(int)

        for turn in recent_turns:
            for tag in turn.get('context_tags', []):
                if not tag.startswith('emotion_'):
                    topic_counts[tag] += 1

        if not topic_counts:
            return []

        # Return top 3 topics
        return sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    def _calculate_engagement_level(self):
        """Calculate user engagement level"""
        if not self.conversation_history:
            return 'low'

        recent_turns = list(self.conversation_history)[-5:]

        # Simple engagement metrics
        input_lengths = []
        for turn in recent_turns:
            try:
                input_lengths.append(len(turn['user_input'].split()))
            except:
                input_lengths.append(0)

        emotion_intensities = []
        for turn in recent_turns:
            try:
                emotion_intensities.append(turn['emotion_analysis'].get('intensity', 1.0))
            except:
                emotion_intensities.append(1.0)

        if not input_lengths or not emotion_intensities:
            return 'low'

        avg_input_length = sum(input_lengths) / len(input_lengths)
        avg_intensity = sum(emotion_intensities) / len(emotion_intensities)

        if avg_input_length > 10 and avg_intensity > 1.2:
            return 'high'
        elif avg_input_length > 5 and avg_intensity > 0.8:
            return 'medium'
        else:
            return 'low'

    def get_emotional_baseline(self):
        """Get user's emotional baseline"""
        baseline = self.user_profile.get('emotional_baseline', {})
        if not baseline or baseline['total_interactions'] == 0:
            return {}

        emotion_counts = baseline['emotion_counts']
        total = baseline['total_interactions']

        # Calculate baseline metrics
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral'

        intensity_values = baseline.get('intensity_values', [1.0])
        valence_values = baseline.get('valence_values', [0.0])

        avg_intensity = sum(intensity_values) / len(intensity_values)
        avg_valence = sum(valence_values) / len(valence_values)

        return {
            'dominant_emotion': dominant_emotion,
            'emotion_distribution': {k: v / total for k, v in emotion_counts.items()},
            'average_intensity': avg_intensity,
            'average_valence': avg_valence,
            'interaction_count': total,
            'emotional_stability': self._calculate_emotional_stability(baseline)
        }

    def _calculate_emotional_stability(self, baseline):
        """Calculate emotional stability score"""
        valence_values = baseline.get('valence_values', [0.0])
        intensity_values = baseline.get('intensity_values', [1.0])

        if len(valence_values) < 2 or len(intensity_values) < 2:
            return 0.5

        try:
            valence_std = np.std(valence_values)
            intensity_std = np.std(intensity_values)

            # Lower std = more stable
            stability = 1.0 - min((valence_std + intensity_std) / 2, 1.0)
            return max(stability, 0.0)
        except:
            return 0.5

    def save_context(self, filepath):
        """Save context to file"""
        try:
            context_data = {
                'conversation_history': list(self.conversation_history),
                'user_profile': self.user_profile,
                'emotional_baseline': self.emotional_baseline,
                'save_timestamp': datetime.now().isoformat()
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(context_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving context: {e}")
            return False

    def load_context(self, filepath):
        """Load context from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                context_data = json.load(f)

            self.conversation_history = deque(context_data.get('conversation_history', []),
                                              maxlen=self.context_window)
            self.user_profile = context_data.get('user_profile', {})
            self.emotional_baseline = context_data.get('emotional_baseline', {})

            return True
        except Exception as e:
            print(f"Error loading context: {e}")
            return False

    def get_context_summary(self):
        """Get comprehensive context summary"""
        current_context = self.get_current_context()
        emotional_baseline = self.get_emotional_baseline()

        return {
            'current_state': current_context,
            'user_baseline': emotional_baseline,
            'conversation_metrics': {
                'total_turns': len(self.conversation_history),
                'recent_engagement': current_context.get('user_engagement', 'low'),
                'emotional_trend': current_context.get('emotional_trend', 'stable')
            },
            'personalization_data': {
                'common_topics': dict(self.user_profile.get('common_topics', {})),
                'emotion_patterns': self.user_profile.get('conversation_patterns', {})
            }
        }