import numpy as np
from collections import defaultdict, Counter
from datetime import datetime, timedelta


class PatternDetector:
    def __init__(self):
        self.pattern_threshold = 0.7
        self.min_pattern_occurrences = 3

    def detect_emotional_patterns(self, emotion_data):
        """Detect various emotional patterns"""
        if len(emotion_data) < 5:
            return {}

        patterns = {
            'daily_patterns': self._detect_daily_patterns(emotion_data),
            'weekly_cycles': self._detect_weekly_cycles(emotion_data),
            'trigger_patterns': self._detect_trigger_patterns(emotion_data),
            'recovery_patterns': self._detect_recovery_patterns(emotion_data),
            'escalation_patterns': self._detect_escalation_patterns(emotion_data)
        }

        return patterns

    def _detect_daily_patterns(self, emotion_data):
        """Detect patterns that repeat daily"""
        hourly_emotions = defaultdict(list)

        for entry in emotion_data:
            if 'timestamp' in entry:
                hour = datetime.fromisoformat(entry['timestamp']).hour
                hourly_emotions[hour].append(entry['emotion'])

        daily_patterns = {}
        for hour, emotions in hourly_emotions.items():
            if len(emotions) >= self.min_pattern_occurrences:
                emotion_counter = Counter(emotions)
                total = sum(emotion_counter.values())

                for emotion, count in emotion_counter.items():
                    frequency = count / total
                    if frequency >= self.pattern_threshold:
                        if hour not in daily_patterns:
                            daily_patterns[hour] = []
                        daily_patterns[hour].append({
                            'emotion': emotion,
                            'frequency': frequency,
                            'occurrences': count
                        })

        return daily_patterns

    def _detect_weekly_cycles(self, emotion_data):
        """Detect weekly emotional cycles"""
        weekday_emotions = defaultdict(list)

        for entry in emotion_data:
            if 'timestamp' in entry:
                weekday = datetime.fromisoformat(entry['timestamp']).weekday()
                weekday_emotions[weekday].append(entry['emotion'])

        weekly_patterns = {}
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        for weekday, emotions in weekday_emotions.items():
            if len(emotions) >= self.min_pattern_occurrences:
                emotion_counter = Counter(emotions)
                dominant_emotion, count = emotion_counter.most_common(1)[0]
                frequency = count / len(emotions)

                if frequency >= self.pattern_threshold:
                    weekly_patterns[weekday_names[weekday]] = {
                        'dominant_emotion': dominant_emotion,
                        'frequency': frequency,
                        'sample_size': len(emotions)
                    }

        return weekly_patterns

    def _detect_trigger_patterns(self, emotion_data):
        """Detect patterns that trigger emotional changes"""
        if len(emotion_data) < 10:
            return {}

        # Analyze emotional transitions
        transitions = []
        for i in range(1, len(emotion_data)):
            prev_emotion = emotion_data[i - 1]['emotion']
            current_emotion = emotion_data[i]['emotion']

            if prev_emotion != current_emotion:
                transitions.append({
                    'from': prev_emotion,
                    'to': current_emotion,
                    'timestamp': emotion_data[i]['timestamp']
                })

        # Find common triggers
        transition_counts = Counter([f"{t['from']}_to_{t['to']}" for t in transitions])
        common_triggers = {}

        for transition, count in transition_counts.most_common(5):
            frequency = count / len(transitions)
            if frequency >= 0.1:  # At least 10% of transitions
                from_emotion, to_emotion = transition.split('_to_')
                common_triggers[transition] = {
                    'from_emotion': from_emotion,
                    'to_emotion': to_emotion,
                    'frequency': frequency,
                    'occurrences': count
                }

        return common_triggers

    def _detect_recovery_patterns(self, emotion_data):
        """Detect patterns of emotional recovery"""
        negative_emotions = ['sad', 'angry', 'fear']
        positive_emotions = ['happy', 'surprise']

        recovery_sequences = []
        current_negative_sequence = []

        for entry in emotion_data:
            emotion = entry['emotion']

            if emotion in negative_emotions:
                current_negative_sequence.append(entry)
            elif emotion in positive_emotions and current_negative_sequence:
                # Negative sequence followed by positive emotion
                recovery_sequences.append({
                    'negative_sequence': current_negative_sequence.copy(),
                    'recovery_emotion': emotion,
                    'recovery_timestamp': entry['timestamp'],
                    'duration_minutes': self._calculate_sequence_duration(current_negative_sequence)
                })
                current_negative_sequence = []
            else:
                current_negative_sequence = []

        # Analyze recovery patterns
        if not recovery_sequences:
            return {}

        avg_recovery_time = np.mean([seq['duration_minutes'] for seq in recovery_sequences])
        common_recovery_emotions = Counter([seq['recovery_emotion'] for seq in recovery_sequences])

        return {
            'total_recovery_sequences': len(recovery_sequences),
            'average_recovery_time_minutes': avg_recovery_time,
            'common_recovery_emotions': dict(common_recovery_emotions),
            'recovery_efficiency': self._calculate_recovery_efficiency(recovery_sequences)
        }

    def _detect_escalation_patterns(self, emotion_data):
        """Detect patterns of emotional escalation"""
        intensity_escalations = []

        for i in range(1, len(emotion_data)):
            prev_intensity = emotion_data[i - 1].get('intensity', 1.0)
            current_intensity = emotion_data[i].get('intensity', 1.0)

            if current_intensity > prev_intensity * 1.5:  # 50% increase
                intensity_escalations.append({
                    'from_intensity': prev_intensity,
                    'to_intensity': current_intensity,
                    'increase_percentage': (current_intensity - prev_intensity) / prev_intensity * 100,
                    'timestamp': emotion_data[i]['timestamp'],
                    'emotion': emotion_data[i]['emotion']
                })

        escalation_analysis = {}
        if intensity_escalations:
            avg_increase = np.mean([esc['increase_percentage'] for esc in intensity_escalations])
            common_escalation_emotions = Counter([esc['emotion'] for esc in intensity_escalations])

            escalation_analysis = {
                'total_escalations': len(intensity_escalations),
                'average_increase_percentage': avg_increase,
                'common_escalation_emotions': dict(common_escalation_emotions),
                'escalation_frequency': len(intensity_escalations) / len(emotion_data)
            }

        return escalation_analysis

    def _calculate_sequence_duration(self, sequence):
        """Calculate duration of emotional sequence in minutes"""
        if len(sequence) < 2:
            return 0

        start_time = datetime.fromisoformat(sequence[0]['timestamp'])
        end_time = datetime.fromisoformat(sequence[-1]['timestamp'])

        return (end_time - start_time).total_seconds() / 60

    def _calculate_recovery_efficiency(self, recovery_sequences):
        """Calculate efficiency of emotional recovery"""
        if not recovery_sequences:
            return 0

        recovery_times = [seq['duration_minutes'] for seq in recovery_sequences]
        avg_recovery_time = np.mean(recovery_times)

        # Lower recovery time = higher efficiency
        max_expected_recovery = 120  # 2 hours
        efficiency = 1 - min(avg_recovery_time / max_expected_recovery, 1)

        return efficiency

    def detect_behavioral_patterns(self, emotion_data, context_data=None):
        """Detect behavioral patterns from emotional data"""
        patterns = self.detect_emotional_patterns(emotion_data)

        behavioral_insights = {
            'emotional_resilience': self._assess_emotional_resilience(patterns),
            'mood_stability': self._assess_mood_stability(emotion_data),
            'stress_patterns': self._identify_stress_patterns(patterns),
            'wellness_indicators': self._calculate_wellness_indicators(emotion_data)
        }

        if context_data:
            behavioral_insights['contextual_patterns'] = self._analyze_contextual_patterns(
                emotion_data, context_data
            )

        return behavioral_insights

    def _assess_emotional_resilience(self, patterns):
        """Assess emotional resilience from patterns"""
        recovery_patterns = patterns.get('recovery_patterns', {})

        if not recovery_patterns:
            return {'score': 0.5, 'level': 'unknown'}

        efficiency = recovery_patterns.get('recovery_efficiency', 0.5)
        recovery_sequences = recovery_patterns.get('total_recovery_sequences', 0)

        # Calculate resilience score
        resilience_score = efficiency * min(recovery_sequences / 10, 1.0)

        if resilience_score > 0.7:
            level = 'high'
        elif resilience_score > 0.4:
            level = 'medium'
        else:
            level = 'low'

        return {
            'score': resilience_score,
            'level': level,
            'recovery_efficiency': efficiency,
            'recovery_opportunities': recovery_sequences
        }

    def _assess_mood_stability(self, emotion_data):
        """Assess mood stability from emotional data"""
        if len(emotion_data) < 5:
            return {'score': 0.5, 'level': 'insufficient_data'}

        emotions = [entry['emotion'] for entry in emotion_data]
        emotion_changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i - 1])

        change_rate = emotion_changes / len(emotions)
        stability_score = 1 - change_rate

        if stability_score > 0.8:
            level = 'very_stable'
        elif stability_score > 0.6:
            level = 'stable'
        elif stability_score > 0.4:
            level = 'moderate'
        else:
            level = 'volatile'

        return {
            'score': stability_score,
            'level': level,
            'change_frequency': change_rate,
            'total_changes': emotion_changes
        }

    def _identify_stress_patterns(self, patterns):
        """Identify stress-related patterns"""
        stress_indicators = {}

        # Check for negative emotion patterns
        escalation_patterns = patterns.get('escalation_patterns', {})
        if escalation_patterns:
            stress_indicators['escalation_frequency'] = escalation_patterns.get('escalation_frequency', 0)

        # Check for recovery patterns
        recovery_patterns = patterns.get('recovery_patterns', {})
        if recovery_patterns:
            stress_indicators['recovery_efficiency'] = recovery_patterns.get('recovery_efficiency', 0)

        # Check for trigger patterns
        trigger_patterns = patterns.get('trigger_patterns', {})
        negative_triggers = sum(1 for trigger in trigger_patterns.values()
                                if trigger['to_emotion'] in ['sad', 'angry', 'fear'])

        stress_indicators['negative_trigger_ratio'] = negative_triggers / len(
            trigger_patterns) if trigger_patterns else 0

        return stress_indicators

    def _calculate_wellness_indicators(self, emotion_data):
        """Calculate overall wellness indicators"""
        if not emotion_data:
            return {}

        positive_emotions = ['happy', 'surprise']
        negative_emotions = ['sad', 'angry', 'fear']

        positive_count = sum(1 for entry in emotion_data if entry['emotion'] in positive_emotions)
        negative_count = sum(1 for entry in emotion_data if entry['emotion'] in negative_emotions)
        total_count = len(emotion_data)

        positivity_ratio = positive_count / total_count if total_count > 0 else 0
        negativity_ratio = negative_count / total_count if total_count > 0 else 0

        wellness_score = positivity_ratio - negativity_ratio

        return {
            'positivity_ratio': positivity_ratio,
            'negativity_ratio': negativity_ratio,
            'wellness_score': wellness_score,
            'positive_emotions': positive_count,
            'negative_emotions': negative_count
        }

    def _analyze_contextual_patterns(self, emotion_data, context_data):
        """Analyze patterns in specific contexts"""
        # This would integrate with context data from ContextAnalyzer
        # For now, return basic contextual analysis
        return {
            'context_aware_patterns': 'requires_context_integration',
            'situation_specific_triggers': []
        }