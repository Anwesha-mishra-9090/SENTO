import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict, Counter
import statistics
from datetime import datetime


class CorrelationAnalyzer:
    def __init__(self):
        self.correlation_threshold = 0.3
        self.min_sample_size = 5

    def analyze_correlations(self, emotion_data, external_factors=None):
        """Analyze correlations between emotions and external factors"""
        correlations = {
            'emotional_correlations': self._analyze_emotional_correlations(emotion_data),
            'temporal_correlations': self._analyze_temporal_correlations(emotion_data),
            'intensity_correlations': self._analyze_intensity_correlations(emotion_data),
            'external_correlations': self._analyze_external_correlations(emotion_data,
                                                                         external_factors) if external_factors else {}
        }

        # Calculate overall correlation strength
        correlations['correlation_strength'] = self._calculate_correlation_strength(correlations)

        return correlations

    def _analyze_emotional_correlations(self, emotion_data):
        """Analyze correlations between different emotional states"""
        if len(emotion_data) < self.min_sample_size:
            return {}

        # Convert to time series of emotional valence
        emotion_valences = {
            'happy': 1, 'excited': 1,
            'sad': -1, 'angry': -1, 'fear': -1,
            'surprise': 0.5, 'neutral': 0
        }

        # Create valence timeline
        valence_timeline = []
        for entry in emotion_data:
            if 'timestamp' in entry:
                valence = emotion_valences.get(entry['emotion'], 0)
                valence_timeline.append({
                    'timestamp': entry['timestamp'],
                    'valence': valence,
                    'intensity': entry.get('intensity', 1.0)
                })

        # Analyze autocorrelation (emotional persistence)
        if len(valence_timeline) > 1:
            valences = [v['valence'] for v in valence_timeline]
            autocorrelation = self._calculate_autocorrelation(valences)
        else:
            autocorrelation = 0

        # Analyze emotion clusters (which emotions often occur together in time)
        emotion_clusters = self._find_emotion_clusters(emotion_data)

        return {
            'emotional_persistence': autocorrelation,
            'emotion_clusters': emotion_clusters,
            'valence_stability': self._calculate_valence_stability(valence_timeline)
        }

    def _calculate_autocorrelation(self, series, lag=1):
        """Calculate autocorrelation for time series"""
        if len(series) <= lag:
            return 0

        x = series[:-lag]
        y = series[lag:]

        if len(x) < 2:
            return 0

        correlation = np.corrcoef(x, y)[0, 1]
        return correlation if not np.isnan(correlation) else 0

    def _find_emotion_clusters(self, emotion_data, time_window_hours=2):
        """Find clusters of emotions that occur close in time"""
        if not emotion_data:
            return {}

        # Sort by timestamp
        sorted_data = sorted(emotion_data, key=lambda x: x.get('timestamp', ''))

        clusters = []
        current_cluster = []

        for i, entry in enumerate(sorted_data):
            if not current_cluster:
                current_cluster.append(entry)
                continue

            # Check time difference
            prev_time = datetime.fromisoformat(sorted_data[i - 1]['timestamp'])
            current_time = datetime.fromisoformat(entry['timestamp'])
            time_diff = (current_time - prev_time).total_seconds() / 3600  # hours

            if time_diff <= time_window_hours:
                current_cluster.append(entry)
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [entry]

        if len(current_cluster) > 1:
            clusters.append(current_cluster)

        # Analyze cluster patterns
        cluster_analysis = {}
        for cluster in clusters:
            emotions = [entry['emotion'] for entry in cluster]
            emotion_set = frozenset(emotions)

            if len(emotion_set) > 1:  # Only clusters with multiple emotions
                cluster_analysis[str(emotion_set)] = cluster_analysis.get(str(emotion_set), 0) + 1

        return cluster_analysis

    def _calculate_valence_stability(self, valence_timeline):
        """Calculate stability of emotional valence"""
        if len(valence_timeline) < 2:
            return 1.0  # Perfect stability with insufficient data

        valences = [v['valence'] for v in valence_timeline]
        changes = [abs(valences[i] - valences[i - 1]) for i in range(1, len(valences))]

        if not changes:
            return 1.0

        average_change = statistics.mean(changes)
        stability = 1.0 - min(average_change, 1.0)  # Normalize to 0-1

        return stability

    def _analyze_temporal_correlations(self, emotion_data):
        """Analyze correlations with time-based factors"""
        temporal_correlations = {}

        if len(emotion_data) < self.min_sample_size:
            return temporal_correlations

        # Analyze day-of-week correlations
        weekday_emotions = defaultdict(list)
        weekend_emotions = []

        for entry in emotion_data:
            if 'timestamp' in entry:
                timestamp = datetime.fromisoformat(entry['timestamp'])
                day_of_week = timestamp.weekday()

                if day_of_week < 5:  # Weekday
                    weekday_emotions[day_of_week].append(entry['emotion'])
                else:  # Weekend
                    weekend_emotions.append(entry['emotion'])

        # Compare weekday vs weekend
        if weekday_emotions and weekend_emotions:
            weekday_dominant = self._get_dominant_emotion(
                [emotion for emotions in weekday_emotions.values() for emotion in emotions])
            weekend_dominant = self._get_dominant_emotion(weekend_emotions)

            if weekday_dominant != weekend_dominant:
                temporal_correlations['weekday_weekend_difference'] = {
                    'weekday_dominant': weekday_dominant,
                    'weekend_dominant': weekend_dominant,
                    'significance': 0.7  # Placeholder
                }

        # Analyze time-of-day correlations
        time_periods = {
            'morning': (5, 12),
            'afternoon': (12, 17),
            'evening': (17, 22),
            'night': (22, 5)
        }

        period_emotions = {}
        for period, (start, end) in time_periods.items():
            period_data = []
            for entry in emotion_data:
                if 'timestamp' in entry:
                    hour = datetime.fromisoformat(entry['timestamp']).hour
                    if start < end:
                        if start <= hour < end:
                            period_data.append(entry['emotion'])
                    else:  # Overnight period
                        if hour >= start or hour < end:
                            period_data.append(entry['emotion'])

            if len(period_data) >= self.min_sample_size:
                dominant_emotion = self._get_dominant_emotion(period_data)
                period_emotions[period] = dominant_emotion

        temporal_correlations['time_period_patterns'] = period_emotions

        return temporal_correlations

    def _get_dominant_emotion(self, emotions):
        """Get the most frequent emotion from a list"""
        if not emotions:
            return 'neutral'

        return Counter(emotions).most_common(1)[0][0]

    def _analyze_intensity_correlations(self, emotion_data):
        """Analyze correlations with emotional intensity"""
        intensity_correlations = {}

        if len(emotion_data) < self.min_sample_size:
            return intensity_correlations

        # Analyze which emotions have highest intensity
        emotion_intensities = defaultdict(list)

        for entry in emotion_data:
            emotion = entry['emotion']
            intensity = entry.get('intensity', 1.0)
            emotion_intensities[emotion].append(intensity)

        # Calculate average intensity per emotion
        avg_intensities = {}
        for emotion, intensities in emotion_intensities.items():
            if len(intensities) >= 3:  # Minimum samples
                avg_intensities[emotion] = statistics.mean(intensities)

        if avg_intensities:
            highest_emotion = max(avg_intensities.items(), key=lambda x: x[1])
            lowest_emotion = min(avg_intensities.items(), key=lambda x: x[1])

            intensity_correlations['intensity_by_emotion'] = avg_intensities
            intensity_correlations['highest_intensity_emotion'] = highest_emotion[0]
            intensity_correlations['lowest_intensity_emotion'] = lowest_emotion[0]

        # Analyze intensity trends over time
        intensity_timeline = []
        for entry in emotion_data:
            if 'timestamp' in entry:
                intensity_timeline.append({
                    'timestamp': entry['timestamp'],
                    'intensity': entry.get('intensity', 1.0)
                })

        if len(intensity_timeline) > 1:
            intensities = [point['intensity'] for point in intensity_timeline]
            intensity_autocorrelation = self._calculate_autocorrelation(intensities)
            intensity_correlations['intensity_persistence'] = intensity_autocorrelation

        return intensity_correlations

    def _analyze_external_correlations(self, emotion_data, external_factors):
        """Analyze correlations with external factors"""
        external_correlations = {}

        if not external_factors or len(emotion_data) < self.min_sample_size:
            return external_correlations

        # This would integrate with external data sources
        # For now, provide structure for future implementation

        external_correlations['external_integration'] = {
            'status': 'requires_external_data',
            'potential_factors': ['weather', 'sleep', 'social_interactions', 'workload'],
            'integration_guide': 'Connect external APIs or manual data entry'
        }

        return external_correlations

    def _calculate_correlation_strength(self, correlations):
        """Calculate overall strength of detected correlations"""
        strength_indicators = []

        # Emotional persistence strength
        emotional = correlations.get('emotional_correlations', {})
        persistence = abs(emotional.get('emotional_persistence', 0))
        strength_indicators.append(persistence)

        # Temporal pattern strength
        temporal = correlations.get('temporal_correlations', {})
        if temporal.get('weekday_weekend_difference'):
            strength_indicators.append(0.7)

        time_patterns = temporal.get('time_period_patterns', {})
        if len(time_patterns) >= 2:
            strength_indicators.append(0.5)

        # Intensity correlation strength
        intensity = correlations.get('intensity_correlations', {})
        if intensity.get('intensity_persistence', 0) > 0:
            strength_indicators.append(abs(intensity['intensity_persistence']))

        return statistics.mean(strength_indicators) if strength_indicators else 0

    def generate_correlation_report(self, emotion_data, external_factors=None):
        """Generate comprehensive correlation analysis report"""
        correlations = self.analyze_correlations(emotion_data, external_factors)

        report = {
            'correlation_summary': self._summarize_correlations(correlations),
            'key_findings': self._extract_correlation_findings(correlations),
            'correlation_strength': correlations.get('correlation_strength', 0),
            'recommendations': self._generate_correlation_recommendations(correlations),
            'detailed_analysis': correlations
        }

        return report

    def _summarize_correlations(self, correlations):
        """Summarize correlation findings"""
        summary = []

        # Emotional correlations summary
        emotional = correlations.get('emotional_correlations', {})
        persistence = emotional.get('emotional_persistence', 0)
        if abs(persistence) > self.correlation_threshold:
            trend = "persistent" if persistence > 0 else "variable"
            summary.append(f"Emotional states show {trend} patterns")

        # Temporal correlations summary
        temporal = correlations.get('temporal_correlations', {})
        if temporal.get('weekday_weekend_difference'):
            summary.append("Different emotional patterns on weekdays vs weekends")

        # Intensity correlations summary
        intensity = correlations.get('intensity_correlations', {})
        if intensity.get('highest_intensity_emotion'):
            summary.append(f"Highest intensity emotions: {intensity['highest_intensity_emotion']}")

        if not summary:
            summary.append("Continue tracking to identify meaningful correlations")

        return summary

    def _extract_correlation_findings(self, correlations):
        """Extract key correlation findings"""
        findings = []

        # Emotional persistence finding
        emotional = correlations.get('emotional_correlations', {})
        persistence = emotional.get('emotional_persistence', 0)
        if persistence > 0.3:
            findings.append("Emotional states tend to persist over time")
        elif persistence < -0.3:
            findings.append("Emotional states frequently alternate")

        # Temporal findings
        temporal = correlations.get('temporal_correlations', {})
        time_patterns = temporal.get('time_period_patterns', {})
        if len(time_patterns) >= 2:
            findings.append("Emotions vary significantly by time of day")

        # Intensity findings
        intensity = correlations.get('intensity_correlations', {})
        high_intensity_emotion = intensity.get('highest_intensity_emotion')
        if high_intensity_emotion and high_intensity_emotion != 'neutral':
            findings.append(f"'{high_intensity_emotion}' emotions are experienced most intensely")

        if not findings:
            findings.append("Correlation analysis will improve with more emotional data")

        return findings

    def _generate_correlation_recommendations(self, correlations):
        """Generate recommendations based on correlation analysis"""
        recommendations = []

        # Recommendations based on emotional persistence
        emotional = correlations.get('emotional_correlations', {})
        persistence = emotional.get('emotional_persistence', 0)

        if persistence > 0.5:
            recommendations.append("Practice mindfulness to become aware of emotional patterns")
        elif persistence < -0.3:
            recommendations.append("Consider establishing routines for emotional stability")

        # Recommendations based on temporal patterns
        temporal = correlations.get('temporal_correlations', {})
        if temporal.get('weekday_weekend_difference'):
            recommendations.append("Plan activities that balance weekday and weekend emotional needs")

        # Recommendations based on intensity
        intensity = correlations.get('intensity_correlations', {})
        high_intensity_emotion = intensity.get('highest_intensity_emotion')
        if high_intensity_emotion in ['angry', 'fear']:
            recommendations.append("Develop coping strategies for high-intensity negative emotions")

        if not recommendations:
            recommendations.append("Continue emotional tracking for personalized recommendations")

        return recommendations