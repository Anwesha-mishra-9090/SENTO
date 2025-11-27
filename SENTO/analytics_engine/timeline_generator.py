import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TimelineGenerator:
    def __init__(self):
        self.emotion_colors = {
            'happy': '#FFD700',  # Gold
            'sad': '#1E90FF',  # DodgerBlue
            'angry': '#FF4500',  # OrangeRed
            'fear': '#8A2BE2',  # BlueViolet
            'surprise': '#00FF7F',  # SpringGreen
            'neutral': '#A9A9A9'  # DarkGray
        }

    def generate_emotional_timeline(self, emotion_data, time_period="7d"):
        """Generate emotional timeline for specified period"""
        if not emotion_data:
            return self._get_empty_timeline()

        # Convert to DataFrame for easier manipulation
        df = self._prepare_emotion_dataframe(emotion_data)

        # Filter by time period
        df_filtered = self._filter_by_time_period(df, time_period)

        if df_filtered.empty:
            return self._get_empty_timeline()

        # Generate timeline analysis
        timeline_analysis = {
            'daily_summary': self._generate_daily_summary(df_filtered),
            'hourly_patterns': self._analyze_hourly_patterns(df_filtered),
            'emotional_cycles': self._detect_emotional_cycles(df_filtered),
            'trend_analysis': self._analyze_emotional_trends(df_filtered),
            'statistics': self._calculate_emotional_statistics(df_filtered)
        }

        return timeline_analysis

    def _prepare_emotion_dataframe(self, emotion_data):
        """Convert emotion data to pandas DataFrame"""
        records = []
        for entry in emotion_data:
            if 'timestamp' in entry and 'emotion' in entry:
                record = {
                    'timestamp': pd.to_datetime(entry['timestamp']),
                    'emotion': entry['emotion'],
                    'confidence': entry.get('confidence', 0.5),
                    'valence': entry.get('valence', 0.0),
                    'intensity': entry.get('intensity', 1.0),
                    'input_type': entry.get('input_type', 'unknown')
                }
                records.append(record)

        return pd.DataFrame(records)

    def _filter_by_time_period(self, df, time_period):
        """Filter data by time period"""
        if df.empty:
            return df

        end_date = df['timestamp'].max()

        if time_period == "1d":
            start_date = end_date - timedelta(days=1)
        elif time_period == "7d":
            start_date = end_date - timedelta(days=7)
        elif time_period == "30d":
            start_date = end_date - timedelta(days=30)
        else:
            start_date = df['timestamp'].min()

        return df[df['timestamp'] >= start_date]

    def _generate_daily_summary(self, df):
        """Generate daily emotional summary"""
        if df.empty:
            return {}

        df_daily = df.set_index('timestamp').resample('D').agg({
            'emotion': lambda x: x.mode()[0] if not x.mode().empty else 'neutral',
            'confidence': 'mean',
            'valence': 'mean',
            'intensity': 'mean'
        }).reset_index()

        daily_summary = []
        for _, row in df_daily.iterrows():
            daily_summary.append({
                'date': row['timestamp'].strftime('%Y-%m-%d'),
                'dominant_emotion': row['emotion'],
                'average_valence': float(row['valence']),
                'average_intensity': float(row['intensity']),
                'confidence': float(row['confidence'])
            })

        return daily_summary

    def _analyze_hourly_patterns(self, df):
        """Analyze emotional patterns by hour of day"""
        if df.empty:
            return {}

        df['hour'] = df['timestamp'].dt.hour

        hourly_patterns = {}
        for emotion in self.emotion_colors.keys():
            emotion_data = df[df['emotion'] == emotion]
            if not emotion_data.empty:
                hourly_counts = emotion_data.groupby('hour').size()
                hourly_patterns[emotion] = {
                    'peak_hours': hourly_counts.nlargest(3).index.tolist(),
                    'frequency': int(hourly_counts.sum()),
                    'distribution': hourly_counts.to_dict()
                }

        return hourly_patterns

    def _detect_emotional_cycles(self, df):
        """Detect emotional cycles and patterns"""
        if len(df) < 5:
            return {}

        # Calculate emotion transitions
        df_sorted = df.sort_values('timestamp')
        df_sorted['prev_emotion'] = df_sorted['emotion'].shift(1)
        df_sorted['transition'] = df_sorted['emotion'] + '_to_' + df_sorted['prev_emotion']

        transitions = df_sorted['transition'].value_counts().head(5).to_dict()

        # Detect emotional stability
        emotion_changes = (df_sorted['emotion'] != df_sorted['prev_emotion']).sum()
        stability_score = 1 - (emotion_changes / len(df_sorted))

        return {
            'common_transitions': transitions,
            'stability_score': float(stability_score),
            'emotional_volatility': emotion_changes / len(df_sorted)
        }

    def _analyze_emotional_trends(self, df):
        """Analyze emotional trends over time"""
        if len(df) < 3:
            return {}

        # Calculate rolling averages
        df_sorted = df.sort_values('timestamp')
        df_sorted['valence_rolling'] = df_sorted['valence'].rolling(window=3, min_periods=1).mean()
        df_sorted['intensity_rolling'] = df_sorted['intensity'].rolling(window=3, min_periods=1).mean()

        # Trend analysis
        recent_valence = df_sorted['valence_rolling'].iloc[-5:].mean()
        overall_valence = df_sorted['valence'].mean()

        trend_direction = "improving" if recent_valence > overall_valence else "declining" if recent_valence < overall_valence else "stable"

        return {
            'trend_direction': trend_direction,
            'recent_valence': float(recent_valence),
            'overall_valence': float(overall_valence),
            'trend_strength': abs(recent_valence - overall_valence)
        }

    def _calculate_emotional_statistics(self, df):
        """Calculate comprehensive emotional statistics"""
        if df.empty:
            return {}

        return {
            'emotion_distribution': df['emotion'].value_counts().to_dict(),
            'average_confidence': float(df['confidence'].mean()),
            'average_valence': float(df['valence'].mean()),
            'average_intensity': float(df['intensity'].mean()),
            'total_entries': len(df),
            'time_span_days': (df['timestamp'].max() - df['timestamp'].min()).days
        }

    def _get_empty_timeline(self):
        """Return empty timeline structure"""
        return {
            'daily_summary': [],
            'hourly_patterns': {},
            'emotional_cycles': {},
            'trend_analysis': {},
            'statistics': {}
        }

    def generate_visualization_data(self, emotion_data, time_period="7d"):
        """Generate data for visualization"""
        timeline = self.generate_emotional_timeline(emotion_data, time_period)

        visualization_data = {
            'emotion_timeline': self._prepare_timeline_visualization(emotion_data),
            'emotion_distribution': timeline.get('statistics', {}).get('emotion_distribution', {}),
            'hourly_heatmap': self._prepare_hourly_heatmap_data(emotion_data),
            'trend_metrics': timeline.get('trend_analysis', {})
        }

        return visualization_data

    def _prepare_timeline_visualization(self, emotion_data):
        """Prepare data for timeline visualization"""
        if not emotion_data:
            return []

        timeline_points = []
        for entry in emotion_data:
            if 'timestamp' in entry and 'emotion' in entry:
                timeline_points.append({
                    'x': entry['timestamp'],
                    'y': entry['emotion'],
                    'confidence': entry.get('confidence', 0.5),
                    'valence': entry.get('valence', 0.0),
                    'color': self.emotion_colors.get(entry['emotion'], '#A9A9A9')
                })

        return timeline_points

    def _prepare_hourly_heatmap_data(self, emotion_data):
        """Prepare data for hourly heatmap"""
        if not emotion_data:
            return {}

        df = self._prepare_emotion_dataframe(emotion_data)
        if df.empty:
            return {}

        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.date

        heatmap_data = {}
        for emotion in self.emotion_colors.keys():
            emotion_data = df[df['emotion'] == emotion]
            if not emotion_data.empty:
                hourly_counts = emotion_data.groupby('hour').size()
                heatmap_data[emotion] = {
                    'hours': hourly_counts.index.tolist(),
                    'counts': hourly_counts.values.tolist()
                }

        return heatmap_data