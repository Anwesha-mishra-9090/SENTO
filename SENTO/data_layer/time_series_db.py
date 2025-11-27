import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


class TimeSeriesDB:
    def __init__(self, db_path="sentio_emotions.db"):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Emotions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                intensity REAL DEFAULT 1.0,
                valence REAL DEFAULT 0.0,
                input_type TEXT DEFAULT 'unknown',
                context_tags TEXT DEFAULT '[]',
                raw_features TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Emotional patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotional_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Coaching interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS coaching_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_input TEXT,
                system_response TEXT,
                emotion_before TEXT,
                emotion_after TEXT,
                intervention_level TEXT,
                effectiveness_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Risk assessments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_factors TEXT NOT NULL,
                intervention_provided TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def store_emotion_data(self, emotion_entry):
        """Store emotion data in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO emotions (
                timestamp, emotion, confidence, intensity, valence, 
                input_type, context_tags, raw_features
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            emotion_entry.get('timestamp', datetime.now().isoformat()),
            emotion_entry.get('emotion', 'neutral'),
            emotion_entry.get('confidence', 0.5),
            emotion_entry.get('intensity', 1.0),
            emotion_entry.get('valence', 0.0),
            emotion_entry.get('input_type', 'unknown'),
            json.dumps(emotion_entry.get('context_tags', [])),
            json.dumps(emotion_entry.get('raw_features', {}))
        ))

        conn.commit()
        conn.close()
        return cursor.lastrowid

    def get_emotion_data(self, start_date=None, end_date=None, limit=1000):
        """Retrieve emotion data within date range"""
        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM emotions WHERE 1=1"
        params = []

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Parse JSON fields
        if not df.empty:
            df['context_tags'] = df['context_tags'].apply(json.loads)
            df['raw_features'] = df['raw_features'].apply(json.loads)

        return df

    def get_emotional_timeline(self, time_period="7d"):
        """Get emotional timeline for specified period"""
        end_date = datetime.now()

        if time_period == "1d":
            start_date = end_date - timedelta(days=1)
        elif time_period == "7d":
            start_date = end_date - timedelta(days=7)
        elif time_period == "30d":
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=7)  # Default

        emotions_df = self.get_emotion_data(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )

        return emotions_df

    def store_emotional_pattern(self, pattern_type, pattern_data, confidence=0.5):
        """Store detected emotional patterns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Calculate date range from pattern data
        start_date = datetime.now().isoformat()
        end_date = (datetime.now() + timedelta(days=7)).isoformat()

        cursor.execute('''
            INSERT INTO emotional_patterns (
                pattern_type, pattern_data, confidence, start_date, end_date
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            pattern_type,
            json.dumps(pattern_data),
            confidence,
            start_date,
            end_date
        ))

        conn.commit()
        conn.close()
        return cursor.lastrowid

    def get_recent_patterns(self, pattern_type=None, limit=10):
        """Get recent emotional patterns"""
        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM emotional_patterns WHERE 1=1"
        params = []

        if pattern_type:
            query += " AND pattern_type = ?"
            params.append(pattern_type)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if not df.empty:
            df['pattern_data'] = df['pattern_data'].apply(json.loads)

        return df

    def store_coaching_interaction(self, interaction_data):
        """Store coaching interaction data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO coaching_interactions (
                timestamp, user_input, system_response, emotion_before,
                emotion_after, intervention_level, effectiveness_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            interaction_data.get('timestamp', datetime.now().isoformat()),
            interaction_data.get('user_input', ''),
            interaction_data.get('system_response', ''),
            interaction_data.get('emotion_before', 'unknown'),
            interaction_data.get('emotion_after', 'unknown'),
            interaction_data.get('intervention_level', 'low'),
            interaction_data.get('effectiveness_score', 0.0)
        ))

        conn.commit()
        conn.close()
        return cursor.lastrowid

    def get_coaching_effectiveness_stats(self, days=30):
        """Get coaching effectiveness statistics"""
        conn = sqlite3.connect(self.db_path)

        start_date = (datetime.now() - timedelta(days=days)).isoformat()

        query = """
            SELECT 
                intervention_level,
                COUNT(*) as total_interactions,
                AVG(effectiveness_score) as avg_effectiveness,
                SUM(CASE WHEN effectiveness_score > 0 THEN 1 ELSE 0 END) as positive_outcomes,
                SUM(CASE WHEN effectiveness_score < 0 THEN 1 ELSE 0 END) as negative_outcomes
            FROM coaching_interactions 
            WHERE timestamp >= ?
            GROUP BY intervention_level
            ORDER BY total_interactions DESC
        """

        df = pd.read_sql_query(query, conn, params=[start_date])
        conn.close()

        return df

    def store_risk_assessment(self, risk_data):
        """Store risk assessment data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO risk_assessments (
                timestamp, risk_level, risk_score, risk_factors, intervention_provided
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            risk_data.get('timestamp', datetime.now().isoformat()),
            risk_data.get('risk_level', 'low_risk'),
            risk_data.get('risk_score', 0.0),
            json.dumps(risk_data.get('risk_factors', [])),
            json.dumps(risk_data.get('intervention_provided', {}))
        ))

        conn.commit()
        conn.close()
        return cursor.lastrowid

    def get_risk_trends(self, days=30):
        """Get risk assessment trends"""
        conn = sqlite3.connect(self.db_path)

        start_date = (datetime.now() - timedelta(days=days)).isoformat()

        query = """
            SELECT 
                DATE(timestamp) as date,
                risk_level,
                COUNT(*) as assessment_count,
                AVG(risk_score) as avg_risk_score
            FROM risk_assessments 
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp), risk_level
            ORDER BY date DESC
        """

        df = pd.read_sql_query(query, conn, params=[start_date])
        conn.close()

        return df

    def calculate_emotional_metrics(self, days=7):
        """Calculate key emotional metrics"""
        emotions_df = self.get_emotion_data(
            start_date=(datetime.now() - timedelta(days=days)).isoformat()
        )

        if emotions_df.empty:
            return {}

        metrics = {
            'total_entries': len(emotions_df),
            'emotion_distribution': emotions_df['emotion'].value_counts().to_dict(),
            'average_intensity': emotions_df['intensity'].mean(),
            'average_confidence': emotions_df['confidence'].mean(),
            'emotional_volatility': self._calculate_volatility(emotions_df),
            'most_common_emotion': emotions_df['emotion'].mode().iloc[0] if not emotions_df[
                'emotion'].mode().empty else 'neutral',
            'positive_negative_ratio': self._calculate_pos_neg_ratio(emotions_df)
        }

        return metrics

    def _calculate_volatility(self, emotions_df):
        """Calculate emotional volatility"""
        if len(emotions_df) < 2:
            return 0.0

        valence_changes = emotions_df['valence'].diff().abs().mean()
        emotion_changes = (emotions_df['emotion'] != emotions_df['emotion'].shift()).sum() / len(emotions_df)

        return (valence_changes + emotion_changes) / 2

    def _calculate_pos_neg_ratio(self, emotions_df):
        """Calculate positive to negative emotion ratio"""
        positive_emotions = ['happy', 'excited', 'surprise']
        negative_emotions = ['sad', 'angry', 'fear']

        positive_count = emotions_df[emotions_df['emotion'].isin(positive_emotions)].shape[0]
        negative_count = emotions_df[emotions_df['emotion'].isin(negative_emotions)].shape[0]

        total = positive_count + negative_count
        if total == 0:
            return 0.5  # Neutral

        return positive_count / total

    def export_data(self, export_format='json', include_tables=None):
        """Export database data in specified format"""
        if include_tables is None:
            include_tables = ['emotions', 'emotional_patterns', 'coaching_interactions', 'risk_assessments']

        conn = sqlite3.connect(self.db_path)
        export_data = {}

        for table in include_tables:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            export_data[table] = df.to_dict('records')

        conn.close()

        if export_format == 'json':
            return json.dumps(export_data, indent=2, default=str)
        elif export_format == 'csv':
            # Create separate CSV files for each table
            csv_data = {}
            for table, records in export_data.items():
                df = pd.DataFrame(records)
                csv_data[table] = df.to_csv(index=False)
            return csv_data
        else:
            return export_data

    def cleanup_old_data(self, retention_days=90):
        """Clean up data older than retention period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=retention_days)).isoformat()

        tables = ['emotions', 'emotional_patterns', 'coaching_interactions', 'risk_assessments']
        deleted_counts = {}

        for table in tables:
            cursor.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff_date,))
            deleted_counts[table] = cursor.rowcount

        conn.commit()
        conn.close()

        return deleted_counts

    def get_database_stats(self):
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}
        tables = ['emotions', 'emotional_patterns', 'coaching_interactions', 'risk_assessments']

        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[f"{table}_count"] = cursor.fetchone()[0]

            cursor.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM {table}")
            min_max = cursor.fetchone()
            stats[f"{table}_date_range"] = {
                'oldest': min_max[0],
                'newest': min_max[1]
            }

        conn.close()
        return stats