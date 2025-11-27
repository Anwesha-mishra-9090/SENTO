import json
import requests
from datetime import datetime
import threading
import time
from collections import deque


class MobileIntegration:
    def __init__(self, api_gateway, push_notification_handler=None):
        self.api_gateway = api_gateway
        self.push_handler = push_notification_handler
        self.mobile_sessions = {}
        self.notification_queue = deque()

        # Mobile-specific configurations
        self.mobile_config = {
            'session_timeout': 3600,  # 1 hour
            'max_notifications_per_hour': 10,
            'emergency_check_interval': 300,  # 5 minutes
            'data_sync_interval': 600  # 10 minutes
        }

    def create_mobile_session(self, user_id, device_info=None):
        """Create a new mobile session for a user"""
        session_id = f"mobile_{user_id}_{int(time.time())}"

        self.mobile_sessions[session_id] = {
            'user_id': user_id,
            'device_info': device_info or {},
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'emotional_context': {},
            'notification_preferences': self._get_default_preferences(),
            'session_data': {
                'emotional_checkins': 0,
                'coaching_sessions': 0,
                'stress_alerts': 0
            }
        }

        print(f"Mobile session created: {session_id}")
        return session_id

    def _get_default_preferences(self):
        """Get default notification preferences"""
        return {
            'emotional_checkin_reminders': True,
            'stress_alerts': True,
            'coaching_suggestions': True,
            'progress_updates': True,
            'quiet_hours': {'start': '22:00', 'end': '07:00'},
            'notification_frequency': 'moderate'  # low, moderate, high
        }

    def process_mobile_emotional_checkin(self, session_id, emotion_data, context=None):
        """Process emotional checkin from mobile app"""
        if session_id not in self.mobile_sessions:
            return {'error': 'Invalid session'}

        try:
            # Update session activity
            self.mobile_sessions[session_id]['last_activity'] = datetime.now().isoformat()
            self.mobile_sessions[session_id]['session_data']['emotional_checkins'] += 1

            # Store emotional context
            if context:
                self.mobile_sessions[session_id]['emotional_context'] = context

            # Analyze emotion using API gateway
            analysis_result = self._analyze_emotion_mobile(emotion_data, context)

            # Check for interventions
            intervention_needed = self._check_intervention_needed(analysis_result, session_id)

            # Prepare mobile response
            mobile_response = {
                'analysis': analysis_result,
                'intervention_suggested': intervention_needed,
                'timestamp': datetime.now().isoformat(),
                'session_metrics': self.mobile_sessions[session_id]['session_data']
            }

            if intervention_needed:
                coaching_response = self._get_mobile_coaching(analysis_result, context)
                mobile_response['coaching_suggestion'] = coaching_response

                # Queue notification if needed
                self._queue_notification(session_id, 'coaching_suggestion', coaching_response)

            return mobile_response

        except Exception as e:
            return {'error': f"Checkin processing failed: {str(e)}"}

    def _analyze_emotion_mobile(self, emotion_data, context):
        """Analyze emotion data for mobile integration"""
        # Use the API gateway for emotion analysis
        endpoint = '/api/analyze/voice' if emotion_data.get('type') == 'voice' else '/api/analyze/text'

        try:
            # This would make an internal API call in a real implementation
            # For now, simulate the response structure
            return {
                'emotion': emotion_data.get('emotion', 'neutral'),
                'confidence': emotion_data.get('confidence', 0.5),
                'intensity': emotion_data.get('intensity', 1.0),
                'valence': emotion_data.get('valence', 0.0),
                'analysis_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': f"Emotion analysis failed: {str(e)}"}

    def _check_intervention_needed(self, analysis_result, session_id):
        """Check if intervention is needed based on emotional analysis"""
        if 'error' in analysis_result:
            return False

        emotion = analysis_result.get('emotion', 'neutral')
        intensity = analysis_result.get('intensity', 1.0)
        valence = analysis_result.get('valence', 0.0)

        # High intensity negative emotions might need intervention
        if intensity > 1.5 and valence < -0.3:
            return True

        # Persistent neutral/low arousal might need engagement
        session_data = self.mobile_sessions[session_id]['session_data']
        if (emotion == 'neutral' and intensity < 0.8 and
                session_data['emotional_checkins'] > 5):
            return True

        return False

    def _get_mobile_coaching(self, analysis_result, context):
        """Get coaching suggestion for mobile"""
        emotion = analysis_result.get('emotion', 'neutral')
        intensity = analysis_result.get('intensity', 1.0)

        coaching_map = {
            'high_intensity': {
                'sad': "Take a moment for deep breathing. Inhale for 4 counts, exhale for 6.",
                'angry': "Step away and count to 10. Focus on your breathing.",
                'fear': "Name three things you can see around you. This can help ground you.",
                'happy': "Savor this positive moment! What made you feel this way?"
            },
            'medium_intensity': {
                'sad': "Would you like to share what's on your mind? I'm here to listen.",
                'angry': "Try the 5-4-3-2-1 grounding technique.",
                'fear': "Remember, this feeling will pass. You've gotten through hard times before.",
                'happy': "It's wonderful to see you experiencing positive emotions!",
                'neutral': "How are you really feeling beneath the surface?"
            }
        }

        intensity_key = 'high_intensity' if intensity > 1.5 else 'medium_intensity'

        return coaching_map.get(intensity_key, {}).get(
            emotion,
            "Thanks for checking in. How can I support you today?"
        )

    def _queue_notification(self, session_id, notification_type, content):
        """Queue a notification for mobile delivery"""
        if session_id not in self.mobile_sessions:
            return

        session = self.mobile_sessions[session_id]
        preferences = session['notification_preferences']

        # Check if notifications are enabled for this type
        if not preferences.get(notification_type, True):
            return

        # Check quiet hours
        if self._in_quiet_hours(preferences.get('quiet_hours')):
            print("Quiet hours - notification queued for later")
            # Would store for delivery later
            return

        notification = {
            'session_id': session_id,
            'user_id': session['user_id'],
            'type': notification_type,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'priority': 'medium'
        }

        self.notification_queue.append(notification)
        print(f"Notification queued: {notification_type}")

    def _in_quiet_hours(self, quiet_hours):
        """Check if current time is within quiet hours"""
        if not quiet_hours:
            return False

        try:
            current_time = datetime.now().time()
            start_time = datetime.strptime(quiet_hours['start'], '%H:%M').time()
            end_time = datetime.strptime(quiet_hours['end'], '%H:%M').time()

            if start_time < end_time:
                return start_time <= current_time <= end_time
            else:
                return current_time >= start_time or current_time <= end_time

        except:
            return False

    def send_push_notification(self, session_id, title, message, data=None):
        """Send push notification to mobile device"""
        if session_id not in self.mobile_sessions:
            return False

        if not self.push_handler:
            print(f"[SIMULATED PUSH] {title}: {message}")
            return True

        try:
            session = self.mobile_sessions[session_id]
            device_token = session['device_info'].get('push_token')

            if not device_token:
                print("No device token available for push notification")
                return False

            # This would integrate with actual push notification service
            # (Firebase Cloud Messaging, Apple Push Notification Service, etc.)
            success = self.push_handler.send_notification(
                device_token, title, message, data
            )

            return success

        except Exception as e:
            print(f"Push notification failed: {e}")
            return False

    def process_mobile_stress_alert(self, session_id, stress_data):
        """Process stress alert from mobile app"""
        if session_id not in self.mobile_sessions:
            return {'error': 'Invalid session'}

        try:
            # Update session
            self.mobile_sessions[session_id]['last_activity'] = datetime.now().isoformat()
            self.mobile_sessions[session_id]['session_data']['stress_alerts'] += 1

            # Analyze stress level
            stress_level = stress_data.get('level', 0)
            stress_context = stress_data.get('context', {})

            response = {
                'stress_level': stress_level,
                'risk_category': self._classify_stress_risk(stress_level),
                'timestamp': datetime.now().isoformat(),
                'immediate_suggestions': self._get_stress_coping_suggestions(stress_level)
            }

            # Send urgent notification for high stress
            if stress_level > 0.7:
                self.send_push_notification(
                    session_id,
                    "High Stress Alert",
                    "I'm detecting high stress levels. Let's work through this together.",
                    {'type': 'stress_alert', 'level': 'high'}
                )

            return response

        except Exception as e:
            return {'error': f"Stress alert processing failed: {str(e)}"}

    def _classify_stress_risk(self, stress_level):
        """Classify stress risk level"""
        if stress_level < 0.3:
            return 'low'
        elif stress_level < 0.6:
            return 'moderate'
        else:
            return 'high'

    def _get_stress_coping_suggestions(self, stress_level):
        """Get stress coping suggestions"""
        if stress_level > 0.7:
            return [
                "Practice deep breathing: 4-7-8 technique",
                "Step away from current situation",
                "Drink a glass of water",
                "Use 5-4-3-2-1 grounding technique"
            ]
        elif stress_level > 0.4:
            return [
                "Take 3 deep breaths",
                "Stretch your body",
                "Listen to calming music",
                "Practice mindfulness for 2 minutes"
            ]
        else:
            return [
                "Maintain regular breaks",
                "Practice good sleep hygiene",
                "Stay hydrated throughout the day"
            ]

    def get_mobile_insights(self, session_id, time_period="7d"):
        """Get personalized insights for mobile display"""
        if session_id not in self.mobile_sessions:
            return {'error': 'Invalid session'}

        session = self.mobile_sessions[session_id]

        # This would integrate with analytics engine
        # For now, provide simulated insights
        insights = {
            'emotional_patterns': self._get_simulated_patterns(session),
            'weekly_summary': self._get_weekly_summary(session),
            'personalized_recommendations': self._get_mobile_recommendations(session),
            'progress_metrics': session['session_data']
        }

        return insights

    def _get_simulated_patterns(self, session):
        """Get simulated emotional patterns (would use real analytics)"""
        return {
            'most_common_emotion': 'neutral',
            'emotional_volatility': 'low',
            'peak_emotional_times': ['Morning', 'Evening'],
            'stress_patterns': 'Moderate stress on weekdays'
        }

    def _get_weekly_summary(self, session):
        """Get weekly emotional summary"""
        return {
            'emotional_checkins': session['session_data']['emotional_checkins'],
            'coaching_sessions': session['session_data']['coaching_sessions'],
            'average_mood': 'balanced',
            'wellness_trend': 'improving'
        }

    def _get_mobile_recommendations(self, session):
        """Get personalized recommendations for mobile"""
        recommendations = []

        checkin_count = session['session_data']['emotional_checkins']

        if checkin_count < 5:
            recommendations.append("Try checking in 3 times daily for better insights")
        else:
            recommendations.append("Great consistency! Keep tracking your emotions")

        recommendations.append("Practice morning intention setting")
        recommendations.append("Schedule weekly emotional reflection")

        return recommendations

    def cleanup_expired_sessions(self):
        """Clean up expired mobile sessions"""
        current_time = datetime.now()
        expired_sessions = []

        for session_id, session in self.mobile_sessions.items():
            last_activity = datetime.fromisoformat(session['last_activity'])
            time_diff = (current_time - last_activity).total_seconds()

            if time_diff > self.mobile_config['session_timeout']:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.mobile_sessions[session_id]
            print(f"Cleaned up expired session: {session_id}")

        return len(expired_sessions)


# Push notification handler (placeholder)
class PushNotificationHandler:
    """Placeholder for actual push notification service integration"""

    def __init__(self):
        self.sent_notifications = []

    def send_notification(self, device_token, title, message, data=None):
        """Send push notification (placeholder implementation)"""
        # In real implementation, this would integrate with FCM/APNS
        notification = {
            'device_token': device_token,
            'title': title,
            'message': message,
            'data': data,
            'sent_at': datetime.now().isoformat()
        }

        self.sent_notifications.append(notification)
        print(f"[PUSH] {title}: {message}")
        return True

    def get_notification_stats(self):
        """Get notification statistics"""
        return {
            'total_sent': len(self.sent_notifications),
            'last_sent': self.sent_notifications[-1]['sent_at'] if self.sent_notifications else None
        }