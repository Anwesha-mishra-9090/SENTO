import random
from datetime import datetime, timedelta  # Fixed missing import
from collections import defaultdict


class CoachingEngine:
    def __init__(self):
        self.coaching_strategies = self._initialize_coaching_strategies()
        self.intervention_levels = self._initialize_intervention_levels()
        self.user_progress = defaultdict(list)

    def _initialize_coaching_strategies(self):
        """Initialize coaching strategies for different emotional states"""
        return {
            'stress': {
                'immediate': [
                    "Take three deep breaths - inhale for 4 counts, hold for 4, exhale for 6",
                    "Step away from your current situation for 5 minutes",
                    "Drink a glass of water and focus on the sensation"
                ],
                'short_term': [
                    "Practice the 5-4-3-2-1 grounding technique: notice 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste",
                    "Write down what's causing your stress and break it into smaller, manageable parts",
                    "Listen to calming music or nature sounds for 10 minutes"
                ],
                'long_term': [
                    "Establish a daily meditation practice, starting with 5 minutes",
                    "Identify and limit exposure to your main stress triggers",
                    "Develop a consistent sleep schedule and bedtime routine"
                ]
            },
            'anxiety': {
                'immediate': [
                    "Practice box breathing: inhale 4 counts, hold 4, exhale 4, hold 4",
                    "Name three things you can control right now",
                    "Place your hand on your chest and feel your heartbeat, breathing slowly to calm it"
                ],
                'short_term': [
                    "Challenge anxious thoughts by asking 'What's the evidence for this worry?'",
                    "Create a worry period - schedule 15 minutes later to address worries",
                    "Engage in light physical activity like walking or stretching"
                ],
                'long_term': [
                    "Practice progressive muscle relaxation daily",
                    "Keep a thought record to identify anxiety patterns",
                    "Gradually expose yourself to anxiety triggers in controlled ways"
                ]
            },
            'sadness': {
                'immediate': [
                    "Reach out to someone you trust and share how you're feeling",
                    "Engage in a small, comforting activity you enjoy",
                    "Write down three things you're grateful for right now"
                ],
                'short_term': [
                    "Schedule pleasant activities for the coming days",
                    "Practice self-compassion - talk to yourself as you would a good friend",
                    "Get outside for some sunlight and fresh air"
                ],
                'long_term': [
                    "Establish a routine that includes social connection",
                    "Consider talking to a mental health professional",
                    "Engage in regular physical exercise"
                ]
            },
            'anger': {
                'immediate': [
                    "Count slowly to 10 before responding",
                    "Remove yourself from the situation temporarily",
                    "Squeeze a stress ball or press your palms together firmly"
                ],
                'short_term': [
                    "Express your feelings using 'I' statements",
                    "Channel the energy into physical activity",
                    "Write a letter expressing your anger (you don't have to send it)"
                ],
                'long_term': [
                    "Identify your anger triggers and develop coping strategies",
                    "Practice assertive communication skills",
                    "Learn conflict resolution techniques"
                ]
            },
            'general_wellness': {
                'daily': [
                    "Start your day with 5 minutes of intention setting",
                    "Take regular breaks to stretch and move throughout the day",
                    "Practice ending your work day with a shutdown ritual"
                ],
                'weekly': [
                    "Schedule one activity purely for enjoyment each week",
                    "Connect with friends or family members",
                    "Review your accomplishments and learning from the week"
                ]
            }
        }

    def _initialize_intervention_levels(self):
        """Initialize intervention levels based on emotional intensity"""
        return {
            'low': {
                'frequency': 'occasional',
                'intensity': 'gentle_reminders',
                'approach': 'preventive'
            },
            'moderate': {
                'frequency': 'regular',
                'intensity': 'guided_practices',
                'approach': 'supportive'
            },
            'high': {
                'frequency': 'frequent',
                'intensity': 'immediate_interventions',
                'approach': 'crisis_support'
            }
        }

    def generate_coaching_response(self, current_emotion, emotion_intensity, context=None):
        """Generate appropriate coaching response based on current emotional state"""
        # Determine intervention level
        intervention_level = self._determine_intervention_level(emotion_intensity)

        # Select appropriate strategies
        strategies = self._select_strategies(current_emotion, intervention_level, context)

        # Generate personalized response
        response = self._craft_response(current_emotion, strategies, context)

        # Track coaching interaction
        self._track_coaching_interaction(current_emotion, intervention_level, strategies)

        return {
            'coaching_response': response,
            'strategies': strategies,
            'intervention_level': intervention_level,
            'follow_up_timing': self._determine_follow_up_timing(intervention_level),
            'timestamp': datetime.now().isoformat()
        }

    def _determine_intervention_level(self, intensity):
        """Determine appropriate intervention level based on emotional intensity"""
        if intensity < 1.2:
            return 'low'
        elif intensity < 1.8:
            return 'moderate'
        else:
            return 'high'

    def _select_strategies(self, emotion, intervention_level, context):
        """Select appropriate coaching strategies"""
        strategies = {}

        # Get emotion-specific strategies
        if emotion in self.coaching_strategies:
            emotion_strategies = self.coaching_strategies[emotion]

            # Select based on intervention level
            if intervention_level == 'high':
                strategies['immediate'] = random.choice(emotion_strategies['immediate'])
                strategies['short_term'] = random.choice(emotion_strategies['short_term'])
            elif intervention_level == 'moderate':
                strategies['short_term'] = random.choice(emotion_strategies['short_term'])
                strategies['long_term'] = random.choice(emotion_strategies['long_term'])
            else:
                strategies['long_term'] = random.choice(emotion_strategies['long_term'])

        # Add general wellness strategies for lower intensity
        if intervention_level in ['low', 'moderate']:
            wellness_strategies = self.coaching_strategies['general_wellness']
            strategies['wellness'] = random.choice(wellness_strategies['daily'])

        return strategies

    def _craft_response(self, emotion, strategies, context):
        """Craft a natural, empathetic coaching response"""
        empathy_phrases = {
            'stress': "I notice you're feeling stressed right now. That's completely understandable.",
            'anxiety': "It sounds like you're experiencing some anxiety. That can feel overwhelming.",
            'sadness': "I hear the sadness in your experience. Thank you for sharing that with me.",
            'anger': "I can sense the frustration you're feeling. That's a valid emotional response.",
            'neutral': "Thanks for checking in about how you're feeling.",
            'happy': "It's wonderful to hear you're feeling positive!"
        }

        # Start with empathy
        empathy = empathy_phrases.get(emotion, "I appreciate you sharing how you're feeling.")

        # Add strategy explanations
        strategy_text = []
        for category, strategy in strategies.items():
            if category == 'immediate':
                strategy_text.append(f"Right now, you might try: {strategy}")
            elif category == 'short_term':
                strategy_text.append(f"In the next little while, consider: {strategy}")
            elif category == 'long_term':
                strategy_text.append(f"For ongoing support, you could: {strategy}")
            elif category == 'wellness':
                strategy_text.append(f"For general wellness: {strategy}")

        # Combine into final response
        response_parts = [empathy] + strategy_text
        return " ".join(response_parts)

    def _determine_follow_up_timing(self, intervention_level):
        """Determine when to follow up based on intervention level"""
        timing_map = {
            'high': '1-2 hours',
            'moderate': '4-6 hours',
            'low': '24 hours'
        }
        return timing_map.get(intervention_level, '24 hours')

    def _track_coaching_interaction(self, emotion, intervention_level, strategies):
        """Track coaching interactions for personalization"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'emotion': emotion,
            'intervention_level': intervention_level,
            'strategies_used': list(strategies.keys()),
            'user_engagement': 'pending'  # Would be updated based on user response
        }

        self.user_progress['coaching_interactions'].append(interaction)

        # Keep only recent interactions
        if len(self.user_progress['coaching_interactions']) > 50:
            self.user_progress['coaching_interactions'] = self.user_progress['coaching_interactions'][-50:]

    def assess_coaching_effectiveness(self, emotion_before, emotion_after, time_gap_minutes=60):
        """Assess effectiveness of coaching interventions"""
        emotion_valences = {
            'happy': 1, 'excited': 1,
            'sad': -1, 'angry': -1, 'fear': -1,
            'surprise': 0.5, 'neutral': 0
        }

        valence_before = emotion_valences.get(emotion_before, 0)
        valence_after = emotion_valences.get(emotion_after, 0)

        effectiveness = valence_after - valence_before  # Positive change is good

        # Adjust for time gap (quicker improvement is better)
        time_factor = max(0, 1 - (time_gap_minutes / 120))  # 2-hour window

        final_score = effectiveness * time_factor

        return {
            'effectiveness_score': final_score,
            'emotional_shift': f"{emotion_before} → {emotion_after}",
            'improvement': 'positive' if final_score > 0 else 'negative' if final_score < 0 else 'neutral',
            'time_to_change_minutes': time_gap_minutes
        }

    def generate_progress_report(self, period="7d"):
        """Generate coaching progress report"""
        interactions = self.user_progress.get('coaching_interactions', [])

        if not interactions:
            return {"message": "No coaching interactions recorded yet"}

        # Filter by period if needed
        if period != "all":
            cutoff_date = datetime.now() - timedelta(days=int(period[:-1]))
            interactions = [i for i in interactions
                            if datetime.fromisoformat(i['timestamp']) > cutoff_date]

        # Calculate metrics
        total_interactions = len(interactions)
        emotion_counts = defaultdict(int)
        intervention_counts = defaultdict(int)

        for interaction in interactions:
            emotion_counts[interaction['emotion']] += 1
            intervention_counts[interaction['intervention_level']] += 1

        # Most common emotions addressed
        common_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            'report_period': period,
            'total_coaching_sessions': total_interactions,
            'most_addressed_emotions': dict(common_emotions),
            'intervention_distribution': dict(intervention_counts),
            'average_sessions_per_week': total_interactions / max(1, int(period[:-1])),
            'coaching_engagement': 'high' if total_interactions > 10 else 'moderate' if total_interactions > 5 else 'low'
        }

    def personalize_coaching_approach(self, user_preferences, historical_effectiveness):
        """Personalize coaching based on user preferences and historical effectiveness"""
        personalized_strategies = self.coaching_strategies.copy()

        # Adjust strategies based on user preferences
        if 'preferred_approach' in user_preferences:
            approach = user_preferences['preferred_approach']
            if approach == 'mindfulness_based':
                # Emphasize mindfulness strategies
                for emotion_strategies in personalized_strategies.values():
                    if 'immediate' in emotion_strategies:
                        emotion_strategies['immediate'] = [s for s in emotion_strategies['immediate']
                                                           if 'breath' in s.lower() or 'mindful' in s.lower()]

        # Adjust based on historical effectiveness
        effective_strategies = historical_effectiveness.get('effective_strategies', [])
        for strategy in effective_strategies:
            # Promote effective strategies to higher priority
            pass  # Implementation would track strategy effectiveness

        return personalized_strategies