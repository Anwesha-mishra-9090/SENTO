import random  # Added missing import
from collections import deque, defaultdict
from datetime import datetime, timedelta


class DialogueManager:
    def __init__(self, max_context_length=10):
        self.conversation_history = deque(maxlen=max_context_length)
        self.dialogue_states = self._initialize_dialogue_states()
        self.user_engagement_level = 'medium'
        self.session_start_time = datetime.now()
        self.current_state = 'greeting'  # Initialize current_state

    def _initialize_dialogue_states(self):
        """Initialize possible dialogue states and transitions"""
        return {
            'greeting': {
                'transitions': ['emotional_checkin', 'direct_issue', 'general_support'],
                'prompts': [
                    "Hello! How are you feeling today?",
                    "Welcome back! What's on your mind?",
                    "Good to see you. How can I support you today?"
                ]
            },
            'emotional_checkin': {
                'transitions': ['emotional_exploration', 'coping_strategies', 'validation'],
                'prompts': [
                    "I hear that you're feeling {emotion}. Tell me more about that.",
                    "What's contributing to these feelings of {emotion}?",
                    "How long have you been feeling {emotion}?"
                ]
            },
            'emotional_exploration': {
                'transitions': ['coping_strategies', 'problem_solving', 'resource_referral'],
                'prompts': [
                    "That sounds really challenging. What's that experience been like for you?",
                    "I can understand why that would bring up feelings of {emotion}.",
                    "What else is coming up as we talk about this?"
                ]
            },
            'coping_strategies': {
                'transitions': ['strategy_implementation', 'follow_up', 'emotional_checkin'],
                'prompts': [
                    "Would you like to try a coping strategy for these feelings?",
                    "What's helped you manage similar feelings in the past?",
                    "Let's explore some ways to work with these emotions."
                ]
            },
            'problem_solving': {
                'transitions': ['action_planning', 'resource_referral', 'emotional_checkin'],
                'prompts': [
                    "What would be most helpful to focus on right now?",
                    "Let's break this down into manageable steps.",
                    "What resources or support might be available to you?"
                ]
            },
            'validation': {
                'transitions': ['emotional_exploration', 'coping_strategies', 'follow_up'],
                'prompts': [
                    "Your feelings are completely valid and understandable.",
                    "It makes sense that you'd feel this way given what you're experiencing.",
                    "Thank you for sharing that with me. That takes courage."
                ]
            },
            'follow_up': {
                'transitions': ['emotional_checkin', 'closure', 'resource_referral'],
                'prompts': [
                    "How are you feeling after our conversation?",
                    "What would you like to take away from our discussion?",
                    "Is there anything else you'd like to explore?"
                ]
            },
            'closure': {
                'transitions': ['greeting'],
                'prompts': [
                    "I'm glad we could talk today. Remember I'm here whenever you need support.",
                    "Thank you for our conversation. Take good care of yourself.",
                    "I appreciate you sharing with me today. Be gentle with yourself."
                ]
            }
        }

    def add_conversation_turn(self, user_input, system_response, emotional_context):
        """Add a conversation turn to the history"""
        turn = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'system_response': system_response,
            'emotional_context': emotional_context,
            'dialogue_state': self.current_state
        }

        self.conversation_history.append(turn)
        self._update_engagement_level(user_input, system_response)

    def _update_engagement_level(self, user_input, system_response):
        """Update user engagement level based on interaction patterns"""
        input_length = len(user_input.split())
        response_ratio = len(system_response.split()) / max(1, input_length)

        # Analyze engagement indicators
        engagement_indicators = []

        if input_length > 15:
            engagement_indicators.append('high_input')
        elif input_length > 5:
            engagement_indicators.append('medium_input')
        else:
            engagement_indicators.append('low_input')

        if response_ratio > 2:
            engagement_indicators.append('high_engagement')
        elif response_ratio > 1:
            engagement_indicators.append('medium_engagement')
        else:
            engagement_indicators.append('low_engagement')

        # Determine overall engagement level
        if engagement_indicators.count('high_input') + engagement_indicators.count('high_engagement') >= 1:
            self.user_engagement_level = 'high'
        elif engagement_indicators.count('low_input') + engagement_indicators.count('low_engagement') >= 2:
            self.user_engagement_level = 'low'
        else:
            self.user_engagement_level = 'medium'

    def get_next_dialogue_state(self, current_emotion, emotion_intensity, user_input):
        """Determine the next appropriate dialogue state"""
        current_state_info = self.dialogue_states.get(self.current_state, {})
        possible_transitions = current_state_info.get('transitions', [])

        # Determine next state based on emotional context and conversation flow
        if emotion_intensity > 1.5 and 'emotional_exploration' in possible_transitions:
            next_state = 'emotional_exploration'
        elif emotion_intensity < 0.8 and 'problem_solving' in possible_transitions:
            next_state = 'problem_solving'
        elif len(self.conversation_history) >= 3 and 'follow_up' in possible_transitions:
            next_state = 'follow_up'
        else:
            # Choose most appropriate transition based on context
            next_state = self._select_optimal_transition(possible_transitions, current_emotion)

        self.current_state = next_state
        return next_state

    def _select_optimal_transition(self, possible_transitions, current_emotion):
        """Select the optimal transition based on conversation context"""
        # Simple heuristic for transition selection
        if not self.conversation_history:
            return 'greeting'

        # Get recent emotional patterns
        recent_emotions = [turn['emotional_context'].get('emotion', 'neutral')
                           for turn in list(self.conversation_history)[-3:]]

        # If same emotion persists, move to coping strategies
        if len(recent_emotions) >= 2 and all(e == current_emotion for e in recent_emotions[-2:]):
            if 'coping_strategies' in possible_transitions:
                return 'coping_strategies'

        # If emotional intensity is decreasing, move to problem-solving
        recent_intensities = [turn['emotional_context'].get('intensity', 1.0)
                              for turn in list(self.conversation_history)[-3:]]
        if len(recent_intensities) >= 2 and recent_intensities[-1] < recent_intensities[-2]:
            if 'problem_solving' in possible_transitions:
                return 'problem_solving'

        # Default to emotional exploration
        if 'emotional_exploration' in possible_transitions:
            return 'emotional_exploration'

        return possible_transitions[0] if possible_transitions else 'greeting'

    def generate_state_prompt(self, dialogue_state, emotional_context=None):
        """Generate appropriate prompt for the current dialogue state"""
        state_info = self.dialogue_states.get(dialogue_state, {})
        prompts = state_info.get('prompts', ["How can I help you?"])

        # Select and personalize prompt
        selected_prompt = random.choice(prompts)

        if emotional_context and '{emotion}' in selected_prompt:
            emotion = emotional_context.get('emotion', 'this way')
            selected_prompt = selected_prompt.format(emotion=emotion)

        return selected_prompt

    def manage_conversation_flow(self, user_input, emotional_analysis):
        """Manage the overall conversation flow"""
        # Add current turn to history
        current_turn = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'emotional_analysis': emotional_analysis,
            'engagement_level': self.user_engagement_level
        }

        # Determine next dialogue state
        next_state = self.get_next_dialogue_state(
            emotional_analysis.get('emotion', 'neutral'),
            emotional_analysis.get('intensity', 1.0),
            user_input
        )

        # Generate appropriate prompt
        prompt = self.generate_state_prompt(next_state, emotional_analysis)

        # Prepare conversation context
        conversation_context = {
            'current_state': next_state,
            'engagement_level': self.user_engagement_level,
            'conversation_length': len(self.conversation_history),
            'session_duration': (datetime.now() - self.session_start_time).total_seconds(),
            'suggested_prompt': prompt,
            'possible_transitions': self.dialogue_states.get(next_state, {}).get('transitions', [])
        }

        return conversation_context

    def detect_conversation_patterns(self):
        """Detect patterns in the conversation history"""
        if len(self.conversation_history) < 3:
            return {}

        patterns = {
            'emotional_flow': self._analyze_emotional_flow(),
            'engagement_trends': self._analyze_engagement_trends(),
            'topic_consistency': self._analyze_topic_consistency(),
            'conversation_depth': self._assess_conversation_depth()
        }

        return patterns

    def _analyze_emotional_flow(self):
        """Analyze how emotions flow through the conversation"""
        emotions = [turn['emotional_context'].get('emotion', 'neutral')
                    for turn in self.conversation_history]
        intensities = [turn['emotional_context'].get('intensity', 1.0)
                       for turn in self.conversation_history]

        # Calculate emotional variability
        emotion_changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i - 1])
        emotion_variability = emotion_changes / (len(emotions) - 1) if len(emotions) > 1 else 0

        # Calculate intensity trend
        if len(intensities) >= 2:
            intensity_trend = 'decreasing' if intensities[-1] < intensities[0] else 'increasing' if intensities[-1] > intensities[0] else 'stable'
        else:
            intensity_trend = 'stable'

        return {
            'emotional_variability': emotion_variability,
            'intensity_trend': intensity_trend,
            'dominant_emotion': max(set(emotions), key=emotions.count) if emotions else 'neutral',
            'emotional_range': len(set(emotions))
        }

    def _analyze_engagement_trends(self):
        """Analyze user engagement trends throughout conversation"""
        engagement_scores = []

        for turn in self.conversation_history:
            input_length = len(turn['user_input'].split())
            # Simple engagement score based on input length and emotional intensity
            intensity = turn['emotional_context'].get('intensity', 1.0)
            engagement_score = min(input_length * intensity / 10, 1.0)
            engagement_scores.append(engagement_score)

        if len(engagement_scores) >= 2:
            engagement_trend = 'increasing' if engagement_scores[-1] > engagement_scores[0] else 'decreasing' if engagement_scores[-1] < engagement_scores[0] else 'stable'
        else:
            engagement_trend = 'stable'

        return {
            'average_engagement': sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0,
            'engagement_trend': engagement_trend,
            'peak_engagement': max(engagement_scores) if engagement_scores else 0
        }

    def _analyze_topic_consistency(self):
        """Analyze how consistent the conversation topics are"""
        # Simple topic consistency based on emotional consistency
        emotions = [turn['emotional_context'].get('emotion', 'neutral')
                    for turn in self.conversation_history]

        if len(emotions) < 2:
            return {'consistency_score': 1.0}

        same_emotion_count = sum(1 for i in range(1, len(emotions)) if emotions[i] == emotions[i - 1])
        consistency_score = same_emotion_count / (len(emotions) - 1)

        return {
            'consistency_score': consistency_score,
            'topic_stability': 'high' if consistency_score > 0.7 else 'low' if consistency_score < 0.3 else 'medium'
        }

    def _assess_conversation_depth(self):
        """Assess the depth of the conversation"""
        depth_indicators = []

        # Input length indicator
        avg_input_length = sum(len(turn['user_input'].split()) for turn in self.conversation_history) / len(
            self.conversation_history)
        if avg_input_length > 10:
            depth_indicators.append('detailed_sharing')

        # Emotional intensity indicator
        avg_intensity = sum(
            turn['emotional_context'].get('intensity', 1.0) for turn in self.conversation_history) / len(
            self.conversation_history)
        if avg_intensity > 1.5:
            depth_indicators.append('high_emotional_engagement')

        # Conversation length indicator
        if len(self.conversation_history) > 5:
            depth_indicators.append('extended_dialogue')

        return {
            'depth_level': 'deep' if len(depth_indicators) >= 2 else 'moderate' if depth_indicators else 'surface',
            'indicators_present': depth_indicators,
            'conversation_complexity': 'high' if len(depth_indicators) >= 2 else 'low'
        }

    def suggest_conversation_shifts(self, current_patterns):
        """Suggest potential conversation shifts based on patterns"""
        suggestions = []

        patterns = self.detect_conversation_patterns()
        emotional_flow = patterns.get('emotional_flow', {})
        engagement_trends = patterns.get('engagement_trends', {})

        # Suggest shifts based on emotional stagnation
        if emotional_flow.get('emotional_variability', 0) < 0.2:
            suggestions.append("Consider exploring different emotional aspects")

        # Suggest shifts based on decreasing engagement
        if engagement_trends.get('engagement_trend') == 'decreasing':
            suggestions.append("Might be time to shift focus or take a break")

        # Suggest deepening based on surface-level engagement
        conversation_depth = patterns.get('conversation_depth', {})
        if conversation_depth.get('depth_level') == 'surface' and len(self.conversation_history) > 3:
            suggestions.append("Opportunity to explore topics more deeply")

        return suggestions

    def get_conversation_summary(self):
        """Generate a summary of the current conversation"""
        if not self.conversation_history:
            return {"message": "No conversation history available"}

        patterns = self.detect_conversation_patterns()
        emotional_flow = patterns.get('emotional_flow', {})
        engagement_trends = patterns.get('engagement_trends', {})

        summary = {
            'session_duration_minutes': round((datetime.now() - self.session_start_time).total_seconds() / 60, 1),
            'total_exchanges': len(self.conversation_history),
            'primary_emotions_discussed': list(set(
                turn['emotional_context'].get('emotion', 'neutral')
                for turn in self.conversation_history
            )),
            'current_engagement_level': self.user_engagement_level,
            'emotional_progression': emotional_flow.get('intensity_trend', 'stable'),
            'conversation_quality': patterns.get('conversation_depth', {}).get('depth_level', 'surface'),
            'suggested_next_steps': self.suggest_conversation_shifts(patterns)
        }

        return summary

    def reset_conversation(self):
        """Reset the conversation for a new session"""
        self.conversation_history.clear()
        self.current_state = 'greeting'
        self.user_engagement_level = 'medium'
        self.session_start_time = datetime.now()