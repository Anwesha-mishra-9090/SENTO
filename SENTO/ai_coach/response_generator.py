import random
from datetime import datetime


class ResponseGenerator:
    def __init__(self):
        self.response_templates = self._initialize_response_templates()
        self.personalization_factors = self._initialize_personalization_factors()

    def _initialize_response_templates(self):
        """Initialize response templates for different scenarios"""
        return {
            'emotional_support': {
                'validation': [
                    "It makes complete sense that you'd feel {emotion} given what you're experiencing.",
                    "Your feelings of {emotion} are completely valid and understandable.",
                    "I can see why you'd feel {emotion} in this situation.",
                    "Thank you for sharing that you're feeling {emotion}. That takes courage."
                ],
                'normalization': [
                    "Many people experience {emotion} when facing similar circumstances.",
                    "Feeling {emotion} is a natural human response to what you're describing.",
                    "It's actually very common to feel {emotion} in situations like this."
                ],
                'empowerment': [
                    "Even though you're feeling {emotion}, you have strengths that can help you through this.",
                    "Your awareness of feeling {emotion} shows great self-understanding.",
                    "Remember that feelings like {emotion} are temporary, even when they feel overwhelming."
                ]
            },
            'problem_solving': {
                'reframing': [
                    "What if we looked at this situation from a different perspective?",
                    "Sometimes it helps to consider alternative ways of viewing this.",
                    "Let's explore what this situation might be teaching you."
                ],
                'action_oriented': [
                    "What's one small step you could take right now that might help?",
                    "Breaking this down into smaller pieces might make it feel more manageable.",
                    "What resources or support do you have available to you?"
                ],
                'solution_focused': [
                    "What has helped you cope with similar feelings in the past?",
                    "Imagine this situation resolved - what would be different?",
                    "What strengths have you used before that could help here?"
                ]
            },
            'mindfulness': {
                'present_focus': [
                    "Let's take a moment to just notice what you're experiencing, without judgment.",
                    "What sensations are you aware of in your body right now?",
                    "Can you describe what you're feeling without trying to change it?"
                ],
                'acceptance': [
                    "Sometimes the first step is simply accepting that we feel what we feel.",
                    "All emotions have value - even the difficult ones.",
                    "What if you could make space for this feeling instead of fighting it?"
                ],
                'grounding': [
                    "Let's connect with the present moment. What's one thing you can see right now?",
                    "Take a moment to notice your breath, just as it is.",
                    "What sounds can you hear in your environment right now?"
                ]
            },
            'encouragement': {
                'strength_based': [
                    "I've noticed how resilient you've been in similar situations.",
                    "Your self-awareness in recognizing {emotion} is really impressive.",
                    "You have more inner resources than you might realize right now."
                ],
                'progress_focused': [
                    "Look how far you've come in understanding your emotions.",
                    "Every time you acknowledge how you feel, you're building emotional intelligence.",
                    "The fact that you're reaching out for support shows great self-care."
                ],
                'hope_oriented': [
                    "This feeling won't last forever, even though it might feel that way now.",
                    "Many people find that difficult emotions like this actually lead to growth.",
                    "You've gotten through hard times before - you can get through this too."
                ]
            }
        }

    def _initialize_personalization_factors(self):
        """Initialize factors for personalizing responses"""
        return {
            'communication_style': ['direct', 'gentle', 'metaphorical', 'practical'],
            'response_length': ['brief', 'detailed', 'moderate'],
            'focus_area': ['emotional', 'practical', 'mindfulness', 'balanced']
        }

    def generate_personalized_response(self, emotion_data, user_context, coaching_strategies):
        """Generate a personalized coaching response"""
        # Analyze the emotional context
        emotion_analysis = self._analyze_emotional_context(emotion_data, user_context)

        # Select appropriate response type
        response_type = self._select_response_type(emotion_analysis, coaching_strategies)

        # Personalize based on user preferences and history
        personalization = self._determine_personalization(user_context)

        # Generate the actual response
        response = self._craft_personalized_response(
            emotion_analysis, response_type, personalization, coaching_strategies
        )

        # Add follow-up questions or prompts
        follow_up = self._generate_follow_up_prompt(emotion_analysis, response_type)

        return {
            'response': response,
            'response_type': response_type,
            'personalization_factors': personalization,
            'follow_up_prompt': follow_up,
            'timestamp': datetime.now().isoformat(),
            'emotional_context': emotion_analysis
        }

    def _analyze_emotional_context(self, emotion_data, user_context):
        """Analyze the emotional context for response generation"""
        current_emotion = emotion_data.get('emotion', 'neutral')
        intensity = emotion_data.get('intensity', 1.0)
        valence = emotion_data.get('valence', 0.0)

        # Determine emotional urgency
        urgency = 'high' if intensity > 1.8 else 'moderate' if intensity > 1.2 else 'low'

        # Determine support needed
        if valence < -0.3:
            support_type = 'emotional_support'
        elif valence > 0.3:
            support_type = 'encouragement'
        else:
            support_type = 'mindfulness'

        return {
            'current_emotion': current_emotion,
            'emotional_intensity': intensity,
            'emotional_valence': valence,
            'urgency_level': urgency,
            'primary_support_needed': support_type,
            'contextual_factors': self._extract_contextual_factors(user_context)
        }

    def _extract_contextual_factors(self, user_context):
        """Extract relevant contextual factors"""
        factors = {}

        if user_context.get('recent_stress_events'):
            factors['stress_context'] = True

        if user_context.get('social_isolation'):
            factors['social_context'] = 'isolated'

        if user_context.get('work_pressure'):
            factors['environment_context'] = 'high_pressure'

        return factors

    def _select_response_type(self, emotion_analysis, coaching_strategies):
        """Select the most appropriate response type"""
        primary_emotion = emotion_analysis['current_emotion']
        urgency = emotion_analysis['urgency_level']
        support_needed = emotion_analysis['primary_support_needed']

        # High urgency emotions need immediate emotional support
        if urgency == 'high':
            return 'emotional_support'

        # Match response type to coaching strategies
        strategy_categories = list(coaching_strategies.keys())
        if 'immediate' in strategy_categories:
            return 'emotional_support'
        elif 'short_term' in strategy_categories:
            return 'problem_solving'
        else:
            return support_needed

    def _determine_personalization(self, user_context):
        """Determine personalization factors for the user"""
        # This would typically use user preference data
        # For now, use balanced defaults
        return {
            'communication_style': user_context.get('preferred_style', 'gentle'),
            'response_length': user_context.get('preferred_length', 'moderate'),
            'focus_area': user_context.get('preferred_focus', 'balanced'),
            'use_metaphors': user_context.get('likes_metaphors', True),
            'include_exercises': user_context.get('likes_exercises', True)
        }

    def _craft_personalized_response(self, emotion_analysis, response_type, personalization, coaching_strategies):
        """Craft a personalized response using templates and strategies"""
        emotion = emotion_analysis['current_emotion']
        templates = self.response_templates.get(response_type, {})

        # Build response components
        response_parts = []

        # Start with appropriate opening based on response type
        if response_type == 'emotional_support':
            opening = random.choice(templates['validation']).format(emotion=emotion)
            response_parts.append(opening)

            # Add normalization for difficult emotions
            if emotion_analysis['emotional_valence'] < 0:
                normalization = random.choice(templates['normalization']).format(emotion=emotion)
                response_parts.append(normalization)

        elif response_type == 'problem_solving':
            opening = random.choice(templates['reframing'])
            response_parts.append(opening)

        elif response_type == 'mindfulness':
            opening = random.choice(templates['present_focus'])
            response_parts.append(opening)

        elif response_type == 'encouragement':
            opening = random.choice(templates['strength_based']).format(emotion=emotion)
            response_parts.append(opening)

        # Add coaching strategies
        for strategy_category, strategy_text in coaching_strategies.items():
            if strategy_category == 'immediate':
                response_parts.append(f"Here's something you can try right now: {strategy_text}")
            elif strategy_category == 'short_term':
                response_parts.append(f"For the next little while: {strategy_text}")
            elif strategy_category == 'long_term':
                response_parts.append(f"Something to consider for ongoing support: {strategy_text}")
            elif strategy_category == 'wellness':
                response_parts.append(f"For general wellness: {strategy_text}")

        # Add closing based on personalization
        closing = self._generate_closing_statement(emotion_analysis, personalization)
        response_parts.append(closing)

        return " ".join(response_parts)

    def _generate_closing_statement(self, emotion_analysis, personalization):
        """Generate an appropriate closing statement"""
        emotion = emotion_analysis['current_emotion']
        intensity = emotion_analysis['emotional_intensity']

        if intensity > 1.5:
            closings = [
                "Be gentle with yourself as you navigate these feelings.",
                "Remember to practice self-compassion right now.",
                "You're doing the best you can in this moment."
            ]
        elif emotion in ['sad', 'fear']:
            closings = [
                "This feeling will pass in its own time.",
                "You have the strength to move through this.",
                "Be patient with yourself and the process."
            ]
        else:
            closings = [
                "Keep checking in with how you're feeling.",
                "I'm here whenever you need to talk.",
                "Take things one moment at a time."
            ]

        return random.choice(closings)

    def _generate_follow_up_prompt(self, emotion_analysis, response_type):
        """Generate appropriate follow-up questions or prompts"""
        emotion = emotion_analysis['current_emotion']

        if response_type == 'emotional_support':
            prompts = [
                "Would you like to share more about what's contributing to these feelings?",
                "What's coming up for you as we talk about this?",
                "How does it feel to acknowledge these emotions?"
            ]
        elif response_type == 'problem_solving':
            prompts = [
                "What thoughts come to mind about possible next steps?",
                "Which of these suggestions feels most doable for you?",
                "What support would be most helpful right now?"
            ]
        elif response_type == 'mindfulness':
            prompts = [
                "What are you noticing in your body as we focus on the present?",
                "How does it feel to simply observe your experience?",
                "What's one thing you're aware of right now?"
            ]
        else:
            prompts = [
                "How are you feeling after our conversation?",
                "What would you like to focus on next?",
                "Is there anything else coming up for you?"
            ]

        return random.choice(prompts)

    def adjust_response_tone(self, response, personalization_factors):
        """Adjust response tone based on personalization factors"""
        style = personalization_factors.get('communication_style', 'gentle')
        length = personalization_factors.get('response_length', 'moderate')

        # Adjust based on communication style
        if style == 'direct':
            # Make language more direct and actionable
            response = response.replace("might try", "could try")
            response = response.replace("consider", "try")
        elif style == 'metaphorical':
            # Add metaphorical language if appropriate
            metaphors = {
                'stress': "Like a tree bending in the wind, sometimes we need to be flexible",
                'anxiety': "Anxiety can be like weather patterns - they come and go",
                'sadness': "Sadness can be like rain - it waters our growth"
            }
            # Would add relevant metaphor based on context

        # Adjust length
        if length == 'brief':
            # Shorten response by removing some clauses
            sentences = response.split('. ')
            if len(sentences) > 3:
                response = '. '.join(sentences[:3]) + '.'
        elif length == 'detailed':
            # Ensure response has sufficient detail
            if len(response.split()) < 50:
                # Add more explanatory content
                pass

        return response

    def generate_multiple_response_options(self, emotion_data, user_context, coaching_strategies, num_options=3):
        """Generate multiple response options for user choice"""
        options = []

        for i in range(num_options):
            # Vary personalization slightly for each option
            varied_context = user_context.copy()
            varied_context['preferred_style'] = random.choice(['direct', 'gentle', 'metaphorical'])

            response = self.generate_personalized_response(
                emotion_data, varied_context, coaching_strategies
            )
            options.append(response)

        return {
            'response_options': options,
            'selection_criteria': 'Choose the response that feels most helpful right now',
            'timestamp': datetime.now().isoformat()
        }