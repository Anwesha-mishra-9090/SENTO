import numpy as np
from datetime import datetime, timedelta


class InterventionSystem:
    def __init__(self):
        self.crisis_keywords = self._initialize_crisis_keywords()
        self.intervention_protocols = self._initialize_intervention_protocols()
        self.risk_assessment_factors = self._initialize_risk_factors()
        self.emergency_contacts = self._initialize_emergency_contacts()

    def _initialize_crisis_keywords(self):
        """Initialize keywords that indicate potential crisis situations"""
        return {
            'high_risk': [
                'suicide', 'kill myself', 'end it all', 'want to die', 'not worth living',
                'harm myself', 'self harm', 'cutting', 'overdose', 'jump off',
                'gun', 'pills', 'rope', 'bridge', 'no way out'
            ],
            'moderate_risk': [
                'can\'t go on', 'can\'t take it', 'hopeless', 'helpless', 'desperate',
                'overwhelmed', 'drowning', 'breaking point', 'losing control',
                'panic attack', 'anxiety attack', 'meltdown'
            ],
            'low_risk': [
                'depressed', 'anxious', 'stressed', 'overwhelmed', 'sad',
                'lonely', 'scared', 'frightened', 'nervous'
            ]
        }

    def _initialize_intervention_protocols(self):
        """Initialize intervention protocols for different risk levels"""
        return {
            'crisis_immediate': {
                'priority': 'highest',
                'actions': [
                    "I'm very concerned about what you're sharing. Your safety is the most important thing right now.",
                    "Please call emergency services immediately or go to your nearest emergency room.",
                    "You can also call a crisis helpline: National Suicide Prevention Lifeline at 1-800-273-8255",
                    "I'm here with you while you take these steps. You don't have to go through this alone."
                ],
                'follow_up': 'immediate_emergency_services'
            },
            'high_risk': {
                'priority': 'high',
                'actions': [
                    "What you're describing sounds very serious and I'm concerned about your safety.",
                    "It's really important that you connect with a mental health professional right away.",
                    "Would you be willing to call a crisis line or reach out to someone you trust?",
                    "I can help you find local resources if that would be helpful."
                ],
                'follow_up': 'within_hours_professional_support'
            },
            'moderate_risk': {
                'priority': 'medium',
                'actions': [
                    "I hear how much pain you're in right now. That sounds incredibly difficult.",
                    "Let's make sure you have some immediate support. Is there someone you can reach out to?",
                    "Would you like me to share some coping strategies that might help right now?",
                    "Remember that these intense feelings, while overwhelming, are temporary."
                ],
                'follow_up': 'within_24_hours_checkin'
            },
            'low_risk': {
                'priority': 'low',
                'actions': [
                    "I hear that you're really struggling right now. Thank you for sharing that.",
                    "Let's work together to find some ways to help you through this.",
                    "What kind of support would feel most helpful to you right now?",
                    "Remember that it's okay to ask for help when you need it."
                ],
                'follow_up': 'routine_followup'
            }
        }

    def _initialize_risk_factors(self):
        """Initialize factors for risk assessment"""
        return {
            'emotional_indicators': ['hopelessness', 'helplessness', 'worthlessness', 'guilt'],
            'behavioral_indicators': ['isolation', 'substance_use', 'risk_taking', 'agitation'],
            'verbal_indicators': ['goodbye_statements', 'burden_statements', 'no_future_statements'],
            'contextual_indicators': ['recent_loss', 'trauma', 'major_stress', 'health_issues']
        }

    def _initialize_emergency_contacts(self):
        """Initialize emergency contact information"""
        return {
            'national_suicide_prevention': {
                'phone': '1-800-273-8255',
                'website': 'https://suicidepreventionlifeline.org',
                'text': 'Text HOME to 741741'
            },
            'crisis_text_line': {
                'phone': 'Text HOME to 741741',
                'website': 'https://www.crisistextline.org'
            },
            'emergency_services': {
                'phone': '911',
                'note': 'For immediate life-threatening emergencies'
            }
        }

    def assess_risk_level(self, user_input, emotional_analysis, conversation_context):
        """Assess risk level based on multiple factors"""
        risk_score = 0.0
        risk_factors = []

        # 1. Text-based risk assessment
        text_risk = self._assess_text_risk(user_input)
        risk_score += text_risk['score']
        risk_factors.extend(text_risk['factors'])

        # 2. Emotional risk assessment
        emotional_risk = self._assess_emotional_risk(emotional_analysis)
        risk_score += emotional_risk['score']
        risk_factors.extend(emotional_risk['factors'])

        # 3. Contextual risk assessment
        contextual_risk = self._assess_contextual_risk(conversation_context)
        risk_score += contextual_risk['score']
        risk_factors.extend(contextual_risk['factors'])

        # Determine overall risk level
        risk_level = self._classify_risk_level(risk_score)

        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'assessment_timestamp': datetime.now().isoformat(),
            'requires_immediate_action': risk_level in ['crisis_immediate', 'high_risk']
        }

    def _assess_text_risk(self, user_input):
        """Assess risk based on text content"""
        risk_score = 0.0
        factors = []

        user_input_lower = user_input.lower()

        # Check for high-risk keywords
        for keyword in self.crisis_keywords['high_risk']:
            if keyword in user_input_lower:
                risk_score += 3.0
                factors.append(f"High-risk keyword: {keyword}")
                break

        # Check for moderate-risk keywords
        for keyword in self.crisis_keywords['moderate_risk']:
            if keyword in user_input_lower:
                risk_score += 1.5
                factors.append(f"Moderate-risk keyword: {keyword}")

        # Check for low-risk keywords
        for keyword in self.crisis_keywords['low_risk']:
            if keyword in user_input_lower:
                risk_score += 0.5
                factors.append(f"Emotional distress keyword: {keyword}")

        # Assess sentence structure and intensity
        if '!' in user_input:
            risk_score += 0.3
            factors.append("Exclamatory language indicating high intensity")

        if user_input_lower.count('i ') > 5:
            risk_score += 0.2
            factors.append("High self-reference indicating personal distress")

        return {'score': risk_score, 'factors': factors}

    def _assess_emotional_risk(self, emotional_analysis):
        """Assess risk based on emotional analysis"""
        risk_score = 0.0
        factors = []

        emotion = emotional_analysis.get('emotion', 'neutral')
        intensity = emotional_analysis.get('intensity', 1.0)
        valence = emotional_analysis.get('valence', 0.0)

        # High-risk emotions
        if emotion in ['angry', 'fear'] and intensity > 1.8:
            risk_score += 2.0
            factors.append(f"High-intensity {emotion}")

        # Moderate-risk emotions
        elif emotion in ['sad', 'fear'] and intensity > 1.5:
            risk_score += 1.0
            factors.append(f"Moderate-intensity {emotion}")

        # Valence-based risk
        if valence < -0.7:
            risk_score += 1.0
            factors.append("Extremely negative emotional valence")

        # Intensity-based risk
        if intensity > 2.0:
            risk_score += 1.5
            factors.append("Very high emotional intensity")

        return {'score': risk_score, 'factors': factors}

    def _assess_contextual_risk(self, conversation_context):
        """Assess risk based on conversation context"""
        risk_score = 0.0
        factors = []

        # Engagement level risk
        engagement = conversation_context.get('engagement_level', 'medium')
        if engagement == 'low':
            risk_score += 0.5
            factors.append("Low engagement may indicate withdrawal")

        # Conversation length risk (very long sessions might indicate crisis)
        session_duration = conversation_context.get('session_duration', 0)
        if session_duration > 1800:  # 30 minutes
            risk_score += 0.5
            factors.append("Extended conversation session")

        # Emotional pattern risk
        emotional_flow = conversation_context.get('emotional_flow', {})
        if emotional_flow.get('emotional_variability', 0) < 0.1:
            risk_score += 0.3
            factors.append("Stuck in same emotional state")

        return {'score': risk_score, 'factors': factors}

    def _classify_risk_level(self, risk_score):
        """Classify overall risk level based on score"""
        if risk_score >= 4.0:
            return 'crisis_immediate'
        elif risk_score >= 2.5:
            return 'high_risk'
        elif risk_score >= 1.0:
            return 'moderate_risk'
        else:
            return 'low_risk'

    def generate_intervention_response(self, risk_assessment, user_context):
        """Generate appropriate intervention response based on risk level"""
        risk_level = risk_assessment['risk_level']
        protocol = self.intervention_protocols.get(risk_level, self.intervention_protocols['low_risk'])

        response = {
            'intervention_level': risk_level,
            'priority': protocol['priority'],
            'immediate_actions': protocol['actions'],
            'emergency_contacts': self._get_relevant_contacts(risk_level),
            'safety_planning': self._generate_safety_plan(risk_level, user_context),
            'follow_up_protocol': protocol['follow_up'],
            'risk_factors': risk_assessment['risk_factors'],
            'timestamp': datetime.now().isoformat()
        }

        return response

    def _get_relevant_contacts(self, risk_level):
        """Get relevant emergency contacts based on risk level"""
        contacts = {}

        if risk_level in ['crisis_immediate', 'high_risk']:
            contacts['immediate_help'] = self.emergency_contacts['national_suicide_prevention']
            contacts['emergency'] = self.emergency_contacts['emergency_services']
        elif risk_level == 'moderate_risk':
            contacts['crisis_support'] = self.emergency_contacts['crisis_text_line']
            contacts['therapeutic_support'] = "Consider contacting a mental health professional"

        return contacts

    def _generate_safety_plan(self, risk_level, user_context):
        """Generate safety planning suggestions"""
        safety_plan = {}

        if risk_level in ['crisis_immediate', 'high_risk']:
            safety_plan['immediate_steps'] = [
                "Remove access to any means of self-harm",
                "Contact emergency services or crisis line",
                "Go to a safe location where you're not alone",
                "Reach out to someone you trust immediately"
            ]

        if risk_level in ['high_risk', 'moderate_risk']:
            safety_plan['coping_strategies'] = [
                "Use grounding techniques (5-4-3-2-1 method)",
                "Contact your support network",
                "Engage in distracting activities",
                "Practice deep breathing exercises"
            ]

            safety_plan['support_network'] = [
                "Identify 2-3 people you can contact in crisis",
                "Save crisis numbers in your phone",
                "Plan regular check-ins with supportive people"
            ]

        return safety_plan

    def monitor_escalation(self, current_risk, previous_risks, time_window_minutes=60):
        """Monitor for risk escalation over time"""
        if len(previous_risks) < 2:
            return {'escalation_detected': False, 'trend': 'insufficient_data'}

        # Get recent risk assessments within time window
        recent_cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_risks = [r for r in previous_risks
                        if datetime.fromisoformat(r['timestamp']) > recent_cutoff]

        if len(recent_risks) < 2:
            return {'escalation_detected': False, 'trend': 'insufficient_data'}

        # Calculate risk trend
        risk_scores = [r['risk_score'] for r in recent_risks] + [current_risk['risk_score']]

        if len(risk_scores) >= 3:
            # Simple trend analysis
            recent_trend = risk_scores[-3:]
            if recent_trend[2] > recent_trend[1] > recent_trend[0]:
                return {
                    'escalation_detected': True,
                    'trend': 'increasing',
                    'escalation_rate': (recent_trend[2] - recent_trend[0]) / 2,
                    'recommendation': 'Consider increasing intervention level'
                }

        return {'escalation_detected': False, 'trend': 'stable'}

    def generate_crisis_report(self, risk_assessment, intervention_response, user_context):
        """Generate a comprehensive crisis report"""
        return {
            'crisis_report': {
                'assessment_time': risk_assessment['assessment_timestamp'],
                'risk_level': risk_assessment['risk_level'],
                'risk_score': risk_assessment['risk_score'],
                'primary_risk_factors': risk_assessment['risk_factors'][:5],
                'intervention_provided': intervention_response['immediate_actions'],
                'emergency_contacts_shared': intervention_response['emergency_contacts'],
                'safety_planning': intervention_response['safety_planning'],
                'user_context': {
                    'current_emotion': user_context.get('current_emotion', 'unknown'),
                    'engagement_level': user_context.get('engagement_level', 'unknown'),
                    'session_duration': user_context.get('session_duration', 0)
                },
                'follow_up_requirements': intervention_response['follow_up_protocol'],
                'report_generated': datetime.now().isoformat()
            },
            'action_items': [
                "Document the intervention provided",
                "Schedule follow-up check-in",
                "Review for potential professional consultation",
                "Update safety planning as needed"
            ]
        }

    def validate_intervention_effectiveness(self, pre_intervention_risk, post_intervention_risk):
        """Validate effectiveness of intervention"""
        if pre_intervention_risk['risk_level'] == post_intervention_risk['risk_level']:
            effectiveness = 'no_change'
        elif self._compare_risk_levels(pre_intervention_risk['risk_level'], post_intervention_risk['risk_level']) < 0:
            effectiveness = 'improved'
        else:
            effectiveness = 'worsened'

        return {
            'effectiveness': effectiveness,
            'risk_change': f"{pre_intervention_risk['risk_level']} → {post_intervention_risk['risk_level']}",
            'score_change': post_intervention_risk['risk_score'] - pre_intervention_risk['risk_score'],
            'recommendation': self._generate_effectiveness_recommendation(effectiveness)
        }

    def _compare_risk_levels(self, level1, level2):
        """Compare two risk levels"""
        risk_order = ['low_risk', 'moderate_risk', 'high_risk', 'crisis_immediate']
        try:
            return risk_order.index(level1) - risk_order.index(level2)
        except ValueError:
            return 0

    def _generate_effectiveness_recommendation(self, effectiveness):
        """Generate recommendations based on intervention effectiveness"""
        if effectiveness == 'improved':
            return "Continue current support strategies"
        elif effectiveness == 'no_change':
            return "Consider escalating intervention approach"
        else:
            return "Immediate escalation to professional intervention required"