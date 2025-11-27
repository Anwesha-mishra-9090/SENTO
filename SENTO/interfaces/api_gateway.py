from flask import Flask, request, jsonify, Response
import json
import threading
from datetime import datetime
import traceback


class APIGateway:
    def __init__(self, emotion_orchestrator, coaching_engine, model_server):
        self.app = Flask(__name__)
        self.emotion_orchestrator = emotion_orchestrator
        self.coaching_engine = coaching_engine
        self.model_server = model_server
        self.setup_routes()

        # API metrics
        self.request_count = 0
        self.error_count = 0
        self.response_times = []

    def setup_routes(self):
        """Setup API routes"""

        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            start_time = datetime.now()

            try:
                health_data = {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'services': {
                        'emotion_orchestrator': 'active',
                        'coaching_engine': 'active',
                        'model_server': 'active'
                    }
                }

                # Check model server health
                model_health = self.model_server.get_serving_metrics()['system_health']
                if model_health['status'] != 'healthy':
                    health_data['status'] = 'degraded'
                    health_data['issues'] = model_health['issues']

                response_time = (datetime.now() - start_time).total_seconds()
                self._record_metrics('health', response_time, False)

                return jsonify(health_data)

            except Exception as e:
                self._record_metrics('health', 0, True)
                return jsonify({'status': 'error', 'error': str(e)}), 500

        @self.app.route('/api/analyze/voice', methods=['POST'])
        def analyze_voice():
            """Analyze emotion from voice"""
            start_time = datetime.now()

            try:
                data = request.get_json()

                if not data or 'audio_features' not in data:
                    return jsonify({'error': 'Missing audio_features'}), 400

                # Analyze emotion
                result = self.emotion_orchestrator.analyze_emotion(
                    data['audio_features'],
                    input_type='voice'
                )

                response_time = (datetime.now() - start_time).total_seconds()
                self._record_metrics('analyze_voice', response_time, False)

                return jsonify(result)

            except Exception as e:
                self._record_metrics('analyze_voice', 0, True)
                return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

        @self.app.route('/api/analyze/text', methods=['POST'])
        def analyze_text():
            """Analyze emotion from text"""
            start_time = datetime.now()

            try:
                data = request.get_json()

                if not data or 'text' not in data:
                    return jsonify({'error': 'Missing text'}), 400

                # Analyze emotion
                result = self.emotion_orchestrator.analyze_emotion(
                    data['text'],
                    input_type='text'
                )

                response_time = (datetime.now() - start_time).total_seconds()
                self._record_metrics('analyze_text', response_time, False)

                return jsonify(result)

            except Exception as e:
                self._record_metrics('analyze_text', 0, True)
                return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

        @self.app.route('/api/coach/response', methods=['POST'])
        def get_coaching_response():
            """Get coaching response based on emotional state"""
            start_time = datetime.now()

            try:
                data = request.get_json()

                required_fields = ['emotion', 'intensity']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Missing {field}'}), 400

                # Get coaching response
                result = self.coaching_engine.generate_coaching_response(
                    data['emotion'],
                    data['intensity'],
                    data.get('context', {})
                )

                response_time = (datetime.now() - start_time).total_seconds()
                self._record_metrics('coaching_response', response_time, False)

                return jsonify(result)

            except Exception as e:
                self._record_metrics('coaching_response', 0, True)
                return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

        @self.app.route('/api/predict/stress', methods=['POST'])
        def predict_stress():
            """Predict stress levels"""
            start_time = datetime.now()

            try:
                data = request.get_json()

                required_fields = ['temporal_features', 'emotional_context']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Missing {field}'}), 400

                # Predict stress
                result = self.model_server.predict_stress_level(
                    data['temporal_features'],
                    data['emotional_context']
                )

                response_time = (datetime.now() - start_time).total_seconds()
                self._record_metrics('predict_stress', response_time, False)

                return jsonify(result)

            except Exception as e:
                self._record_metrics('predict_stress', 0, True)
                return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

        @self.app.route('/api/insights/patterns', methods=['GET'])
        def get_emotional_patterns():
            """Get emotional pattern insights"""
            start_time = datetime.now()

            try:
                time_period = request.args.get('time_period', '7d')

                # Get emotional insights
                insights = self.emotion_orchestrator.get_emotional_insights(time_period)

                response_time = (datetime.now() - start_time).total_seconds()
                self._record_metrics('emotional_patterns', response_time, False)

                return jsonify(insights)

            except Exception as e:
                self._record_metrics('emotional_patterns', 0, True)
                return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

        @self.app.route('/api/metrics', methods=['GET'])
        def get_api_metrics():
            """Get API performance metrics"""
            try:
                metrics = {
                    'total_requests': self.request_count,
                    'error_count': self.error_count,
                    'error_rate': self.error_count / max(1, self.request_count),
                    'average_response_time': sum(self.response_times) / len(
                        self.response_times) if self.response_times else 0,
                    'timestamp': datetime.now().isoformat()
                }

                return jsonify(metrics)

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/models/status', methods=['GET'])
        def get_model_status():
            """Get model serving status"""
            try:
                status = self.model_server.get_serving_metrics()
                return jsonify(status)

            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def _record_metrics(self, endpoint, response_time, is_error):
        """Record API metrics"""
        self.request_count += 1

        if is_error:
            self.error_count += 1
        else:
            self.response_times.append(response_time)

            # Keep only recent response times
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]

    def start_server(self, host='0.0.0.0', port=5000, debug=False):
        """Start the API server"""
        print(f"Starting SENTIO API Gateway on {host}:{port}")

        # Warm up models before starting
        print("Warming up models...")
        self.model_server.warmup_models()

        # Start Flask server
        self.app.run(host=host, port=port, debug=debug, threaded=True)

    def get_flask_app(self):
        """Get the Flask app instance for external serving"""
        return self.app


# Example usage class for easy integration
class SENTIOAPI:
    def __init__(self, emotion_orchestrator, coaching_engine, model_server):
        self.gateway = APIGateway(emotion_orchestrator, coaching_engine, model_server)

    def start(self, host='0.0.0.0', port=5000):
        """Start the SENTIO API"""
        self.gateway.start_server(host, port)

    def get_app(self):
        """Get Flask app for deployment"""
        return self.gateway.get_flask_app()