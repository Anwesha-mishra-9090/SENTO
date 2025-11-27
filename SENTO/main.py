#!/usr/bin/env python3
"""
SENTIO - Emotional AI Life Coach v1.0
Main entry point for the application
"""

import os
import sys
import logging
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.emotion_orchestrator import EmotionOrchestrator
from voice_analysis.feature_extraction import FeatureExtractor
from text_analysis.sentiment_analyzer import SentimentAnalyzer
from analytics_engine.timeline_generator import TimelineGenerator
from ai_coach.coaching_engine import CoachingEngine
from data_layer.time_series_db import TimeSeriesDB
from ml_models.model_manager import EmotionalModelManager
from ml_models.model_serving import ModelServer
from interfaces.api_gateway import APIGateway
from interfaces.web_dashboard import WebDashboard
from interfaces.voice_interface import VoiceInterface, TextToSpeech
from interfaces.mobile_integration import MobileIntegration, PushNotificationHandler
from model_initializer import check_and_initialize_models


class SENTIO:
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.components = {}

    def setup_logging(self):
        """Setup application logging with Unicode support"""
        # Remove existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler('sentio.log', encoding='utf-8')
        file_handler.setFormatter(formatter)

        # Console handler with proper encoding handling
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler]
        )

    def initialize_components(self):
        """Initialize all SENTIO components"""
        self.logger.info("[TOOL] Initializing SENTIO components...")

        try:
            # Check and create missing models first
            check_and_initialize_models()

            # 1. Data Layer
            self.logger.info("[CHART] Initializing data layer...")
            self.components['data_manager'] = TimeSeriesDB()

            # 2. ML Models
            self.logger.info("[ROBOT] Initializing ML models...")
            self.components['model_manager'] = EmotionalModelManager()
            self.components['model_server'] = ModelServer(self.components['model_manager'])

            # 3. Core Processing
            self.logger.info("[TARGET] Initializing core processors...")
            self.components['feature_extractor'] = FeatureExtractor()
            self.components['sentiment_analyzer'] = SentimentAnalyzer()
            self.components['timeline_generator'] = TimelineGenerator()

            # 4. Emotion Orchestrator
            self.logger.info("[BRAIN] Initializing emotion orchestrator...")
            self.components['emotion_orchestrator'] = EmotionOrchestrator()

            # 5. AI Coach
            self.logger.info("[SPARKLE] Initializing AI coach...")
            self.components['coaching_engine'] = CoachingEngine()

            # 6. Interfaces
            self.logger.info("[GLOBE] Initializing interfaces...")
            self.components['voice_interface'] = VoiceInterface(
                self.components['emotion_orchestrator'],
                self.components['feature_extractor'],
                TextToSpeech()
            )

            self.components['push_handler'] = PushNotificationHandler()
            self.components['mobile_integration'] = MobileIntegration(
                None,  # Will be set after API gateway
                self.components['push_handler']
            )

            self.logger.info("[OK] All components initialized successfully!")

        except Exception as e:
            self.logger.error(f"[ERROR] Component initialization failed: {e}")
            raise

    def start_services(self, start_api=True, start_dashboard=True, start_voice=False):
        """Start SENTIO services"""
        self.logger.info("[ROCKET] Starting SENTIO services...")

        try:
            services = []

            # Start API Gateway
            if start_api:
                self.logger.info("[PLUG] Starting API Gateway...")
                self.components['api_gateway'] = APIGateway(
                    self.components['emotion_orchestrator'],
                    self.components['coaching_engine'],
                    self.components['model_server']
                )
                # Update mobile integration with API gateway
                self.components['mobile_integration'].api_gateway = self.components['api_gateway']

                # Start API in background thread
                import threading
                api_thread = threading.Thread(
                    target=self.components['api_gateway'].start_server,
                    kwargs={'host': '0.0.0.0', 'port': 5000, 'debug': False}
                )
                api_thread.daemon = True
                api_thread.start()
                services.append(('API Gateway', 'http://localhost:5000'))
                print("Starting SENTIO API Gateway on 0.0.0.0:5000")

            # Start Web Dashboard - FIXED: Use 127.0.0.1 instead of 0.0.0.0
            if start_dashboard:
                self.logger.info("[CHART_UP] Starting Web Dashboard...")
                dashboard = WebDashboard(
                    self.components['emotion_orchestrator'],
                    None,  # Analytics engine would go here
                    self.components['data_manager']
                )

                # FIXED: Use 127.0.0.1 for web dashboard
                dashboard_thread = threading.Thread(
                    target=dashboard.start_dashboard,
                    kwargs={'host': '127.0.0.1', 'port': 8000, 'debug': False}  # ← FIXED HERE
                )
                dashboard_thread.daemon = True
                dashboard_thread.start()
                services.append(('Web Dashboard', 'http://localhost:8000'))

            # Start Voice Interface (optional)
            if start_voice:
                self.logger.info("[MICROPHONE] Starting Voice Interface...")
                # Voice interface runs in main thread for user interaction
                voice_thread = threading.Thread(
                    target=self.components['voice_interface'].start_listening
                )
                voice_thread.daemon = True
                voice_thread.start()
                services.append(('Voice Interface', 'Active - Say "Hey SENTIO"'))

            # Display service status
            self.logger.info("\n" + "=" * 50)
            self.logger.info("[PARTY] SENTIO Services Started Successfully!")
            for service, url in services:
                self.logger.info(f"   {service}: {url}")
            self.logger.info("=" * 50)
            self.logger.info("[NOTE] Logs are being written to: sentio.log")
            self.logger.info("[STOP] Press Ctrl+C to stop all services")

            return True

        except Exception as e:
            self.logger.error(f"[ERROR] Service startup failed: {e}")
            return False

    def get_system_status(self):
        """Get system status report"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'components_initialized': list(self.components.keys()),
            'services_running': [],
            'system_health': 'healthy'
        }

        # Check component health
        for name, component in self.components.items():
            try:
                if hasattr(component, 'get_status'):
                    status[f'{name}_status'] = component.get_status()
                else:
                    status[f'{name}_status'] = 'active'
            except:
                status[f'{name}_status'] = 'error'
                status['system_health'] = 'degraded'

        return status

    def shutdown(self):
        """Graceful shutdown of SENTIO"""
        self.logger.info("[STOP] Shutting down SENTIO...")

        # Stop voice interface if running
        if 'voice_interface' in self.components:
            self.components['voice_interface'].stop_listening()

        self.logger.info("[WAVE] SENTIO shutdown complete")


def main():
    """Main application entry point"""
    sentio = SENTIO()

    try:
        # Initialize system
        sentio.initialize_components()

        # Start services
        success = sentio.start_services(
            start_api=True,
            start_dashboard=True,
            start_voice=False  # Set to True to enable voice interface
        )

        if success:
            # Keep the main thread alive
            try:
                while True:
                    # Display status every 5 minutes
                    import time
                    time.sleep(300)
                    status = sentio.get_system_status()
                    sentio.logger.info(f"[CHART] System Status: {status['system_health']}")

            except KeyboardInterrupt:
                sentio.logger.info("\n[STOP] Received interrupt signal...")

    except Exception as e:
        sentio.logger.error(f"[ERROR] SENTIO failed to start: {e}")
        return 1
    finally:
        sentio.shutdown()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


# Alternative simple starter for development
def start_simple():
    """Simple starter for development and testing"""
    print("[ROCKET] Starting SENTIO in simple mode...")

    # Initialize just the core components
    data_manager = TimeSeriesDB()
    emotion_orchestrator = EmotionOrchestrator()

    # Start web dashboard only - FIXED: Use 127.0.0.1
    dashboard = WebDashboard(emotion_orchestrator, None, data_manager)
    dashboard.start_dashboard(host='127.0.0.1', port=8000, debug=True)  # ← FIXED HERE

# Uncomment for simple development mode
# if __name__ == "__main__":
#     start_simple()