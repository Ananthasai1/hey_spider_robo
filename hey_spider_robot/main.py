#!/usr/bin/env python3
"""
Hey Spider Robot with OLED Display - Main Application (Fixed Version)
Complete AI-powered spider robot system with visual feedback and modern OpenAI API integration
"""

import sys
import time
import signal
import json
import traceback
import os
from config.settings import settings

class HeySpiderRobot:
    def __init__(self):
        print("=" * 60)
        print("🕷️  HEY SPIDER ROBOT - INITIALIZATION")
        print("=" * 60)
        print(f"Python version: {sys.version}")
        print(f"Working directory: {os.getcwd()}")
        print(f"OpenAI API Key configured: {'Yes' if settings.OPENAI_API_KEY else 'No (set OPENAI_API_KEY environment variable)'}")
        
        # Initialize components with error handling
        self.oled = None
        self.spider = None
        self.vision = None
        self.ai = None
        self.voice = None
        self.web = None
        self.running = False
        
        # Initialize each component safely
        self._init_oled()
        self._init_spider()
        self._init_vision()
        self._init_ai()
        self._init_voice()
        self._init_web()
        
        print("=" * 60)
        print("🕷️  HEY SPIDER ROBOT - INITIALIZATION COMPLETE")
        print("=" * 60)
        
    def _init_oled(self):
        """Initialize OLED display with error handling"""
        try:
            print("Initializing OLED display...")
            from src.oled_display import OLEDDisplay
            self.oled = OLEDDisplay()
            if self.oled.display:
                self.oled.show_startup_message()
                self.oled.start()
                print("✅ OLED display initialized successfully")
            else:
                print("⚠️  OLED display initialized in mock mode")
        except Exception as e:
            print(f"❌ OLED display initialization failed: {e}")
            traceback.print_exc()
            self.oled = None
            
    def _init_spider(self):
        """Initialize spider controller with error handling"""
        try:
            print("Initializing spider controller...")
            from src.spider_controller import SpiderController
            self.spider = SpiderController(self.oled)
            print("✅ Spider controller initialized successfully")
        except Exception as e:
            print(f"❌ Spider controller initialization failed: {e}")
            traceback.print_exc()
            self.spider = None
            
    def _init_vision(self):
        """Initialize visual monitoring with error handling"""
        try:
            print("Initializing visual monitoring...")
            from src.visual_monitor import VisualMonitor
            self.vision = VisualMonitor(self.oled)
            print("✅ Visual monitoring initialized successfully")
        except Exception as e:
            print(f"❌ Visual monitoring initialization failed: {e}")
            traceback.print_exc()
            self.vision = None
            
    def _init_ai(self):
        """Initialize AI thinking with error handling"""
        try:
            print("Initializing AI thinking...")
            from src.ai_thinking import AIThinking
            self.ai = AIThinking(self.spider, self.vision, self.oled)
            print("✅ AI thinking initialized successfully")
        except Exception as e:
            print(f"❌ AI thinking initialization failed: {e}")
            traceback.print_exc()
            self.ai = None
            
    def _init_voice(self):
        """Initialize voice activation with error handling"""
        try:
            print("Initializing voice activation...")
            from src.voice_activation import VoiceActivation
            self.voice = VoiceActivation(self.handle_voice_command, self.oled)
            print("✅ Voice activation initialized successfully")
        except Exception as e:
            print(f"❌ Voice activation initialization failed: {e}")
            traceback.print_exc()
            self.voice = None
            
    def _init_web(self):
        """Initialize web interface with error handling"""
        try:
            print("Initializing web interface...")
            from src.web_interface import WebInterface
            self.web = WebInterface(self.spider, self.vision, self.ai, self.oled)
            print("✅ Web interface initialized successfully")
        except Exception as e:
            print(f"❌ Web interface initialization failed: {e}")
            traceback.print_exc()
            # Try to create a minimal web interface
            try:
                print("Attempting to create minimal web interface...")
                from flask import Flask, jsonify
                
                class MinimalWebInterface:
                    def __init__(self):
                        self.app = Flask(__name__)
                        self.setup_minimal_routes()
                        
                    def setup_minimal_routes(self):
                        @self.app.route('/')
                        def index():
                            return """
                            <html>
                                <head><title>Hey Spider Robot - Error</title></head>
                                <body>
                                    <h1>🕷️ Hey Spider Robot</h1>
                                    <p>System is running in minimal mode due to initialization errors.</p>
                                    <p>Check the console logs for more details.</p>
                                    <h2>Status:</h2>
                                    <ul>
                                        <li>Spider Controller: {}</li>
                                        <li>Vision System: {}</li>
                                        <li>AI System: {}</li>
                                        <li>Voice System: {}</li>
                                    </ul>
                                </body>
                            </html>
                            """.format(
                                "✅ OK" if self.spider else "❌ Error",
                                "✅ OK" if self.vision else "❌ Error", 
                                "✅ OK" if self.ai else "❌ Error",
                                "✅ OK" if self.voice else "❌ Error"
                            )
                            
                        @self.app.route('/health')
                        def health():
                            return jsonify({
                                'status': 'minimal_mode',
                                'components': {
                                    'spider': bool(self.spider),
                                    'vision': bool(self.vision),
                                    'ai': bool(self.ai),
                                    'voice': bool(self.voice)
                                }
                            })
                            
                    def run(self, host='0.0.0.0', port=5000, debug=False):
                        print(f"Starting minimal web interface on http://{host}:{port}")
                        self.app.run(host=host, port=port, debug=debug)
                
                # Store references for minimal interface
                minimal_web = MinimalWebInterface()
                minimal_web.spider = self.spider
                minimal_web.vision = self.vision
                minimal_web.ai = self.ai
                minimal_web.voice = self.voice
                
                self.web = minimal_web
                print("✅ Minimal web interface created successfully")
                
            except Exception as e2:
                print(f"❌ Even minimal web interface creation failed: {e2}")
                self.web = None
                
    def handle_voice_command(self, command: str):
        """Handle voice commands with error handling"""
        print(f"Processing voice command: {command}")
        
        try:
            if self.oled:
                self.oled.update_command(command)
        except Exception as e:
            print(f"OLED update error: {e}")
            
        try:
            # Basic movement commands
            if any(word in command for word in ['forward', 'walk', 'move']):
                if self.spider:
                    self.spider.walk_forward()
                else:
                    print("Spider controller not available")
                    
            elif 'left' in command:
                if self.spider:
                    self.spider.turn_left()
                else:
                    print("Spider controller not available")
                    
            elif 'right' in command:
                if self.spider:
                    self.spider.turn_right()
                else:
                    print("Spider controller not available")
                    
            elif 'dance' in command:
                if self.spider:
                    self.spider.dance()
                else:
                    print("Spider controller not available")
                    
            elif 'wave' in command:
                if self.spider:
                    self.spider.wave()
                else:
                    print("Spider controller not available")
                    
            elif any(word in command for word in ['photo', 'picture', 'capture']):
                if self.vision:
                    filename = self.vision.capture_photo()
                    print(f"Photo captured: {filename}" if filename else "Photo capture failed")
                else:
                    print("Vision system not available")
                    
            elif 'stop' in command:
                print("Stopping robot...")
                if self.oled:
                    self.oled.update_mode("STOPPED")
                    
            else:
                # Use AI to process unknown commands
                if self.ai:
                    try:
                        ai_response = self.ai.process_command(command)
                        parsed_response = json.loads(ai_response)
                        action = parsed_response.get('action', 'unknown')
                        response_message = parsed_response.get('response', 'Command processed')
                        
                        print(f"AI Response: {response_message}")
                        
                        # Execute AI-determined action
                        if action == 'walk_forward' and self.spider:
                            self.spider.walk_forward()
                        elif action == 'turn_left' and self.spider:
                            self.spider.turn_left()
                        elif action == 'turn_right' and self.spider:
                            self.spider.turn_right()
                        elif action == 'dance' and self.spider:
                            self.spider.dance()
                        elif action == 'wave' and self.spider:
                            self.spider.wave()
                        elif action == 'take_photo' and self.vision:
                            self.vision.capture_photo()
                        elif action == 'unknown':
                            print(f"Unknown command: {command}")
                            
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"AI response parsing error: {e}")
                        print("Could not understand command")
                else:
                    print("AI system not available for command processing")
                
        except Exception as e:
            print(f"Command execution error: {e}")
            traceback.print_exc()
            if self.oled:
                try:
                    self.oled.update_mode("ERROR")
                    time.sleep(2)
                except:
                    pass
                
        finally:
            if self.oled:
                try:
                    self.oled.update_mode("LISTENING")
                except:
                    pass
                
    def start(self):
        """Start all robot systems"""
        print("\n" + "="*60)
        print("🕷️  STARTING HEY SPIDER ROBOT SYSTEMS")
        print("="*60)
        
        self.running = True
        
        # Start subsystems that are available
        available_systems = []
        
        if self.vision:
            try:
                print("Starting visual monitoring...")
                self.vision.start_monitoring()
                available_systems.append("Visual Monitoring")
            except Exception as e:
                print(f"Failed to start visual monitoring: {e}")
        
        if self.ai:
            try:
                print("Starting AI thinking...")
                self.ai.start_thinking()
                available_systems.append("AI Thinking")
            except Exception as e:
                print(f"Failed to start AI thinking: {e}")
        
        if self.voice:
            try:
                print("Starting voice activation...")
                self.voice.start_listening()
                available_systems.append("Voice Activation")
            except Exception as e:
                print(f"Failed to start voice activation: {e}")
        
        if self.oled:
            try:
                self.oled.update_mode("ACTIVE")
            except Exception as e:
                print(f"OLED update error: {e}")
                
        print("\n" + "="*60)
        print("🕷️  HEY SPIDER ROBOT IS NOW ACTIVE!")
        print("="*60)
        print("✅ Available Systems:", ", ".join(available_systems) if available_systems else "None")
        
        if self.spider:
            print("🤖 Spider Controller: Ready for commands")
        else:
            print("❌ Spider Controller: Not available")
            
        print(f"🌐 Web Interface: http://localhost:{settings.WEB_PORT}")
        
        if available_systems:
            print("📢 Voice Commands:")
            print("   - Say 'Hey Spider' followed by a command")
            print("   - Available commands: walk forward, turn left/right, dance, wave, take photo")
        
        print("="*60)
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        # Start web interface (this will block)
        if self.web:
            try:
                self.web.run(host='0.0.0.0', port=settings.WEB_PORT, debug=False)
            except KeyboardInterrupt:
                self.stop()
            except Exception as e:
                print(f"Web interface error: {e}")
                traceback.print_exc()
                self.stop()
        else:
            print("❌ No web interface available. System will run in console mode.")
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()
            
    def stop(self):
        """Stop all robot systems"""
        print("\n" + "="*50)
        print("🛑 STOPPING HEY SPIDER ROBOT")
        print("="*50)
        
        self.running = False
        
        if self.oled:
            try:
                self.oled.update_mode("SHUTDOWN")
            except Exception as e:
                print(f"OLED shutdown error: {e}")
        
        # Stop subsystems safely
        if self.voice:
            try:
                print("Stopping voice activation...")
                self.voice.stop_listening()
            except Exception as e:
                print(f"Voice shutdown error: {e}")
        
        if self.vision:
            try:
                print("Stopping visual monitoring...")
                self.vision.stop_monitoring()
            except Exception as e:
                print(f"Vision shutdown error: {e}")
        
        if self.ai:
            try:
                print("Stopping AI thinking...")
                self.ai.stop_thinking()
            except Exception as e:
                print(f"AI shutdown error: {e}")
        
        if self.oled:
            try:
                print("Stopping OLED display...")
                self.oled.stop()
            except Exception as e:
                print(f"OLED stop error: {e}")
        
        # Cleanup hardware
        if self.spider:
            try:
                print("Cleaning up hardware...")
                self.spider.cleanup()
            except Exception as e:
                print(f"Hardware cleanup error: {e}")
        
        if self.vision:
            try:
                self.vision.cleanup()
            except Exception as e:
                print(f"Vision cleanup error: {e}")
        
        print("="*50)
        print("✅ Hey Spider Robot stopped safely.")
        print("="*50)
        
    def signal_handler(self, signum, frame):
        """Handle system signals"""
        print(f"\n🛑 Received signal {signum}")
        self.stop()
        sys.exit(0)

def main():
    """Main application entry point with comprehensive error handling"""
    print("🕷️ HEY SPIDER ROBOT - Starting up...")
    print("Hardware compatibility: Raspberry Pi with servo/camera/sensor support")
    print("Software compatibility: Can run with or without hardware (mock mode)")
    
    try:
        # Check Python version
        if sys.version_info < (3, 7):
            print("❌ Python 3.7 or higher is required")
            sys.exit(1)
            
        # Check if we're in a virtual environment (recommended)
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("✅ Running in virtual environment")
        else:
            print("⚠️  Not running in virtual environment (recommended)")
            
        # Initialize robot
        robot = HeySpiderRobot()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, robot.signal_handler)
        signal.signal(signal.SIGTERM, robot.signal_handler)
        
        # Start the robot
        robot.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies with: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()