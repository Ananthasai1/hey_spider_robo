import base64
import json
import threading
import time
import traceback
from typing import Optional

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
from src.oled_display import OLEDDisplay

# Inline HTML template to avoid template file issues
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hey Spider Robot Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            background: linear-gradient(45deg, #fff, #ffd700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .header p {
            margin-top: 10px;
            opacity: 0.9;
        }
        
        .connection-status {
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            margin: 10px 0;
            font-weight: bold;
        }
        
        .connected {
            background: rgba(76, 175, 80, 0.3);
            color: #4CAF50;
        }
        
        .disconnected {
            background: rgba(244, 67, 54, 0.3);
            color: #f44336;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr;
            grid-template-rows: auto auto auto;
            gap: 20px;
        }
        
        .panel {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        
        .panel h3 {
            margin-bottom: 15px;
            color: #fff;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            padding-bottom: 8px;
        }
        
        .video-panel {
            grid-row: span 2;
        }
        
        #videoFeed {
            width: 100%;
            max-width: 640px;
            border-radius: 10px;
            display: block;
            margin: 0 auto;
            background: #000;
            min-height: 240px;
        }
        
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 20px;
        }
        
        .btn {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
        }
        
        .btn:active {
            transform: translateY(-1px);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .status-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 10px 0;
            padding: 12px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }
        
        .status-value {
            font-weight: bold;
            color: #ffd700;
        }
        
        .detection-list {
            max-height: 200px;
            overflow-y: auto;
            scrollbar-width: thin;
        }
        
        .detection-item {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 8px 12px;
            background: rgba(0,0,0,0.2);
            border-radius: 6px;
        }
        
        .confidence {
            color: #4CAF50;
            font-weight: bold;
        }
        
        .ai-thought {
            font-style: italic;
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #9C27B0;
            min-height: 60px;
        }
        
        .oled-display {
            background: #000;
            color: #0ff;
            font-family: 'Courier New', monospace;
            padding: 15px;
            border-radius: 8px;
            font-size: 12px;
            line-height: 1.4;
            height: 150px;
            overflow: hidden;
            border: 2px solid #0ff;
            box-shadow: inset 0 0 10px rgba(0, 255, 255, 0.3);
        }
        
        .voice-commands {
            grid-column: span 2;
        }
        
        .command-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        
        .command-item {
            padding: 10px;
            background: rgba(255,255,255,0.1);
            border-radius: 6px;
            border-left: 3px solid #FF5722;
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        @media (max-width: 1200px) {
            .dashboard {
                grid-template-columns: 1fr 1fr;
            }
            .video-panel {
                grid-column: span 2;
                grid-row: span 1;
            }
            .voice-commands {
                grid-column: span 2;
            }
        }
        
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
            .video-panel {
                grid-column: span 1;
            }
            .voice-commands {
                grid-column: span 1;
            }
        }
        
        .error-message {
            background: rgba(244, 67, 54, 0.2);
            border: 1px solid #f44336;
            color: #f44336;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🕷️ Hey Spider Robot</h1>
            <p>AI-Powered Spider Robot with Real-Time Control & Monitoring</p>
            <div id="connectionStatus" class="connection-status disconnected">
                Connecting to robot...
            </div>
        </div>
        
        <div class="dashboard">
            <!-- Video Feed Panel -->
            <div class="panel video-panel">
                <h3>📹 Live Camera Feed</h3>
                <img id="videoFeed" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='640' height='240'%3E%3Crect width='100%' height='100%' fill='%23333'/%3E%3Ctext x='50%' y='50%' font-family='Arial' font-size='18' fill='white' text-anchor='middle' dy='.3em'%3ECamera Initializing...%3C/text%3E%3C/svg%3E" alt="Camera feed">
                
                <div class="controls">
                    <button class="btn" onclick="sendCommand('walk forward')" id="walkBtn">🚶 Walk Forward</button>
                    <button class="btn" onclick="sendCommand('turn left')" id="leftBtn">↺ Turn Left</button>
                    <button class="btn" onclick="sendCommand('turn right')" id="rightBtn">↻ Turn Right</button>
                    <button class="btn" onclick="sendCommand('dance')" id="danceBtn">💃 Dance</button>
                    <button class="btn" onclick="sendCommand('wave')" id="waveBtn">👋 Wave</button>
                    <button class="btn" onclick="sendCommand('take photo')" id="photoBtn">📸 Take Photo</button>
                </div>
            </div>
            
            <!-- Robot Status Panel -->
            <div class="panel">
                <h3>🤖 Robot Status</h3>
                <div class="status-item">
                    <span>Distance Sensor:</span>
                    <span class="status-value" id="distance">-- cm</span>
                </div>
                <div class="status-item">
                    <span>Mode:</span>
                    <span class="status-value" id="mode">INITIALIZING</span>
                </div>
                <div class="status-item">
                    <span>Moving:</span>
                    <span class="status-value" id="moving">No</span>
                </div>
                <div class="status-item">
                    <span>Detections:</span>
                    <span class="status-value" id="detectionCount">0</span>
                </div>
                <div class="status-item">
                    <span>Emotional State:</span>
                    <span class="status-value" id="emotionalState">curious</span>
                </div>
            </div>
            
            <!-- OLED Display Simulation -->
            <div class="panel">
                <h3>📺 OLED Display</h3>
                <div class="oled-display" id="oledDisplay">
                    HEY SPIDER<br>
                    Mode: INITIALIZING<br>
                    Dist: --.-cm<br>
                    Cmd: --<br>
                    Sees: 0 objects<br>
                    Booting up...
                </div>
            </div>
            
            <!-- AI Thoughts Panel -->
            <div class="panel">
                <h3>🧠 AI Thoughts</h3>
                <div class="ai-thought" id="aiThought">
                    "System initializing... Loading personality matrix..."
                </div>
                <div class="status-item">
                    <span>AI Status:</span>
                    <span class="status-value" id="aiStatus">Loading...</span>
                </div>
            </div>
            
            <!-- Recent Detections -->
            <div class="panel">
                <h3>👁️ Object Detection</h3>
                <div class="detection-list" id="detectionList">
                    <p style="text-align: center; opacity: 0.7;">No objects detected yet...</p>
                </div>
                <div class="status-item">
                    <span>Detection Description:</span>
                </div>
                <div id="detectionDescription" style="font-style: italic; opacity: 0.8;">
                    Waiting for camera initialization...
                </div>
            </div>
            
            <!-- Voice Commands -->
            <div class="panel voice-commands">
                <h3>🎙️ Voice Commands</h3>
                <p style="margin-bottom: 15px;">Say <strong>"Hey Spider"</strong> followed by:</p>
                <div class="command-list">
                    <div class="command-item">
                        <strong>Movement:</strong><br>
                        walk forward, turn left, turn right
                    </div>
                    <div class="command-item">
                        <strong>Actions:</strong><br>
                        dance, wave, stop
                    </div>
                    <div class="command-item">
                        <strong>Camera:</strong><br>
                        take a photo, take a picture
                    </div>
                    <div class="command-item">
                        <strong>AI Commands:</strong><br>
                        Natural language is supported!
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Initialize Socket.IO connection
        const socket = io();
        let isConnected = false;
        let commandInProgress = false;
        
        // Connection status
        const statusElement = document.getElementById('connectionStatus');
        
        socket.on('connect', function() {
            console.log('Connected to robot');
            isConnected = true;
            statusElement.textContent = '🟢 Connected to Hey Spider Robot';
            statusElement.className = 'connection-status connected';
            document.getElementById('aiStatus').textContent = 'Online';
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected from robot');
            isConnected = false;
            statusElement.textContent = '🔴 Disconnected from Robot';
            statusElement.className = 'connection-status disconnected';
            document.getElementById('aiStatus').textContent = 'Offline';
        });
        
        // Real-time status updates
        socket.on('status_update', function(data) {
            try {
                // Update video feed
                if (data.video_frame) {
                    document.getElementById('videoFeed').src = 'data:image/jpeg;base64,' + data.video_frame;
                }
                
                // Update status information
                document.getElementById('distance').textContent = data.distance.toFixed(1) + ' cm';
                document.getElementById('moving').textContent = data.is_moving ? 'Yes' : 'No';
                document.getElementById('detectionCount').textContent = data.detections.length;
                document.getElementById('emotionalState').textContent = data.emotional_state || 'curious';
                
                // Update AI thought
                if (data.ai_thought) {
                    document.getElementById('aiThought').textContent = '"' + data.ai_thought + '"';
                }
                
                // Update OLED display simulation
                updateOLEDDisplay(data);
                
                // Update detections
                updateDetections(data.detections);
            } catch (error) {
                console.error('Error updating interface:', error);
            }
        });
        
        function updateOLEDDisplay(data) {
            try {
                const oledDisplay = document.getElementById('oledDisplay');
                const mode = data.is_moving ? 'MOVING' : 'READY';
                const detectionCount = data.detections.length;
                const thought = data.ai_thought ? 
                    (data.ai_thought.length > 20 ? data.ai_thought.substring(0, 20) + '...' : data.ai_thought) : 
                    'Thinking...';
                
                oledDisplay.innerHTML = `HEY SPIDER<br>Mode: ${mode}<br>Dist: ${data.distance.toFixed(1)}cm<br>Cmd: --<br>Sees: ${detectionCount} objects<br>${thought}`;
                
                // Update mode display
                document.getElementById('mode').textContent = mode;
            } catch (error) {
                console.error('Error updating OLED display:', error);
            }
        }
        
        function updateDetections(detections) {
            try {
                const detectionList = document.getElementById('detectionList');
                const detectionDesc = document.getElementById('detectionDescription');
                
                if (!detections || detections.length === 0) {
                    detectionList.innerHTML = '<p style="text-align: center; opacity: 0.7;">No objects detected</p>';
                    detectionDesc.textContent = "I don't see anything interesting.";
                    return;
                }
                
                // Update detection list
                let html = '';
                detections.forEach(detection => {
                    html += `<div class="detection-item">
                        <span>${detection.class}</span>
                        <span class="confidence">${(detection.confidence * 100).toFixed(1)}%</span>
                    </div>`;
                });
                detectionList.innerHTML = html;
                
                // Create natural language description
                const objectCounts = {};
                detections.forEach(det => {
                    objectCounts[det.class] = (objectCounts[det.class] || 0) + 1;
                });
                
                const descriptions = [];
                for (const [objClass, count] of Object.entries(objectCounts)) {
                    descriptions.push(count === 1 ? `1 ${objClass}` : `${count} ${objClass}s`);
                }
                
                let description;
                if (descriptions.length === 1) {
                    description = `I can see ${descriptions[0]}.`;
                } else if (descriptions.length === 2) {
                    description = `I can see ${descriptions[0]} and ${descriptions[1]}.`;
                } else {
                    description = `I can see ${descriptions.slice(0, -1).join(', ')}, and ${descriptions[descriptions.length - 1]}.`;
                }
                
                detectionDesc.textContent = description;
            } catch (error) {
                console.error('Error updating detections:', error);
            }
        }
        
        function sendCommand(command) {
            try {
                if (!isConnected) {
                    alert('Not connected to robot!');
                    return;
                }
                
                if (commandInProgress) {
                    console.log('Command already in progress, ignoring');
                    return;
                }
                
                commandInProgress = true;
                
                // Disable all buttons temporarily
                const buttons = document.querySelectorAll('.btn');
                buttons.forEach(btn => {
                    btn.disabled = true;
                    btn.classList.add('pulse');
                });
                
                console.log('Sending command:', command);
                socket.emit('command', {command: command});
                
                // Visual feedback
                const activeBtn = event ? event.target : null;
                if (activeBtn) {
                    activeBtn.style.background = 'linear-gradient(45deg, #FF5722, #E64A19)';
                }
                
                // Re-enable buttons after delay
                setTimeout(() => {
                    buttons.forEach(btn => {
                        btn.disabled = false;
                        btn.classList.remove('pulse');
                    });
                    if (activeBtn) {
                        activeBtn.style.background = 'linear-gradient(45deg, #4CAF50, #45a049)';
                    }
                    commandInProgress = false;
                }, 2000);
            } catch (error) {
                console.error('Error sending command:', error);
                commandInProgress = false;
            }
        }
        
        socket.on('command_result', function(result) {
            try {
                console.log('Command result:', result);
                if (!result.success) {
                    alert('Command failed: ' + result.message);
                }
            } catch (error) {
                console.error('Error handling command result:', error);
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(event) {
            try {
                if (event.ctrlKey || event.altKey || event.metaKey) return;
                
                switch(event.key.toLowerCase()) {
                    case 'w':
                    case 'arrowup':
                        sendCommand('walk forward');
                        break;
                    case 'a':
                    case 'arrowleft':
                        sendCommand('turn left');
                        break;
                    case 'd':
                    case 'arrowright':
                        sendCommand('turn right');
                        break;
                    case ' ':
                        event.preventDefault();
                        sendCommand('dance');
                        break;
                    case 'p':
                        sendCommand('take photo');
                        break;
                }
            } catch (error) {
                console.error('Error handling keyboard shortcut:', error);
            }
        });
        
        // Initialize connection status
        setTimeout(() => {
            if (!isConnected) {
                statusElement.textContent = '🟡 Attempting to connect...';
            }
        }, 3000);
        
        // Error handling for uncaught errors
        window.addEventListener('error', function(event) {
            console.error('JavaScript error:', event.error);
        });
    </script>
</body>
</html>
'''

class WebInterface:
    def __init__(self, spider_controller, visual_monitor, ai_thinking, 
                 oled_display: Optional[OLEDDisplay] = None):
        self.spider = spider_controller
        self.vision = visual_monitor
        self.ai = ai_thinking
        self.oled = oled_display
        
        print("Initializing web interface...")
        
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'hey_spider_secret_key_2024'
        
        # Initialize SocketIO with error handling
        try:
            self.socketio = SocketIO(self.app, cors_allowed_origins="*", 
                                    logger=False, engineio_logger=False,
                                    ping_timeout=60, ping_interval=25)
            print("SocketIO initialized successfully")
        except Exception as e:
            print(f"SocketIO initialization error: {e}")
            traceback.print_exc()
            self.socketio = None
        
        self.setup_routes()
        if self.socketio:
            self.setup_socket_events()
        
        # Background task control
        self.background_running = False
        
    def setup_routes(self):
        """Setup Flask routes with comprehensive error handling"""
        
        @self.app.route('/')
        def index():
            try:
                return render_template_string(HTML_TEMPLATE)
            except Exception as e:
                print(f"Template rendering error: {e}")
                traceback.print_exc()
                return f"""
                <html>
                    <head><title>Hey Spider Robot - Error</title></head>
                    <body>
                        <h1>Hey Spider Robot</h1>
                        <p>Web interface loading error: {str(e)}</p>
                        <p>Please check the server logs for more details.</p>
                    </body>
                </html>
                """, 500
            
        @self.app.route('/api/status')
        def get_status():
            try:
                # Safely get data with fallbacks
                distance = 50.0
                detections = []
                detection_description = "Camera initializing..."
                ai_thought = "Loading..."
                is_moving = False
                emotional_state = "curious"
                
                try:
                    if self.spider:
                        distance = self.spider.get_distance()
                        is_moving = getattr(self.spider, 'is_moving', False)
                except Exception as e:
                    print(f"Error getting spider status: {e}")
                
                try:
                    if self.vision:
                        detections = self.vision.get_latest_detections() or []
                        detection_description = self.vision.get_detection_description() or "No detections"
                except Exception as e:
                    print(f"Error getting vision status: {e}")
                
                try:
                    if self.ai:
                        ai_thought = self.ai.get_current_thought() or "Thinking..."
                        emotional_state = getattr(self.ai, 'emotional_state', 'curious')
                except Exception as e:
                    print(f"Error getting AI status: {e}")
                
                return jsonify({
                    'distance': distance,
                    'detections': detections,
                    'detection_description': detection_description,
                    'ai_thought': ai_thought,
                    'is_moving': is_moving,
                    'emotional_state': emotional_state,
                    'status': 'online'
                })
            except Exception as e:
                print(f"Status API error: {e}")
                traceback.print_exc()
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500
            
        @self.app.route('/api/command', methods=['POST'])
        def execute_command():
            try:
                data = request.get_json() or {}
                command = data.get('command', '').strip()
                
                if not command:
                    return jsonify({'success': False, 'message': 'No command provided'}), 400
                
                print(f"Web API command received: {command}")
                result = self._execute_command(command)
                return jsonify(result)
                
            except Exception as e:
                print(f"Command API error: {e}")
                traceback.print_exc()
                return jsonify({
                    'success': False, 
                    'message': f'Server error: {str(e)}'
                }), 500
            
        @self.app.route('/api/photo', methods=['POST'])
        def take_photo():
            try:
                filename = ""
                if self.vision:
                    filename = self.vision.capture_photo() or ""
                
                return jsonify({
                    'filename': filename, 
                    'success': bool(filename)
                })
            except Exception as e:
                print(f"Photo API error: {e}")
                return jsonify({
                    'filename': '', 
                    'success': False, 
                    'error': str(e)
                })
            
        @self.app.route('/health')
        def health_check():
            """Simple health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'spider': bool(self.spider),
                'vision': bool(self.vision),
                'ai': bool(self.ai),
                'socketio': bool(self.socketio)
            })
            
    def setup_socket_events(self):
        """Setup SocketIO events with error handling"""
        if not self.socketio:
            return
            
        @self.socketio.on('connect')
        def handle_connect():
            try:
                print('Web client connected')
                emit('status', 'Connected to Hey Spider Robot')
            except Exception as e:
                print(f"Socket connect error: {e}")
            
        @self.socketio.on('disconnect')
        def handle_disconnect():
            try:
                print('Web client disconnected')
            except Exception as e:
                print(f"Socket disconnect error: {e}")
            
        @self.socketio.on('command')
        def handle_command(data):
            try:
                command = data.get('command', '') if data else ''
                print(f"Socket command received: {command}")
                result = self._execute_command(command)
                emit('command_result', result)
            except Exception as e:
                print(f"Socket command error: {e}")
                traceback.print_exc()
                emit('command_result', {
                    'success': False, 
                    'message': f'Error: {str(e)}'
                })
            
    def _execute_command(self, command: str) -> dict:
        """Execute a robot command with comprehensive error handling"""
        if not command:
            return {'success': False, 'message': 'Empty command'}
            
        command = command.lower().strip()
        print(f"Executing command: {command}")
        
        # Update OLED if available
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
                    return {'success': True, 'message': 'Walking forward'}
                else:
                    return {'success': False, 'message': 'Spider controller not available'}
                
            elif 'left' in command:
                if self.spider:
                    self.spider.turn_left()
                    return {'success': True, 'message': 'Turning left'}
                else:
                    return {'success': False, 'message': 'Spider controller not available'}
                
            elif 'right' in command:
                if self.spider:
                    self.spider.turn_right()
                    return {'success': True, 'message': 'Turning right'}
                else:
                    return {'success': False, 'message': 'Spider controller not available'}
                
            elif 'dance' in command:
                if self.spider:
                    self.spider.dance()
                    return {'success': True, 'message': 'Dancing!'}
                else:
                    return {'success': False, 'message': 'Spider controller not available'}
                
            elif 'wave' in command:
                if self.spider:
                    self.spider.wave()
                    return {'success': True, 'message': 'Waving hello!'}
                else:
                    return {'success': False, 'message': 'Spider controller not available'}
                
            elif any(word in command for word in ['photo', 'picture', 'capture']):
                if self.vision:
                    filename = self.vision.capture_photo()
                    return {'success': bool(filename), 'message': f'Photo saved: {filename}' if filename else 'Photo capture failed'}
                else:
                    return {'success': False, 'message': 'Vision system not available'}
                
            elif 'stop' in command:
                return {'success': True, 'message': 'Stopped'}
                
            else:
                # Use AI to process unknown commands
                if self.ai:
                    try:
                        ai_response = self.ai.process_command(command)
                        parsed_response = json.loads(ai_response)
                        action = parsed_response.get('action', 'unknown')
                        
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
                            
                        return {
                            'success': action != 'unknown',
                            'message': parsed_response.get('response', 'Command processed by AI')
                        }
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"AI response parsing error: {e}")
                        return {'success': False, 'message': 'AI could not process command'}
                else:
                    return {'success': False, 'message': 'Unknown command and AI not available'}
                    
        except Exception as e:
            print(f"Command execution error: {e}")
            traceback.print_exc()
            return {'success': False, 'message': f'Error: {str(e)}'}
            
    def start_background_tasks(self):
        """Start background tasks for real-time updates"""
        if not self.socketio:
            print("SocketIO not available, skipping background tasks")
            return
            
        def broadcast_status():
            print("Starting background status broadcast...")
            while self.background_running:
                try:
                    # Prepare status data with safe fallbacks
                    status_data = {
                        'distance': 50.0,
                        'detections': [],
                        'ai_thought': 'System running...',
                        'is_moving': False,
                        'emotional_state': 'curious'
                    }
                    
                    # Safely gather data from each component
                    try:
                        if self.spider:
                            status_data['distance'] = self.spider.get_distance()
                            status_data['is_moving'] = getattr(self.spider, 'is_moving', False)
                    except Exception as e:
                        print(f"Error getting spider data: {e}")
                    
                    try:
                        if self.vision:
                            status_data['detections'] = self.vision.get_latest_detections() or []
                    except Exception as e:
                        print(f"Error getting vision data: {e}")
                    
                    try:
                        if self.ai:
                            status_data['ai_thought'] = self.ai.get_current_thought() or "Thinking..."
                            status_data['emotional_state'] = getattr(self.ai, 'emotional_state', 'curious')
                    except Exception as e:
                        print(f"Error getting AI data: {e}")
                    
                    # Add video frame if available
                    try:
                        if self.vision and OPENCV_AVAILABLE:
                            frame = self.vision.get_latest_frame()
                            if frame is not None:
                                # Resize frame for web streaming
                                frame_resized = cv2.resize(frame, (320, 240))
                                _, buffer = cv2.imencode('.jpg', frame_resized, 
                                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
                                frame_data = base64.b64encode(buffer).decode('utf-8')
                                status_data['video_frame'] = frame_data
                            else:
                                status_data['video_frame'] = None
                        else:
                            status_data['video_frame'] = None
                    except Exception as e:
                        print(f"Frame encoding error: {e}")
                        status_data['video_frame'] = None
                        
                    # Broadcast to all connected clients
                    try:
                        self.socketio.emit('status_update', status_data)
                    except Exception as e:
                        print(f"Socket emit error: {e}")
                    
                    time.sleep(1)  # Update every second
                    
                except Exception as e:
                    print(f"Broadcast error: {e}")
                    traceback.print_exc()
                    time.sleep(5)
                    
        # Start background thread
        self.background_running = True
        threading.Thread(target=broadcast_status, daemon=True).start()
        print("Background tasks started")
        
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the web interface with comprehensive error handling"""
        print(f"Starting web interface on http://{host}:{port}")
        
        # Start background tasks if SocketIO is available
        if self.socketio:
            self.start_background_tasks()
        
        try:
            if self.socketio:
                # Use SocketIO app
                self.socketio.run(
                    self.app, 
                    host=host, 
                    port=port, 
                    debug=debug,
                    allow_unsafe_werkzeug=True,
                    log_output=True
                )
            else:
                # Fallback to regular Flask if SocketIO failed
                print("Running in fallback mode without SocketIO")
                self.app.run(host=host, port=port, debug=debug)
                
        except Exception as e:
            print(f"Web interface startup error: {e}")
            traceback.print_exc()
            raise
        finally:
            self.background_running = False
            
    def stop(self):
        """Stop background tasks"""
        self.background_running = False
        print("Web interface stopped")