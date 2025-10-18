import threading
import time
import json
from typing import Optional, Dict, Any
from openai import OpenAI
from config.settings import settings
from src.oled_display import OLEDDisplay

class AIThinking:
    def __init__(self, spider_controller, visual_monitor, oled_display: Optional[OLEDDisplay] = None):
        self.spider = spider_controller
        self.vision = visual_monitor
        self.oled = oled_display
        self.running = False
        self.think_thread = None
        self.current_thought = ""
        self.emotional_state = "curious"
        
        # Initialize OpenAI client with new API
        if settings.OPENAI_API_KEY:
            try:
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
                print("OpenAI client initialized successfully")
            except Exception as e:
                print(f"OpenAI initialization error: {e}")
                self.client = None
        else:
            self.client = None
            print("OpenAI API key not set - set OPENAI_API_KEY environment variable")
        
    def start_thinking(self):
        """Start the AI thinking thread"""
        if not self.client:
            print("OpenAI client not initialized, AI thinking disabled")
            return
            
        self.running = True
        self.think_thread = threading.Thread(target=self._thinking_loop, daemon=True)
        self.think_thread.start()
        
    def stop_thinking(self):
        """Stop AI thinking"""
        self.running = False
        if self.think_thread:
            self.think_thread.join()
            
    def _thinking_loop(self):
        """Main AI thinking loop"""
        while self.running:
            try:
                self._generate_thought()
                time.sleep(settings.AI_THINKING_INTERVAL)
            except Exception as e:
                print(f"AI thinking error: {e}")
                time.sleep(10)
                
    def _generate_thought(self):
        """Generate an AI thought based on current context"""
        if not self.client:
            return
            
        try:
            # Gather context
            context = self._gather_context()
            
            # Create prompt
            prompt = f"""You are a friendly spider robot with personality. Based on your current situation, 
            generate a brief thought or observation (max 50 characters for display).
            
            Current context:
            - What you see: {context['detections_desc']}
            - Distance to nearest object: {context['distance']:.1f}cm
            - Current emotional state: {self.emotional_state}
            - Recent activity: {context['recent_activity']}
            
            Respond with just the thought, keep it short and personality-filled."""
            
            response = self.client.chat.completions.create(
                model=settings.AI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0.8
            )
            
            thought = response.choices[0].message.content.strip()
            self.current_thought = thought[:50]  # Truncate for display
            
            if self.oled:
                self.oled.update_ai_thought(self.current_thought)
                
            # Update emotional state based on context
            self._update_emotional_state(context)
            
            print(f"AI Thought: {self.current_thought}")
            
        except Exception as e:
            print(f"Error generating AI thought: {e}")
            self.current_thought = "Thinking quietly..."
            if self.oled:
                self.oled.update_ai_thought(self.current_thought)
            
    def _gather_context(self) -> Dict[str, Any]:
        """Gather current context for AI reasoning"""
        try:
            detections = self.vision.get_latest_detections()
            detections_desc = self.vision.get_detection_description()
            distance = self.spider.get_distance()
            
            return {
                'detections': detections,
                'detections_desc': detections_desc,
                'distance': distance,
                'recent_activity': 'idle',
                'is_moving': self.spider.is_moving
            }
        except Exception as e:
            print(f"Error gathering context: {e}")
            return {
                'detections': [],
                'detections_desc': "Nothing visible",
                'distance': 0.0,
                'recent_activity': 'idle',
                'is_moving': False
            }
        
    def _update_emotional_state(self, context: Dict[str, Any]):
        """Update emotional state based on context"""
        try:
            distance = context['distance']
            num_detections = len(context['detections'])
            
            if distance < 20 and num_detections > 0:
                self.emotional_state = "excited"
            elif num_detections > 2:
                self.emotional_state = "interested"
            elif distance > 100:
                self.emotional_state = "lonely"
            else:
                self.emotional_state = "curious"
        except Exception as e:
            print(f"Error updating emotional state: {e}")
            self.emotional_state = "curious"
            
    def process_command(self, command: str) -> str:
        """Process voice command using AI"""
        if not self.client:
            return '{"action": "unknown", "response": "AI not available"}'
            
        try:
            prompt = f"""You are a spider robot AI. Parse this voice command and return a JSON response:
            Command: "{command}"
            
            Available actions: walk_forward, turn_left, turn_right, dance, wave, take_photo, stop
            
            Return JSON like: {{"action": "walk_forward", "parameters": {{"steps": 3}}, "response": "Moving forward!"}}
            
            If unclear, return action "unknown" and ask for clarification. Be friendly and spider-like in responses."""
            
            response = self.client.chat.completions.create(
                model=settings.AI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Command processing error: {e}")
            return '{"action": "unknown", "response": "Sorry, I could not process that command."}'
            
    def get_current_thought(self) -> str:
        """Get the current AI thought"""
        return self.current_thought