import time
import threading
from typing import Optional, List
try:
    import board
    import busio
    import adafruit_ssd1306
    from PIL import Image, ImageDraw, ImageFont
    I2C_AVAILABLE = True
except ImportError as e:
    print(f"I2C/OLED libraries not available: {e}")
    I2C_AVAILABLE = False

from config.settings import settings

class OLEDDisplay:
    def __init__(self):
        self.width = settings.OLED_WIDTH
        self.height = settings.OLED_HEIGHT
        self.running = False
        self.display_thread = None
        
        # Initialize I2C and display
        if I2C_AVAILABLE:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self.display = adafruit_ssd1306.SSD1306_I2C(
                    self.width, self.height, i2c, addr=0x3C
                )
                self.display.fill(0)
                self.display.show()
                print("OLED display initialized successfully")
            except Exception as e:
                print(f"OLED initialization error: {e}")
                self.display = None
        else:
            print("OLED display disabled - hardware libraries not available")
            self.display = None
            
        # Create image and drawing context if display is available
        if self.display:
            self.image = Image.new('1', (self.width, self.height))
            self.draw = ImageDraw.Draw(self.image)
            
            # Load font
            try:
                self.font = ImageFont.load_default()
                self.font_small = ImageFont.load_default()
            except Exception as e:
                print(f"Font loading error: {e}")
                self.font = None
                self.font_small = None
        
        # Display state
        self.current_status = "Initializing..."
        self.current_mode = "IDLE"
        self.last_command = ""
        self.detections = []
        self.ai_thought = ""
        self.distance = 0
        
    def start(self):
        """Start the display update thread"""
        if not self.display:
            return
            
        self.running = True
        self.display_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.display_thread.start()
        
    def stop(self):
        """Stop the display update thread"""
        self.running = False
        if self.display_thread:
            self.display_thread.join(timeout=2)
            
    def _update_loop(self):
        """Main display update loop"""
        while self.running:
            try:
                self._update_display()
                time.sleep(settings.OLED_UPDATE_INTERVAL)
            except Exception as e:
                print(f"Display update error: {e}")
                time.sleep(1)
                
    def _update_display(self):
        """Update the OLED display with current information"""
        if not self.display or not self.draw:
            return
            
        try:
            # Clear the image
            self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
            
            # Line positions
            line_height = 10
            y_pos = 0
            
            # Title
            self.draw.text((0, y_pos), "HEY SPIDER", font=self.font, fill=255)
            y_pos += line_height + 2
            
            # Status line
            self.draw.text((0, y_pos), f"Mode: {self.current_mode}", font=self.font_small, fill=255)
            y_pos += line_height
            
            # Distance
            self.draw.text((0, y_pos), f"Dist: {self.distance:.1f}cm", font=self.font_small, fill=255)
            y_pos += line_height
            
            # Last command
            if self.last_command:
                cmd_display = self.last_command[:15] + "..." if len(self.last_command) > 15 else self.last_command
                self.draw.text((0, y_pos), f"Cmd: {cmd_display}", font=self.font_small, fill=255)
            y_pos += line_height
            
            # Detections
            if self.detections:
                det_text = f"Sees: {len(self.detections)} objects"
                self.draw.text((0, y_pos), det_text, font=self.font_small, fill=255)
            y_pos += line_height
            
            # AI thought (truncated)
            if self.ai_thought:
                thought_display = self.ai_thought[:20] + "..." if len(self.ai_thought) > 20 else self.ai_thought
                self.draw.text((0, y_pos), thought_display, font=self.font_small, fill=255)
            
            # Update the display
            self.display.image(self.image)
            self.display.show()
        except Exception as e:
            print(f"Display render error: {e}")
        
    def update_status(self, status: str):
        """Update the current status"""
        self.current_status = status
        
    def update_mode(self, mode: str):
        """Update the current mode"""
        self.current_mode = mode
        print(f"OLED Mode: {mode}")
        
    def update_command(self, command: str):
        """Update the last command"""
        self.last_command = command
        
    def update_detections(self, detections: List[dict]):
        """Update detected objects"""
        self.detections = detections
        
    def update_ai_thought(self, thought: str):
        """Update AI thought"""
        self.ai_thought = thought
        
    def update_distance(self, distance: float):
        """Update distance reading"""
        self.distance = distance
        
    def show_startup_message(self):
        """Show startup animation"""
        if not self.display or not self.draw:
            print("OLED Startup: Hey Spider Robot Initializing...")
            return
            
        messages = [
            "Booting...",
            "Loading AI...",
            "Camera Ready",
            "Voice Ready",
            "Hey Spider!",
            "Ready to Go!"
        ]
        
        for msg in messages:
            try:
                self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
                # Calculate text position for centering
                try:
                    bbox = self.draw.textbbox((0, 0), msg, font=self.font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except:
                    text_width = len(msg) * 6
                    text_height = 10
                    
                x = max(0, (self.width - text_width) // 2)
                y = max(0, (self.height - text_height) // 2)
                
                self.draw.text((x, y), msg, font=self.font, fill=255)
                self.display.image(self.image)
                self.display.show()
                time.sleep(0.8)
            except Exception as e:
                print(f"Startup message error: {e}")
                break