import time
import threading
import math
from typing import Dict, List, Optional

# Hardware imports with fallbacks
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("RPi.GPIO not available - using mock GPIO")
    GPIO_AVAILABLE = False

try:
    from adafruit_servokit import ServoKit
    SERVOKIT_AVAILABLE = True
except ImportError:
    print("ServoKit not available - servo control disabled")
    SERVOKIT_AVAILABLE = False

try:
    from gpiozero import DistanceSensor
    GPIOZERO_AVAILABLE = True
except ImportError:
    print("gpiozero not available - distance sensor disabled")
    GPIOZERO_AVAILABLE = False

from config.hardware_config import SERVO_PINS, ULTRASONIC_PINS
from src.oled_display import OLEDDisplay

class SpiderController:
    def __init__(self, oled_display: Optional[OLEDDisplay] = None):
        self.oled = oled_display
        self.is_moving = False
        self.current_position = "neutral"
        
        # Initialize servo controller
        if SERVOKIT_AVAILABLE:
            try:
                self.kit = ServoKit(channels=16, address=0x40, frequency=50)
                self.setup_servos()
                print("Servo controller initialized")
            except Exception as e:
                print(f"Servo initialization error: {e}")
                self.kit = None
        else:
            self.kit = None
            print("Servo control disabled - ServoKit not available")
            
        # Initialize ultrasonic sensor
        if GPIOZERO_AVAILABLE:
            try:
                self.distance_sensor = DistanceSensor(
                    echo=ULTRASONIC_PINS['echo'], 
                    trigger=ULTRASONIC_PINS['trigger'],
                    max_distance=4  # Maximum 4 meters
                )
                print("Distance sensor initialized")
            except Exception as e:
                print(f"Distance sensor error: {e}")
                self.distance_sensor = None
        else:
            self.distance_sensor = None
            print("Distance sensor disabled - gpiozero not available")
            
        # Start distance monitoring
        self.start_distance_monitoring()
        
        if self.oled:
            self.oled.update_mode("READY")
            
    def setup_servos(self):
        """Initialize all servos to neutral position"""
        if not self.kit:
            return
            
        self.servo_positions = {
            'leg1_shoulder': 90, 'leg1_elbow': 90, 'leg1_foot': 90,
            'leg2_shoulder': 90, 'leg2_elbow': 90, 'leg2_foot': 90,
            'leg3_shoulder': 90, 'leg3_elbow': 90, 'leg3_foot': 90,
            'leg4_shoulder': 90, 'leg4_elbow': 90, 'leg4_foot': 90
        }
        
        for servo_name, position in self.servo_positions.items():
            try:
                if servo_name in SERVO_PINS:
                    channel = SERVO_PINS[servo_name]
                    self.kit.servo[channel].angle = position
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error setting {servo_name}: {e}")
                
    def start_distance_monitoring(self):
        """Start continuous distance monitoring"""
        def monitor_distance():
            while True:
                try:
                    if self.distance_sensor:
                        # Get distance in centimeters
                        distance_m = self.distance_sensor.distance
                        distance_cm = distance_m * 100 if distance_m and distance_m < 4 else 400
                        if self.oled:
                            self.oled.update_distance(distance_cm)
                    else:
                        # Mock distance for testing
                        if self.oled:
                            self.oled.update_distance(50.0)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Distance monitoring error: {e}")
                    time.sleep(2)
                    
        threading.Thread(target=monitor_distance, daemon=True).start()
        
    def move_servo(self, servo_name: str, angle: int, speed: float = 0.1):
        """Move a single servo to specified angle"""
        if not self.kit or servo_name not in SERVO_PINS:
            print(f"Mock servo move: {servo_name} to {angle} degrees")
            return
            
        try:
            # Clamp angle to valid range
            angle = max(0, min(180, angle))
            
            channel = SERVO_PINS[servo_name]
            current_pos = self.servo_positions.get(servo_name, 90)
            
            # Smooth movement
            steps = abs(angle - current_pos)
            if steps > 0:
                step_size = 1 if angle > current_pos else -1
                for pos in range(current_pos, angle + step_size, step_size):
                    self.kit.servo[channel].angle = pos
                    time.sleep(speed / max(1, steps / 10))  # Adjusted timing
                    
            self.servo_positions[servo_name] = angle
            
        except Exception as e:
            print(f"Error moving servo {servo_name}: {e}")
            
    def walk_forward(self, steps: int = 4):
        """Walk forward using alternating diagonal gait"""
        if self.is_moving:
            print("Already moving, command ignored")
            return
            
        self.is_moving = True
        if self.oled:
            self.oled.update_mode("WALKING")
            
        try:
            print(f"Walking forward {steps} steps")
            for step in range(steps):
                # Phase 1: Lift legs 1 and 4, move legs 2 and 3
                self._lift_legs(['leg1', 'leg4'])
                self._move_legs_forward(['leg2', 'leg3'])
                self._lower_legs(['leg1', 'leg4'])
                time.sleep(0.3)
                
                # Phase 2: Lift legs 2 and 3, move legs 1 and 4
                self._lift_legs(['leg2', 'leg3'])
                self._move_legs_forward(['leg1', 'leg4'])
                self._lower_legs(['leg2', 'leg3'])
                time.sleep(0.3)
                
        except Exception as e:
            print(f"Walk error: {e}")
        finally:
            self.is_moving = False
            if self.oled:
                self.oled.update_mode("READY")
                
    def turn_left(self, steps: int = 2):
        """Turn left by rotating body"""
        if self.is_moving:
            return
            
        self.is_moving = True
        if self.oled:
            self.oled.update_mode("TURNING")
            
        try:
            print(f"Turning left {steps} steps")
            for step in range(steps):
                self._adjust_leg_positions({
                    'leg1_shoulder': 70, 'leg2_shoulder': 110,
                    'leg3_shoulder': 70, 'leg4_shoulder': 110
                })
                time.sleep(0.5)
                self._return_to_neutral()
                time.sleep(0.3)
        except Exception as e:
            print(f"Turn left error: {e}")
        finally:
            self.is_moving = False
            if self.oled:
                self.oled.update_mode("READY")
                
    def turn_right(self, steps: int = 2):
        """Turn right by rotating body"""
        if self.is_moving:
            return
            
        self.is_moving = True
        if self.oled:
            self.oled.update_mode("TURNING")
            
        try:
            print(f"Turning right {steps} steps")
            for step in range(steps):
                self._adjust_leg_positions({
                    'leg1_shoulder': 110, 'leg2_shoulder': 70,
                    'leg3_shoulder': 110, 'leg4_shoulder': 70
                })
                time.sleep(0.5)
                self._return_to_neutral()
                time.sleep(0.3)
        except Exception as e:
            print(f"Turn right error: {e}")
        finally:
            self.is_moving = False
            if self.oled:
                self.oled.update_mode("READY")
                
    def dance(self):
        """Perform a dance sequence"""
        if self.is_moving:
            return
            
        self.is_moving = True
        if self.oled:
            self.oled.update_mode("DANCING")
            
        try:
            print("Dancing!")
            dance_moves = [
                {'leg1_elbow': 45, 'leg3_elbow': 45},
                {'leg2_elbow': 45, 'leg4_elbow': 45},
                {'leg1_elbow': 135, 'leg3_elbow': 135},
                {'leg2_elbow': 135, 'leg4_elbow': 135}
            ]
            
            for move in dance_moves * 3:
                self._adjust_leg_positions(move)
                time.sleep(0.4)
                self._return_to_neutral()
                time.sleep(0.2)
                
        except Exception as e:
            print(f"Dance error: {e}")
        finally:
            self.is_moving = False
            if self.oled:
                self.oled.update_mode("READY")
                
    def wave(self):
        """Wave with front legs"""
        if self.is_moving:
            return
            
        self.is_moving = True
        if self.oled:
            self.oled.update_mode("WAVING")
            
        try:
            print("Waving!")
            for _ in range(3):
                self.move_servo('leg1_elbow', 45)
                self.move_servo('leg2_elbow', 45)
                time.sleep(0.3)
                self.move_servo('leg1_elbow', 135)
                self.move_servo('leg2_elbow', 135)
                time.sleep(0.3)
            self._return_to_neutral()
        except Exception as e:
            print(f"Wave error: {e}")
        finally:
            self.is_moving = False
            if self.oled:
                self.oled.update_mode("READY")
                
    def _lift_legs(self, legs: List[str]):
        """Lift specified legs"""
        for leg in legs:
            self.move_servo(f'{leg}_foot', 45, 0.05)
            
    def _lower_legs(self, legs: List[str]):
        """Lower specified legs"""
        for leg in legs:
            self.move_servo(f'{leg}_foot', 90, 0.05)
            
    def _move_legs_forward(self, legs: List[str]):
        """Move specified legs forward"""
        for leg in legs:
            current = self.servo_positions.get(f'{leg}_shoulder', 90)
            new_angle = max(60, min(120, current + 20))  # Clamp to safe range
            self.move_servo(f'{leg}_shoulder', new_angle, 0.05)
            
    def _adjust_leg_positions(self, positions: Dict[str, int]):
        """Adjust multiple leg positions simultaneously"""
        threads = []
        for servo_name, angle in positions.items():
            thread = threading.Thread(
                target=self.move_servo, 
                args=(servo_name, angle, 0.02),
                daemon=True
            )
            threads.append(thread)
            thread.start()
            
        # Wait for all movements to complete
        for thread in threads:
            thread.join()
        
    def _return_to_neutral(self):
        """Return all servos to neutral position"""
        neutral_positions = {
            'leg1_shoulder': 90, 'leg1_elbow': 90, 'leg1_foot': 90,
            'leg2_shoulder': 90, 'leg2_elbow': 90, 'leg2_foot': 90,
            'leg3_shoulder': 90, 'leg3_elbow': 90, 'leg3_foot': 90,
            'leg4_shoulder': 90, 'leg4_elbow': 90, 'leg4_foot': 90
        }
        self._adjust_leg_positions(neutral_positions)
        
    def get_distance(self) -> float:
        """Get current distance reading"""
        try:
            if self.distance_sensor:
                distance_m = self.distance_sensor.distance
                return distance_m * 100 if distance_m and distance_m < 4 else 400
        except Exception as e:
            print(f"Distance reading error: {e}")
        return 50.0  # Return mock distance for testing
        
    def cleanup(self):
        """Clean up GPIO resources"""
        try:
            if self.distance_sensor:
                self.distance_sensor.close()
            if GPIO_AVAILABLE:
                GPIO.cleanup()
        except Exception as e:
            print(f"Cleanup error: {e}")
            pass