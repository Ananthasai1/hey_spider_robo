"""
Real-Time Object Detection with YOLOv12 and OV5647 Camera
Optimized for Raspberry Pi with Night Vision Support
"""

import threading
import time
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
from collections import deque
import queue

# Import configurations
try:
    from config.yolo_v12_config import (
        YOLOv12Config, OV5647CameraConfig, COCO_CLASSES,
        get_yolo_config, get_camera_config, get_class_color,
        NIGHT_VISION_PRESETS
    )
except ImportError:
    print("Config not found, using defaults")

# Hardware imports with fallbacks
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    print("OpenCV not available")
    OPENCV_AVAILABLE = False

try:
    from picamera2 import Picamera2, Preview
    from libcamera import controls
    PICAMERA2_AVAILABLE = True
except ImportError:
    print("PiCamera2 not available - will try legacy or USB camera")
    PICAMERA2_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("YOLO not available")
    YOLO_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = 'cpu'


class RealTimeYOLOv12Detector:
    """Real-time YOLOv12 object detector with multi-threading"""
    
    def __init__(self, config: YOLOv12Config):
        self.config = config
        self.model = None
        self.detection_queue = queue.Queue(maxsize=30)
        self.running = False
        self.detection_thread = None
        
        # Performance tracking
        self.fps = 0
        self.frame_times = deque(maxlen=30)
        self.detection_count = 0
        
        # Initialize model
        self._init_model()
        
    def _init_model(self):
        """Initialize YOLOv12 model"""
        if not YOLO_AVAILABLE:
            print("YOLO not available - detector disabled")
            return
        
        try:
            model_path = self.config.MODEL_PATH
            if not os.path.exists(model_path):
                # Try alternative paths
                alt_paths = [
                    f"models/{model_path}",
                    f"yolov8{self.config.MODEL_SIZE}.pt",  # Fallback to v8
                    "yolov8n.pt"
                ]
                
                for path in alt_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
                else:
                    print(f"Downloading YOLOv12 model: {model_path}")
            
            # Load model
            self.model = YOLO(model_path)
            
            # Optimize model
            if TORCH_AVAILABLE:
                self.model.to(DEVICE)
                if self.config.HALF_PRECISION and DEVICE != 'cpu':
                    self.model.half()
            
            # Warm up model
            print("Warming up YOLOv12 model...")
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy, verbose=False)
            
            print(f"✅ YOLOv12 loaded: {model_path} on {DEVICE}")
            print(f"📊 Classes: {len(self.model.names)}")
            
        except Exception as e:
            print(f"❌ YOLOv12 initialization failed: {e}")
            self.model = None
    
    def start(self):
        """Start detection thread"""
        if not self.model:
            return
        
        self.running = True
        self.detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True
        )
        self.detection_thread.start()
        print("🎯 Real-time detector started")
    
    def stop(self):
        """Stop detection thread"""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2)
    
    def _detection_loop(self):
        """Main detection processing loop"""
        while self.running:
            try:
                if not self.detection_queue.empty():
                    frame, timestamp = self.detection_queue.get(timeout=0.1)
                    
                    start_time = time.time()
                    results = self._detect_objects(frame)
                    detection_time = time.time() - start_time
                    
                    # Update FPS
                    self.frame_times.append(detection_time)
                    if len(self.frame_times) > 0:
                        self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                    
                    yield results
                else:
                    time.sleep(0.001)
                    
            except Exception as e:
                print(f"Detection error: {e}")
                time.sleep(0.1)
    
    def _detect_objects(self, frame):
        """Detect objects in frame using YOLOv12"""
        if not self.model:
            return []
        
        try:
            # Run detection
            results = self.model(
                frame,
                conf=self.config.CONFIDENCE_THRESHOLD,
                iou=self.config.IOU_THRESHOLD,
                max_det=self.config.MAX_DETECTIONS,
                verbose=False,
                device=DEVICE,
                half=self.config.HALF_PRECISION
            )
            
            detections = []
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Apply filters
                        if not self._passes_filters(box, cls_id):
                            continue
                        
                        detection = {
                            'class': COCO_CLASSES.get(cls_id, f"class_{cls_id}"),
                            'class_id': cls_id,
                            'confidence': conf,
                            'bbox': box.tolist(),
                            'timestamp': time.time(),
                            'area': (box[2] - box[0]) * (box[3] - box[1])
                        }
                        
                        # Add tracking ID if available
                        if hasattr(boxes, 'id') and boxes.id is not None:
                            detection['track_id'] = int(boxes.id[i].cpu().numpy())
                        
                        detections.append(detection)
            
            self.detection_count += len(detections)
            return detections
            
        except Exception as e:
            print(f"Detection processing error: {e}")
            return []
    
    def _passes_filters(self, box, class_id) -> bool:
        """Check if detection passes filters"""
        # Size filter
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        if w < self.config.MIN_DETECTION_SIZE or h < self.config.MIN_DETECTION_SIZE:
            return False
        
        if w > self.config.MAX_DETECTION_SIZE or h > self.config.MAX_DETECTION_SIZE:
            return False
        
        # Aspect ratio filter
        if self.config.ASPECT_RATIO_FILTER:
            aspect = w / h if h > 0 else 0
            if aspect < self.config.MIN_ASPECT_RATIO or aspect > self.config.MAX_ASPECT_RATIO:
                return False
        
        # Class filter
        if self.config.PRIORITY_CLASSES and class_id not in self.config.PRIORITY_CLASSES:
            return False
        
        if self.config.IGNORE_CLASSES and class_id in self.config.IGNORE_CLASSES:
            return False
        
        return True
    
    def add_frame(self, frame, timestamp=None):
        """Add frame to detection queue"""
        if timestamp is None:
            timestamp = time.time()
        
        try:
            self.detection_queue.put_nowait((frame, timestamp))
        except queue.Full:
            # Drop oldest frame if queue full
            try:
                self.detection_queue.get_nowait()
                self.detection_queue.put_nowait((frame, timestamp))
            except:
                pass


class OV5647CameraController:
    """OV5647 Camera controller with night vision support"""
    
    def __init__(self, config: OV5647CameraConfig):
        self.config = config
        self.camera = None
        self.running = False
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Night vision state
        self.night_mode = False
        self.last_brightness = 128
        
        # Initialize camera
        self._init_camera()
    
    def _init_camera(self):
        """Initialize OV5647 camera"""
        if PICAMERA2_AVAILABLE:
            self._init_picamera2()
        elif OPENCV_AVAILABLE:
            self._init_opencv_camera()
        else:
            print("No camera backend available")
    
    def _init_picamera2(self):
        """Initialize using PiCamera2 (recommended)"""
        try:
            print("🎥 Initializing OV5647 with PiCamera2...")
            
            self.camera = Picamera2()
            
            # Configure camera
            camera_config = self.camera.create_still_configuration(
                main={
                    "size": (self.config.CAPTURE_WIDTH, self.config.CAPTURE_HEIGHT),
                    "format": "RGB888"
                },
                buffer_count=self.config.BUFFER_SIZE
            )
            
            self.camera.configure(camera_config)
            
            # Set initial controls
            self._update_camera_settings()
            
            self.camera.start()
            time.sleep(2)  # Camera warm-up
            
            print(f"✅ OV5647 initialized: {self.config.CAPTURE_WIDTH}x{self.config.CAPTURE_HEIGHT}")
            
        except Exception as e:
            print(f"PiCamera2 init failed: {e}")
            self.camera = None
    
    def _init_opencv_camera(self):
        """Initialize using OpenCV (fallback)"""
        try:
            print("🎥 Initializing camera with OpenCV...")
            
            for idx in [0, 1, 2]:
                self.camera = cv2.VideoCapture(idx)
                if self.camera.isOpened():
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAPTURE_WIDTH)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAPTURE_HEIGHT)
                    self.camera.set(cv2.CAP_PROP_FPS, self.config.TARGET_FPS)
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Test read
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        print(f"✅ Camera initialized on index {idx}")
                        return
                    
                    self.camera.release()
            
            self.camera = None
            print("❌ No working camera found")
            
        except Exception as e:
            print(f"OpenCV camera init failed: {e}")
            self.camera = None
    
    def _update_camera_settings(self, night_mode: bool = None):
        """Update camera settings based on mode"""
        if not self.camera or not PICAMERA2_AVAILABLE:
            return
        
        try:
            if night_mode is None:
                night_mode = self.night_mode
            
            if night_mode:
                # Night vision settings
                preset = NIGHT_VISION_PRESETS.get('night', {})
                self.camera.set_controls({
                    "ExposureTime": preset.get('exposure_time', 33333),
                    "AnalogueGain": preset.get('analog_gain', 8.0),
                    "Brightness": preset.get('brightness', 0.3),
                    "Contrast": preset.get('contrast', 1.4),
                })
            else:
                # Day mode settings
                self.camera.set_controls({
                    "ExposureTime": self.config.DAY_EXPOSURE_TIME,
                    "AnalogueGain": self.config.DAY_ANALOG_GAIN,
                    "Brightness": self.config.BRIGHTNESS,
                    "Contrast": self.config.CONTRAST,
                    "Saturation": self.config.SATURATION,
                    "Sharpness": self.config.SHARPNESS,
                })
            
            if self.config.AUTO_EXPOSURE:
                self.camera.set_controls({"AeEnable": True})
            
            if self.config.AUTO_WHITE_BALANCE:
                self.camera.set_controls({"AwbEnable": True})
            
        except Exception as e:
            print(f"Camera settings update failed: {e}")
    
    def start(self):
        """Start camera capture"""
        if not self.camera:
            return False
        
        self.running = True
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )
        self.capture_thread.start()
        return True
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        
        if self.camera:
            if PICAMERA2_AVAILABLE:
                self.camera.stop()
                self.camera.close()
            else:
                self.camera.release()
    
    def _capture_loop(self):
        """Main capture loop"""
        while self.running:
            try:
                frame = self._capture_frame()
                if frame is not None:
                    # Auto night mode
                    if self.config.AUTO_NIGHT_MODE:
                        self._check_night_mode(frame)
                    
                    # Add to queue
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame)
                        except:
                            pass
                
                time.sleep(1.0 / self.config.TARGET_FPS)
                
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)
    
    def _capture_frame(self):
        """Capture single frame"""
        try:
            if PICAMERA2_AVAILABLE and isinstance(self.camera, Picamera2):
                frame = self.camera.capture_array()
                return frame
            elif OPENCV_AVAILABLE:
                ret, frame = self.camera.read()
                if ret:
                    return frame
            return None
        except Exception as e:
            print(f"Frame capture error: {e}")
            return None
    
    def _check_night_mode(self, frame):
        """Check and switch night mode based on brightness"""
        try:
            # Calculate average brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
            avg_brightness = np.mean(gray)
            self.last_brightness = avg_brightness
            
            # Switch to night mode if dark
            should_be_night = avg_brightness < self.config.NIGHT_MODE_THRESHOLD
            
            if should_be_night != self.night_mode:
                self.night_mode = should_be_night
                self._update_camera_settings(self.night_mode)
                mode = "NIGHT" if self.night_mode else "DAY"
                print(f"🌙 Switched to {mode} mode (brightness: {avg_brightness:.1f})")
                
        except Exception as e:
            print(f"Night mode check error: {e}")
    
    def get_frame(self):
        """Get latest frame from queue"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def is_night_mode(self):
        """Check if in night mode"""
        return self.night_mode


class EnhancedVisualMonitor:
    """Enhanced visual monitoring with YOLOv12 and OV5647"""
    
    def __init__(self, oled_display=None):
        self.oled = oled_display
        
        # Configuration
        self.yolo_config = get_yolo_config('balanced')
        self.camera_config = get_camera_config()
        
        # Components
        self.camera = OV5647CameraController(self.camera_config)
        self.detector = RealTimeYOLOv12Detector(self.yolo_config)
        
        # State
        self.running = False
        self.monitoring_thread = None
        
        # Data
        self.latest_frame = None
        self.annotated_frame = None
        self.latest_detections = []
        self.detection_history = deque(maxlen=100)
        
        # Performance
        self.fps = 0
        self.total_frames = 0
        self.total_detections = 0
        
        # Directories
        os.makedirs('images/raw', exist_ok=True)
        os.makedirs('images/detections', exist_ok=True)
        os.makedirs('images/night_vision', exist_ok=True)
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("✅ Enhanced Visual Monitor initialized")
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.running:
            return
        
        print("🎬 Starting enhanced visual monitoring...")
        
        # Start camera
        if not self.camera.start():
            print("❌ Failed to start camera")
            return
        
        # Start detector
        self.detector.start()
        
        # Start monitoring loop
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        print("✅ Visual monitoring active")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        print("🛑 Stopping visual monitoring...")
        self.running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=3)
        
        self.detector.stop()
        self.camera.stop()
        
        print("✅ Visual monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.running:
            try:
                # Get frame from camera
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                self.latest_frame = frame.copy()
                self.total_frames += 1
                
                # Resize for processing
                process_frame = cv2.resize(
                    frame,
                    (self.yolo_config.IMG_SIZE, self.yolo_config.IMG_SIZE)
                )
                
                # Add to detector queue
                self.detector.add_frame(process_frame)
                
                # Get detections (non-blocking)
                detections = self._get_latest_detections()
                if detections:
                    self.latest_detections = detections
                    self.total_detections += len(detections)
                    self._update_detection_history(detections)
                
                # Annotate frame
                self.annotated_frame = self._annotate_frame(frame, detections)
                
                # Update FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    self.fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Update OLED
                if self.oled:
                    self.oled.update_detections(detections)
                
                # Small sleep to prevent CPU overload
                time.sleep(0.001)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(0.1)
    
    def _get_latest_detections(self):
        """Get latest detections from detector"""
        try:
            detection_gen = self.detector._detection_loop()
            return next(detection_gen, [])
        except:
            return []
    
    def _annotate_frame(self, frame, detections):
        """Annotate frame with detection results"""
        if not detections:
            return frame.copy()
        
        annotated = frame.copy()
        
        try:
            # Resize if needed
            h, w = annotated.shape[:2]
            scale_x = w / self.yolo_config.IMG_SIZE
            scale_y = h / self.yolo_config.IMG_SIZE
            
            for det in detections:
                # Scale bbox to original frame size
                bbox = det['bbox']
                x1 = int(bbox[0] * scale_x)
                y1 = int(bbox[1] * scale_y)
                x2 = int(bbox[2] * scale_x)
                y2 = int(bbox[3] * scale_y)
                
                # Get color
                color = get_class_color(det['class_id'])
                
                # Draw bounding box
                thickness = self.yolo_config.BOX_THICKNESS
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                
                # Prepare label
                label = f"{det['class']}"
                if self.yolo_config.SHOW_CONFIDENCE:
                    label += f" {det['confidence']:.2f}"
                
                if 'track_id' in det:
                    label += f" ID:{det['track_id']}"
                
                # Draw label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = self.yolo_config.FONT_SCALE
                (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, 2)
                
                cv2.rectangle(
                    annotated,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 5),
                    font,
                    font_scale,
                    (255, 255, 255),
                    2
                )
            
            # Add FPS and info
            info_text = f"FPS: {self.fps} | Detections: {len(detections)}"
            if self.camera.is_night_mode():
                info_text += " | NIGHT MODE"
            
            cv2.putText(
                annotated,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Night vision indicator
            if self.camera.is_night_mode():
                cv2.putText(
                    annotated,
                    "🌙 NIGHT VISION",
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
            
        except Exception as e:
            self.logger.error(f"Annotation error: {e}")
            return frame.copy()
        
        return annotated
    
    def _update_detection_history(self, detections):
        """Update detection history"""
        self.detection_history.append({
            'timestamp': time.time(),
            'detections': detections.copy(),
            'count': len(detections),
            'night_mode': self.camera.is_night_mode()
        })
    
    def capture_photo(self) -> str:
        """Capture and save photo with detections"""
        try:
            frame_to_save = self.annotated_frame if self.annotated_frame is not None else self.latest_frame
            
            if frame_to_save is None:
                self.logger.warning("No frame available for capture")
                return ""
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Determine save directory
            if self.camera.is_night_mode():
                base_dir = "images/night_vision"
            else:
                base_dir = "images/detections"
            
            # Save annotated image
            filename = f"{base_dir}/capture_{timestamp}.jpg"
            success = cv2.imwrite(filename, frame_to_save)
            
            if success:
                # Save metadata
                metadata = {
                    'timestamp': timestamp,
                    'detections': len(self.latest_detections),
                    'night_mode': self.camera.is_night_mode(),
                    'fps': self.fps,
                    'objects': [d['class'] for d in self.latest_detections]
                }
                
                metadata_file = filename.replace('.jpg', '_metadata.txt')
                with open(metadata_file, 'w') as f:
                    f.write(str(metadata))
                
                self.logger.info(f"📸 Photo saved: {filename}")
                return filename
            else:
                self.logger.error("Failed to save photo")
                return ""
                
        except Exception as e:
            self.logger.error(f"Photo capture error: {e}")
            return ""
    
    def get_latest_detections(self) -> List[Dict]:
        """Get latest detection results"""
        return self.latest_detections.copy()
    
    def get_detection_description(self) -> str:
        """Get natural language description"""
        if not self.latest_detections:
            if self.camera.is_night_mode():
                return "Night vision active - no objects detected in darkness"
            return "No objects detected"
        
        # Count objects
        object_counts = {}
        for det in self.latest_detections:
            cls = det['class']
            object_counts[cls] = object_counts.get(cls, 0) + 1
        
        # Build description
        descriptions = []
        for obj_cls, count in object_counts.items():
            if count == 1:
                descriptions.append(f"1 {obj_cls}")
            else:
                descriptions.append(f"{count} {obj_cls}s")
        
        if len(descriptions) == 1:
            desc = f"I can see {descriptions[0]}"
        elif len(descriptions) == 2:
            desc = f"I can see {descriptions[0]} and {descriptions[1]}"
        else:
            desc = f"I can see {', '.join(descriptions[:-1])}, and {descriptions[-1]}"
        
        if self.camera.is_night_mode():
            desc += " (night vision mode)"
        
        return desc + "."
    
    def get_latest_frame(self):
        """Get latest raw frame"""
        return self.latest_frame
    
    def get_annotated_frame(self):
        """Get annotated frame"""
        return self.annotated_frame
    
    def is_camera_active(self) -> bool:
        """Check if camera is active"""
        return self.running and self.camera.camera is not None
    
    def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        return {
            'fps': self.fps,
            'detector_fps': self.detector.fps,
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'current_objects': len(self.latest_detections),
            'night_mode': self.camera.is_night_mode(),
            'brightness': self.camera.last_brightness,
            'device': DEVICE,
            'model': self.yolo_config.MODEL_PATH
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("🧹 Cleaning up visual monitor...")
        self.stop_monitoring()
        self.latest_detections.clear()
        self.detection_history.clear()
        self.logger.info("✅ Cleanup complete")


# Export main class
__all__ = ['EnhancedVisualMonitor']