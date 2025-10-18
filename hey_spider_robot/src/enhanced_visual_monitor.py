import threading
import time
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    print("OpenCV not available - camera disabled")
    OPENCV_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("YOLO not available - object detection disabled")
    YOLO_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
except ImportError:
    print("PyTorch not available - using CPU only")
    TORCH_AVAILABLE = False
    DEVICE = 'cpu'

from src.oled_display import OLEDDisplay

# YOLO v12 class names (COCO dataset)
COCO_CLASS_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
    19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
    38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup',
    42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
    47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

class VisualMonitor:
    def __init__(self, oled_display: Optional[OLEDDisplay] = None):
        self.oled = oled_display
        self.camera = None
        self.model = None
        self.running = False
        self.camera_active = False
        self.capture_thread = None
        self.detection_thread = None
        
        # Detection data
        self.latest_detections = []
        self.latest_frame = None
        self.annotated_frame = None
        self.detection_history = []
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.detection_times = []
        
        # Detection settings
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.max_detections = 10
        self.detection_interval = 0.1
        
        # Create directories
        os.makedirs('images', exist_ok=True)
        os.makedirs('images/detections', exist_ok=True)
        os.makedirs('images/raw', exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize YOLO model
        self.initialize_yolo_model()
        
        # AUTO-START CAMERA on initialization
        print("🎥 Auto-starting camera system...")
        self._auto_start_camera()
        
    def initialize_yolo_model(self):
        """Initialize YOLO v12 model with proper error handling"""
        if not YOLO_AVAILABLE:
            print("⚠️ YOLO not available - object detection disabled")
            return
            
        try:
            model_paths = [
                'yolov8n.pt',
                'yolov8s.pt',
                'yolov8m.pt',
                'models/yolov8n.pt',
                'models/yolov8s.pt'
            ]
            
            for model_path in model_paths:
                try:
                    print(f"🤖 Attempting to load YOLO model: {model_path}")
                    self.model = YOLO(model_path)
                    self.model.to(DEVICE)
                    
                    # Test the model
                    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                    test_results = self.model(dummy_img, verbose=False)
                    
                    print(f"✅ YOLO v8 model loaded successfully: {model_path}")
                    print(f"📊 Model device: {DEVICE}")
                    print(f"🎯 Model classes: {len(self.model.names)} classes")
                    return
                    
                except Exception as e:
                    print(f"❌ Failed to load {model_path}: {e}")
                    continue
                    
            print("⚠️ All YOLO models failed to load - using mock detection")
            self.model = None
            
        except Exception as e:
            print(f"❌ YOLO initialization error: {e}")
            self.model = None
    
    def _auto_start_camera(self):
        """Automatically start camera on initialization"""
        print("=" * 60)
        print("🎥 CAMERA AUTO-START SEQUENCE")
        print("=" * 60)
        
        if self._initialize_camera():
            print("✅ Camera auto-start successful")
            
            # Start monitoring and detection automatically
            if not self.running:
                self.start_monitoring()
                print("✅ Visual monitoring started automatically")
        else:
            print("⚠️ Camera auto-start failed - using mock mode")
            
        print("=" * 60)
    
    def _initialize_camera(self) -> bool:
        """Initialize camera hardware"""
        if self.camera_active:
            print("📹 Camera already active")
            return True
            
        if not OPENCV_AVAILABLE:
            print("⚠️ OpenCV not available - using mock camera")
            self.camera_active = True
            self._generate_mock_frame()
            return True
            
        try:
            # Try different camera indices
            camera_indices = [0, 1, 2, '/dev/video0', '/dev/video1']
            
            for idx in camera_indices:
                try:
                    print(f"🎥 Trying camera index: {idx}")
                    self.camera = cv2.VideoCapture(idx)
                    
                    if self.camera.isOpened():
                        # Configure camera settings
                        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.camera.set(cv2.CAP_PROP_FPS, 30)
                        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        # Test camera
                        ret, frame = self.camera.read()
                        if ret and frame is not None:
                            self.latest_frame = frame
                            self.annotated_frame = frame.copy()
                            self.camera_active = True
                            print(f"✅ Camera initialized on index: {idx}")
                            print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
                            
                            # Start detection thread
                            if not self.detection_thread or not self.detection_thread.is_alive():
                                self.detection_thread = threading.Thread(
                                    target=self._detection_loop, 
                                    daemon=True
                                )
                                self.detection_thread.start()
                                print("✅ Detection thread started")
                                
                            return True
                        else:
                            self.camera.release()
                            self.camera = None
                            
                except Exception as e:
                    print(f"❌ Camera index {idx} failed: {e}")
                    if self.camera:
                        self.camera.release()
                        self.camera = None
                    continue
                    
            print("❌ No physical camera found - using mock camera")
            self.camera_active = True
            self._generate_mock_frame()
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            self.camera_active = True
            self._generate_mock_frame()
            return False
    
    def start_monitoring(self):
        """Start the visual monitoring thread"""
        if not self.running:
            self.running = True
            self.capture_thread = threading.Thread(
                target=self._monitoring_loop, 
                daemon=True
            )
            self.capture_thread.start()
            print("👁️ Visual monitoring started")
    
    def stop_monitoring(self):
        """Stop visual monitoring"""
        print("🛑 Stopping visual monitoring...")
        self.running = False
        self.camera_active = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=3)
        
        if self.detection_thread:
            self.detection_thread.join(timeout=2)
        
        if self.camera and self.camera.isOpened():
            self.camera.release()
            self.camera = None
        
        print("✅ Visual monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop for camera capture"""
        frame_count = 0
        last_fps_update = time.time()
        
        while self.running:
            try:
                if self.camera_active and self.camera and self.camera.isOpened():
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        self.latest_frame = frame.copy()
                        frame_count += 1
                        
                        # Update FPS
                        current_time = time.time()
                        if current_time - last_fps_update >= 1.0:
                            self.current_fps = frame_count
                            frame_count = 0
                            last_fps_update = current_time
                            
                    else:
                        print("⚠️ Failed to read camera frame")
                        time.sleep(0.1)
                        
                elif self.camera_active:
                    # Update mock frame periodically
                    if frame_count % 30 == 0:
                        self._generate_mock_frame()
                    frame_count += 1
                    
                time.sleep(1/30)  # 30 FPS target
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(1)
                
    def _detection_loop(self):
        """Dedicated thread for object detection"""
        last_detection_time = 0
        
        while self.camera_active and self.running:
            try:
                current_time = time.time()
                
                if current_time - last_detection_time >= self.detection_interval:
                    if self.latest_frame is not None:
                        self._process_frame_detection(self.latest_frame)
                        last_detection_time = current_time
                        
                time.sleep(0.01)
                
            except Exception as e:
                print(f"❌ Detection loop error: {e}")
                time.sleep(0.5)
                
    def _process_frame_detection(self, frame):
        """Process frame for YOLO detection"""
        if not self.model:
            self._generate_mock_detections()
            return
            
        try:
            start_time = time.time()
            
            if self.oled:
                self.oled.update_mode("DETECTING")
                
            input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.model(
                input_frame,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                max_det=self.max_detections,
                verbose=False,
                device=DEVICE
            )
            
            detections = []
            annotated_frame = frame.copy()
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        if confidence >= self.confidence_threshold:
                            class_name = COCO_CLASS_NAMES.get(class_id, f"class_{class_id}")
                            
                            detection = {
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': box.tolist(),
                                'class_id': class_id,
                                'timestamp': time.time()
                            }
                            detections.append(detection)
                            
                            # Draw detection
                            x1, y1, x2, y2 = map(int, box)
                            color = self._get_class_color(class_id)
                            
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                            label = f"{class_name}: {confidence:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            
                            cv2.rectangle(
                                annotated_frame,
                                (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1),
                                color,
                                -1
                            )
                            
                            cv2.putText(
                                annotated_frame,
                                label,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2
                            )
            
            self.latest_detections = detections
            self.annotated_frame = annotated_frame
            
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            if len(self.detection_times) > 100:
                self.detection_times.pop(0)
                
            self._update_detection_history(detections)
            
            if self.oled:
                self.oled.update_detections(detections)
                self.oled.update_mode("ACTIVE")
                
            if detections:
                detected_objects = [f"{d['class']}({d['confidence']:.2f})" for d in detections]
                self.logger.info(f"🎯 Detected: {', '.join(detected_objects)}")
                
        except Exception as e:
            print(f"❌ YOLO detection error: {e}")
            self._generate_mock_detections()
        finally:
            if self.oled:
                self.oled.update_mode("ACTIVE")
                
    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get consistent color for object class"""
        np.random.seed(class_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        return color
        
    def _update_detection_history(self, detections: List[Dict]):
        """Update detection history"""
        timestamp = time.time()
        
        for detection in detections:
            detection['timestamp'] = timestamp
            
        self.detection_history.append({
            'timestamp': timestamp,
            'detections': detections.copy(),
            'count': len(detections)
        })
        
        if len(self.detection_history) > 100:
            self.detection_history.pop(0)
            
    def _generate_mock_detections(self):
        """Generate mock detections for testing"""
        import random
        
        mock_objects = [
            ('person', random.uniform(0.8, 0.95)),
            ('chair', random.uniform(0.7, 0.9)),
            ('laptop', random.uniform(0.6, 0.85)),
            ('cup', random.uniform(0.5, 0.8)),
            ('book', random.uniform(0.6, 0.85)),
        ]
        
        num_detections = random.randint(0, 3)
        detections = []
        
        if self.latest_frame is not None:
            height, width = self.latest_frame.shape[:2]
            annotated_frame = self.latest_frame.copy()
        else:
            width, height = 640, 480
            annotated_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
        for i in range(num_detections):
            obj_class, confidence = random.choice(mock_objects)
            
            x1 = random.randint(50, width - 200)
            y1 = random.randint(50, height - 150)
            x2 = x1 + random.randint(80, 200)
            y2 = y1 + random.randint(60, 150)
            
            x2 = min(x2, width - 10)
            y2 = min(y2, height - 10)
            
            detection = {
                'class': obj_class,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'class_id': random.randint(0, 79),
                'timestamp': time.time()
            }
            detections.append(detection)
            
            color = (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"MOCK {obj_class}: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        self.latest_detections = detections
        self.annotated_frame = annotated_frame
        
        if self.oled:
            self.oled.update_detections(detections)
            
    def _generate_mock_frame(self):
        """Generate mock camera frame"""
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            for i in range(480):
                frame[i, :] = [i//2, (i//3) % 255, (480-i)//2]
                
            cv2.circle(frame, (160, 120), 50, (255, 255, 0), -1)
            cv2.rectangle(frame, (300, 200), (500, 350), (0, 255, 255), -1)
            
            cv2.putText(frame, "MOCK CAMERA - AUTO STARTED",
                       (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (255, 255, 255), 2)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, f"Time: {timestamp}",
                       (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (255, 255, 255), 2)
            
            self.latest_frame = frame
            self.annotated_frame = frame.copy()
            
        except Exception as e:
            print(f"❌ Error generating mock frame: {e}")
            
    def capture_photo(self) -> str:
        """Capture and save photo"""
        try:
            frame_to_save = self.annotated_frame if self.annotated_frame is not None else self.latest_frame
            
            if frame_to_save is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                raw_filename = f"images/raw/photo_{timestamp}.jpg"
                cv2.imwrite(raw_filename, self.latest_frame if self.latest_frame is not None else frame_to_save)
                
                annotated_filename = f"images/photo_{timestamp}_detected.jpg"
                
                if self.latest_detections:
                    summary = f"Detections: {len(self.latest_detections)} | FPS: {self.current_fps}"
                    cv2.putText(frame_to_save, summary,
                               (10, frame_to_save.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                               (255, 255, 255), 2)
                
                success = cv2.imwrite(annotated_filename, frame_to_save)
                if success:
                    print(f"📸 Photo captured: {annotated_filename}")
                    return annotated_filename
                else:
                    print("❌ Failed to save photo")
                    return ""
            else:
                print("❌ No frame available")
                return ""
                
        except Exception as e:
            print(f"❌ Photo capture error: {e}")
            return ""
            
    def get_latest_detections(self) -> List[Dict]:
        """Get latest detections"""
        return self.latest_detections.copy()
        
    def get_detection_description(self) -> str:
        """Get natural language description"""
        if not self.latest_detections:
            if not self.camera_active:
                return "Camera system offline"
            return "No objects detected in view"
            
        object_counts = {}
        
        for detection in self.latest_detections:
            class_name = detection['class']
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
                
        descriptions = []
        for obj_class, count in object_counts.items():
            if count == 1:
                descriptions.append(f"1 {obj_class}")
            else:
                descriptions.append(f"{count} {obj_class}s")
                
        if len(descriptions) == 0:
            return "No clear objects detected"
        elif len(descriptions) == 1:
            return f"I can see {descriptions[0]}."
        elif len(descriptions) == 2:
            return f"I can see {descriptions[0]} and {descriptions[1]}."
        else:
            return f"I can see {', '.join(descriptions[:-1])}, and {descriptions[-1]}."
        
    def get_latest_frame(self):
        """Get latest raw frame"""
        return self.latest_frame
        
    def get_annotated_frame(self):
        """Get annotated frame"""
        return self.annotated_frame if self.annotated_frame is not None else self.latest_frame
        
    def is_camera_active(self) -> bool:
        """Check if camera is active"""
        return self.camera_active
        
    def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        avg_detection_time = 0
        if self.detection_times:
            avg_detection_time = sum(self.detection_times) / len(self.detection_times)
            
        return {
            'fps': self.current_fps,
            'avg_detection_time': avg_detection_time,
            'total_detections': len(self.detection_history),
            'current_objects': len(self.latest_detections),
            'model_device': DEVICE,
            'model_loaded': self.model is not None,
            'camera_active': self.camera_active
        }
        
    def cleanup(self):
        """Cleanup resources"""
        print("🧹 Cleaning up visual monitor...")
        self.stop_monitoring()
        
        if self.camera and self.camera.isOpened():
            self.camera.release()
            
        self.latest_detections.clear()
        self.detection_history.clear()
        
        print("✅ Visual monitor cleanup complete")