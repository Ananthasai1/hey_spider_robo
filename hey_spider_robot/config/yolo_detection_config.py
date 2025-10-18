# YOLO v8/v12 Object Detection Configuration
# Enhanced configuration for optimal object detection performance

import os
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class YOLOConfig:
    """YOLO model configuration settings"""
    
    # Model Selection
    MODEL_SIZE: str = "n"  # n(nano), s(small), m(medium), l(large), x(extra-large)
    MODEL_PATH: str = f"yolov8{MODEL_SIZE}.pt"
    CUSTOM_MODEL_PATH: str = "models/custom_spider_objects.pt"
    USE_CUSTOM_MODEL: bool = False
    
    # Performance Settings
    DEVICE: str = "auto"  # auto, cpu, cuda, mps
    HALF_PRECISION: bool = True  # Use FP16 for faster inference
    BATCH_SIZE: int = 1
    IMG_SIZE: int = 640  # Input image size
    
    # Detection Thresholds
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.4  # Non-Maximum Suppression threshold
    MAX_DETECTIONS: int = 20
    
    # Processing Settings
    DETECTION_INTERVAL: float = 0.1  # Process every 0.1 seconds
    SAVE_DETECTION_IMAGES: bool = True
    SAVE_RAW_IMAGES: bool = False
    
    # Filtering Settings
    MIN_DETECTION_SIZE: int = 32  # Minimum bounding box size
    MAX_DETECTION_SIZE: int = 1000  # Maximum bounding box size
    ASPECT_RATIO_FILTER: bool = True
    MIN_ASPECT_RATIO: float = 0.2
    MAX_ASPECT_RATIO: float = 5.0
    
    # Classes of Interest (COCO dataset indices)
    PRIORITY_CLASSES: List[int] = None  # None = all classes
    IGNORE_CLASSES: List[int] = None    # Classes to ignore
    
    # Performance Monitoring
    ENABLE_PERFORMANCE_TRACKING: bool = True
    LOG_DETECTION_STATS: bool = True
    FPS_SMOOTHING_FACTOR: float = 0.9
    
    def __post_init__(self):
        if self.PRIORITY_CLASSES is None:
            # Default priority classes for indoor robot navigation
            self.PRIORITY_CLASSES = [
                0,   # person
                56,  # chair
                57,  # couch
                58,  # potted plant
                59,  # bed
                60,  # dining table
                62,  # tv
                63,  # laptop
                64,  # mouse
                65,  # remote
                66,  # keyboard
                67,  # cell phone
                39,  # bottle
                41,  # cup
                73,  # book
                74,  # clock
            ]
        
        if self.IGNORE_CLASSES is None:
            # Classes that are less relevant for indoor navigation
            self.IGNORE_CLASSES = [
                1, 2, 3, 4, 5, 6, 7, 8,  # Vehicles
                9, 10, 11, 12,           # Traffic items
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  # Animals
            ]

# COCO Class Names with Enhanced Descriptions
ENHANCED_CLASS_INFO = {
    0: {'name': 'person', 'category': 'people', 'priority': 'high', 'color': (255, 0, 0)},
    1: {'name': 'bicycle', 'category': 'vehicle', 'priority': 'low', 'color': (0, 255, 0)},
    2: {'name': 'car', 'category': 'vehicle', 'priority': 'medium', 'color': (0, 0, 255)},
    # ... (additional classes would be defined here)
    39: {'name': 'bottle', 'category': 'object', 'priority': 'medium', 'color': (255, 255, 0)},
    41: {'name': 'cup', 'category': 'object', 'priority': 'medium', 'color': (255, 0, 255)},
    56: {'name': 'chair', 'category': 'furniture', 'priority': 'high', 'color': (0, 255, 255)},
    57: {'name': 'couch', 'category': 'furniture', 'priority': 'high', 'color': (128, 0, 128)},
    58: {'name': 'potted plant', 'category': 'decoration', 'priority': 'medium', 'color': (0, 128, 0)},
    59: {'name': 'bed', 'category': 'furniture', 'priority': 'high', 'color': (128, 128, 0)},
    60: {'name': 'dining table', 'category': 'furniture', 'priority': 'high', 'color': (128, 0, 0)},
    62: {'name': 'tv', 'category': 'electronics', 'priority': 'medium', 'color': (0, 128, 128)},
    63: {'name': 'laptop', 'category': 'electronics', 'priority': 'medium', 'color': (255, 128, 0)},
    64: {'name': 'mouse', 'category': 'electronics', 'priority': 'low', 'color': (128, 255, 0)},
    65: {'name': 'remote', 'category': 'electronics', 'priority': 'low', 'color': (0, 255, 128)},
    66: {'name': 'keyboard', 'category': 'electronics', 'priority': 'low', 'color': (128, 0, 255)},
    67: {'name': 'cell phone', 'category': 'electronics', 'priority': 'medium', 'color': (255, 0, 128)},
    73: {'name': 'book', 'category': 'object', 'priority': 'low', 'color': (64, 128, 255)},
    74: {'name': 'clock', 'category': 'object', 'priority': 'low', 'color': (255, 64, 128)},
}

# Detection Quality Settings
DETECTION_QUALITY = {
    'high_performance': {
        'model_size': 'n',
        'img_size': 416,
        'confidence': 0.6,
        'iou': 0.5,
        'half_precision': True,
        'detection_interval': 0.05
    },
    'balanced': {
        'model_size': 's',
        'img_size': 640,
        'confidence': 0.5,
        'iou': 0.4,
        'half_precision': True,
        'detection_interval': 0.1
    },
    'high_accuracy': {
        'model_size': 'm',
        'img_size': 832,
        'confidence': 0.4,
        'iou': 0.3,
        'half_precision': False,
        'detection_interval': 0.2
    }
}

# Auto Mode Integration Settings
AUTO_MODE_DETECTION = {
    'obstacle_classes': [0, 56, 57, 58, 59, 60],  # People and furniture
    'safe_distance_threshold': 1.5,  # meters
    'detection_confidence_for_avoidance': 0.7,
    'track_moving_objects': True,
    'predict_movement': False  # Advanced feature
}

# Camera-Specific Optimizations
CAMERA_OPTIMIZATIONS = {
    'raspberry_pi_camera': {
        'brightness': 55,
        'contrast': 50,
        'saturation': 50,
        'sharpness': 50,
        'auto_white_balance': True,
        'exposure_mode': 'auto'
    },
    'usb_camera': {
        'auto_exposure': True,
        'brightness': 128,
        'contrast': 128,
        'saturation': 128,
        'gain': 64
    }
}

# Performance Monitoring Configuration
PERFORMANCE_CONFIG = {
    'track_fps': True,
    'track_detection_time': True,
    'track_memory_usage': True,
    'log_slow_detections': True,
    'slow_detection_threshold': 0.5,  # seconds
    'performance_log_interval': 60    # seconds
}

# Export configuration
def get_yolo_config(quality_preset: str = 'balanced') -> YOLOConfig:
    """Get YOLO configuration with specified quality preset"""
    config = YOLOConfig()
    
    if quality_preset in DETECTION_QUALITY:
        preset = DETECTION_QUALITY[quality_preset]
        config.MODEL_SIZE = preset['model_size']
        config.MODEL_PATH = f"yolov8{preset['model_size']}.pt"
        config.IMG_SIZE = preset['img_size']
        config.CONFIDENCE_THRESHOLD = preset['confidence']
        config.IOU_THRESHOLD = preset['iou']
        config.HALF_PRECISION = preset['half_precision']
        config.DETECTION_INTERVAL = preset['detection_interval']
    
    return config

def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """Get color for object class with fallback"""
    if class_id in ENHANCED_CLASS_INFO:
        return ENHANCED_CLASS_INFO[class_id]['color']
    else:
        # Generate consistent color based on class ID
        import hashlib
        hash_color = hashlib.md5(str(class_id).encode()).hexdigest()
        r = int(hash_color[0:2], 16)
        g = int(hash_color[2:4], 16)
        b = int(hash_color[4:6], 16)
        return (r, g, b)

def get_class_priority(class_id: int) -> str:
    """Get priority level for object class"""
    if class_id in ENHANCED_CLASS_INFO:
        return ENHANCED_CLASS_INFO[class_id]['priority']
    return 'low'

def is_priority_class(class_id: int, config: YOLOConfig) -> bool:
    """Check if class is in priority list"""
    if config.PRIORITY_CLASSES is None:
        return class_id not in (config.IGNORE_CLASSES or [])
    return class_id in config.PRIORITY_CLASSES

# Model download and management
def ensure_model_downloaded(model_path: str) -> bool:
    """Ensure YOLO model is downloaded and available"""
    try:
        if os.path.exists(model_path):
            return True
            
        # Try to download model using ultralytics
        from ultralytics import YOLO
        model = YOLO(model_path)  # This will auto-download if needed
        return os.path.exists(model_path)
        
    except Exception as e:
        print(f"Error downloading model {model_path}: {e}")
        return False

# Export main configuration
__all__ = [
    'YOLOConfig',
    'ENHANCED_CLASS_INFO', 
    'DETECTION_QUALITY',
    'AUTO_MODE_DETECTION',
    'CAMERA_OPTIMIZATIONS',
    'PERFORMANCE_CONFIG',
    'get_yolo_config',
    'get_class_color',
    'get_class_priority',
    'is_priority_class',
    'ensure_model_downloaded'
]