# YOLO v12 Real-Time Object Detection Configuration
# Optimized for OV5647 Camera Module with Night Vision Support

import os
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class YOLOv12Config:
    """YOLO v12 model configuration with OV5647 camera optimization"""
    
    # Model Selection - YOLOv12
    MODEL_VERSION: str = "12"  # YOLO v12
    MODEL_SIZE: str = "n"  # n(nano), s(small), m(medium), l(large), x(extra-large)
    MODEL_PATH: str = f"yolo12{MODEL_SIZE}.pt"
    CUSTOM_MODEL_PATH: str = "models/yolo12_custom.pt"
    USE_CUSTOM_MODEL: bool = False
    
    # Performance Settings (Optimized for Raspberry Pi)
    DEVICE: str = "auto"  # auto, cpu, cuda, mps
    HALF_PRECISION: bool = True  # FP16 for faster inference on supported devices
    BATCH_SIZE: int = 1
    IMG_SIZE: int = 640  # YOLOv12 supports: 320, 640, 1280
    
    # Real-Time Detection Settings
    CONFIDENCE_THRESHOLD: float = 0.45  # Lower for better detection
    IOU_THRESHOLD: float = 0.45  # Non-Maximum Suppression threshold
    MAX_DETECTIONS: int = 30  # Increased for real-time tracking
    
    # Performance Optimization
    DETECTION_INTERVAL: float = 0.033  # ~30 FPS target
    ENABLE_ASYNC_INFERENCE: bool = True  # Async processing for better FPS
    USE_TRT: bool = False  # TensorRT optimization (if available)
    USE_OPENVINO: bool = False  # OpenVINO optimization (for Intel)
    
    # Real-Time Processing
    ENABLE_TRACKING: bool = True  # Object tracking between frames
    TRACKER_TYPE: str = "bytetrack"  # bytetrack, botsort
    TRACK_PERSIST: bool = True  # Persist tracking across frames
    
    # Filtering Settings
    MIN_DETECTION_SIZE: int = 20  # Pixels - smaller for distance detection
    MAX_DETECTION_SIZE: int = 1200
    ASPECT_RATIO_FILTER: bool = True
    MIN_ASPECT_RATIO: float = 0.15
    MAX_ASPECT_RATIO: float = 6.0
    
    # Priority Classes (COCO dataset - optimized for indoor robotics)
    PRIORITY_CLASSES: List[int] = None
    IGNORE_CLASSES: List[int] = None
    
    # Visualization Settings
    SHOW_LABELS: bool = True
    SHOW_CONFIDENCE: bool = True
    SHOW_BOUNDING_BOXES: bool = True
    BOX_THICKNESS: int = 2
    FONT_SCALE: float = 0.6
    
    # Recording & Logging
    SAVE_DETECTION_IMAGES: bool = True
    SAVE_DETECTION_VIDEO: bool = False
    LOG_DETECTION_STATS: bool = True
    
    def __post_init__(self):
        if self.PRIORITY_CLASSES is None:
            # Enhanced priority classes for robotics
            self.PRIORITY_CLASSES = [
                0,   # person - CRITICAL
                1,   # bicycle
                2,   # car
                14,  # bird
                15,  # cat
                16,  # dog
                39,  # bottle
                41,  # cup
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
                73,  # book
                74,  # clock
            ]
        
        if self.IGNORE_CLASSES is None:
            # Less relevant for indoor navigation
            self.IGNORE_CLASSES = [
                3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  # Vehicles & traffic
            ]


@dataclass  
class OV5647CameraConfig:
    """Configuration for OV5647 Camera Module with Night Vision"""
    
    # Camera Hardware
    CAMERA_TYPE: str = "OV5647"  # Camera sensor model
    USE_PICAMERA2: bool = True  # Use PiCamera2 library (recommended)
    
    # Resolution Settings
    CAPTURE_WIDTH: int = 1640  # OV5647 max: 2592
    CAPTURE_HEIGHT: int = 1232  # OV5647 max: 1944
    DISPLAY_WIDTH: int = 640
    DISPLAY_HEIGHT: int = 480
    
    # Performance Settings
    TARGET_FPS: int = 30  # Target frame rate
    BUFFER_SIZE: int = 2  # Frame buffer size
    
    # Image Quality Settings
    BRIGHTNESS: float = 0.1  # -1.0 to 1.0
    CONTRAST: float = 1.2  # 0.0 to 2.0
    SATURATION: float = 1.0  # 0.0 to 2.0
    SHARPNESS: float = 1.2  # 0.0 to 16.0
    
    # Night Vision Settings
    ENABLE_NIGHT_VISION: bool = True
    AUTO_NIGHT_MODE: bool = True  # Automatically switch based on light level
    NIGHT_MODE_THRESHOLD: int = 30  # Brightness threshold (0-255)
    
    # Night Vision Parameters
    NIGHT_EXPOSURE_TIME: int = 33333  # Microseconds (33ms for low light)
    NIGHT_ANALOG_GAIN: float = 8.0  # Analog gain (1.0-16.0)
    NIGHT_DENOISE: str = "cdn_hq"  # Denoise mode: off, cdn_off, cdn_fast, cdn_hq
    
    # Day Mode Parameters
    DAY_EXPOSURE_TIME: int = 10000  # Microseconds
    DAY_ANALOG_GAIN: float = 2.0
    
    # Auto Exposure/White Balance
    AUTO_EXPOSURE: bool = True
    EXPOSURE_MODE: str = "normal"  # normal, short, long, custom
    AUTO_WHITE_BALANCE: bool = True
    AWB_MODE: str = "auto"  # auto, incandescent, tungsten, fluorescent, daylight, cloudy
    
    # HDR & Advanced Features
    ENABLE_HDR: bool = False  # High Dynamic Range
    ENABLE_DENOISE: bool = True
    DENOISE_MODE: str = "cdn_hq"  # Color Denoise
    
    # IR Filter (if available)
    HAS_IR_CUT_FILTER: bool = False  # Does camera have IR cut filter?
    AUTO_IR_FILTER: bool = True  # Auto switch IR filter for night vision


# COCO Dataset Class Names (80 classes for YOLOv12)
COCO_CLASSES = {
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

# Enhanced Class Information with Colors
CLASS_COLORS = {
    0: (255, 0, 0),      # person - red
    15: (255, 165, 0),   # cat - orange
    16: (255, 255, 0),   # dog - yellow
    39: (0, 255, 255),   # bottle - cyan
    41: (255, 0, 255),   # cup - magenta
    56: (0, 128, 255),   # chair - blue
    57: (128, 0, 255),   # couch - purple
    62: (0, 255, 128),   # tv - green
    63: (255, 128, 0),   # laptop - orange
}

# YOLOv12 Performance Presets
PERFORMANCE_PRESETS = {
    'ultra_fast': {
        'model_size': 'n',
        'img_size': 320,
        'confidence': 0.5,
        'iou': 0.5,
        'half_precision': True,
        'detection_interval': 0.05,
        'enable_tracking': False
    },
    'balanced': {
        'model_size': 'n',
        'img_size': 640,
        'confidence': 0.45,
        'iou': 0.45,
        'half_precision': True,
        'detection_interval': 0.033,
        'enable_tracking': True
    },
    'high_accuracy': {
        'model_size': 's',
        'img_size': 640,
        'confidence': 0.4,
        'iou': 0.4,
        'half_precision': True,
        'detection_interval': 0.066,
        'enable_tracking': True
    },
    'maximum_quality': {
        'model_size': 'm',
        'img_size': 1280,
        'confidence': 0.35,
        'iou': 0.35,
        'half_precision': False,
        'detection_interval': 0.1,
        'enable_tracking': True
    }
}

# Night Vision Optimization
NIGHT_VISION_PRESETS = {
    'low_light': {
        'exposure_time': 20000,
        'analog_gain': 4.0,
        'brightness': 0.2,
        'contrast': 1.3,
        'denoise': 'cdn_hq',
        'confidence_boost': 0.05  # Boost confidence in low light
    },
    'night': {
        'exposure_time': 33333,
        'analog_gain': 8.0,
        'brightness': 0.3,
        'contrast': 1.4,
        'denoise': 'cdn_hq',
        'confidence_boost': 0.1
    },
    'extreme_night': {
        'exposure_time': 50000,
        'analog_gain': 12.0,
        'brightness': 0.4,
        'contrast': 1.5,
        'denoise': 'cdn_hq',
        'confidence_boost': 0.15
    }
}

def get_yolo_config(preset: str = 'balanced') -> YOLOv12Config:
    """Get YOLO v12 configuration with preset"""
    config = YOLOv12Config()
    
    if preset in PERFORMANCE_PRESETS:
        p = PERFORMANCE_PRESETS[preset]
        config.MODEL_SIZE = p['model_size']
        config.MODEL_PATH = f"yolo12{p['model_size']}.pt"
        config.IMG_SIZE = p['img_size']
        config.CONFIDENCE_THRESHOLD = p['confidence']
        config.IOU_THRESHOLD = p['iou']
        config.HALF_PRECISION = p['half_precision']
        config.DETECTION_INTERVAL = p['detection_interval']
        config.ENABLE_TRACKING = p['enable_tracking']
    
    return config

def get_camera_config() -> OV5647CameraConfig:
    """Get OV5647 camera configuration"""
    return OV5647CameraConfig()

def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """Get color for object class"""
    if class_id in CLASS_COLORS:
        return CLASS_COLORS[class_id]
    
    # Generate consistent color for unknown classes
    import hashlib
    h = hashlib.md5(str(class_id).encode()).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (r, g, b)

def ensure_model_downloaded(model_path: str) -> bool:
    """Download YOLOv12 model if not present"""
    try:
        if os.path.exists(model_path):
            return True
        
        from ultralytics import YOLO
        print(f"Downloading YOLOv12 model: {model_path}")
        model = YOLO(model_path)
        return os.path.exists(model_path)
    except Exception as e:
        print(f"Model download error: {e}")
        return False

__all__ = [
    'YOLOv12Config',
    'OV5647CameraConfig',
    'COCO_CLASSES',
    'CLASS_COLORS',
    'PERFORMANCE_PRESETS',
    'NIGHT_VISION_PRESETS',
    'get_yolo_config',
    'get_camera_config',
    'get_class_color',
    'ensure_model_downloaded'
]