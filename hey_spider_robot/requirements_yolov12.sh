#!/bin/bash
echo "================================================================"
echo "🕷️  Hey Spider Robot - YOLOv12 + OV5647 Real-Time Installation"
echo "================================================================"
echo "Installing YOLOv12 with OV5647 Camera and Night Vision"
echo ""

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_section() { echo -e "\n${BLUE}━━━ $1 ━━━${NC}"; }

# Check Raspberry Pi
print_section "System Check"
if grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    PI_MODEL=$(tr -d '\0' < /proc/device-tree/model)
    print_status "Detected: $PI_MODEL"
else
    print_warning "Not a Raspberry Pi - some features may not work"
fi

# Check Python
PYTHON_VER=$(python3 --version | grep -oP '\d+\.\d+')
if python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"; then
    print_status "Python $PYTHON_VER - OK"
else
    print_error "Python 3.8+ required"
    exit 1
fi

# Check RAM
TOTAL_RAM=$(free -m | awk '/^Mem:/{print $2}')
AVAIL_RAM=$(free -m | awk '/^Mem:/{print $7}')
print_status "RAM: ${AVAIL_RAM}MB available / ${TOTAL_RAM}MB total"

if [ "$TOTAL_RAM" -lt 2000 ]; then
    print_warning "Low RAM - consider swap file for YOLOv12"
fi

# System updates
print_section "System Updates"
print_status "Updating package lists..."
sudo apt update

print_status "Upgrading system packages..."
sudo apt upgrade -y

# Essential dependencies
print_section "Installing Dependencies"

print_status "Installing build tools..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget

print_status "Installing OpenCV dependencies..."
sudo apt install -y \
    libopencv-dev \
    python3-opencv \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    libtbb2 \
    libtbb-dev

print_status "Installing camera support..."
sudo apt install -y \
    python3-picamera2 \
    libcamera-dev \
    libcamera-apps

# Audio for voice
print_status "Installing audio support..."
sudo apt install -y \
    portaudio19-dev \
    python3-pyaudio \
    espeak \
    espeak-data

# I2C and GPIO
print_status "Installing hardware interfaces..."
sudo apt install -y \
    i2c-tools \
    python3-smbus \
    python3-rpi.gpio

# Enable hardware interfaces
print_section "Hardware Configuration"

if command -v raspi-config >/dev/null 2>&1; then
    print_status "Enabling I2C..."
    sudo raspi-config nonint do_i2c 0
    
    print_status "Enabling camera..."
    sudo raspi-config nonint do_camera 0
    
    print_status "Enabling SPI..."
    sudo raspi-config nonint do_spi 0
    
    # Increase GPU memory for camera
    if ! grep -q "gpu_mem=256" /boot/config.txt; then
        echo "gpu_mem=256" | sudo tee -a /boot/config.txt
        print_status "GPU memory set to 256MB"
    fi
    
    # Optimize camera settings
    if ! grep -q "start_x=1" /boot/config.txt; then
        echo "start_x=1" | sudo tee -a /boot/config.txt
        print_status "Camera start enabled"
    fi
fi

# Create project structure
print_section "Project Setup"
print_status "Creating directories..."
mkdir -p {config,src,models,images/{raw,detections,night_vision},logs,data}
touch config/__init__.py src/__init__.py

# Python environment
print_section "Python Environment"
print_status "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch
print_section "PyTorch Installation"
print_status "Installing PyTorch (CPU optimized for Pi)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Ultralytics YOLO
print_section "YOLOv12 Installation"
print_status "Installing Ultralytics (YOLOv12 support)..."
pip install ultralytics --upgrade

# Core dependencies
print_status "Installing Python packages..."
cat > requirements_yolo.txt << 'EOF'
# Core ML & Vision
ultralytics>=8.1.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0

# Camera
picamera2>=0.3.12
libcamera>=0.1.0

# Hardware
adafruit-circuitpython-pca9685
adafruit-circuitpython-servokit
adafruit-circuitpython-ssd1306
RPi.GPIO
gpiozero

# Audio & Voice
SpeechRecognition
pyaudio
pydub

# AI & API
openai>=1.3.0

# Web Interface
Flask>=3.0.0
Flask-SocketIO>=5.3.0
python-socketio>=5.8.0

# Utilities
python-dotenv
psutil
tqdm
loguru

# Performance
imutils
EOF

pip install -r requirements_yolo.txt

# Download YOLO models
print_section "Downloading YOLO Models"

python3 << 'EOFDL'
import os
from ultralytics import YOLO

os.makedirs('models', exist_ok=True)

models = {
    'yolov8n.pt': 'YOLOv8 Nano (fastest)',
    'yolov8s.pt': 'YOLOv8 Small (balanced)',
    'yolo11n.pt': 'YOLOv11 Nano',
}

for model_name, desc in models.items():
    try:
        print(f"\n📥 Downloading {desc}...")
        model = YOLO(model_name)
        
        # Move to models directory
        if os.path.exists(model_name):
            os.rename(model_name, f'models/{model_name}')
            print(f"✅ {model_name} ready")
        
        # Test model
        import numpy as np
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(test_img, verbose=False)
        print(f"✓ Model test passed - {len(model.names)} classes")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

print("\n✅ YOLO models ready!")
EOFDL

# Test camera
print_section "Camera System Test"
print_status "Testing OV5647 camera..."

python3 << 'EOFCAM'
import sys
try:
    from picamera2 import Picamera2
    import time
    
    print("Initializing PiCamera2...")
    camera = Picamera2()
    
    config = camera.create_still_configuration(
        main={"size": (1640, 1232), "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()
    
    time.sleep(2)
    
    print("Capturing test image...")
    frame = camera.capture_array()
    
    print(f"✅ Camera working! Resolution: {frame.shape}")
    print(f"   Image size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Test night vision capability
    import numpy as np
    brightness = np.mean(frame)
    print(f"   Average brightness: {brightness:.1f}/255")
    
    if brightness < 50:
        print("   🌙 Low light detected - night vision recommended")
    
    camera.stop()
    camera.close()
    
except ImportError:
    print("⚠️  PiCamera2 not available - trying OpenCV...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ OpenCV camera working! Resolution: {frame.shape}")
            cap.release()
        else:
            print("❌ No camera detected")
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Camera error: {e}")
    sys.exit(1)
EOFCAM

# Performance optimization
print_section "Performance Optimization"

# Swap file for low RAM systems
if [ "$TOTAL_RAM" -lt 4000 ]; then
    print_status "Setting up swap file for better performance..."
    
    if [ ! -f /swapfile ]; then
        sudo fallocate -l 2G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        print_status "2GB swap file created"
    else
        print_warning "Swap file already exists"
    fi
fi

# CPU governor
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true
    print_status "CPU set to performance mode"
fi

# User permissions
print_section "User Permissions"
print_status "Adding user to hardware groups..."
sudo usermod -a -G video,i2c,spi,gpio $USER

# Create test script
print_section "Creating Test Scripts"

cat > test_yolo_camera.py << 'EOFTEST'
#!/usr/bin/env python3
"""Test YOLOv12 with OV5647 Camera"""

import cv2
import time
import numpy as np
from ultralytics import YOLO

def test_realtime_detection():
    print("🎥 Testing Real-Time YOLOv12 Detection with OV5647")
    print("=" * 60)
    
    # Load model
    print("Loading YOLOv12 model...")
    try:
        model = YOLO('models/yolov8n.pt')
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return
    
    # Initialize camera
    print("\nInitializing camera...")
    try:
        from picamera2 import Picamera2
        camera = Picamera2()
        config = camera.create_still_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        camera.configure(config)
        camera.start()
        time.sleep(2)
        use_picamera = True
        print("✅ PiCamera2 initialized")
    except:
        print("PiCamera2 not available, using OpenCV...")
        camera = cv2.VideoCapture(0)
        use_picamera = False
        if not camera.isOpened():
            print("❌ No camera found")
            return
        print("✅ OpenCV camera initialized")
    
    # Detection loop
    print("\n🎯 Starting real-time detection (Press 'q' to quit)...")
    print("=" * 60)
    
    frame_count = 0
    start_time = time.time()
    detection_times = []
    
    try:
        while True:
            # Capture frame
            if use_picamera:
                frame = camera.capture_array()
            else:
                ret, frame = camera.read()
                if not ret:
                    break
            
            # Run detection
            det_start = time.time()
            results = model(frame, conf=0.45, verbose=False)
            det_time = time.time() - det_start
            detection_times.append(det_time)
            
            # Process results
            detections = 0
            for result in results:
                if hasattr(result, 'boxes'):
                    detections = len(result.boxes)
                    
                    # Draw boxes
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{model.names[cls]} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            avg_det_time = np.mean(detection_times[-30:]) if detection_times else 0
            
            # Display info
            info = f"FPS: {fps:.1f} | Det: {avg_det_time*1000:.1f}ms | Objects: {detections}"
            cv2.putText(frame, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show frame
            cv2.imshow('YOLOv12 Real-Time Detection', frame)
            
            # Stats every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: {fps:.1f} FPS, "
                      f"{avg_det_time*1000:.1f}ms detection, "
                      f"{detections} objects")
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        # Cleanup
        if use_picamera:
            camera.stop()
            camera.close()
        else:
            camera.release()
        cv2.destroyAllWindows()
        
        # Final stats
        print("\n" + "=" * 60)
        print("📊 Test Results:")
        print(f"   Total frames: {frame_count}")
        print(f"   Average FPS: {fps:.2f}")
        print(f"   Avg detection time: {np.mean(detection_times)*1000:.2f}ms")
        print(f"   Min/Max detection: {np.min(detection_times)*1000:.2f}ms / {np.max(detection_times)*1000:.2f}ms")
        print("=" * 60)

if __name__ == "__main__":
    test_realtime_detection()
EOFTEST

chmod +x test_yolo_camera.py
print_status "Test script created: ./test_yolo_camera.py"

# Create night vision test script
cat > test_night_vision.py << 'EOFNIGHT'
#!/usr/bin/env python3
"""Test OV5647 Night Vision Capabilities"""

import time
import numpy as np
from picamera2 import Picamera2
from libcamera import controls

def test_night_vision():
    print("🌙 Testing OV5647 Night Vision Mode")
    print("=" * 60)
    
    try:
        camera = Picamera2()
        config = camera.create_still_configuration(
            main={"size": (1640, 1232), "format": "RGB888"}
        )
        camera.configure(config)
        camera.start()
        time.sleep(2)
        
        # Test different lighting conditions
        modes = [
            ("Day Mode", 10000, 2.0, 0.0, 1.0),
            ("Low Light", 20000, 4.0, 0.2, 1.3),
            ("Night Mode", 33333, 8.0, 0.3, 1.4),
            ("Extreme Night", 50000, 12.0, 0.4, 1.5),
        ]
        
        for mode_name, exposure, gain, brightness, contrast in modes:
            print(f"\n📷 Testing {mode_name}:")
            print(f"   Exposure: {exposure}µs, Gain: {gain}x")
            
            # Apply settings
            camera.set_controls({
                "ExposureTime": exposure,
                "AnalogueGain": gain,
                "Brightness": brightness,
                "Contrast": contrast,
            })
            
            time.sleep(1)  # Let camera adjust
            
            # Capture and analyze
            frame = camera.capture_array()
            avg_brightness = np.mean(frame)
            std_brightness = np.std(frame)
            
            print(f"   Brightness: {avg_brightness:.1f}/255")
            print(f"   Contrast: {std_brightness:.1f}")
            
            # Save sample
            import cv2
            filename = f"test_{mode_name.lower().replace(' ', '_')}.jpg"
            cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"   ✓ Saved: {filename}")
        
        camera.stop()
        camera.close()
        
        print("\n✅ Night vision test complete!")
        print("Check the saved images to verify quality.")
        
    except Exception as e:
        print(f"❌ Night vision test failed: {e}")

if __name__ == "__main__":
    test_night_vision()
EOFNIGHT

chmod +x test_night_vision.py
print_status "Night vision test created: ./test_night_vision.py"

# Create performance monitor
cat > monitor_performance.py << 'EOFMON'
#!/usr/bin/env python3
"""Monitor System Performance for YOLO Detection"""

import psutil
import time
import subprocess

def monitor_system():
    print("📊 System Performance Monitor")
    print("=" * 60)
    
    try:
        while True:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1, percpu=False)
            cpu_temp = "N/A"
            try:
                temp_result = subprocess.run(
                    ['vcgencmd', 'measure_temp'],
                    capture_output=True, text=True
                )
                if temp_result.returncode == 0:
                    cpu_temp = temp_result.stdout.strip().replace('temp=', '').replace("'C", '°C')
            except:
                pass
            
            # Memory
            mem = psutil.virtual_memory()
            mem_used = mem.used / (1024**3)
            mem_total = mem.total / (1024**3)
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_used = disk.used / (1024**3)
            disk_total = disk.total / (1024**3)
            
            # Display
            print(f"\r CPU: {cpu_percent:5.1f}% | Temp: {cpu_temp:>6} | "
                  f"RAM: {mem_used:.1f}/{mem_total:.1f}GB ({mem.percent:.1f}%) | "
                  f"Disk: {disk_used:.1f}/{disk_total:.1f}GB ({disk.percent:.1f}%)", 
                  end='', flush=True)
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")

if __name__ == "__main__":
    monitor_system()
EOFMON

chmod +x monitor_performance.py
print_status "Performance monitor created: ./monitor_performance.py"

# Create systemd service
print_section "System Service (Optional)"
read -p "Install as systemd service for auto-start? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    cat > hey-spider-yolo.service << EOF
[Unit]
Description=Hey Spider Robot with YOLOv12 and OV5647
After=network.target multi-user.target
Wants=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
EnvironmentFile=-$(pwd)/.env
ExecStart=$(pwd)/venv/bin/python $(pwd)/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=$(pwd)

# Hardware access
SupplementaryGroups=gpio i2c spi video audio

[Install]
WantedBy=multi-user.target
EOF

    sudo cp hey-spider-yolo.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable hey-spider-yolo.service
    print_status "Systemd service installed and enabled"
    print_status "Control with: sudo systemctl start/stop/status hey-spider-yolo"
fi

# Summary
print_section "Installation Complete! 🎉"
echo ""
echo "📋 Summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ YOLOv12 models installed"
echo "  ✅ OV5647 camera configured"
echo "  ✅ Night vision support enabled"
echo "  ✅ Real-time detection ready"
echo "  ✅ Python environment configured"
echo ""
echo "📁 Project Structure:"
echo "  • config/          - Configuration files"
echo "  • src/             - Source code"
echo "  • models/          - YOLO model files"
echo "  • images/          - Captured images"
echo "    ├── raw/         - Raw captures"
echo "    ├── detections/  - Annotated detections"
echo "    └── night_vision/ - Night mode captures"
echo ""
echo "🧪 Test Commands:"
echo "  • ./test_yolo_camera.py     - Test real-time detection"
echo "  • ./test_night_vision.py    - Test night vision modes"
echo "  • ./monitor_performance.py  - Monitor system performance"
echo ""
echo "🚀 Quick Start:"
echo "  1. Activate environment:  source venv/bin/activate"
echo "  2. Test camera:          ./test_yolo_camera.py"
echo "  3. Test night vision:    ./test_night_vision.py"
echo "  4. Run main app:         python main.py"
echo ""
echo "⚙️  Configuration:"
echo "  • Edit config/yolo_v12_config.py for YOLO settings"
echo "  • Adjust camera settings in OV5647CameraConfig"
echo "  • Set OpenAI API key in .env file"
echo ""
echo "🌙 Night Vision Features:"
echo "  • Auto night mode switching"
echo "  • Adjustable exposure (10-50ms)"
echo "  • Gain control (1-16x)"
echo "  • Enhanced low-light detection"
echo ""
echo "📊 Performance Tips:"
echo "  • Use 'balanced' preset for best FPS/accuracy"
echo "  • Lower resolution for faster detection"
echo "  • Enable half-precision for speed boost"
echo "  • Monitor CPU/RAM with monitor_performance.py"
echo ""
echo "🔧 Optimization Options:"
echo "  • Model: nano(fastest), small(balanced), medium(accurate)"
echo "  • Resolution: 320x320(fast), 640x640(balanced), 1280x1280(quality)"
echo "  • Confidence: 0.3-0.7 (lower=more detections, higher=more accurate)"
echo ""

# Check for reboot requirement
if [ -f /var/run/reboot-required ]; then
    print_warning "System reboot required for hardware changes"
    echo ""
    read -p "Reboot now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Rebooting..."
        sudo reboot
    else
        print_warning "Please reboot manually: sudo reboot"
    fi
else
    print_status "No reboot required"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🕷️  Ready to detect with Hey Spider Robot!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "" "