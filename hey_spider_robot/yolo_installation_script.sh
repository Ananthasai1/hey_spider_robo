#!/bin/bash
echo "==============================================================="
echo "🕷️  Enhanced Hey Spider Robot - YOLO v8/v12 Installation"
echo "==============================================================="
echo "Installing advanced object detection with YOLO v8 (v12 features)"
echo ""

# Exit on any error
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo -e "${BLUE}[SECTION]${NC} $1"
}

# Check if running on supported system
print_section "System Compatibility Check"
if command -v lsb_release > /dev/null 2>&1; then
    OS_INFO=$(lsb_release -si)
    OS_VERSION=$(lsb_release -sr)
    print_status "Detected OS: $OS_INFO $OS_VERSION"
else
    print_warning "Could not detect OS version"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
REQUIRED_PYTHON="3.8"
if python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"; then
    print_status "Python version: $(python3 --version)"
else
    print_error "Python 3.8+ required. Found: $(python3 --version)"
    exit 1
fi

# Check available memory (YOLO needs sufficient RAM)
AVAILABLE_RAM=$(free -m | awk '/^Mem:/{print $7}')
if [ "$AVAILABLE_RAM" -lt 1000 ]; then
    print_warning "Low available RAM: ${AVAILABLE_RAM}MB. YOLO may run slowly."
else
    print_status "Available RAM: ${AVAILABLE_RAM}MB - Good for YOLO"
fi

# Check for GPU support
print_section "GPU Support Detection"
if command -v nvidia-smi > /dev/null 2>&1; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "GPU detection failed")
    if [ "$GPU_INFO" != "GPU detection failed" ]; then
        print_status "NVIDIA GPU detected: $GPU_INFO"
        USE_GPU=true
    else
        print_warning "NVIDIA drivers found but GPU not accessible"
        USE_GPU=false
    fi
elif lspci | grep -i nvidia > /dev/null 2>&1; then
    print_warning "NVIDIA hardware detected but drivers not installed"
    USE_GPU=false
else
    print_status "No NVIDIA GPU detected - will use CPU"
    USE_GPU=false
fi

# Update system packages
print_section "System Package Updates"
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
print_section "Installing System Dependencies"
print_status "Installing required system packages..."

# Essential build tools
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    git

# OpenCV dependencies
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
    libtbb-dev \
    libdc1394-22-dev

# Audio processing (for voice commands)
sudo apt install -y \
    portaudio19-dev \
    python3-pyaudio \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev

# I2C and GPIO tools
sudo apt install -y \
    i2c-tools \
    python3-smbus

# Enable I2C and camera interfaces
print_section "Hardware Interface Setup"
if command -v raspi-config >/dev/null 2>&1; then
    print_status "Enabling I2C interface..."
    sudo raspi-config nonint do_i2c 0
    
    print_status "Enabling camera interface..."
    sudo raspi-config nonint do_camera 0
    
    print_status "Enabling SPI interface (optional)..."
    sudo raspi-config nonint do_spi 0
else
    print_warning "raspi-config not found - manual hardware setup may be needed"
fi

# Create project directory structure
print_section "Project Directory Setup"
print_status "Creating project directories..."
mkdir -p models data images/{raw,detections,thumbnails} logs backups

# Create Python virtual environment
print_section "Python Environment Setup"
print_status "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install wheel
print_status "Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with appropriate version
print_section "PyTorch Installation"
if [ "$USE_GPU" = true ]; then
    print_status "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    print_status "Installing PyTorch (CPU-only version)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install Ultralytics YOLO v8
print_section "YOLO v8 Installation"
print_status "Installing Ultralytics YOLO v8 (with v12 features)..."
pip install ultralytics

# Install other Python dependencies
print_section "Python Dependencies Installation"
print_status "Installing enhanced Python packages..."
pip install -r requirements.txt

# Download YOLO models
print_section "YOLO Model Download"
print_status "Downloading YOLO v8 models..."

python3 << 'EOF'
try:
    from ultralytics import YOLO
    import os
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Download different model sizes
    models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
    
    for model_name in models:
        try:
            print(f"Downloading {model_name}...")
            model = YOLO(model_name)
            
            # Move to models directory
            if os.path.exists(model_name):
                os.rename(model_name, f'models/{model_name}')
            
            print(f"✅ {model_name} downloaded successfully")
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
    
    # Test YOLO installation
    print("\n🧪 Testing YOLO installation...")
    test_model = YOLO('models/yolov8n.pt')
    print(f"✅ YOLO test successful - {len(test_model.names)} classes available")
    
    # Print some class names
    print(f"Sample classes: {list(test_model.names.values())[:10]}")
    
except Exception as e:
    print(f"❌ YOLO installation test failed: {e}")
    exit(1)
EOF

# Test camera functionality
print_section "Camera System Test"
print_status "Testing camera functionality..."

python3 << 'EOF'
try:
    import cv2
    import numpy as np
    
    # Try to open camera
    for camera_id in [0, 1, 2]:
        try:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Camera {camera_id} working - Resolution: {frame.shape}")
                    cap.release()
                    break
                cap.release()
        except Exception as e:
            print(f"❌ Camera {camera_id} failed: {e}")
    else:
        print("⚠️ No working camera found - will use mock camera")
        
except Exception as e:
    print(f"❌ Camera test error: {e}")
EOF

# Set up user permissions
print_section "User Permissions Setup"
print_status "Setting up user permissions..."
sudo usermod -a -G i2c,spi,gpio,video $USER 2>/dev/null || print_warning "Some groups may not exist"

# Create systemd service (optional)
print_section "System Service Setup"
read -p "Create systemd service for auto-start? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Creating systemd service..."
    
    cat > hey-spider-enhanced.service << EOF
[Unit]
Description=Enhanced Hey Spider Robot with YOLO v8
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

# Security settings
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

    sudo cp hey-spider-enhanced.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable hey-spider-enhanced.service
    print_status "Systemd service installed and enabled"
fi

# Performance optimization
print_section "Performance Optimization"
print_status "Applying performance optimizations..."

# Increase GPU memory split for Pi Camera (if Raspberry Pi)
if [ -f /boot/config.txt ]; then
    if ! grep -q "gpu_mem=" /boot/config.txt; then
        echo "gpu_mem=128" | sudo tee -a /boot/config.txt
        print_status "GPU memory split increased to 128MB"
    fi
fi

# Set CPU governor to performance (if available)
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true
    print_status "CPU governor set to performance mode"
fi

# Create performance monitoring script
cat > check_performance.py << 'EOF'
#!/usr/bin/env python3
import psutil
import subprocess
import sys

def check_system_performance():
    print("🔍 System Performance Check")
    print("=" * 40)
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_percent}%")
    
    # Memory usage
    memory = psutil.virtual_memory()
    print(f"RAM Usage: {memory.percent}% ({memory.used // (1024*1024)}MB / {memory.total // (1024*1024)}MB)")
    
    # GPU temperature (if available)
    try:
        temp_result = subprocess.run(['vcgencmd', 'measure_temp'], capture_output=True, text=True)
        if temp_result.returncode == 0:
            temp = temp_result.stdout.strip().replace('temp=', '').replace("'C", '°C')
            print(f"GPU Temperature: {temp}")
    except:
        pass
    
    # Disk usage
    disk = psutil.disk_usage('/')
    print(f"Disk Usage: {disk.percent}% ({disk.used // (1024*1024*1024)}GB / {disk.total // (1024*1024*1024)}GB)")
    
    # Check if GPU is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA GPU: Available ({torch.cuda.get_device_name(0)})")
        else:
            print("CUDA GPU: Not available")
    except:
        print("CUDA GPU: PyTorch not available")
    
    print("\n" + "=" * 40)
    
    # Performance recommendations
    if cpu_percent > 80:
        print("⚠️ High CPU usage detected")
    if memory.percent > 85:
        print("⚠️ High memory usage detected")
    if disk.percent > 90:
        print("⚠️ Low disk space")

if __name__ == "__main__":
    check_system_performance()
EOF

chmod +x check_performance.py

# Final system information
print_section "Installation Summary"
print_status "Installation completed successfully!"
echo ""
echo "📊 System Information:"
echo "  • Python: $(python3 --version)"
echo "  • PyTorch: $(python3 -c 'import torch; print(