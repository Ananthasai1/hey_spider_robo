#!/bin/bash
echo "======================================================="
echo "🕷️  Hey Spider Robot Installation with OLED Display"
echo "======================================================="
echo "Updated version with modern OpenAI API integration"
echo ""

# Exit on any error
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Check if running on Raspberry Pi
if [[ ! -f /proc/device-tree/model ]] || ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    print_warning "This doesn't appear to be a Raspberry Pi."
    print_warning "The system will work in mock mode for testing."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

print_status "Installing system dependencies..."
sudo apt install -y python3-pip python3-venv git cmake build-essential
sudo apt install -y portaudio19-dev python3-dev libffi-dev libjpeg-dev
sudo apt install -y espeak espeak-data libespeak1 libespeak-dev
sudo apt install -y i2c-tools python3-opencv libatlas-base-dev

# Enable I2C
print_status "Enabling I2C interface..."
sudo raspi-config nonint do_i2c 0

# Enable camera (if available)
if command -v raspi-config >/dev/null 2>&1; then
    print_status "Enabling camera interface..."
    sudo raspi-config nonint do_camera 0
else
    print_warning "raspi-config not found - camera setup skipped"
fi

# Create project directory structure
print_status "Setting up project directory..."
mkdir -p config src templates systemd images

# Create __init__.py files
touch config/__init__.py src/__init__.py

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install Python packages
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Download YOLO model
print_status "Downloading YOLO model (this may take a moment)..."
python3 -c "
try:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    print('YOLO model downloaded successfully')
except Exception as e:
    print(f'YOLO model download failed: {e}')
    print('Object detection will be disabled')
"

# Set up permissions
print_status "Setting up user permissions..."
sudo usermod -a -G i2c $USER 2>/dev/null || print_warning "