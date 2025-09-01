#!/bin/bash

# YOLO Drowsiness Detection Project - Installation Script
# This script creates a virtual environment and installs all required dependencies

set -e  # Exit on any error

echo "🚀 YOLO Drowsiness Detection - Installation Script"
echo "=================================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 is not installed. Please install pip3 first."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf .venv
fi

python3 -m venv .venv
echo "✅ Virtual environment created successfully!"

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "📥 Installing required packages..."
echo "   This may take a few minutes depending on your internet connection..."

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ All packages installed successfully!"
else
    echo "❌ Error: requirements.txt not found!"
    exit 1
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python -c "
import sys
print(f'Python version: {sys.version}')
print(f'Virtual environment: {sys.prefix}')

try:
    import ultralytics
    import cv2
    import sklearn
    import matplotlib
    import serial
    print('✅ All main libraries imported successfully!')
    print('')
    print('📋 Installed package versions:')
    print(f'   ultralytics: {ultralytics.__version__}')
    print(f'   opencv-python: {cv2.__version__}')
    print(f'   scikit-learn: {sklearn.__version__}')
    print(f'   matplotlib: {matplotlib.__version__}')
    import serial
    print(f'   pyserial: {serial.VERSION}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

echo ""
echo "🎉 Installation completed successfully!"
echo "=================================================="
echo ""
echo "📝 Usage Instructions:"
echo "   1. Activate the virtual environment:"
echo "      source .venv/bin/activate"
echo ""
echo "   2. Run your Python scripts:"
echo "      python '1) split_dataset.py'"
echo "      python '2) train_yolov8_drowsiness.py'"
echo "      python '3) confusion_matrix.py'"
echo "      python '4) test_realtime_detection.py'"
echo "      python serialtest.py"
echo ""
echo "   3. Deactivate when done:"
echo "      deactivate"
echo ""
echo "🔧 Troubleshooting:"
echo "   - If you get permission errors, run: chmod +x install.sh"
echo "   - If CUDA is not detected, PyTorch will use CPU (this is normal)"
echo "   - For ESP32 connection issues, check USB drivers and port permissions"
echo ""
echo "✨ Happy coding!"
