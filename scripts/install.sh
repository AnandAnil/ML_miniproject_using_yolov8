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

# Download YOLOv8m Face Detection Model
echo ""
echo "📥 Downloading YOLOv8m Face Detection Model..."
echo "🎯 Target: YOLOv8m model from Yusepp's Google Drive"

# Check if gdown is installed, if not install it
echo "🔧 Installing gdown for Google Drive downloads..."
pip install gdown

# Create models directory
mkdir -p models

echo "📥 Starting download with gdown..."
echo "🔗 Using Python script for reliable Google Drive download"
echo ""

# Run the Python download script
python << 'EOF'
import gdown
import os
import sys

# Google Drive file ID for YOLOv8m face model from Yusepp
file_id = "1IJZBcyMHGhzAi0G4aZLcqryqZSjPsps-"
output_file = "models/yolov8m-face.pt"

# Google Drive download URL
url = f"https://drive.google.com/uc?id={file_id}"

print(f"🔗 Download URL: {url}")
print(f"📁 Output file: {output_file}")
print("⏳ Downloading... This may take a few minutes (197 MB)")
print("")

try:
    # Download with gdown
    gdown.download(url, output_file, quiet=False)
    
    # Check if file exists and has reasonable size
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"\n✅ Download completed!")
        print(f"📁 File: {output_file}")
        print(f"📏 Size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 100:  # Expected size around 197MB
            print("🎯 SUCCESS! File size looks correct for YOLOv8m model!")
            sys.exit(0)
        else:
            print("⚠️  File seems too small, might be incomplete.")
            sys.exit(1)
    else:
        print("❌ File not found after download")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Download failed: {str(e)}")
    print("\n💡 Trying alternative method...")
    
    # Try fuzzy download (handles confirmation pages)
    try:
        gdown.download(url, output_file, quiet=False, fuzzy=True)
        
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            file_size_mb = file_size / (1024 * 1024)
            print(f"\n✅ Alternative download completed!")
            print(f"📏 Size: {file_size_mb:.2f} MB")
            if file_size_mb > 100:
                sys.exit(0)
        
    except Exception as e2:
        print(f"❌ Alternative download also failed: {str(e2)}")
        print(f"\n💡 Manual download required:")
        print("   1. Visit: https://github.com/Yusepp/YOLOv8-Face")
        print("   2. Click: 'YOLOv8 medium (v0.2)' Google Drive link")
        print("   3. Download and save as: models/yolov8m-face.pt")
        sys.exit(1)
EOF

# Check Python script exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ YOLOv8m face model downloaded successfully!"
    echo "📁 Location: models/yolov8m-face.pt"
    echo "🔗 Source: https://github.com/Yusepp/YOLOv8-Face"
else
    echo ""
    echo "⚠️  Face model download failed, but installation continues..."
    echo "💡 You can download manually later from: https://github.com/Yusepp/YOLOv8-Face"
fi

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
echo "   3. Face detection model location:"
echo "      models/yolov8m-face.pt (automatically downloaded)"
echo ""
echo "   4. Deactivate when done:"
echo "      deactivate"
echo ""
echo "🔧 Troubleshooting:"
echo "   - If you get permission errors, run: chmod +x install.sh"
echo "   - If CUDA is not detected, PyTorch will use CPU (this is normal)"
echo "   - For ESP32 connection issues, check USB drivers and port permissions"
echo ""
echo "✨ Happy coding!"
