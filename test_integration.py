#!/usr/bin/env python3
"""
Quick Integration Test for ESP32 Auto-Detection
"""

import sys
import os
sys.path.append('.')

# Import the auto-detection function
from esp32_autodetect import find_esp32_port

def test_integration():
    print("🧪 Testing ESP32 Auto-Detection Integration")
    print("=" * 50)
    
    # Test auto-detection
    port, baud = find_esp32_port()
    
    if port and baud:
        print(f"\n✅ Auto-detection successful!")
        print(f"📱 Port: {port}")
        print(f"🔧 Baud: {baud}")
        
        print(f"\n📋 Integration Code:")
        print("=" * 30)
        print(f"detector = FaceDrowsinessDetector(")
        print(f"    port='{port}',")
        print(f"    baud={baud},")
        print(f"    face_model_path='yolov8m-face.pt',")
        print(f"    drowsiness_model_path='yolo_drowsiness/yolov8m_cls_drowsy/weights/best.pt'")
        print(f")")
        
        print(f"\n🚀 Ready to run main detection:")
        print(f"python3 '5) test_realtime_detection.py'")
        
        return True
    else:
        print("\n❌ Auto-detection failed")
        print("💡 Manual setup required:")
        print("detector = FaceDrowsinessDetector(port='/dev/ttyUSB0', baud=115200)")
        return False

if __name__ == "__main__":
    success = test_integration()
    
    if success:
        print("\n🎯 System ready for demonstration!")
    else:
        print("\n⚠️  Manual configuration required.")
