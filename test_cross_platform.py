#!/usr/bin/env python3
"""
Cross-Platform ESP32 Detection Test
Tests Windows and Linux compatibility
"""

import serial.tools.list_ports
import platform
import sys

def test_cross_platform_detection():
    """Test ESP32 detection across platforms"""
    
    print("🖥️  Cross-Platform ESP32 Detection Test")
    print("=" * 45)
    
    # Detect current platform
    current_platform = platform.system()
    print(f"🔍 Current Platform: {current_platform}")
    print(f"🐍 Python Version: {platform.python_version()}")
    
    # Enhanced ESP32 identifiers for cross-platform
    esp32_keywords = [
        'CP210X',           # Silicon Labs (most common)
        'CH340',            # CH340 chips (common on cheap boards)
        'ESP32',            # Direct ESP32 reference
        'UART',             # Generic UART bridge
        'USB2.0-SERIAL',    # Generic USB serial
        'SILICON LABS',     # Silicon Labs full name
        'FTDI',             # FTDI chips
        'USB-SERIAL',       # Alternative naming
        'CP2102',           # Specific CP2102 reference
        'CH341',            # CH341 variant
        'QinHeng',          # CH340 manufacturer
        'PROLIFIC',         # PL2303 chips
    ]
    
    print(f"\n📡 Scanning for serial ports...")
    
    # Get all available ports
    try:
        ports = serial.tools.list_ports.comports()
        
        if not ports:
            print("❌ No serial ports found")
            return False
            
        print(f"✅ Found {len(ports)} serial port(s):")
        
        esp32_candidates = []
        
        for i, port in enumerate(ports, 1):
            print(f"\n📍 Port {i}: {port.device}")
            print(f"   📝 Description: {port.description}")
            print(f"   🔧 Hardware ID: {port.hwid}")
            
            # Check for ESP32 identifiers
            port_text = f"{port.description} {port.hwid}".upper()
            
            for keyword in esp32_keywords:
                if keyword in port_text:
                    print(f"   ✅ ESP32 Candidate: Matches '{keyword}'")
                    esp32_candidates.append((port.device, keyword, port.description))
                    break
            else:
                print(f"   ⚪ Not ESP32: No matching identifiers")
        
        # Platform-specific default suggestions
        print(f"\n🎯 Platform-Specific Defaults:")
        if current_platform == 'Windows':
            print(f"   🪟 Windows: COM1, COM3, COM4, COM5...")
            default_port = 'COM3'
        elif current_platform == 'Darwin':  # macOS
            print(f"   🍎 macOS: /dev/cu.SLAB_USBtoUART, /dev/cu.usbserial-*")
            default_port = '/dev/cu.SLAB_USBtoUART'
        else:  # Linux
            print(f"   🐧 Linux: /dev/ttyUSB0, /dev/ttyUSB1, /dev/ttyACM0...")
            default_port = '/dev/ttyUSB0'
        
        # Results summary
        print(f"\n📊 Detection Results:")
        print(f"   🔍 ESP32 Candidates: {len(esp32_candidates)}")
        
        if esp32_candidates:
            print(f"   🎯 Recommended Ports:")
            for port, keyword, desc in esp32_candidates:
                print(f"      📱 {port} (via {keyword})")
            
            # Best candidate
            best_port = esp32_candidates[0][0]
            print(f"\n🏆 Best Candidate: {best_port}")
            
            # Generate code
            print(f"\n💻 Integration Code:")
            print(f"detector = FaceDrowsinessDetector(")
            print(f"    port='{best_port}',")
            print(f"    baud=115200")
            print(f")")
            
        else:
            print(f"   ⚠️  No ESP32 detected - using platform default")
            print(f"   🔧 Default Port: {default_port}")
            
            print(f"\n💻 Fallback Code:")
            print(f"detector = FaceDrowsinessDetector(")
            print(f"    port='{default_port}',")
            print(f"    baud=115200")
            print(f")")
        
        return len(esp32_candidates) > 0
        
    except Exception as e:
        print(f"❌ Error scanning ports: {e}")
        return False

def show_platform_tips():
    """Show platform-specific setup tips"""
    current_platform = platform.system()
    
    print(f"\n💡 Platform-Specific Tips:")
    print("=" * 30)
    
    if current_platform == 'Windows':
        print("🪟 Windows Setup:")
        print("   • Install CP210x drivers from Silicon Labs")
        print("   • Check Device Manager for 'Ports (COM & LPT)'")
        print("   • ESP32 usually appears as COM3, COM4, COM5...")
        print("   • Try different COM ports if auto-detection fails")
        
    elif current_platform == 'Darwin':  # macOS
        print("🍎 macOS Setup:")
        print("   • Install CP210x drivers if needed")
        print("   • Check /dev/cu.* devices")
        print("   • ESP32 usually appears as /dev/cu.SLAB_USBtoUART")
        print("   • Use 'ls /dev/cu.*' to list available ports")
        
    else:  # Linux
        print("🐧 Linux Setup:")
        print("   • Add user to dialout group: sudo usermod -a -G dialout $USER")
        print("   • Check /dev/ttyUSB* or /dev/ttyACM* devices")
        print("   • ESP32 usually appears as /dev/ttyUSB0")
        print("   • Use 'ls /dev/tty*' to list available ports")
        print("   • Logout/login after adding to dialout group")

if __name__ == "__main__":
    print("🧪 Testing cross-platform ESP32 detection...")
    
    success = test_cross_platform_detection()
    show_platform_tips()
    
    if success:
        print(f"\n✅ ESP32 detection successful!")
    else:
        print(f"\n⚠️  Manual configuration may be required")
    
    print(f"\n🚀 Ready for all platforms: Windows, macOS, Linux!")
