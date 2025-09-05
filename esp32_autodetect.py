#!/usr/bin/env python3
"""
Simple ESP32 Auto-Detection for Integration
"""

import serial
import serial.tools.list_ports
import time

def find_esp32_port():
    """
    Enhanced ESP32 port detection (Linux/Windows compatible)
    Specifically targets CP2102 and CH340 while avoiding Bluetooth
    Returns: (port_device, baud_rate) or (None, None) if not found
    """
    
    # Prioritized ESP32 identifiers
    esp32_identifiers = [
        ('CP2102 USB to UART Bridge Controller', 10),  # Exact CP2102 match
        ('CP210X', 9),                                 # Silicon Labs family
        ('Silicon Labs CP210x', 9),                    # Full name
        ('CH340', 8),                                  # CH340 chips
        ('CH341', 8),                                  # CH341 variant  
        ('USB2.0-Serial', 7),                          # Generic USB serial
        ('USB-SERIAL CH340', 8),                       # CH340 specific
        ('QinHeng Electronics', 7),                    # CH340 manufacturer
    ]
    
    # Patterns to exclude (Bluetooth, modems, etc.)
    exclusion_patterns = [
        'BLUETOOTH', 'BT', 'WIRELESS', 'MODEM', 'FAX', 'DIAL'
    ]
    
    print("🔍 Auto-detecting ESP32 port (Enhanced CP2102/CH340 detection)...")
    
    # Get all available ports
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("❌ No serial ports found on system")
        return None, None
    
    candidates = []
    
    print(f"📡 Scanning {len(ports)} serial port(s)...")
    
    for port in ports:
        port_desc = port.description.upper()
        port_hwid = port.hwid.upper() 
        port_full = f"{port_desc} {port_hwid}"
        
        print(f"   📍 {port.device}: {port.description}")
        
        # Skip excluded devices
        if any(excl in port_full for excl in exclusion_patterns):
            print(f"      ⏭️  Excluded (matches: {[e for e in exclusion_patterns if e in port_full]})")
            continue
        
        # Check for ESP32 identifiers
        for identifier, score in esp32_identifiers:
            if identifier.upper() in port_full:
                print(f"      ✅ ESP32 candidate (matched: {identifier}, score: {score})")
                candidates.append((port.device, score, port.description))
                break
        else:
            print(f"      ⚪ Not ESP32 (no matching identifiers)")
    
    if not candidates:
        print("❌ No ESP32 candidates found")
        return None, None
    
    # Sort by score and test communication
    candidates.sort(key=lambda x: x[1], reverse=True)
    print(f"\n🧪 Testing {len(candidates)} candidate(s) in priority order...")
    
    for port_device, score, desc in candidates:
        print(f"   🔌 Testing {port_device} (score: {score})...")
        
        # Test communication with different baud rates
        for baud in [115200, 9600]:
            try:
                ser = serial.Serial(port_device, baud, timeout=1)
                time.sleep(0.5)
                
                # Test basic communication
                ser.write(b'AT\r\n')
                time.sleep(0.2)
                
                ser.close()
                print(f"      ✅ Communication successful at {baud} baud")
                print(f"📋 Device: {desc}")
                return port_device, baud
                
            except Exception as e:
                try:
                    ser.close()
                except:
                    pass
                print(f"      ❌ Failed at {baud} baud: {type(e).__name__}")
                continue
    
    print("❌ No ESP32 devices responded to communication test")
    return None, None

# Quick test function
def test_esp32_detection():
    port, baud = find_esp32_port()
    if port:
        print(f"\n🎯 Use this in your code:")
        print(f"detector = FaceDrowsinessDetector(port='{port}', baud={baud})")
    else:
        print("\n⚠️  ESP32 not detected.")
        print("🔍 Manual troubleshooting:")
        
        # Show all available ports for manual selection
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        
        if ports:
            print("\n📋 Available serial ports:")
            for i, port in enumerate(ports, 1):
                print(f"   {i}. {port.device}: {port.description}")
            
            print(f"\n💡 Try these ports manually:")
            # Platform-specific suggestions
            import platform
            if platform.system() == 'Windows':
                print("   # Windows - try each COM port:")
                for port in ports:
                    if port.device.startswith('COM'):
                        print(f"   detector = FaceDrowsinessDetector(port='{port.device}', baud=115200)")
            else:
                print("   # Linux - try USB/ACM ports:")
                for port in ports:
                    if '/ttyUSB' in port.device or '/ttyACM' in port.device:
                        print(f"   detector = FaceDrowsinessDetector(port='{port.device}', baud=115200)")
        else:
            print("❌ No serial ports found on system")
            print("🔌 Check ESP32 USB connection")

if __name__ == "__main__":
    test_esp32_detection()
