#!/usr/bin/env python3
"""
CP2102 USB-to-UART Bridge Detection Test
Specifically tests for ESP32 boards with CP2102 chips
Avoids Bluetooth and other non-ESP32 serial devices
"""

import serial.tools.list_ports
import platform

def detailed_cp2102_scan():
    """Detailed scan specifically for CP2102 devices"""
    
    print("🔍 CP2102 USB-to-UART Bridge Detection")
    print("=" * 45)
    
    # Get system info
    current_platform = platform.system()
    print(f"🖥️  Platform: {current_platform}")
    
    # Get all serial ports
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("❌ No serial ports found")
        return []
    
    print(f"📡 Found {len(ports)} serial port(s) total")
    print("\n📋 Detailed Port Analysis:")
    print("-" * 40)
    
    cp2102_devices = []
    bluetooth_devices = []
    other_devices = []
    
    for i, port in enumerate(ports, 1):
        print(f"\n📍 Port {i}: {port.device}")
        print(f"   📝 Description: {port.description}")
        print(f"   🔧 Hardware ID: {port.hwid}")
        print(f"   🏭 Manufacturer: {getattr(port, 'manufacturer', 'Unknown')}")
        print(f"   🆔 Product ID: {getattr(port, 'product', 'Unknown')}")
        print(f"   📊 VID:PID: {getattr(port, 'vid', 'N/A')}:{getattr(port, 'pid', 'N/A')}")
        
        # Create full text for analysis
        full_text = f"{port.description} {port.hwid} {getattr(port, 'manufacturer', '')}".upper()
        
        # Check for specific patterns
        if 'CP2102' in full_text and 'UART BRIDGE' in full_text:
            print(f"   ✅ CONFIRMED CP2102 USB-to-UART Bridge")
            cp2102_devices.append(port)
            
        elif 'CP210X' in full_text or 'SILICON LABS' in full_text:
            print(f"   🎯 SILICON LABS CP210x Family Device")
            cp2102_devices.append(port)
            
        elif 'CH340' in full_text or 'CH341' in full_text:
            print(f"   🎯 CH340/CH341 USB-to-Serial (Common on ESP32)")
            cp2102_devices.append(port)
            
        elif any(bt in full_text for bt in ['BLUETOOTH', 'BT', 'WIRELESS']):
            print(f"   🔵 Bluetooth/Wireless Device (EXCLUDED)")
            bluetooth_devices.append(port)
            
        elif any(modem in full_text for modem in ['MODEM', 'FAX', 'DIAL']):
            print(f"   📞 Modem/Dial-up Device (EXCLUDED)")
            other_devices.append(port)
            
        else:
            print(f"   ❓ Unknown/Other Serial Device")
            other_devices.append(port)
    
    # Summary
    print(f"\n📊 Detection Summary:")
    print(f"=" * 25)
    print(f"✅ ESP32 Candidates (CP2102/CH340): {len(cp2102_devices)}")
    print(f"🔵 Bluetooth Devices (Excluded): {len(bluetooth_devices)}")
    print(f"❓ Other Devices: {len(other_devices)}")
    
    # Show ESP32 candidates
    if cp2102_devices:
        print(f"\n🎯 ESP32 Device Candidates:")
        for i, device in enumerate(cp2102_devices, 1):
            print(f"   {i}. {device.device}: {device.description}")
            
        # Recommended device
        best_device = cp2102_devices[0]
        print(f"\n🏆 Recommended Device: {best_device.device}")
        print(f"📋 Description: {best_device.description}")
        
        # Generate code
        print(f"\n💻 Integration Code:")
        print(f"detector = FaceDrowsinessDetector(")
        print(f"    port='{best_device.device}',")
        print(f"    baud=115200")
        print(f")")
        
        return cp2102_devices
    
    else:
        print(f"\n❌ No ESP32 devices detected")
        
        if bluetooth_devices:
            print(f"💡 Note: {len(bluetooth_devices)} Bluetooth device(s) were excluded")
            print(f"   This is normal - Bluetooth serial ports are not ESP32 devices")
        
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Check ESP32 USB connection")
        print(f"   2. Install CP210x drivers if needed")
        print(f"   3. Try different USB cable/port")
        print(f"   4. Verify ESP32 is powered on")
        
        return []

def show_windows_specific_tips():
    """Show Windows-specific tips for CP2102 detection"""
    if platform.system() == 'Windows':
        print(f"\n🪟 Windows-Specific Tips:")
        print(f"=" * 25)
        print(f"1. 🔧 Device Manager:")
        print(f"   • Open Device Manager (devmgmt.msc)")
        print(f"   • Look under 'Ports (COM & LPT)'")
        print(f"   • ESP32 should show as 'Silicon Labs CP210x USB to UART Bridge'")
        print(f"   • Note the COM port number (e.g., COM4, COM5)")
        
        print(f"\n2. 🚫 Avoid These COM Ports:")
        print(f"   • COM1, COM2 (usually built-in ports)")
        print(f"   • Any port labeled 'Bluetooth'")
        print(f"   • Modem or Fax devices")
        
        print(f"\n3. 📥 Driver Installation:")
        print(f"   • Download CP210x drivers from Silicon Labs website")
        print(f"   • Install if ESP32 not recognized")
        print(f"   • Restart after driver installation")

if __name__ == "__main__":
    cp2102_devices = detailed_cp2102_scan()
    show_windows_specific_tips()
    
    if cp2102_devices:
        print(f"\n✅ CP2102 detection successful!")
        print(f"🚀 Ready to use ESP32 drowsiness detection system!")
    else:
        print(f"\n⚠️  Manual configuration may be required")
        print(f"💡 Check hardware connections and drivers")
