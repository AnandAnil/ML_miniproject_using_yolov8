#!/usr/bin/env python3
"""
ESP32 Port Detection Script
Automatically finds ESP32 devices connected via USB/Serial
"""

import serial
import serial.tools.list_ports
import time
import sys

class ESP32PortFinder:
    def __init__(self):
        self.esp32_identifiers = [
            # Common ESP32 USB-to-Serial chip identifiers (Windows & Linux)
            'CP210X',           # Silicon Labs CP2102/CP2104
            'CH340',            # CH340 series
            'FT232R',           # FTDI FT232R
            'USB2.0-SERIAL',    # Generic USB-to-Serial
            'UART',             # Generic UART
            'ESP32',            # Direct ESP32 reference
            'SILICON LABS',     # Silicon Labs full name
            'FTDI',             # FTDI chips
            'USB-SERIAL',       # Alternative naming
            'CP2102',           # Specific CP2102 reference
            'CH341',            # CH341 variant
            # ESP32 specific identifiers
            'ESP32',
            'ESPRESSIF',
            # Device IDs
            '10C4:EA60',        # Silicon Labs CP210x
            '1A86:7523',        # CH340 series
            '0403:6001',        # FTDI FT232R
        ]
        
        self.test_commands = [
            b'AT\r\n',          # Basic AT command
            b'ESP32\r\n',       # ESP32 identifier
            b'?\r\n',           # Query command
        ]

    def scan_all_ports(self):
        """Scan all available serial ports"""
        ports = serial.tools.list_ports.comports()
        found_ports = []
        
        print("🔍 Scanning all available serial ports...")
        print("-" * 60)
        
        for port in ports:
            port_info = {
                'device': port.device,
                'description': port.description,
                'hwid': port.hwid,
                'manufacturer': getattr(port, 'manufacturer', 'Unknown'),
                'product': getattr(port, 'product', 'Unknown'),
                'vid': getattr(port, 'vid', None),
                'pid': getattr(port, 'pid', None),
                'esp32_likely': False
            }
            
            # Check if port matches ESP32 identifiers
            search_text = f"{port.description} {port.hwid} {port_info['manufacturer']} {port_info['product']}".upper()
            
            for identifier in self.esp32_identifiers:
                if identifier.upper() in search_text:
                    port_info['esp32_likely'] = True
                    break
            
            found_ports.append(port_info)
            
            # Display port information
            status = "🟢 LIKELY ESP32" if port_info['esp32_likely'] else "⚪ Generic"
            print(f"{status}")
            print(f"  Device: {port_info['device']}")
            print(f"  Description: {port_info['description']}")
            print(f"  Manufacturer: {port_info['manufacturer']}")
            print(f"  Hardware ID: {port_info['hwid']}")
            if port_info['vid'] and port_info['pid']:
                print(f"  VID:PID: {port_info['vid']:04X}:{port_info['pid']:04X}")
            print()
        
        return found_ports

    def test_esp32_communication(self, port_device, baud_rates=[115200, 9600, 38400, 57600]):
        """Test communication with potential ESP32 on given port"""
        print(f"🔌 Testing communication on {port_device}...")
        
        for baud in baud_rates:
            print(f"  Trying baud rate: {baud}")
            try:
                # Open serial connection
                ser = serial.Serial(
                    port=port_device,
                    baudrate=baud,
                    timeout=2,
                    write_timeout=2
                )
                
                time.sleep(1)  # Allow connection to stabilize
                
                # Clear any existing data
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                
                # Test different commands
                for cmd in self.test_commands:
                    try:
                        ser.write(cmd)
                        time.sleep(0.5)
                        
                        # Read response
                        if ser.in_waiting > 0:
                            response = ser.read(ser.in_waiting)
                            if response:
                                print(f"    ✅ Response received: {response}")
                                ser.close()
                                return True, baud
                                
                    except Exception as e:
                        print(f"    ⚠️ Command failed: {e}")
                        continue
                
                # Test by sending a simple message and checking for any response
                try:
                    ser.write(b'HELLO\r\n')
                    time.sleep(1)
                    if ser.in_waiting > 0:
                        response = ser.read(ser.in_waiting)
                        print(f"    ✅ Device responds: {response}")
                        ser.close()
                        return True, baud
                except:
                    pass
                
                ser.close()
                
            except Exception as e:
                print(f"    ❌ Failed to open port at {baud}: {e}")
                continue
        
        return False, None

    def find_esp32_ports(self, test_communication=True):
        """Main function to find ESP32 ports"""
        print("🚀 ESP32 Port Detection Tool")
        print("=" * 60)
        
        # Get all ports
        all_ports = self.scan_all_ports()
        
        if not all_ports:
            print("❌ No serial ports found!")
            return []
        
        # Filter likely ESP32 ports
        likely_esp32_ports = [port for port in all_ports if port['esp32_likely']]
        
        print(f"📊 Summary:")
        print(f"  Total ports found: {len(all_ports)}")
        print(f"  Likely ESP32 ports: {len(likely_esp32_ports)}")
        print()
        
        verified_ports = []
        
        if test_communication:
            print("🔬 Testing communication with likely ESP32 ports...")
            print("-" * 60)
            
            for port_info in likely_esp32_ports:
                is_responding, working_baud = self.test_esp32_communication(port_info['device'])
                if is_responding:
                    port_info['verified'] = True
                    port_info['working_baud'] = working_baud
                    verified_ports.append(port_info)
                    print(f"✅ {port_info['device']} - ESP32 CONFIRMED (Baud: {working_baud})")
                else:
                    port_info['verified'] = False
                    print(f"❌ {port_info['device']} - No response")
                print()
        
        # Final results
        print("🎯 FINAL RESULTS:")
        print("=" * 60)
        
        if verified_ports:
            print("✅ VERIFIED ESP32 PORTS:")
            for port in verified_ports:
                print(f"  🟢 {port['device']} (Baud: {port['working_baud']})")
                print(f"     Description: {port['description']}")
        else:
            print("⚠️  NO VERIFIED ESP32 PORTS FOUND")
            if likely_esp32_ports:
                print("\n📝 LIKELY ESP32 PORTS (unverified):")
                for port in likely_esp32_ports:
                    print(f"  🟡 {port['device']}")
                    print(f"     Description: {port['description']}")
        
        print()
        return verified_ports if verified_ports else likely_esp32_ports

    def get_best_esp32_port(self):
        """Get the most likely ESP32 port for use in code"""
        ports = self.find_esp32_ports(test_communication=True)
        
        if not ports:
            print("❌ No ESP32 ports found!")
            return None, None
        
        # Return the first verified port
        best_port = ports[0]
        device = best_port['device']
        baud = best_port.get('working_baud', 115200)
        
        print(f"🎯 RECOMMENDED FOR YOUR CODE:")
        print(f"   Port: '{device}'")
        print(f"   Baud: {baud}")
        print()
        print(f"💻 Code snippet:")
        print(f"   detector = FaceDrowsinessDetector(port='{device}', baud={baud})")
        
        return device, baud

def main():
    finder = ESP32PortFinder()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--best':
        # Just return the best port for scripting
        device, baud = finder.get_best_esp32_port()
        if device:
            print(f"{device},{baud}")
    else:
        # Full scan and analysis
        finder.find_esp32_ports(test_communication=True)

if __name__ == "__main__":
    main()
