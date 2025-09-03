#!/usr/bin/env python3
"""
Serial communication with ESP32
Class-based implementation for sending data to ESP32 via serial port
"""

import serial
import time
import sys

class ESP32Serial:
    """
    ESP32 Serial Communication Class
    Handles serial communication with ESP32 microcontroller
    """
    
    def __init__(self, baudrate=115200, timeout=1):
        """
        Initialize ESP32 serial communication
        
        Args:
            baudrate (int): Serial communication baud rate (default: 115200)
            timeout (int): Serial timeout in seconds (default: 1)
        """
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.connected_port = None
        self.is_connected = False
        
        # Common ESP32 serial ports on different systems
        self.possible_ports = [
            '/dev/ttyUSB0',   # Common on Linux
            '/dev/ttyUSB1',   # Alternative on Linux
            '/dev/ttyACM0',   # Some ESP32 boards on Linux
            '/dev/ttyACM1',   # Alternative on Linux
            'COM3',           # Common on Windows
            'COM4',           # Alternative on Windows
            'COM5',           # Alternative on Windows
        ]
    
    def connect(self, port=None):
        """
        Connect to ESP32 via serial port
        
        Args:
            port (str): Specific port to connect to. If None, auto-detect.
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        if port:
            # Try specific port
            ports_to_try = [port]
        else:
            # Auto-detect
            ports_to_try = self.possible_ports
        
        for port in ports_to_try:
            try:
                print(f"Trying to connect to {port}...")
                self.ser = serial.Serial(
                    port=port,
                    baudrate=self.baudrate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=self.timeout
                )
                
                if self.ser.is_open:
                    self.connected_port = port
                    self.is_connected = True
                    print(f"✅ Successfully connected to ESP32 on {port}")
                    
                    # Give ESP32 time to initialize
                    print("Waiting for ESP32 to initialize...")
                    time.sleep(2)
                    return True
                    
            except serial.SerialException as e:
                print(f"❌ Could not connect to {port}: {e}")
                continue
            except Exception as e:
                print(f"❌ Unexpected error with {port}: {e}")
                continue
        
        # Connection failed
        self.is_connected = False
        print("❌ Could not connect to ESP32 on any port!")
        self._print_troubleshooting_tips()
        return False
    
    def disconnect(self):
        """
        Disconnect from ESP32 serial port
        """
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"🔌 Disconnected from {self.connected_port}")
            self.is_connected = False
    
    def send_character(self, char, wait_for_response=False, response_timeout=0.5):
        """
        Send a single character to ESP32
        
        Args:
            char (str): Character to send
            wait_for_response (bool): Whether to wait and check for ESP32 response
            response_timeout (float): Time to wait for response in seconds
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.is_connected or not self.ser:
            print("❌ Not connected to ESP32!")
            return False
        
        try:
            print(f"Sending character: '{char}'")
            self.ser.write(char.encode('utf-8'))
            
            # Optional response handling
            if wait_for_response:
                print("Waiting for response...")
                time.sleep(response_timeout)
                
                # Read response if available
                if self.ser.in_waiting > 0:
                    response = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    print(f"ESP32 response: {response}")
                else:
                    print("No response from ESP32")
            
            print("✅ Character sent successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error sending character: {e}")
            return False
    
    def send_multiple(self, characters, wait_for_response=False, delay_between=0.1):
        """
        Send multiple characters to ESP32
        
        Args:
            characters (str): String of characters to send one by one
            wait_for_response (bool): Whether to wait for responses
            delay_between (float): Delay between sending characters
            
        Returns:
            bool: True if all characters sent successfully, False otherwise
        """
        if not self.is_connected:
            print("❌ Not connected to ESP32!")
            return False
        
        success_count = 0
        for char in characters:
            if self.send_character(char, wait_for_response=wait_for_response):
                success_count += 1
            if delay_between > 0:
                time.sleep(delay_between)
        
        print(f"📊 Sent {success_count}/{len(characters)} characters successfully")
        return success_count == len(characters)
    
    def read_response(self):
        """
        Read any available response from ESP32
        
        Returns:
            str: Response from ESP32 or empty string if none
        """
        if not self.is_connected or not self.ser:
            return ""
        
        try:
            if self.ser.in_waiting > 0:
                response = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                return response
            return ""
        except Exception as e:
            print(f"❌ Error reading response: {e}")
            return ""
    
    def _print_troubleshooting_tips(self):
        """
        Print troubleshooting tips for connection issues
        """
        print("\nTroubleshooting tips:")
        print("1. Make sure ESP32 is connected via USB")
        print("2. Check if ESP32 is recognized by system:")
        print("   - Linux: ls /dev/tty*")
        print("   - Windows: Device Manager > Ports")
        print("3. Install ESP32 drivers if needed")
        print("4. Try different USB cable or port")
    
    def __enter__(self):
        """
        Context manager entry
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - ensures proper cleanup
        """
        self.disconnect()

def test_basic_usage():
    """Test basic ESP32 serial communication"""
    print("📋 TEST 1: Basic Usage")
    print("=" * 50)
    
    esp32 = ESP32Serial()
    
    if esp32.connect():
        # Test different send methods
        esp32.send_character('a')  # No response wait
        esp32.send_character('b', wait_for_response=True)  # With response wait
        esp32.send_multiple('xyz')  # Multiple characters
        esp32.disconnect()
        print("✅ Basic usage test completed!")
    else:
        print("❌ Basic usage test failed!")
    print()

def test_context_manager():
    """Test context manager approach (recommended)"""
    print("📋 TEST 2: Context Manager (Recommended)")
    print("=" * 50)
    
    try:
        with ESP32Serial() as esp32:
            if esp32.connect():
                esp32.send_character('c')
                esp32.send_character('d', wait_for_response=True)
                print("✅ Context manager test completed!")
            else:
                print("❌ Context manager test failed!")
    except Exception as e:
        print(f"❌ Context manager error: {e}")
    print()

def test_manual_with_exception_handling():
    """Test manual connection with proper exception handling"""
    print("📋 TEST 3: Manual with Exception Handling")
    print("=" * 50)
    
    esp32 = ESP32Serial()
    try:
        if esp32.connect():
            print("Connected! You can now:")
            print("- Type characters to send (press Enter)")
            print("- Type 'q' to quit")
            print("- Press Ctrl+C to test interrupt handling")
            
            # Simulate a few automatic sends instead of user input for testing
            test_chars = ['e', 'f', 'g']
            for char in test_chars:
                print(f"Auto-sending: {char}")
                esp32.send_character(char)
                time.sleep(0.5)
            
            print("✅ Manual exception handling test completed!")
        else:
            print("❌ Manual exception handling test failed!")
            
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C pressed, closing serial port...")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        esp32.disconnect()
    print()

def test_signal_handling():
    """Test signal handling for advanced control"""
    print("📋 TEST 4: Signal Handling")
    print("=" * 50)
    
    import signal
    
    esp32 = ESP32Serial()
    
    def signal_handler(sig, frame):
        print(f"\n🛑 Signal {sig} received, closing serial port...")
        esp32.disconnect()
        sys.exit(0)
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if esp32.connect():
            print("Signal handler registered. Sending test data...")
            esp32.send_character('h')
            esp32.send_character('i', wait_for_response=True)
            print("✅ Signal handling test completed!")
            esp32.disconnect()
        else:
            print("❌ Signal handling test failed!")
    except Exception as e:
        print(f"❌ Signal handling error: {e}")
    finally:
        # Reset signal handler
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    print()

def test_interactive_mode():
    """Test interactive mode with quit option (simulated)"""
    print("📋 TEST 5: Interactive Mode (Simulated)")
    print("=" * 50)
    
    esp32 = ESP32Serial()
    
    if esp32.connect():
        print("Interactive mode simulation...")
        
        # Simulate user commands
        commands = ['j', 'k', 'l', 'q']
        
        for cmd in commands:
            print(f"Simulated command: '{cmd}'")
            
            if cmd.lower() == 'q':
                print("Quit command received!")
                esp32.disconnect()
                break
            else:
                esp32.send_character(cmd)
                time.sleep(0.3)
        
        print("✅ Interactive mode test completed!")
    else:
        print("❌ Interactive mode test failed!")
    print()

def test_response_variations():
    """Test different response handling options"""
    print("📋 TEST 6: Response Handling Variations")
    print("=" * 50)
    
    with ESP32Serial() as esp32:
        if esp32.connect():
            print("Testing response variations...")
            
            # No response wait (fastest)
            esp32.send_character('m')
            
            # Wait for response with default timeout
            esp32.send_character('n', wait_for_response=True)
            
            # Wait for response with custom timeout
            esp32.send_character('o', wait_for_response=True, response_timeout=1.0)
            
            # Send multiple with response checking
            esp32.send_multiple('pqr', wait_for_response=True, delay_between=0.2)
            
            # Manual response reading
            esp32.send_character('s')
            time.sleep(0.1)
            response = esp32.read_response()
            if response:
                print(f"Manual read response: {response}")
            else:
                print("No response available")
            
            print("✅ Response variations test completed!")
        else:
            print("❌ Response variations test failed!")
    print()

def main():
    """
    Main function demonstrating all ESP32 serial communication implementations
    
    This tests all the different approaches mentioned in the comments:
    1. Basic usage with manual disconnect
    2. Context manager (recommended)
    3. Manual with exception handling for Ctrl+C
    4. Signal handling for advanced control
    5. Interactive mode with quit option
    6. Different response handling variations
    """
    print("🚀 ESP32 Serial Communication - All Implementations Test")
    print("=" * 60)
    print("Class-based implementation with multiple approaches")
    print("Baud rate: 115200")
    print("=" * 60)
    print()
    
    # Run all tests
    test_basic_usage()
    test_context_manager()
    test_manual_with_exception_handling()
    test_signal_handling()
    test_interactive_mode()
    test_response_variations()
    
    print("🎯 ALL TESTS COMPLETED!")
    print("=" * 60)
    print("Summary of tested implementations:")
    print("✅ Basic usage with manual connection/disconnection")
    print("✅ Context manager with automatic cleanup")
    print("✅ Manual exception handling (Ctrl+C protection)")
    print("✅ Signal handling for advanced interrupt control")
    print("✅ Interactive mode with quit commands")
    print("✅ Various response handling options")
    print("=" * 60)

if __name__ == "__main__":
    main()
