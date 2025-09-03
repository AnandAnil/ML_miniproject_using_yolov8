import serial
import time

possible_ports = [
    # common ubuntu ports
    '/dev/ttyUSB0',
    '/dev/ttyUSB1',
    '/dev/ttyUSB2',
    '/dev/ttyUSB3',
    '/dev/ttyUSB4',
    '/dev/ttyUSB5',
    '/dev/ttyACM0',
    '/dev/ttyACM1',
    '/dev/ttyACM2',
    '/dev/ttyACM3',
    '/dev/ttyACM4',
    '/dev/ttyACM5',
    # common windows ports
    'COM1',
    'COM2',
    'COM3',
    'COM4',
    'COM5',
    'COM6',
    'COM7',
    'COM8',
    'COM9',
    'COM10',
    'COM11',
    'COM12',
    'COM13',
    'COM14',
    'COM15'
]

baud = 115200
for port in possible_ports:
    try:
        print(f"Trying to connect to {port}...")
        ser = serial.Serial(port, baud, timeout=1)
        if ser.is_open:
            connected_port = port
            is_connected = True
            print(f"Successfully connected to ESP32 on {port}")
            print("Waiting for ESP32 to initialize...")
            time.sleep(2)
            break
    except serial.SerialException as e:
        print(f"Could not connect to {port}: {e}")
        continue
    except Exception as e:
        print(f"Unexpected error with {port}: {e}")
        continue

# Send test characters
test_chars = ['H', 'M', 'N', 'X']

for char in test_chars:
    print(f"Sending: {char}")
    ser.write(char.encode())
    
    # Read response
    time.sleep(0.001)
    while ser.in_waiting > 0:
        response = ser.readline().decode().strip()
        print(f"Received: {response}")
    

ser.close()