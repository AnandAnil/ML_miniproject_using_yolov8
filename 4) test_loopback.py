import serial
import time

possible_ports = []
for i in range(0,6):
    possible_ports.append(f'/dev/ttyUSB{i}')
for i in range(0,6):
    possible_ports.append(f'/dev/ttyACM{i}')
for i in range(5,16):
    possible_ports.append(f'COM{i}')

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
    time.sleep(0.01)
    while ser.in_waiting > 0:
        response = ser.readline().decode().strip()
        print(f"Received: {response}")
    

ser.close()