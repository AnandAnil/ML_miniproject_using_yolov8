import serial
import time

# Test your loopback
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)

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