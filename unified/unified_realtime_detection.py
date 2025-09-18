#!/usr/bin/env python3
"""
Unified Face + Drowsiness Real-time Detection
Uses a single trained model that detects faces and classifies drowsiness in one pass

This replaces the two-step process (face detection + drowsiness classification) 
with a single unified model for better performance.
"""

from ultralytics import YOLO
import cv2
import os
import sys
import time
import serial
import serial.tools.list_ports
import platform

def find_mcu():
    """Auto-detect ESP32 port - copied from your original code"""
    esp32_identifiers = [('CP2102 USB to UART Bridge Controller', 10),('CP210X', 9),('Silicon Labs CP210x', 9),('CH340', 8),('CH341', 8),('USB2.0-Serial', 7),('USB-SERIAL CH340', 8),('QinHeng Electronics', 7)]
    exclusion_patterns = ['BLUETOOTH','BT','WIRELESS','MODEM','FAX','DIAL']
    candidates = []
    print("Auto-detecting ESP32 port (CP2102/CH340 detection)...")
    ports = serial.tools.list_ports.comports()
    for port in ports:
        port_desc = port.description.upper()
        port_hwid = port.hwid.upper()
        port_full = f"{port_desc} {port_hwid}"
        if any(excl in port_full for excl in exclusion_patterns):
            print(f"Skipping {port.device}: {port.description} (excluded device)")
            continue
        for identifier, score in esp32_identifiers:
            if identifier.upper() in port_full:
                print(f"Found ESP32 candidate on {port.device}: {port.description}")
                candidates.append((port.device, score, port.description))
                break

    candidates.sort(key=lambda x: x[1], reverse=True)
    
    for port_device, score, desc in candidates:
        print(f"Testing {port_device} (score: {score})...")
        
        for baud in [115200, 9600]:
            try:
                ser = serial.Serial(port_device, baud, timeout=1)
                time.sleep(0.5)
                ser.write('A'.encode())
                time.sleep(0.2)
                messages = ''
                while ser.in_waiting > 0:
                    try:
                        message = ser.readline().decode().strip()
                        if message:
                            messages += (message)
                    except:
                        break
                if messages == 'SYSTEM_READY':
                    ser.close()
                    print(f"ESP32 confirmed on {port_device} at {baud} baud")
                    print(f"Device: {desc}")
                    return port_device, baud
            except Exception as e:
                try:
                    ser.close()
                except:
                    pass
                continue
    
    if platform.system() == 'Windows':
        port_num = input(f"Check Device Manager > Ports (COM & LPT) for 'Silicon Labs CP210x' and enter here if found. If not found just hit enter: ")
        port_desc = "COM" + port_num
        if port_desc == "COM":
            print("MCU Not connected")
            sys.exit(1)
        return port_desc, 115200
    else:
        print(f"\nDefaulting to /dev/ttyUSB0 - please verify your ESP32 port!")
        port_num = input(f"Linux tip: Run 'ls /dev/ttyUSB* /dev/ttyACM*' to see USB serial devices and enter the part including USB or ACM. If not found just hit enter: ")
        port_desc = '/dev/tty'+port_num
        if port_desc == '/dev/tty':
            print("MCU Not Connected")
            sys.exit(1)
        return port_desc, 115200

class UnifiedFaceDrowsinessDetector:
    def __init__(self, port='/dev/ttyUSB0', baud=115200, timeout=1, 
                 unified_model_path="runs/detect/unified_face_drowsy2/weights/best.pt"):
        self.connected_port = port
        self.last_smell_time = 0
        self.smell_cooldown = 20
        self.mcu_message_time = 0
        self.message_showed = False
        self.more_data = False
        self.mcu_name = "ESP32"
        self.mcu_message = ''
        
        # Initialize serial connection
        try:
            print(f"Trying to connect to {port}...")
            self.ser = serial.Serial(port, baud, timeout=timeout)

            if self.ser.is_open:
                self.connected_port = port
                self.is_connected = True
                print(f"Successfully connected to {self.mcu_name} on {port} with baudrate {baud}")
  
                # Give mcu time to initialize
                print(f"Waiting for {self.mcu_name} to initialize...")
                time.sleep(2)
        except serial.SerialException as e:
            print(f"Could not connect to {port}: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error with {port}: {e}")
            sys.exit(1)

        # Load unified model
        if os.path.exists(unified_model_path):
            print(f"Loading unified model: {unified_model_path}")
            self.unified_model = YOLO(unified_model_path)
            self.class_names = ['alert_face', 'drowsy_face']
            print("Unified model loaded successfully!")
        else:
            print(f"Unified model not found at {unified_model_path}")
            print("Please train the unified model first using train_unified_model.py")
            sys.exit(1)

    def serialsend(self, char):
        """Send command to MCU"""
        self.ser.write(char.encode())

    def serialreceive(self):
        """Receive messages from MCU"""
        messages = []
        while self.ser.in_waiting > 0:
            try:
                message = self.ser.readline().decode().strip()
                if message:  # Skip empty lines
                    messages.append(message)
                    self.message_showed = False
            except:
                break
        return messages  # Returns list of all messages
    
    def disconnect(self):
        """Disconnect from MCU"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"Disconnected from {self.connected_port}")
            self.is_connected = False

    def find_area_of_bbox(self, bbox):
        """Calculate area of bounding box"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return width * height
    
    def detect_faces_and_drowsiness(self, image, conf_threshold=0.5):
        """
        Single unified detection - detects faces and classifies drowsiness in one pass
        Returns list of detections with face bbox and drowsiness classification
        """
        results = self.unified_model(image, conf=conf_threshold, verbose=self.more_data)
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Get class name and drowsiness state
                    class_name = self.class_names[cls]
                    drowsy_state = "Drowsy" if cls == 1 else "Non Drowsy"
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': class_name,
                        'drowsy_state': drowsy_state,
                        'calculate': False
                    })
        
        return detections

    def draw_info_textbox(self, frame, info_text="", box_color=(50, 50, 50), text_color=(255, 255, 255), bottom=False):
        """Draw information text box - copied from your original code"""
        if not info_text:
            return frame
            
        h, w = frame.shape[:2]
        
        # Split text into lines
        lines = info_text.split('\\n') if '\\n' in info_text else [info_text]
        
        # Calculate text box dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 25
        padding = 10
        
        # Get maximum text width
        max_width = 0
        for line in lines:
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            max_width = max(max_width, text_size[0])
        
        # Calculate box dimensions
        box_width = max_width + (padding * 2)
        box_height = (len(lines) * line_height) + (padding * 2)
        
        # Center the box horizontally
        if not bottom:
            x1 = (w - box_width) // 2
            y1 = 10
            x2 = x1 + box_width
            y2 = y1 + box_height
        else:
            x1 = (w - box_width) // 2
            y1 = h - box_height - 10
            x2 = x1 + box_width
            y2 = y1 + box_height

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
        
        # Draw text lines
        for i, line in enumerate(lines):
            text_y = y1 + padding + (i + 1) * line_height
            # Center text horizontally within the box
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            text_x = x1 + (box_width - text_size[0]) // 2
            cv2.putText(frame, line, (text_x, text_y), font, font_scale, text_color, thickness)
        
        return frame

    def process_video(self, video_path, sample_duration=2):
        """Process video with unified model - simplified from your original code"""
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Time-based tracking variables
        start_time = time.time()
        last_reset_time = start_time
        
        # Per-sample tracking (resets every N seconds)
        sample_frame_count = 0
        sample_drowsy_count = 0
        sample_alert_count = 0
        
        # Overall session tracking
        total_samples = 0
        total_drowsy_count = 0
        total_alert_count = 0
        
        # Alert state tracking
        current_alert_state = None
        last_alert_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            elapsed_time = current_time - start_time
            sample_elapsed = current_time - last_reset_time
            
            # Handle MCU messages
            mcu_messages = self.serialreceive()
            if mcu_messages and not self.message_showed:
                for message in mcu_messages:
                    self.mcu_message = message
                    self.mcu_message_time = current_time
                    print(f"MCU: {message}")
                self.message_showed = True

            # Single unified detection call
            detections = self.detect_faces_and_drowsiness(frame)
            
            if detections:
                if len(detections) == 1:
                    detections[0]['calculate'] = True
                else:
                    # Find largest face (like in your original code)
                    face_areas = [self.find_area_of_bbox(det['bbox']) for det in detections]
                    largest_face_index = face_areas.index(max(face_areas))
                    detections[largest_face_index]['calculate'] = True

            # Process detections and draw bounding boxes
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                
                if detection['calculate']:
                    drowsy_state = detection['drowsy_state']
                    conf = detection['confidence']
                    
                    # Update sample counts
                    sample_frame_count += 1
                    
                    # Choose color and count based on drowsiness
                    if drowsy_state == "Drowsy":
                        color = (0, 0, 255)  # Red
                        label = f"DROWSY {conf:.2f}"
                        sample_drowsy_count += 1
                    else:  # Non Drowsy
                        color = (0, 255, 0)  # Green
                        label = f"ALERT {conf:.2f}"
                        sample_alert_count += 1
                else:
                    color = (0, 255, 255)  # Yellow
                    label = f"{detection['class'].upper()} {detection['confidence']:.2f}"
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Sample-based analysis (every N seconds)
            if sample_elapsed >= sample_duration and sample_frame_count > 0:
                total_samples += 1
                total_drowsy_count += sample_drowsy_count
                total_alert_count += sample_alert_count
                
                # Calculate sample percentages
                sample_total = sample_drowsy_count + sample_alert_count
                if sample_total > 0:
                    drowsy_percentage = (sample_drowsy_count / sample_total) * 100
                    alert_percentage = (sample_alert_count / sample_total) * 100
                    
                    # Determine alert state
                    if drowsy_percentage > 70:
                        new_alert_state = "HIGH"
                    elif drowsy_percentage > 40:
                        new_alert_state = "MEDIUM"
                    else:
                        new_alert_state = "LOW"
                    
                    # Send MCU commands based on alert state
                    if new_alert_state != current_alert_state:
                        current_alert_state = new_alert_state
                        last_alert_time = current_time
                        
                        if current_alert_state == "HIGH":
                            print(f"HIGH DROWSINESS ALERT! ({drowsy_percentage:.1f}% drowsy)")
                            if current_time - self.last_smell_time > self.smell_cooldown:
                                self.serialsend('S')
                                self.last_smell_time = current_time
                            else:
                                self.serialsend('H')
                        elif current_alert_state == "MEDIUM":
                            print(f"Medium drowsiness detected ({drowsy_percentage:.1f}% drowsy)")
                            self.serialsend('M')
                
                # Reset sample counters
                sample_frame_count = 0
                sample_drowsy_count = 0
                sample_alert_count = 0
                last_reset_time = current_time

            # Display information
            info_text = f"Session: {elapsed_time:.0f}s | Model: UNIFIED"
            frame = self.draw_info_textbox(frame, info_text)
            
            # Display MCU message
            if self.mcu_message and (current_time - self.mcu_message_time) < 3:
                mcu_text = f"MCU: {self.mcu_message}"
                frame = self.draw_info_textbox(frame, mcu_text, bottom=True, box_color=(0, 100, 0))
            
            # Display sample statistics
            if total_samples > 0:
                total_detections = total_drowsy_count + total_alert_count
                if total_detections > 0:
                    overall_drowsy_pct = (total_drowsy_count / total_detections) * 100
                    stats_text = f"Samples: {total_samples} | Overall Drowsy: {overall_drowsy_pct:.1f}%"
                    
                    # Add stats to top info
                    info_text += f" | {stats_text}"
                    frame = self.draw_info_textbox(frame, info_text)

            # Show frame
            cv2.imshow('Unified Face + Drowsiness Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                self.more_data = not self.more_data
                print(f"Verbose mode: {'ON' if self.more_data else 'OFF'}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.disconnect()

def main():
    print("Unified Face + Drowsiness Detection")
    print("Using single model for face detection + drowsiness classification")
    print("=" * 60)
    
    # Check if unified model exists
    unified_model_path = "runs/detect/unified_face_drowsy_production/weights/best.pt"
    if not os.path.exists(unified_model_path):
        print(f"Unified model not found at {unified_model_path}")
        print("Please train the unified model first:")
        print("1. Run create_unified_dataset.py")
        print("2. Run train_unified_model.py")
        return
    
    # Auto-detect ESP32 port
    print("Starting ESP32 auto-detection...")
    esp32_port, esp32_baud = find_mcu()
    
    print(f"\nConnecting to ESP32...")
    
    # Initialize detector with unified model
    detector = UnifiedFaceDrowsinessDetector(
        port=esp32_port,
        baud=esp32_baud,
        unified_model_path=unified_model_path
    )    
    
    print("\nStarting webcam detection...")
    print("Press 'q' to quit, 'v' to toggle verbose mode")
    
    # Start webcam detection
    detector.process_video(0, sample_duration=2)

if __name__ == "__main__":
    main()