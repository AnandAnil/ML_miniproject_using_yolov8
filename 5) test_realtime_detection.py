from ultralytics import YOLO
import cv2
import os
import sys
import time
import serial
import serial.tools.list_ports

use_model = 'yolov8m'

def auto_detect_esp32():
    """Auto-detect ESP32 port (Linux/Windows compatible) - Enhanced CP2102 detection"""
    # Specific ESP32 USB-to-Serial identifiers with priority
    esp32_identifiers = [
        ('CP2102 USB to UART Bridge Controller', 10),  # Exact CP2102 match (highest priority)
        ('CP210X', 9),                                 # Silicon Labs CP210x family
        ('Silicon Labs CP210x', 9),                    # Full Silicon Labs name
        ('CH340', 8),                                  # CH340 chips
        ('CH341', 8),                                  # CH341 variant
        ('USB2.0-Serial', 7),                          # Generic but common on ESP32
        ('USB-SERIAL CH340', 8),                       # Specific CH340 variant
        ('QinHeng Electronics', 7),                    # CH340 manufacturer
    ]
    
    # Exclusion patterns for non-ESP32 devices
    exclusion_patterns = [
        'BLUETOOTH',
        'BT',
        'WIRELESS',
        'MODEM',
        'FAX',
        'DIAL'
    ]
    
    print("Auto-detecting ESP32 port (CP2102/CH340 detection)...")
    ports = serial.tools.list_ports.comports()
    
    candidates = []
    
    for port in ports:
        port_desc = port.description.upper()
        port_hwid = port.hwid.upper()
        port_full = f"{port_desc} {port_hwid}"
        
        # Skip excluded devices (like Bluetooth)
        if any(excl in port_full for excl in exclusion_patterns):
            print(f"Skipping {port.device}: {port.description} (excluded device)")
            continue
        
        # Check for ESP32 identifiers with scoring
        for identifier, score in esp32_identifiers:
            if identifier.upper() in port_full:
                print(f"Found ESP32 candidate on {port.device}: {port.description}")
                candidates.append((port.device, score, port.description))
                break
    
    # Sort by score (highest first) and test communication
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    for port_device, score, desc in candidates:
        print(f"Testing {port_device} (score: {score})...")
        
        for baud in [115200, 9600]:
            try:
                ser = serial.Serial(port_device, baud, timeout=1)
                time.sleep(0.5)
                ser.write(b'AT\r\n')
                time.sleep(0.2)
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
    
    # Return platform-appropriate default (but warn user)
    import platform
    if platform.system() == 'Windows':
        print(f"\nDefaulting to COM1 - PLEASE VERIFY YOUR ESP32 PORT!")
        print(f"Windows tip: Check Device Manager > Ports (COM & LPT) for 'Silicon Labs CP210x'")
        return 'COM1', 115200
    else:
        print(f"\nDefaulting to /dev/ttyUSB0 - please verify your ESP32 port!")
        print(f"Linux tip: Run 'ls /dev/ttyUSB* /dev/ttyACM*' to see USB serial devices")
        return '/dev/ttyUSB0', 115200

class FaceDrowsinessDetector:
    def __init__(self,port='/dev/ttyUSB0',baud=115200,timeout=1,face_model_path="yolov8m-face.pt",drowsiness_model_path="yolo_drowsiness/yolov8m_cls_drowsy/weights/best.pt"):
        self.connected_port = port
        self.last_smell_time = 0
        self.smell_cooldown = 20
        self.mcu_message_time = 0
        self.message_showed = False
        self.more_data = False
        self.mcu_name = "ESP32"
        self.mcu_message = ''
        
        try:
            print(f"Trying to connect to {port}...")
            self.ser=serial.Serial(port,baud,timeout=timeout)

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

        if os.path.exists(face_model_path):
            self.face_model = YOLO(face_model_path)
        else:
            print("Face Detection model not found. Please download it from github ( Yusepp/YOLOv8-Face/"+use_model+" )")
            sys.exit(1)

        if drowsiness_model_path and os.path.exists(drowsiness_model_path):
            self.drowsiness_model = YOLO(drowsiness_model_path)
        else:
            self.drowsiness_model = None
            print("Drowsiness model not loaded. Train it first using the training scripts.")
            sys.exit(1)

    def serialsend(self,char):
        self.ser.write(char.encode())

    def serialreceive(self):
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
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"Disconnected from {self.connected_port}")
            self.is_connected = False

    def find_area_of_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return width * height
    
    def detect_faces(self, image, conf_threshold=0.5):
        results = self.face_model(image, conf=conf_threshold, verbose=self.more_data)
        faces = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    faces.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'calculate': False
                    })
        
        return faces
    
    def classify_drowsiness(self, face_crop):
        face_resized = cv2.resize(face_crop, (224, 224))
        results = self.drowsiness_model(face_resized, verbose=self.more_data)
        
        if results and len(results) > 0:
            probs = results[0].probs
            if probs is not None:
                class_names = ['Drowsy', 'Non Drowsy']
                predicted_class = class_names[probs.top1]
                confidence = probs.top1conf.cpu().numpy()
                return predicted_class, float(confidence)
        return "Unknown", 0.0
    
    def draw_info_textbox(self, frame, info_text="", box_color=(50, 50, 50), text_color=(255, 255, 255),bottom=False):
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

        # Current sample data
        current_drowsy_percentage = 0
        last_completed_sample_info = ""
        show_results = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # check for new message each frame
            current_time = time.time()
            sample_frame_count += 1
            mcu_messages = self.serialreceive()
            if mcu_messages:
                latest_mcu_msg = mcu_messages[-1]
                self.mcu_message = f"{self.mcu_name}: {latest_mcu_msg}"
                self.mcu_message_time = time.time()
                
            if self.mcu_message and not self.message_showed and (current_time - self.mcu_message_time) < 5:
                frame = self.draw_info_textbox(frame, self.mcu_message, bottom=True)
            elif (current_time - self.mcu_message_time) >= 5:
                self.mcu_message = ''
                self.message_showed = True

            # Check if we need to reset for new sample period
            if current_time - last_reset_time >= sample_duration:
                # Calculate percentage for the completed sample
                if sample_frame_count > 0:
                    current_drowsy_percentage = (sample_drowsy_count / (sample_drowsy_count + sample_alert_count)) * 100 if (sample_drowsy_count + sample_alert_count) > 0 else 0
                    if current_drowsy_percentage > 70:
                        if current_time - self.last_smell_time >= self.smell_cooldown:
                            self.serialsend('S')
                            self.last_smell_time = current_time
                        else:
                            self.serialsend('H')
                    elif current_drowsy_percentage > 40:
                        self.serialsend('M')
                    # Update overall session averages
                    total_samples += 1                    
                    # Store the completed sample results
                    total_detections = sample_drowsy_count + sample_alert_count
                    last_completed_sample_info = f"Last {sample_duration}s Sample: {current_drowsy_percentage:.1f}% Drowsy ({sample_drowsy_count}D/{sample_alert_count}A/{total_detections}T)\\nSamples taken till now: {total_samples}"
                    show_results = True
                
                # Reset for new sample period
                last_reset_time = current_time
                sample_frame_count = 0
                sample_drowsy_count = 0
                sample_alert_count = 0
            
            # Detect faces
            faces = self.detect_faces(frame)
            face_areas = []
            if faces:
                if len(faces) == 1:
                    faces[0]['calculate'] = True
                else:
                    for face in faces:
                        face_areas.append(self.find_area_of_bbox(face['bbox']))
                    largest_face_index = face_areas.index(max(face_areas))
                    faces[largest_face_index]['calculate'] = True

            for face in faces:
                x1, y1, x2, y2 = face['bbox']

                # Crop face
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    # Classify drowsiness
                    if face['calculate']:
                        drowsy_class, drowsy_conf = self.classify_drowsiness(face_crop)
                        # Choose color based on drowsiness
                        if drowsy_class == "Drowsy":
                            color = (0, 0, 255)  # Red for drowsy
                            label = f"DROWSY {drowsy_conf:.2f}"
                            sample_drowsy_count += 1

                        elif drowsy_class == "Non Drowsy":
                            color = (0, 255, 0)  # Green for alert
                            label = f"ALERT {drowsy_conf:.2f}"
                            sample_alert_count += 1
                    else:
                        color = (0, 255, 255)  # Yellow for unknown
                        label = f"UNKNOWN"
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Calculate time remaining in current sample
            time_in_sample = current_time - last_reset_time
            time_remaining = sample_duration - time_in_sample
            
            # Prepare info text for the top text box
            # Only show results after sample period is complete
            if show_results and last_completed_sample_info:
                info_text = last_completed_sample_info
            else:
                # Show collecting data message during sample period
                info_text = f"Collecting data... Time remaining: {time_remaining:.1f}s\\n"
                if total_samples > 0:
                    info_text += f"Completed Samples: {total_samples}"
                else:
                    info_text += "Waiting for first sample to complete..."
            
            # Draw the info text box at the top
            time_remains = self.smell_cooldown - (current_time-self.last_smell_time)
            info_text += f"\\ntime remaining to redeploy smelly {max(0,time_remains):.1f}"
            frame = self.draw_info_textbox(frame, info_text)
            
            # Display frame
            cv2.imshow('Face Drowsiness Detection', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        self.disconnect()
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Auto-detect ESP32 port with enhanced detection
    print("Starting ESP32 auto-detection...")
    esp32_port, esp32_baud = auto_detect_esp32()
    
    print(f"\nConnecting to ESP32...")
    
    # Initialize detector with auto-detected port
    detector = FaceDrowsinessDetector(
        port=esp32_port,
        baud=esp32_baud,
        face_model_path='yolov8m-face.pt',
        drowsiness_model_path='yolo_drowsiness/'+use_model+'_cls_drowsy/weights/best.pt'
    )    
    # Start webcam detection with 2-second samples
    detector.process_video(0, sample_duration=2)

if __name__ == "__main__":
    main()