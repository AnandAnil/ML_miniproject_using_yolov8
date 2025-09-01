from ultralytics import YOLO
import cv2
import os
import time
from collections import deque

use_model = 'yolov8n'

class FaceDrowsinessDetector:
    def __init__(self, 
                 face_model_path="yolov8m-face.pt",
                 drowsiness_model_path="yolo_drowsiness/yolov8m_cls_drowsy/weights/best.pt"):
        """
        Initialize face detection and drowsiness classification models
        
        Args:
            face_model_path: Path to YOLOv8 face detection model
            drowsiness_model_path: Path to trained drowsiness classification model
        """
        # Load face detection model
        self.face_model = YOLO(face_model_path)
        
        # Load drowsiness classification model (after training)
        if drowsiness_model_path and os.path.exists(drowsiness_model_path):
            self.drowsiness_model = YOLO(drowsiness_model_path)
        else:
            self.drowsiness_model = None
            print("Drowsiness model not loaded. Train it first using your training script.")
        
        # Drowsiness tracking system
        self.drowsiness_history = deque(maxlen=60)  # Store last 60 frames (2 seconds at 30fps)
        self.last_alert_time = 0
        self.alert_cooldown = 3.0  # 3 seconds between alerts
        self.current_alert_status = None
        self.alert_start_time = 0
        self.alert_duration = 3.0  # Show alert for 3 seconds
    
    def detect_faces(self, image, conf_threshold=0.5):
        """
        Detect faces in the image
        
        Args:
            image: Input image (numpy array)
            conf_threshold: Confidence threshold for face detection
            
        Returns:
            List of face bounding boxes and confidences
        """
        results = self.face_model(image, conf=conf_threshold)
        faces = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    faces.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf)
                    })
        
        return faces
    
    def classify_drowsiness(self, face_crop):
        """
        Classify if the face shows drowsiness
        
        Args:
            face_crop: Cropped face image
            
        Returns:
            Drowsiness prediction and confidence
        """
        if self.drowsiness_model is None:
            return "Model not loaded", 0.0
        
        # Resize face crop to model input size
        face_resized = cv2.resize(face_crop, (224, 224))
        
        # Run drowsiness classification
        results = self.drowsiness_model(face_resized)
        
        if results and len(results) > 0:
            # Get prediction
            probs = results[0].probs
            if probs is not None:
                class_names = ['Drowsy', 'Non Drowsy']  # Adjust based on your classes
                predicted_class = class_names[probs.top1]
                confidence = probs.top1conf.cpu().numpy()
                return predicted_class, float(confidence)
        
        return "Unknown", 0.0
    
    def draw_info_textbox(self, frame, info_text="", box_color=(50, 50, 50), text_color=(255, 255, 255)):
        """
        Draw a customizable text box at the top of the frame
        
        Args:
            frame: Input frame to draw on
            info_text: Text to display (can be multiple lines separated by \\n)
            box_color: Background color of the text box (B, G, R)
            text_color: Text color (B, G, R)
            
        Returns:
            Frame with text box overlay
        """
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
        x1 = (w - box_width) // 2
        y1 = 10
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
        """
        Process video for real-time face detection and drowsiness classification
        
        Args:
            video_path: Path to input video or 0 for webcam
            sample_duration: Duration in seconds to calculate drowsiness percentage (default: 2)
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Time-based tracking variables
        start_time = time.time()
        last_reset_time = start_time
        
        # FPS tracking
        fps_start_time = start_time
        fps_frame_count = 0
        current_fps = 0
        last_fps_update = start_time
        
        # Per-sample tracking (resets every N seconds)
        sample_frame_count = 0
        sample_drowsy_count = 0
        sample_alert_count = 0
        
        # Overall session tracking
        total_samples = 0
        total_drowsy_percentage = 0
        
        # Current sample data
        current_drowsy_percentage = 0
        last_completed_sample_info = ""
        show_results = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            sample_frame_count += 1
            fps_frame_count += 1
            
            # Calculate FPS every second
            if current_time - last_fps_update >= 1.0:
                current_fps = fps_frame_count / (current_time - last_fps_update)
                fps_frame_count = 0
                last_fps_update = current_time
            
            # Check if we need to reset for new sample period
            if current_time - last_reset_time >= sample_duration:
                # Calculate percentage for the completed sample
                if sample_frame_count > 0:
                    current_drowsy_percentage = (sample_drowsy_count / (sample_drowsy_count + sample_alert_count)) * 100 if (sample_drowsy_count + sample_alert_count) > 0 else 0
                    
                    # Update overall session averages
                    total_samples += 1
                    total_drowsy_percentage = ((total_drowsy_percentage * (total_samples - 1)) + current_drowsy_percentage) / total_samples
                    
                    # Store the completed sample results
                    total_detections = sample_drowsy_count + sample_alert_count
                    last_completed_sample_info = f"Last {sample_duration}s Sample: {current_drowsy_percentage:.1f}% Drowsy ({sample_drowsy_count}D/{sample_alert_count}A/{total_detections}T)\\n"
                    last_completed_sample_info += f"Session Average: {total_drowsy_percentage:.1f}% Drowsy | Samples: {total_samples} | FPS: {current_fps:.1f}"
                    show_results = True
                
                # Reset for new sample period
                last_reset_time = current_time
                sample_frame_count = 0
                sample_drowsy_count = 0
                sample_alert_count = 0
            
            # Detect faces
            faces = self.detect_faces(frame)
            current_faces = len(faces)
            
            # Process each detected face
            for face in faces:
                x1, y1, x2, y2 = face['bbox']
                
                # Crop face
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    # Classify drowsiness
                    drowsy_class, drowsy_conf = self.classify_drowsiness(face_crop)
                    
                    # Count drowsy vs alert for current sample
                    if drowsy_class == "Drowsy":
                        sample_drowsy_count += 1
                    elif drowsy_class == "Non Drowsy":
                        sample_alert_count += 1
                    
                    # Choose color based on drowsiness
                    if drowsy_class == "Drowsy":
                        color = (0, 0, 255)  # Red for drowsy
                        label = f"DROWSY {drowsy_conf:.2f}"
                    elif drowsy_class == "Non Drowsy":
                        color = (0, 255, 0)  # Green for alert
                        label = f"ALERT {drowsy_conf:.2f}"
                    else:
                        color = (255, 255, 0)  # Yellow for unknown
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
                info_text = f"Collecting data... Time remaining: {time_remaining:.1f}s | FPS: {current_fps:.1f}\\n"
                if total_samples > 0:
                    info_text += f"Session Average: {total_drowsy_percentage:.1f}% Drowsy | Completed Samples: {total_samples}"
                else:
                    info_text += "Waiting for first sample to complete..."
            
            # Draw the info text box at the top
            frame = self.draw_info_textbox(frame, info_text)
            
            # Display frame
            cv2.imshow('Face Drowsiness Detection', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Face Drowsiness Detection - Real-time Testing")
    print("Starting webcam drowsiness detection...")
    print("Press 'q' to quit")
    
    # Initialize detector
    detector = FaceDrowsinessDetector(
        face_model_path='yolov8m-face.pt',
        drowsiness_model_path='yolo_drowsiness/'+use_model+'_cls_drowsy/weights/best.pt'
    )

    # Start webcam detection with 2-second samples
    detector.process_video(0, sample_duration=2)