from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
from collections import deque

class FaceDrowsinessDetector:
    def __init__(self, 
                 face_model_path="yolov8m-face.pt",  # or "yolov8n-face.pt" for faster processing
                 drowsiness_model_path="yolo_drowsiness/yolov8n_cls_drowsy/weights/best.pt"):
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
    
    def draw_alert_overlay(self, image):
        """Draw simple drowsy/alert indicator in top-left corner"""
        h, w = image.shape[:2]
        
        # Only proceed if we have enough data (at least 30 frames = 1 second)
        if len(self.drowsiness_history) < 30:
            # Show "ANALYZING..." in top-left
            cv2.rectangle(image, (10, 10), (200, 60), (50, 50, 50), -1)
            cv2.rectangle(image, (10, 10), (200, 60), (255, 255, 0), 2)
            cv2.putText(image, "ANALYZING...", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(image, f"Data: {len(self.drowsiness_history)}/30", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            return image
        
        # Get last 60 frames (2 seconds at 30 FPS) for analysis
        recent_data = list(self.drowsiness_history)[-60:] if len(self.drowsiness_history) >= 60 else list(self.drowsiness_history)
        
        # Count drowsy vs alert predictions
        drowsy_count = sum(1 for data in recent_data if data['is_drowsy'])
        total_count = len(recent_data)
        alert_count = total_count - drowsy_count
        
        # Determine if more drowsy or more alert
        is_more_drowsy = drowsy_count > alert_count
        
        # Set colors and text based on result
        if is_more_drowsy:
            bg_color = (0, 0, 150)  # Dark red
            border_color = (0, 0, 255)  # Bright red
            text_color = (255, 255, 255)  # White
            main_text = "DROWSY DETECTED"
        else:
            bg_color = (0, 100, 0)  # Dark green
            border_color = (0, 255, 0)  # Bright green
            text_color = (255, 255, 255)  # White
            main_text = "DRIVER ALERT"
        
        # Draw the indicator box in top-left corner
        box_width = 220
        box_height = 70
        
        # Draw background
        cv2.rectangle(image, (10, 10), (10 + box_width, 10 + box_height), bg_color, -1)
        
        # Draw border
        cv2.rectangle(image, (10, 10), (10 + box_width, 10 + box_height), border_color, 3)
        
        # Draw main status text
        cv2.putText(image, main_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Draw statistics
        stats_text = f"Past 2s: {drowsy_count}/{total_count} drowsy"
        cv2.putText(image, stats_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        return image
    
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
    
    def update_drowsiness_tracking(self, is_drowsy, confidence):
        """
        Update drowsiness tracking history and determine alert status
        
        Args:
            is_drowsy: Boolean indicating if current frame shows drowsiness
            confidence: Confidence score of the prediction
        """
        current_time = time.time()
        
        # Add current prediction to history
        self.drowsiness_history.append({
            'is_drowsy': is_drowsy,
            'confidence': confidence,
            'timestamp': current_time
        })
        
        # Only analyze if we have at least 1 second of data (30 frames)
        if len(self.drowsiness_history) >= 30:
            # Count drowsy predictions in last 2 seconds
            recent_predictions = list(self.drowsiness_history)
            drowsy_count = sum(1 for pred in recent_predictions if pred['is_drowsy'])
            total_count = len(recent_predictions)
            drowsy_percentage = drowsy_count / total_count
            
            # Check if we should trigger an alert
            if drowsy_percentage >= 0.7:  # If 70% of last 2 seconds were drowsy
                if (current_time - self.last_alert_time) > self.alert_cooldown:
                    self.current_alert_status = "DROWSINESS DETECTED!"
                    self.alert_start_time = current_time
                    self.last_alert_time = current_time
                    print(f"🚨 DROWSINESS ALERT: {drowsy_percentage:.1%} drowsy in last 2 seconds")
            
            elif drowsy_percentage <= 0.3:  # If 70% of last 2 seconds were alert
                if (current_time - self.last_alert_time) > self.alert_cooldown:
                    self.current_alert_status = "DRIVER ALERT"
                    self.alert_start_time = current_time
                    self.last_alert_time = current_time
        
        # Clear alert after duration
        if (self.current_alert_status and 
            current_time - self.alert_start_time > self.alert_duration):
            self.current_alert_status = None
    
    def draw_alert_overlay(self, frame):
        """
        Draw alert overlay on the frame
        
        Args:
            frame: Input frame to draw on
            
        Returns:
            Frame with alert overlay
        """
        if self.current_alert_status:
            h, w = frame.shape[:2]
            
            # Create alert box in top-right corner
            box_width = 300
            box_height = 80
            x1 = w - box_width - 20
            y1 = 20
            x2 = w - 20
            y2 = y1 + box_height
            
            # Choose color based on alert type
            if "DROWSINESS" in self.current_alert_status:
                color = (0, 0, 255)  # Red
                text_color = (255, 255, 255)  # White
            else:
                color = (0, 255, 0)  # Green
                text_color = (0, 0, 0)  # Black
            
            # Draw semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Draw border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Split text into lines
            lines = self.current_alert_status.split()
            if len(lines) > 1:
                line1 = " ".join(lines[:2])
                line2 = " ".join(lines[2:]) if len(lines) > 2 else ""
            else:
                line1 = self.current_alert_status
                line2 = ""
            
            # Draw text lines
            text_y = y1 + 30
            cv2.putText(frame, line1, (x1 + 10, text_y), font, font_scale, text_color, thickness)
            if line2:
                cv2.putText(frame, line2, (x1 + 10, text_y + 25), font, font_scale, text_color, thickness)
            
            # Add blinking effect for drowsiness alert
            if "DROWSINESS" in self.current_alert_status:
                current_time = time.time()
                if int(current_time * 3) % 2:  # Blink 3 times per second
                    cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 5)
        
        # Draw statistics in bottom-left corner
        if len(self.drowsiness_history) >= 10:
            recent_predictions = list(self.drowsiness_history)[-30:]  # Last 30 frames
            drowsy_count = sum(1 for pred in recent_predictions if pred['is_drowsy'])
            total_count = len(recent_predictions)
            drowsy_percentage = drowsy_count / total_count
            
            # Statistics text
            stats_text = f"Drowsiness: {drowsy_percentage:.1%}"
            cv2.putText(frame, stats_text, (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_image(self, image_path, output_path=None):
        """
        Process single image for face detection and drowsiness classification
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return
        
        # Detect faces
        faces = self.detect_faces(image)
        
        # Process each detected face
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face['bbox']
            face_conf = face['confidence']
            
            # Crop face
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                # Classify drowsiness
                drowsy_class, drowsy_conf = self.classify_drowsiness(face_crop)
                
                # Update drowsiness tracking
                is_drowsy = drowsy_class == "Drowsy"
                self.update_drowsiness_tracking(is_drowsy, drowsy_conf)
                
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
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Add label with background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(image, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                print(f"Face {i+1}: {drowsy_class} (confidence: {drowsy_conf:.2f})")
        
        # Draw alert overlay
        image = self.draw_alert_overlay(image)
        
        # Save or display result
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Annotated image saved to: {output_path}")
        else:
            cv2.imshow('Face Drowsiness Detection', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def process_video(self, video_path, output_path=None):
        """
        Process video for real-time face detection and drowsiness classification
        
        Args:
            video_path: Path to input video or 0 for webcam
            output_path: Path to save output video
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Process each detected face
            for face in faces:
                x1, y1, x2, y2 = face['bbox']
                
                # Crop face
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    # Classify drowsiness
                    drowsy_class, drowsy_conf = self.classify_drowsiness(face_crop)
                    
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
            
            # Display frame
            cv2.imshow('Face Drowsiness Detection', frame)
            
            # Save frame if output path specified
            if output_path:
                out.write(frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Initialize detector with trained drowsiness model
    detector = FaceDrowsinessDetector(
        face_model_path='yolov8m-face.pt',
        drowsiness_model_path='yolo_drowsiness/yolov8n_cls_drowsy/weights/best.pt'
    )
    
    # Test with an image
    # detector.process_image('path_to_test_image.jpg', 'output_annotated.jpg')
    
    # Test with webcam (uncomment to use)
    # detector.process_video(0)  # 0 for webcam
    
    # Test with video file (uncomment to use)
    # detector.process_video('input_video.mp4', 'output_video.mp4')
    
    print("Face drowsiness detection system ready!")
    print("Uncomment the lines above to test with images, webcam, or video files.")
