#!/usr/bin/env python3
"""
Unified Model Image Tester
Select an image and see face detection + drowsiness classification results
"""

import cv2
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox

class UnifiedModelTester:
    def __init__(self, unified_model_path="runs/detect/unified_face_drowsy2/weights/best.pt"):
        self.unified_model_path = unified_model_path
        self.class_names = ['alert_face', 'drowsy_face']
        
        # Load unified model
        if os.path.exists(unified_model_path):
            print(f"Loading unified model: {unified_model_path}")
            self.unified_model = YOLO(unified_model_path)
            print("Unified model loaded successfully!")
        else:
            print(f"Unified model not found at {unified_model_path}")
            available_models = list(Path("runs/detect").glob("*/weights/best.pt")) if Path("runs/detect").exists() else []
            if available_models:
                print("Available models:")
                for i, model in enumerate(available_models):
                    print(f"  {i+1}. {model}")
                choice = input("Enter model number or 'q' to quit: ")
                if choice.isdigit() and 1 <= int(choice) <= len(available_models):
                    self.unified_model_path = available_models[int(choice)-1]
                    self.unified_model = YOLO(self.unified_model_path)
                    print(f"Loaded model: {self.unified_model_path}")
                else:
                    sys.exit(1)
            else:
                print("No trained models found. Please train a model first.")
                sys.exit(1)
    
    def detect_and_classify(self, image_path, conf_threshold=0.3):
        """Detect faces and classify drowsiness in a single image"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            return None, []
        
        # Run inference
        results = self.unified_model(image, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Get class name and drowsiness state
                    class_name = self.class_names[cls]
                    drowsy_state = "Drowsy" if cls == 1 else "Alert"
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': class_name,
                        'drowsy_state': drowsy_state
                    })
        
        return image, detections
    
    def draw_results(self, image, detections):
        """Draw bounding boxes and labels on image"""
        output_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            drowsy_state = detection['drowsy_state']
            conf = detection['confidence']
            
            # Choose color based on drowsiness
            if drowsy_state == "Drowsy":
                color = (0, 0, 255)  # Red
                label = f"DROWSY {conf:.2f}"
            else:
                color = (0, 255, 0)  # Green
                label = f"ALERT {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)
            
            # Add label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(output_image, (x1, y1 - 35), (x1 + label_size[0] + 10, y1), color, -1)
            cv2.putText(output_image, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return output_image
    
    def test_image_gui(self):
        """GUI version with file picker"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        while True:
            # File picker
            file_path = filedialog.askopenfilename(
                title="Select an image to test",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                break
            
            print(f"\nTesting image: {file_path}")
            
            # Process image
            image, detections = self.detect_and_classify(file_path)
            
            if image is None:
                messagebox.showerror("Error", "Could not load the selected image")
                continue
            
            # Display results
            if detections:
                print(f"Found {len(detections)} face(s):")
                for i, det in enumerate(detections):
                    print(f"  Face {i+1}: {det['drowsy_state']} (confidence: {det['confidence']:.3f})")
                
                # Draw results
                result_image = self.draw_results(image, detections)
                
                # Resize for display if too large
                h, w = result_image.shape[:2]
                if h > 800 or w > 1200:
                    scale = min(800/h, 1200/w)
                    new_h, new_w = int(h*scale), int(w*scale)
                    result_image = cv2.resize(result_image, (new_w, new_h))
                
                # Show result
                cv2.imshow('Unified Model Test Result', result_image)
                
                # Show original for comparison
                if h > 400 or w > 600:
                    scale = min(400/h, 600/w)
                    new_h, new_w = int(h*scale), int(w*scale)
                    display_original = cv2.resize(image, (new_w, new_h))
                else:
                    display_original = image
                cv2.imshow('Original Image', display_original)
                
                print("Press any key to continue, 'q' to quit, 's' to save result...")
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('s'):
                    # Save result
                    save_path = f"test_result_{Path(file_path).stem}.jpg"
                    cv2.imwrite(save_path, result_image)
                    print(f"Result saved as: {save_path}")
                    messagebox.showinfo("Saved", f"Result saved as: {save_path}")
                
                cv2.destroyAllWindows()
                
                if key == ord('q'):
                    break
            else:
                print("No faces detected in the image")
                messagebox.showinfo("No Detection", "No faces detected in the image")
        
        root.destroy()
    
    def test_image_cli(self, image_path):
        """Command line version"""
        print(f"\nTesting image: {image_path}")
        
        # Process image
        image, detections = self.detect_and_classify(image_path)
        
        if image is None:
            return
        
        # Display results
        if detections:
            print(f"Found {len(detections)} face(s):")
            for i, det in enumerate(detections):
                print(f"  Face {i+1}: {det['drowsy_state']} (confidence: {det['confidence']:.3f})")
            
            # Draw results
            result_image = self.draw_results(image, detections)
            
            # Resize for display if too large
            h, w = result_image.shape[:2]
            if h > 800 or w > 1200:
                scale = min(800/h, 1200/w)
                new_h, new_w = int(h*scale), int(w*scale)
                result_image = cv2.resize(result_image, (new_w, new_h))
            
            # Show result
            cv2.imshow('Unified Model Test Result', result_image)
            
            print("Press any key to continue, 's' to save result...")
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('s'):
                # Save result
                save_path = f"test_result_{Path(image_path).stem}.jpg"
                cv2.imwrite(save_path, result_image)
                print(f"Result saved as: {save_path}")
            
            cv2.destroyAllWindows()
        else:
            print("No faces detected in the image")

def main():
    print("Unified Model Image Tester")
    print("=" * 40)
    
    # Initialize tester
    tester = UnifiedModelTester()
    
    # Check if command line argument provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            tester.test_image_cli(image_path)
        else:
            print(f"Image not found: {image_path}")
    else:
        # Use GUI file picker
        print("Opening file picker...")
        tester.test_image_gui()

if __name__ == "__main__":
    main()