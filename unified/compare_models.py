#!/usr/bin/env python3
"""
Model Comparison Tool
Compare your original two-step approach vs unified model on the same images
"""

import cv2
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

class ModelComparison:
    def __init__(self, 
                 face_model_path="yolov8m-face.pt",
                 drowsiness_model_path="yolo_drowsiness/yolov8m_cls_drowsy/weights/best.pt",
                 unified_model_path="runs/detect/unified_face_drowsy_test5ep/weights/best.pt"):
        
        # Load original face detection model
        if os.path.exists(face_model_path):
            print(f"Loading face model: {face_model_path}")
            self.face_model = YOLO(face_model_path)
        else:
            print(f"Face model not found: {face_model_path}")
            sys.exit(1)
        
        # Load original drowsiness classification model
        if os.path.exists(drowsiness_model_path):
            print(f"Loading drowsiness model: {drowsiness_model_path}")
            self.drowsiness_model = YOLO(drowsiness_model_path)
        else:
            print(f"Drowsiness model not found: {drowsiness_model_path}")
            self.drowsiness_model = None
        
        # Load unified model
        if os.path.exists(unified_model_path):
            print(f"Loading unified model: {unified_model_path}")
            self.unified_model = YOLO(unified_model_path)
            self.unified_class_names = ['alert_face', 'drowsy_face']
        else:
            print(f"Unified model not found: {unified_model_path}")
            # Try to find available unified models
            available_models = list(Path("runs/detect").glob("*/weights/best.pt")) if Path("runs/detect").exists() else []
            if available_models:
                print("Available unified models:")
                for i, model in enumerate(available_models):
                    print(f"  {i+1}. {model}")
                choice = input("Enter model number or 'q' to quit: ")
                if choice.isdigit() and 1 <= int(choice) <= len(available_models):
                    unified_model_path = available_models[int(choice)-1]
                    self.unified_model = YOLO(unified_model_path)
                    print(f"Loaded unified model: {unified_model_path}")
                else:
                    sys.exit(1)
            else:
                print("No unified models found.")
                self.unified_model = None
    
    def original_approach(self, image, conf_threshold=0.5):
        """Original two-step approach: face detection + drowsiness classification"""
        results = []
        
        # Step 1: Detect faces
        face_results = self.face_model(image, conf=conf_threshold, verbose=False)
        
        for result in face_results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    face_conf = box.conf[0].cpu().numpy()
                    
                    # Step 2: Crop face and classify drowsiness
                    face_crop = image[int(y1):int(y2), int(x1):int(x2)]
                    
                    if face_crop.size > 0 and self.drowsiness_model:
                        # Resize face for classification
                        face_resized = cv2.resize(face_crop, (224, 224))
                        drowsy_results = self.drowsiness_model(face_resized, verbose=False)
                        
                        if drowsy_results and len(drowsy_results) > 0:
                            probs = drowsy_results[0].probs
                            if probs is not None:
                                class_names = ['Drowsy', 'Non Drowsy']
                                predicted_class = class_names[probs.top1]
                                drowsy_conf = probs.top1conf.cpu().numpy()
                                
                                results.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'face_confidence': float(face_conf),
                                    'drowsy_state': predicted_class,
                                    'drowsy_confidence': float(drowsy_conf),
                                    'method': 'Original'
                                })
        
        return results
    
    def unified_approach(self, image, conf_threshold=0.3):
        """Unified single-step approach"""
        results = []
        
        if not self.unified_model:
            return results
        
        unified_results = self.unified_model(image, conf=conf_threshold, verbose=False)
        
        for result in unified_results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Get class name and drowsiness state
                    class_name = self.unified_class_names[cls]
                    drowsy_state = "Drowsy" if cls == 1 else "Non Drowsy"
                    
                    results.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'drowsy_state': drowsy_state,
                        'method': 'Unified'
                    })
        
        return results
    
    def draw_comparison(self, image, original_results, unified_results):
        """Draw side-by-side comparison"""
        h, w = image.shape[:2]
        
        # Create side-by-side image
        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
        comparison[:, :w] = image  # Original approach on left
        comparison[:, w:] = image  # Unified approach on right
        
        # Draw original results on left side
        for result in original_results:
            x1, y1, x2, y2 = result['bbox']
            drowsy_state = result['drowsy_state']
            
            if drowsy_state == "Drowsy":
                color = (0, 0, 255)  # Red
                label = f"ORIG: DROWSY {result['drowsy_confidence']:.2f}"
            else:
                color = (0, 255, 0)  # Green
                label = f"ORIG: ALERT {result['drowsy_confidence']:.2f}"
            
            cv2.rectangle(comparison, (x1, y1), (x2, y2), color, 2)
            cv2.putText(comparison, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw unified results on right side
        for result in unified_results:
            x1, y1, x2, y2 = result['bbox']
            x1 += w  # Shift to right side
            x2 += w
            drowsy_state = result['drowsy_state']
            
            if drowsy_state == "Drowsy":
                color = (0, 0, 255)  # Red
                label = f"UNI: DROWSY {result['confidence']:.2f}"
            else:
                color = (0, 255, 0)  # Green
                label = f"UNI: ALERT {result['confidence']:.2f}"
            
            cv2.rectangle(comparison, (x1, y1), (x2, y2), color, 2)
            cv2.putText(comparison, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add labels
        cv2.putText(comparison, "ORIGINAL (Face + Drowsy)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, "UNIFIED MODEL", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return comparison
    
    def test_image(self, image_path):
        """Test both approaches on an image"""
        print(f"\nTesting: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        
        # Test both approaches
        print("Running original approach...")
        original_results = self.original_approach(image)
        
        print("Running unified approach...")
        unified_results = self.unified_approach(image)
        
        # Print results
        print(f"\nOriginal approach found {len(original_results)} faces:")
        for i, result in enumerate(original_results):
            print(f"  Face {i+1}: {result['drowsy_state']} (confidence: {result.get('drowsy_confidence', 0):.3f})")
        
        print(f"\nUnified approach found {len(unified_results)} faces:")
        for i, result in enumerate(unified_results):
            print(f"  Face {i+1}: {result['drowsy_state']} (confidence: {result['confidence']:.3f})")
        
        # Create comparison image
        comparison = self.draw_comparison(image, original_results, unified_results)
        
        # Resize if too large
        h, w = comparison.shape[:2]
        if h > 600 or w > 1400:
            scale = min(600/h, 1400/w)
            new_h, new_w = int(h*scale), int(w*scale)
            comparison = cv2.resize(comparison, (new_w, new_h))
        
        # Show comparison
        cv2.imshow('Model Comparison: Original vs Unified', comparison)
        
        print("\nPress any key to continue, 's' to save comparison...")
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('s'):
            save_path = f"comparison_{Path(image_path).stem}.jpg"
            cv2.imwrite(save_path, comparison)
            print(f"Comparison saved as: {save_path}")
        
        cv2.destroyAllWindows()
    
    def test_gui(self):
        """GUI version with file picker"""
        root = tk.Tk()
        root.withdraw()
        
        while True:
            file_path = filedialog.askopenfilename(
                title="Select an image to compare both models",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                break
            
            self.test_image(file_path)
        
        root.destroy()
    
    def test_dataset_samples(self):
        """Test on sample images from your dataset"""
        dataset_paths = [
            "split_dataset/val/Drowsy",
            "split_dataset/val/Non Drowsy",
            "Driver Drowsiness Dataset (DDD)/Drowsy",
            "Driver Drowsiness Dataset (DDD)/Non Drowsy"
        ]
        
        for dataset_path in dataset_paths:
            if Path(dataset_path).exists():
                print(f"\nTesting samples from: {dataset_path}")
                images = list(Path(dataset_path).glob("*.png"))[:3]  # Test first 3 images
                for img_path in images:
                    self.test_image(img_path)
                    input("Press Enter to continue to next image...")
                break

def main():
    print("Model Comparison Tool")
    print("=" * 50)
    print("Compare Original (Face + Drowsy) vs Unified Model")
    
    # Initialize comparison
    comparator = ModelComparison()
    
    if len(sys.argv) > 1:
        # Test specific image
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            comparator.test_image(image_path)
        else:
            print(f"Image not found: {image_path}")
    else:
        print("\nChoose testing mode:")
        print("1. GUI file picker")
        print("2. Test dataset samples")
        
        choice = input("Enter choice (1 or 2): ")
        
        if choice == "1":
            comparator.test_gui()
        elif choice == "2":
            comparator.test_dataset_samples()
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()