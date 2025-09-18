#!/usr/bin/env python3
"""
Unified Face + Drowsiness Detection Dataset Creator
Creates a YOLO detection dataset from your existing split_dataset structure

This uses your existing yolov8m-face.pt model to detect faces in images and 
creates bounding boxes for training a unified model.
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class UnifiedDatasetCreator:
    def __init__(self, source_dataset_path="split_dataset", 
                 face_model_path="yolov8m-face.pt",
                 output_path="unified_face_drowsy_dataset"):
        self.source_path = Path(source_dataset_path)
        self.output_path = Path(output_path)
        self.face_model_path = face_model_path
        self.class_names = ['alert_face', 'drowsy_face']
        
        # Load face detection model
        if os.path.exists(face_model_path):
            print(f"Loading face detection model: {face_model_path}")
            self.face_model = YOLO(face_model_path)
        else:
            print(f"Face model not found at {face_model_path}")
            print("Please ensure yolov8m-face.pt is in the current directory")
            exit(1)
        
        # Create output directory structure
        self.create_directory_structure()
    
    def create_directory_structure(self):
        """Create the YOLO dataset directory structure"""
        for split in ['train', 'val', 'test']:
            (self.output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        print(f"Created dataset structure at: {self.output_path}")
    
    def detect_faces_in_image(self, image_path, conf_threshold=0.5):
        """
        Detect faces in image using your yolov8m-face.pt model
        Returns list of normalized bounding boxes (x_center, y_center, width, height)
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return []
            
            height, width = image.shape[:2]
            
            # Use your face model to detect faces
            results = self.face_model(image, conf=conf_threshold, verbose=False)
            
            faces = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        # Convert to YOLO format (normalized center coordinates)
                        x_center = ((x1 + x2) / 2) / width
                        y_center = ((y1 + y2) / 2) / height
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height
                        
                        faces.append({
                            'bbox': (x_center, y_center, box_width, box_height),
                            'confidence': float(conf)
                        })
            
            # If no faces detected, skip this image
            if not faces:
                print(f"No faces detected in {image_path}")
                return []
            
            # Return the most confident face (like in your original code)
            if len(faces) > 1:
                faces.sort(key=lambda x: x['confidence'], reverse=True)
            
            return [faces[0]]  # Return only the best face
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return []
    
    def process_existing_splits(self):
        """Process the existing train/val/test splits from split_dataset"""
        
        splits = ['train', 'val', 'test']
        total_processed = 0
        total_skipped = 0
        
        for split in splits:
            print(f"\nProcessing {split} split...")
            
            split_path = self.source_path / split
            if not split_path.exists():
                print(f"Warning: {split} folder not found, skipping...")
                continue
            
            # Process Drowsy images (class 1 = drowsy_face)
            drowsy_path = split_path / "Drowsy"
            drowsy_processed, drowsy_skipped = 0, 0
            if drowsy_path.exists():
                drowsy_processed, drowsy_skipped = self.process_class_folder(drowsy_path, split, 1, "drowsy")
            
            # Process Non Drowsy images (class 0 = alert_face)  
            alert_path = split_path / "Non Drowsy"
            alert_processed, alert_skipped = 0, 0
            if alert_path.exists():
                alert_processed, alert_skipped = self.process_class_folder(alert_path, split, 0, "alert")
            
            split_total = drowsy_processed + alert_processed
            split_skipped = drowsy_skipped + alert_skipped
            total_processed += split_total
            total_skipped += split_skipped
            
            print(f"  {split}: {alert_processed} alert + {drowsy_processed} drowsy = {split_total} processed")
            if split_skipped > 0:
                print(f"  {split}: {split_skipped} images skipped (no face detected)")
        
        print(f"\nTotal images processed: {total_processed}")
        print(f"Total images skipped: {total_skipped}")
        return total_processed
    
    def process_class_folder(self, class_folder, split, class_id, class_name):
        """Process all images in a class folder"""
        processed_count = 0
        skipped_count = 0
        
        # Get all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(class_folder.glob(ext))
        
        total_images = len(image_files)
        print(f"    Processing {total_images} {class_name} images...")
        
        for i, img_path in enumerate(image_files):
            try:
                # Detect faces in image
                faces = self.detect_faces_in_image(img_path)
                if not faces:
                    skipped_count += 1
                    continue
                
                # Use the first (best) face
                face = faces[0]
                bbox = face['bbox']
                
                # Create new filename
                new_img_name = f"{class_name}_{img_path.stem}_{processed_count:04d}{img_path.suffix}"
                dest_img_path = self.output_path / 'images' / split / new_img_name
                
                # Copy image to destination
                shutil.copy2(img_path, dest_img_path)
                
                # Create YOLO label file
                label_content = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                label_path = self.output_path / 'labels' / split / f"{dest_img_path.stem}.txt"
                
                with open(label_path, 'w') as f:
                    f.write(label_content)
                
                processed_count += 1
                
                if processed_count % 50 == 0:
                    print(f"      Processed {processed_count}/{total_images} {class_name} images")
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                skipped_count += 1
        
        print(f"    {class_name}: {processed_count} processed, {skipped_count} skipped")
        return processed_count, skipped_count
    
    def create_dataset_yaml(self):
        """Create the dataset configuration YAML file"""
        yaml_content = f"""# Unified Face Detection + Drowsiness Classification Dataset
# Generated from existing split_dataset structure using yolov8m-face.pt

path: {self.output_path.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
nc: 2  # number of classes
names: ['alert_face', 'drowsy_face']

# Dataset info
# Class 0: alert_face (Non Drowsy faces)
# Class 1: drowsy_face (Drowsy faces)
"""
        
        yaml_path = self.output_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created dataset configuration: {yaml_path}")
        return yaml_path
    
    def generate_statistics(self):
        """Generate dataset statistics"""
        stats = {
            'train': {'alert_face': 0, 'drowsy_face': 0},
            'val': {'alert_face': 0, 'drowsy_face': 0},
            'test': {'alert_face': 0, 'drowsy_face': 0}
        }
        
        for split in ['train', 'val', 'test']:
            label_dir = self.output_path / 'labels' / split
            if label_dir.exists():
                for label_file in label_dir.glob("*.txt"):
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.strip():  # Skip empty lines
                                class_id = int(line.split()[0])
                                if class_id == 0:
                                    stats[split]['alert_face'] += 1
                                else:
                                    stats[split]['drowsy_face'] += 1
        
        print("\nDataset Statistics:")
        print("=" * 50)
        total_images = 0
        for split, counts in stats.items():
            split_total = sum(counts.values())
            total_images += split_total
            if split_total > 0:
                print(f"{split.upper()}: {split_total} total images")
                for class_name, count in counts.items():
                    percentage = (count / split_total * 100) if split_total > 0 else 0
                    print(f"  {class_name}: {count} ({percentage:.1f}%)")
            else:
                print(f"{split.upper()}: No images")
        
        if total_images > 0:
            print(f"\nTOTAL DATASET: {total_images} images")
            
            # Overall class distribution
            total_alert = sum(stats[split]['alert_face'] for split in stats.keys())
            total_drowsy = sum(stats[split]['drowsy_face'] for split in stats.keys())
            alert_pct = (total_alert / total_images * 100) if total_images > 0 else 0
            drowsy_pct = (total_drowsy / total_images * 100) if total_images > 0 else 0
            
            print(f"\nOverall Distribution:")
            print(f"  Alert faces: {total_alert} ({alert_pct:.1f}%)")
            print(f"  Drowsy faces: {total_drowsy} ({drowsy_pct:.1f}%)")
        
        return stats

def main():
    print("Creating Unified Face + Drowsiness Detection Dataset")
    print("Using existing split_dataset structure with yolov8m-face.pt")
    print("=" * 60)
    
    # Check if source dataset exists
    source_path = "split_dataset"
    if not Path(source_path).exists():
        print(f"Error: Source dataset not found at {source_path}")
        print("Please ensure the split_dataset folder exists in the current directory")
        return
    
    # Check if face model exists
    face_model = "yolov8m-face.pt"
    if not Path(face_model).exists():
        print(f"Error: Face model not found at {face_model}")
        print("Please ensure yolov8m-face.pt is in the current directory")
        return
    
    # Check required folders
    required_folders = ['train', 'val', 'test']
    for folder in required_folders:
        folder_path = Path(source_path) / folder
        if not folder_path.exists():
            print(f"Warning: {folder} folder not found in {source_path}")
    
    # Create unified dataset
    creator = UnifiedDatasetCreator(
        source_dataset_path=source_path,
        face_model_path=face_model,
        output_path="unified_face_drowsy_dataset"
    )
    
    # Process the existing splits
    total_processed = creator.process_existing_splits()
    
    if total_processed == 0:
        print("No images were processed. Please check your dataset structure.")
        return
    
    # Create dataset configuration
    yaml_path = creator.create_dataset_yaml()
    
    # Generate statistics
    stats = creator.generate_statistics()
    
    print("\n" + "=" * 60)
    print("Dataset creation completed successfully!")
    print(f"Dataset location: {creator.output_path.absolute()}")
    print(f"Configuration file: {yaml_path}")
    print("\nNext steps:")
    print("1. Run train_unified_model.py to train the unified model")
    print("2. Use unified_realtime_detection.py for real-time detection")

if __name__ == "__main__":
    main()