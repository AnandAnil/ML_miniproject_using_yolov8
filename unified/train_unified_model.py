#!/usr/bin/env python3
"""
Unified Face + Drowsiness Detection Model Trainer
Trains a single YOLOv8 model based on yolov8m-face.pt that can detect faces and classify drowsiness

Uses your existing yolov8m-face.pt as the base model for better face detection performance.
"""

import os
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml

class UnifiedModelTrainer:
    def __init__(self, dataset_yaml_path="unified_face_drowsy_dataset/dataset.yaml",
                 base_model_path="yolov8m-face.pt", 
                 project_name="unified_face_drowsy"):
        self.dataset_yaml = dataset_yaml_path
        self.base_model_path = base_model_path
        self.project_name = project_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Check if base model exists
        if not Path(self.base_model_path).exists():
            print(f"Error: Base model not found at {self.base_model_path}")
            print("Please ensure yolov8m-face.pt is in the current directory")
            exit(1)
        
        print(f"Using base model: {self.base_model_path}")
    
    def verify_dataset(self):
        """Verify the dataset exists and is properly formatted"""
        if not Path(self.dataset_yaml).exists():
            print(f"Error: Dataset configuration not found at {self.dataset_yaml}")
            print("Please run create_unified_dataset.py first")
            return False
        
        # Load and verify dataset configuration
        with open(self.dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        dataset_path = Path(dataset_config['path'])
        
        # Check if directories exist and count files
        splits_info = {}
        required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
        
        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                print(f"Error: Required directory not found: {dir_path}")
                return False
            
            # Count files
            if 'images' in dir_name:
                file_count = len(list(dir_path.glob("*.jpg"))) + len(list(dir_path.glob("*.png")))
                split_name = dir_name.split('/')[1]
                splits_info[split_name] = splits_info.get(split_name, {})
                splits_info[split_name]['images'] = file_count
            else:
                file_count = len(list(dir_path.glob("*.txt")))
                split_name = dir_name.split('/')[1]
                splits_info[split_name] = splits_info.get(split_name, {})
                splits_info[split_name]['labels'] = file_count
        
        # Verify images and labels match
        print("Dataset verification:")
        total_images = 0
        for split, counts in splits_info.items():
            images = counts.get('images', 0)
            labels = counts.get('labels', 0)
            total_images += images
            print(f"  {split}: {images} images, {labels} labels", end="")
            if images == labels and images > 0:
                print(" ✓")
            elif images == 0:
                print(" (empty)")
            else:
                print(f" ✗ (mismatch!)")
                return False
        
        if total_images == 0:
            print("Error: No images found in dataset")
            return False
        
        print(f"Classes: {dataset_config['names']}")
        print(f"Total images: {total_images}")
        return True
    
    def check_class_balance(self):
        """Check class distribution in the dataset"""
        if not Path(self.dataset_yaml).exists():
            return None
            
        with open(self.dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        dataset_path = Path(dataset_config['path'])
        class_counts = {'train': [0, 0], 'val': [0, 0]}  # [alert_count, drowsy_count]
        
        for split in ['train', 'val']:
            label_dir = dataset_path / 'labels' / split
            if label_dir.exists():
                for label_file in label_dir.glob("*.txt"):
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.split()[0])
                                class_counts[split][class_id] += 1
        
        print("\nDataset Class Distribution:")
        for split, counts in class_counts.items():
            total = sum(counts)
            if total > 0:
                alert_pct = counts[0] / total * 100
                drowsy_pct = counts[1] / total * 100
                print(f"  {split}: Alert={counts[0]} ({alert_pct:.1f}%), Drowsy={counts[1]} ({drowsy_pct:.1f}%)")
                
                # Calculate class weights for imbalanced data
                if split == 'train':
                    # Weight inversely proportional to frequency
                    alert_weight = total / (2 * counts[0]) if counts[0] > 0 else 1.0
                    drowsy_weight = total / (2 * counts[1]) if counts[1] > 0 else 1.0
                    return [alert_weight, drowsy_weight]
        
        return None

    def train_model(self, epochs=100, imgsz=640, batch=16, patience=20, lr0=0.01):
        """Train the unified model with optimizations"""
        
        print("\nStarting model training...")
        print("=" * 50)
        print(f"Base model: {self.base_model_path}")
        print(f"Dataset: {self.dataset_yaml}")
        print(f"Epochs: {epochs}")
        print(f"Image size: {imgsz}")
        print(f"Batch size: {batch}")
        print(f"Device: {self.device}")
        print("=" * 50)
        
        # Check class balance and calculate weights
        class_weights = self.check_class_balance()
        if class_weights:
            print(f"Calculated class weights: Alert={class_weights[0]:.2f}, Drowsy={class_weights[1]:.2f}")
        
        # Load your face model as base
        model = YOLO(self.base_model_path)
        
        # Freeze backbone layers to preserve excellent face detection
        # Layers 0-9 are the backbone (feature extraction) - CRITICAL for face detection
        # Only train the neck (10-21) and head (22) for drowsiness classification
        freeze_layers = list(range(10))  # Freeze backbone layers 0-9
        print(f"Freezing backbone layers {freeze_layers} to preserve face detection performance")
        print("Training only neck and detection head for drowsiness classification")
        
        # Training parameters optimized for face detection + drowsiness classification
        train_args = {
            'data': self.dataset_yaml,
            'epochs': epochs,
            'imgsz': imgsz,          # Use passed image size
            'batch': batch,          # Use passed batch size
            'workers': 4,            # Parallel data loading
            'device': '0',           # Use GPU if available
            'project': None,         # Use default runs directory
            'name': self.project_name,
            'exist_ok': True,        # Overwrite existing
            'pretrained': True,      # Use pretrained weights
            'optimizer': 'auto',
            'verbose': True,
            'patience': patience,    # Use passed patience
            'single_cls': False,     # CRITICAL: Multi-class detection for drowsy/alert
            'freeze': freeze_layers, # Freeze backbone to preserve face detection
            
            # Data augmentation - moderate for faces
            'hsv_h': 0.010,      # Reduced hue augmentation
            'hsv_s': 0.5,        # Moderate saturation
            'hsv_v': 0.3,        # Moderate value
            'degrees': 5,        # Small rotation for faces
            'translate': 0.1,    # Small translation
            'scale': 0.5,        # Moderate scaling
            'shear': 2.0,        # Small shear
            'perspective': 0.0,  # No perspective for faces
            'flipud': 0.0,       # No vertical flip for faces
            'fliplr': 0.5,       # 50% horizontal flip
            # Advanced augmentation - reduced for face preservation
            'mosaic': 0.1,       # Minimal mosaic to preserve face detection quality
            'mixup': 0.05,       # Reduced mixup for cleaner face learning
            'copy_paste': 0.05,  # Reduced copy-paste for face preservation            # Learning rate and optimization
            'lr0': lr0,          # Use passed learning rate
            'lrf': 0.01,         # Lower final learning rate
            'momentum': 0.937,   # SGD momentum
            'weight_decay': 0.0005,  # Weight decay
            
            # Validation settings
            'val': True,          # Validate during training
            'plots': True,        # Save training plots
            'save_period': 10,    # Save checkpoint every N epochs
        }
        
        # Add class weights if calculated
        if class_weights:
            # Create a custom loss function or use focal loss
            train_args['cls'] = 1.5  # Increase classification loss weight even more
        
        # Train the model
        results = model.train(**train_args)
        
        # Return the trained model path
        model_path = f"runs/detect/{self.project_name}/weights/best.pt"
        print(f"\nTraining completed! Best model saved at: {model_path}")
        return model_path
    
    def validate_model(self, model_path=None):
        """Validate the trained model"""
        if model_path is None:
            model_path = f"runs/detect/{self.project_name}/weights/best.pt"
        
        if not Path(model_path).exists():
            print(f"Model not found at {model_path}")
            return None
        
        print(f"\nValidating model: {model_path}")
        model = YOLO(model_path)
        
        # Run validation
        metrics = model.val(data=self.dataset_yaml)
        
        # Print key metrics
        print("\nValidation Results:")
        print("=" * 30)
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        
        # Per-class metrics
        if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
            class_names = ['alert_face', 'drowsy_face']
            print("\nPer-class mAP50:")
            for i, class_name in enumerate(class_names):
                if i < len(metrics.box.maps):
                    print(f"  {class_name}: {metrics.box.maps[i]:.4f}")
        
        return metrics
    
    def test_inference(self, model_path=None, test_image=None):
        """Test inference on a sample image"""
        if model_path is None:
            model_path = f"runs/detect/{self.project_name}/weights/best.pt"
        
        if not Path(model_path).exists():
            print(f"Model not found at {model_path}")
            return
        
        model = YOLO(model_path)
        
        # Find a test image if not provided
        if test_image is None:
            test_dirs = [
                "unified_face_drowsy_dataset/images/val",
                "unified_face_drowsy_dataset/images/test",
                "split_dataset/val/Drowsy",
                "split_dataset/val/Non Drowsy"
            ]
            
            for test_dir in test_dirs:
                test_path = Path(test_dir)
                if test_path.exists():
                    test_images = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))
                    if test_images:
                        test_image = test_images[0]
                        break
        
        if test_image and Path(test_image).exists():
            print(f"\nTesting inference on: {test_image}")
            
            # Run inference
            results = model(test_image, conf=0.3)
            
            # Save annotated result
            output_path = f"unified_model_test_result.jpg"
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                results[0].save(output_path)
                print(f"Test result saved to: {output_path}")
                
                # Print detection results
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = ['alert_face', 'drowsy_face'][cls]
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            print(f"Detected: {class_name} (confidence: {conf:.3f}) at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
            else:
                print("No detections found in test image")
                # Still save the image for reference
                results[0].save(output_path)
                print(f"Test image (no detections) saved to: {output_path}")
        else:
            print("No test image found")
    
    def export_model(self, model_path=None, formats=['onnx']):
        """Export the model to different formats"""
        if model_path is None:
            model_path = f"runs/detect/{self.project_name}/weights/best.pt"
        
        if not Path(model_path).exists():
            print(f"Model not found at {model_path}")
            return
        
        model = YOLO(model_path)
        
        for format_type in formats:
            try:
                print(f"Exporting to {format_type}...")
                model.export(format=format_type)
                print(f"Export to {format_type} completed")
            except Exception as e:
                print(f"Failed to export to {format_type}: {e}")

def main():
    print("Unified Face + Drowsiness Detection Model Trainer")
    print("Using yolov8m-face.pt as base model")
    print("=" * 60)
    
    # Configuration - Full production training
    EPOCHS = 50           # Full training for best results
    BATCH_SIZE = 16       # Larger batch for stable training
    IMAGE_SIZE = 416      # Higher resolution for better accuracy
    PATIENCE = 15         # More patience for convergence
    LEARNING_RATE = 0.001 # Optimized learning rate for fine-tuning
    
    # Initialize trainer with optimized settings
    trainer = UnifiedModelTrainer(
        base_model_path="yolov8m-face.pt",
        project_name="unified_face_drowsy_production"  # Production model
    )
    
    # Verify dataset
    if not trainer.verify_dataset():
        print("Dataset verification failed. Please check your dataset.")
        print("Run create_unified_dataset.py if you haven't already.")
        return
    
    # Display training configuration
    print(f"\nTraining configuration:")
    print(f"  Base model: {trainer.base_model_path} (face-optimized)")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Device: {trainer.device}")
    
    # Ask user for confirmation
    response = input("\nProceed with training? (y/n): ").lower()
    if response != 'y':
        print("Training cancelled.")
        return
    
    # Train the model
    try:
        model_path = trainer.train_model(
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMAGE_SIZE,
            patience=PATIENCE,
            lr0=LEARNING_RATE
        )
        
        # Validate the model
        trainer.validate_model(model_path)
        
        # Test inference
        trainer.test_inference(model_path)
        
        # Export model (optional)
        export_response = input("\nExport model to ONNX format? (y/n): ").lower()
        if export_response == 'y':
            trainer.export_model(model_path, formats=['onnx'])
        
        print("\n" + "=" * 60)
        print("Training pipeline completed successfully!")
        print(f"Best model: {model_path}")
        print("Next step: Use unified_realtime_detection.py for real-time detection")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        print("Check your GPU memory, batch size, or dataset.")

if __name__ == "__main__":
    main()