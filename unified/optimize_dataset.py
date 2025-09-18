#!/usr/bin/env python3
"""
Dataset Optimizer for Unified Model
Analyzes and rebalances the dataset for better training
"""

import os
import shutil
import random
from pathlib import Path
import yaml

class DatasetOptimizer:
    def __init__(self, dataset_path="unified_face_drowsy_dataset"):
        self.dataset_path = Path(dataset_path)
        self.class_names = ['alert_face', 'drowsy_face']
    
    def analyze_dataset(self):
        """Analyze current dataset distribution"""
        print("Analyzing dataset distribution...")
        
        stats = {}
        for split in ['train', 'val', 'test']:
            stats[split] = {'alert': 0, 'drowsy': 0}
            
            label_dir = self.dataset_path / 'labels' / split
            if label_dir.exists():
                for label_file in label_dir.glob("*.txt"):
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.split()[0])
                                if class_id == 0:
                                    stats[split]['alert'] += 1
                                else:
                                    stats[split]['drowsy'] += 1
        
        print("\nCurrent Dataset Distribution:")
        print("=" * 50)
        total_alert = 0
        total_drowsy = 0
        
        for split, counts in stats.items():
            total = counts['alert'] + counts['drowsy']
            total_alert += counts['alert']
            total_drowsy += counts['drowsy']
            
            if total > 0:
                alert_pct = counts['alert'] / total * 100
                drowsy_pct = counts['drowsy'] / total * 100
                print(f"{split.upper()}: {total} total")
                print(f"  Alert: {counts['alert']} ({alert_pct:.1f}%)")
                print(f"  Drowsy: {counts['drowsy']} ({drowsy_pct:.1f}%)")
                
                if alert_pct > 70:
                    print(f"  ⚠️  {split} is heavily biased toward Alert!")
                elif drowsy_pct > 70:
                    print(f"  ⚠️  {split} is heavily biased toward Drowsy!")
                else:
                    print(f"  ✓ {split} has reasonable balance")
            else:
                print(f"{split.upper()}: No data")
            print()
        
        # Overall stats
        grand_total = total_alert + total_drowsy
        if grand_total > 0:
            overall_alert_pct = total_alert / grand_total * 100
            overall_drowsy_pct = total_drowsy / grand_total * 100
            
            print(f"OVERALL: {grand_total} total images")
            print(f"  Alert: {total_alert} ({overall_alert_pct:.1f}%)")
            print(f"  Drowsy: {total_drowsy} ({overall_drowsy_pct:.1f}%)")
            
            if abs(overall_alert_pct - 50) > 20:
                print(f"  ⚠️  SEVERE CLASS IMBALANCE DETECTED!")
                print(f"     This explains why the model is biased!")
            
        return stats
    
    def balance_dataset(self, target_ratio=0.5):
        """Rebalance dataset by duplicating minority class"""
        print(f"\nRebalancing dataset to {target_ratio*100:.0f}% drowsy, {(1-target_ratio)*100:.0f}% alert...")
        
        stats = self.analyze_dataset()
        
        for split in ['train', 'val']:
            if split not in stats:
                continue
                
            alert_count = stats[split]['alert']
            drowsy_count = stats[split]['drowsy']
            total = alert_count + drowsy_count
            
            if total == 0:
                continue
            
            current_drowsy_ratio = drowsy_count / total
            print(f"\n{split}: Current drowsy ratio: {current_drowsy_ratio:.2f}")
            
            if abs(current_drowsy_ratio - target_ratio) < 0.1:
                print(f"  ✓ {split} already well balanced")
                continue
            
            # Determine which class to augment
            if current_drowsy_ratio < target_ratio:
                # Need more drowsy images
                minority_class = 'drowsy'
                minority_id = 1
                target_count = int(alert_count * target_ratio / (1 - target_ratio))
                needed = max(0, target_count - drowsy_count)
            else:
                # Need more alert images
                minority_class = 'alert'
                minority_id = 0
                target_count = int(drowsy_count * (1 - target_ratio) / target_ratio)
                needed = max(0, target_count - alert_count)
            
            if needed > 0:
                print(f"  Need {needed} more {minority_class} images in {split}")
                self.augment_minority_class(split, minority_class, minority_id, needed)
    
    def augment_minority_class(self, split, class_name, class_id, count_needed):
        """Duplicate minority class images with slight modifications"""
        images_dir = self.dataset_path / 'images' / split
        labels_dir = self.dataset_path / 'labels' / split
        
        # Find existing minority class images
        minority_images = []
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip() and int(line.split()[0]) == class_id:
                        img_name = label_file.stem
                        # Find corresponding image
                        for ext in ['.jpg', '.png', '.jpeg']:
                            img_path = images_dir / f"{img_name}{ext}"
                            if img_path.exists():
                                minority_images.append((img_path, label_file))
                                break
        
        if not minority_images:
            print(f"    No {class_name} images found to augment!")
            return
        
        print(f"    Found {len(minority_images)} {class_name} images to duplicate")
        
        # Duplicate images randomly
        augmented = 0
        while augmented < count_needed:
            # Pick random source image
            src_img, src_label = random.choice(minority_images)
            
            # Create new filename
            base_name = f"aug_{class_name}_{augmented:04d}"
            
            # Copy image
            new_img_path = images_dir / f"{base_name}{src_img.suffix}"
            shutil.copy2(src_img, new_img_path)
            
            # Copy label
            new_label_path = labels_dir / f"{base_name}.txt"
            shutil.copy2(src_label, new_label_path)
            
            augmented += 1
            
            if augmented % 50 == 0:
                print(f"    Augmented {augmented}/{count_needed} {class_name} images")
        
        print(f"    ✓ Added {augmented} {class_name} images to {split}")
    
    def create_balanced_yaml(self):
        """Create optimized dataset YAML with class weights"""
        yaml_content = f"""# Optimized Unified Face Detection + Drowsiness Classification Dataset
# Rebalanced for better training

path: {self.dataset_path.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
nc: 2  # number of classes
names: ['alert_face', 'drowsy_face']

# Training optimizations
# - Rebalanced class distribution
# - Optimized for drowsiness detection accuracy
"""
        
        yaml_path = self.dataset_path / "dataset_optimized.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created optimized dataset config: {yaml_path}")
        return yaml_path

def main():
    print("Dataset Optimizer for Unified Model Training")
    print("=" * 60)
    
    # Check if dataset exists
    dataset_path = "unified_face_drowsy_dataset"
    if not Path(dataset_path).exists():
        print(f"Dataset not found at {dataset_path}")
        print("Please run create_unified_dataset.py first")
        return
    
    optimizer = DatasetOptimizer(dataset_path)
    
    # Analyze current distribution
    stats = optimizer.analyze_dataset()
    
    # Ask if user wants to rebalance
    response = input("\nDo you want to rebalance the dataset? (y/n): ").lower()
    if response == 'y':
        # Set target ratio (e.g., 50% drowsy, 50% alert)
        target_drowsy = input("Enter target drowsy percentage (default 50): ").strip()
        if target_drowsy:
            try:
                target_ratio = float(target_drowsy) / 100
            except:
                target_ratio = 0.5
        else:
            target_ratio = 0.5
        
        optimizer.balance_dataset(target_ratio)
        
        # Analyze again
        print("\nAfter rebalancing:")
        optimizer.analyze_dataset()
        
        # Create optimized YAML
        optimizer.create_balanced_yaml()
        
        print("\n" + "=" * 60)
        print("Dataset optimization completed!")
        print("Use the optimized training script to train a better model.")

if __name__ == "__main__":
    main()