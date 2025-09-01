import splitfolders
import os
import shutil

# Configuration
DATASET_PATH = 'Driver Drowsiness Dataset (DDD)'
SPLIT_PATH = 'split_dataset'
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Remove existing split folder if it exists
if os.path.exists(SPLIT_PATH):
    shutil.rmtree(SPLIT_PATH)
    print(f"Removed existing folder: {SPLIT_PATH}")

print(f"Splitting dataset from '{DATASET_PATH}' to '{SPLIT_PATH}'")
print(f"Split ratios - Train: {TRAIN_RATIO}, Validation: {VAL_RATIO}, Test: {TEST_RATIO}")

# Split the dataset using splitfolders
splitfolders.ratio(
    DATASET_PATH,           # Input folder path
    output=SPLIT_PATH,      # Output folder path
    seed=42,                # Random seed for reproducibility
    ratio=(TRAIN_RATIO, VAL_RATIO, TEST_RATIO),  # Train, validation, test ratios
    group_prefix=None,      # To ignore prefix in file names
    move=False              # Copy files instead of moving them
)

print("Dataset split complete. Structure:")
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(SPLIT_PATH, split)
    if os.path.exists(split_dir):
        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"{split}/{cls}: {count} images")
