import os
import shutil
import random

DATASET_PATH = 'Driver Drowsiness Dataset (DDD)'
SPLIT_PATH = 'split_dataset'
CLASSES = ['Drowsy', 'Non Drowsy']
SPLITS = {'train': 0.7, 'val': 0.15, 'test': 0.15}

random.seed(42)

for split in SPLITS:
    for cls in CLASSES:
        split_dir = os.path.join(SPLIT_PATH, split, cls)
        os.makedirs(split_dir, exist_ok=True)

for cls in CLASSES:
    src_dir = os.path.join(DATASET_PATH, cls)
    images = [f for f in os.listdir(src_dir) if f.lower().endswith('.png')]
    random.shuffle(images)
    n_total = len(images)
    n_train = int(n_total * SPLITS['train'])
    n_val = int(n_total * SPLITS['val'])
    n_test = n_total - n_train - n_val
    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train+n_val],
        'test': images[n_train+n_val:]
    }
    for split, files in splits.items():
        for fname in files:
            src = os.path.join(src_dir, fname)
            dst = os.path.join(SPLIT_PATH, split, cls, fname)
            shutil.copy2(src, dst)
    print(f"{cls}: {n_train} train, {n_val} val, {n_test} test images")

print("Dataset split complete. Structure:")
for split in SPLITS:
    for cls in CLASSES:
        split_dir = os.path.join(SPLIT_PATH, split, cls)
        print(split_dir, len(os.listdir(split_dir)), "images")
