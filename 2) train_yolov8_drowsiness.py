from ultralytics import YOLO
import os

# Paths

# Use split dataset
DATASET_PATH = 'split_dataset'
CLASSES = ['Drowsy', 'Non Drowsy']

# Check split folders
for split in ['train', 'val', 'test']:
    for cls in CLASSES:
        folder = os.path.join(DATASET_PATH, split, cls)
        if not os.path.isdir(folder):
            raise Exception(f"Missing folder: {folder}")
        print(f"Found {split} class folder: {folder}, {len(os.listdir(folder))} images")

# Train YOLOv8 classifier
model = YOLO('yolov8m-cls.pt')

results = model.train(
    data=DATASET_PATH,
    epochs=20,
    imgsz=224,
    project='yolo_drowsiness',
    name='yolov8n_cls_drowsy',
    exist_ok=True,
    split='val'
)

print("Training complete!")
print("Results summary:")
print(f"Top-1 Accuracy: {results.top1:.4f}")
print(f"Top-5 Accuracy: {results.top5:.4f}")
print(f"Fitness Score: {results.fitness:.4f}")

# The best weights are automatically saved in the project directory
best_weights_path = 'yolo_drowsiness/yolov8n_cls_drowsy/weights/best.pt'
print(f"Best weights saved at: {best_weights_path}")
