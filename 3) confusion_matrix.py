from ultralytics import YOLO
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
test_model='yolov8m'
# Paths
MODEL_PATH = 'yolo_drowsiness/'+test_model+'_cls_drowsy/weights/best.pt'
DATASET_PATH = 'split_dataset'
CLASSES = ['Drowsy', 'Non Drowsy']

# Load model
model = YOLO(MODEL_PATH)
print(f'Model loaded from: {MODEL_PATH}')

# Collect all images and true labels from train, val, test
image_paths = []
true_labels = []
for split in ['train', 'val', 'test']:
    for idx, cls in enumerate(CLASSES):
        folder = os.path.join(DATASET_PATH, split, cls)
        for fname in os.listdir(folder):
            if fname.lower().endswith('.png'):
                image_paths.append(os.path.join(folder, fname))
                true_labels.append(idx)

# Predict
pred_labels = []
for img_path in image_paths:
    results = model(img_path)
    pred = int(np.argmax(results[0].probs.data.cpu().numpy()))
    pred_labels.append(pred)

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Drowsiness Detection')
plt.show()

# Print confusion matrix
print('Confusion Matrix:')
print(cm)
