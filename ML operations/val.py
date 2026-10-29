from ultralytics import YOLO
import torch

# Load the weights from the 200-epoch mark
model = YOLO("runs/detect/R2_robotics/YOLO26n_R2_Take3/weights/last.pt")

print("📊 AUDITING EPOCH 50 PERFORMANCE...")
metrics = model.val(data="whole_dataset_with_json_labels/Train Dataset/data.yaml", device="mps")

print("-" * 30)
print(f"Current mAP50: {metrics.box.map50:.4f}")
print(f"Current mAP50-95: {metrics.box.map:.4f}")
print(f"Fitness Score: {metrics.fitness:.4f}")
print("-" * 30)