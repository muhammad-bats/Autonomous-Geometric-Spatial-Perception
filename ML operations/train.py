import os
import ultralytics
from ultralytics import YOLO

# Take 3 part I | comment out when needed.
"""
model = YOLO("yolo26n.pt")

# training arguments (tentative)
results = model.train(

    data="/Users/muhammadhassan/Robotics/whole_dataset_with_json_labels/Train Dataset/data.yaml",
    project="R2_robotics",
    name="YOLO26n_R2_Take3",
    

    device="mps",
    batch=32,
    workers=8,
    cache=False,
    imgsz=640,
    

    epochs=300,
    patience=50,
    optimizer="AdamW", # MuSGD or AdamW | AdamW rn so no bottleneck
    nbs=64,
    seed=42,
    
    amp=True,
    close_mosaic=10,
    overlap_mask=True,
    val=True,
    

    exist_ok=True,
    save=True,
)

# Take 3 part II | comment out when needed.
# resume=True loads arguments in the 'args.yaml' in the same folder
"""
model = YOLO("/Users/muhammadhassan/Robotics/whole_dataset_with_json_labels/runs/detect/R2_robotics/YOLO26n_R2_Take3/weights/last.pt")

print("RESUMING FROM EPOCH 228...")

results = model.train(resume=True)

print("Training complete.\n Starting Export")

# tensorrt export
model.export(format="engine")

# coreml xml export
model.export(format="coreml", nms=True, int8=True) 

# onnx export
model.export(format="onnx", opset=12, dynamic=True)

print("All formats exported successfully.")