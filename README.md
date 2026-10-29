# Edge Vision & Geometric Pose Estimation Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Geometric_CV-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Ultralytics-orange)
![Edge](https://img.shields.io/badge/Deployment-Edge_Robotics-lightgrey)

A real-time, lightweight computer vision pipeline engineered for autonomous robotic target acquisition. Developed for high-stakes competition environments (ABU ROBOCON), this module fuses deep learning-based object detection with geometric mathematical models to provide spatial awareness, distance estimation, and angular bearing to the control systems without relying on depth sensors.

## System Architecture

The pipeline is designed to operate on compute-constrained edge devices, converting raw RGB camera feeds into actionable 3D spatial coordinates and mechanical handshake triggers.

### 1. Neural Perception Layer
- **Model:** A custom-trained YOLO architecture (`YOLO26n_R2_Take2`), heavily optimized for edge inference.
- **Classification:** The model is trained to differentiate between adversarial/decoy targets and genuine objectives (`"fake"` vs `"real"` class definitions).

### 2. Geometric Computer Vision Layer
Once a valid target is isolated, the system bypasses heavy 3D-pose estimation networks in favor of high-speed, deterministic geometric approximations using the pinhole camera model:

- **Distance Estimation:** 
  Calculated using focal length ($f_{px}$) and the known physical width of the target ($W_{real}$):
  $$D = \frac{W_{real} \cdot f_{px}}{w_{px}}$$
- **Bearing Calculation:** 
  The angular offset from the camera's optical center ($c_x$):
  $$\theta_{bearing} = \arctan\left(\frac{x_{center} - c_x}{f_{px}}\right)$$
- **Surface Alignment Estimation:** 
  Approximated by calculating the deformation of the bounding box's aspect ratio against the target's known true physical aspect ratio.

### 3. Control System Handshake
The vision module operates continuously until specific spatial thresholds are met. When the target distance falls below $30.0$ cm and the bearing aligns within $\pm 2.0^{\circ}$, the system successfully completes the terminal handshake, yielding execution to the robotic manipulator sequence.

## Repository Structure
- `/detect/output.py`: Core inference and geometric calculation loop.
- `/train/`: Scripts and hyperparameter configurations for the YOLO training cycles.
- `/weights/`: Exported `.pt` model files.
- `/config/`: Camera intrinsic matrices and focal length calibration parameters.

## Local Execution

### Prerequisites
Ensure a local environment with PyTorch and Ultralytics installed. 

```bash
git clone https://github.com/muhammad-bats/Autonomous-Geometric-Spatial-Perception.git
cd Geometric-Edge-CV
pip install -r requirements.txt
```

### Running the Inference Pipeline
Attach a calibrated webcam and execute the detection loop:

``` bash
python output.py
```
>Note: If utilizing a different camera sensor, you must recalculate `FOCAL_LENGTH_PX` using a standard checkerboard calibration script to ensure distance accuracy.