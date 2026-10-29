import cv2
import numpy as np
from ultralytics import YOLO


REAL_BOX_WIDTH = 20.0       # true physical width of the boxes (cm)
REAL_BOX_ASPECT_RATIO = 1.0 # width/height ratio of the box's front face
FOCAL_LENGTH_PX = 520.0     # pixel focal length of your lens; have to calculate

# handshake thresholds
DISTANCE_PICKUP_THRESHOLD = 30.0  # stop and switch sequence when 30cm away
ANGLE_ALIGN_THRESHOLD = 2.0       # within +/- 2 degrees of center axis

def main():

    model = YOLO("runs/detect/R2_robotics/YOLO26n_R2_Take2/weights/best.pt")    

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera feed unavailable.")
        return

    # camera frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cx = frame_width // 2  # optical center point X
    
    print("Output sequence initiated. Scanning for targets...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # running inference
        results = model(frame, verbose=False, conf=0.5)[0]
        
        target_detected = False
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # real vs fake classification
            if "fake" in class_name:
                print("0")
                continue
                
            elif "real" in class_name:
                print("1")
                target_detected = True
                
                xyxy = box.xyxy[0].cpu().numpy()
                w_px = xyxy[2] - xyxy[0]
                h_px = xyxy[3] - xyxy[1]
                x_center = xyxy[0] + (w_px / 2)
                
                # distance calculation
                distance = (REAL_BOX_WIDTH * FOCAL_LENGTH_PX) / w_px
                
                # angle calculation

                bearing_rad = np.arctan((x_center - cx) / FOCAL_LENGTH_PX)
                bearing_deg = np.degrees(bearing_rad)
                
                perceived_ratio = w_px / h_px
                ratio_error = perceived_ratio - REAL_BOX_ASPECT_RATIO
                face_angle_est = ratio_error * 45.0
                

                print(f"[TARGET LOCKED] Dist: {distance:.1f}cm | Bearing: {bearing_deg:.1f}° | Alignment: {face_angle_est:.1f}°")
                
                # --- OPERATION TERMINATION HANDSHAKE ---

                if distance <= DISTANCE_PICKUP_THRESHOLD and abs(bearing_deg) <= ANGLE_ALIGN_THRESHOLD:
                    print("[HANDSHAKE SUCCESS] Robot perfectly positioned. Terminating tracking sequence.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                
                break
                
        if not target_detected:
            print("Scanning... No real target visible.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()