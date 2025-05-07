import os
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "input/3c.mp4"  # Change as needed
OUTPUT_DIR = "output/silhouettes"
MODEL_PATH = "yolov8x-seg.pt"  # Download from Ultralytics if not present

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
MAX_FRAMES = 200

while True:
    ret, frame = cap.read()
    if not ret or frame_idx >= MAX_FRAMES:
        break
    frame_idx += 1
    results = model(frame)
    for r in results:
        if not hasattr(r, 'masks') or r.masks is None or r.masks.data is None:
            continue
        for j, mask in enumerate(r.masks.data):
            if int(r.boxes.cls[j]) != 0:  # Only person class
                continue
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            # Ensure mask is the same size as the frame
            if mask_np.shape != frame.shape[:2]:
                mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            if np.count_nonzero(mask_np) == 0:
                print("Empty mask, skipping.")
                continue
            track_id = int(r.boxes.id[j]) if hasattr(r.boxes, 'id') and r.boxes.id is not None else j
            person_dir = os.path.join(OUTPUT_DIR, f"track_{track_id}")
            os.makedirs(person_dir, exist_ok=True)
            mask_filename = os.path.join(person_dir, f"{frame_idx:04d}.png")
            cv2.imwrite(mask_filename, mask_np)
            print(f"Saved: {mask_filename}, shape: {mask_np.shape}")
cap.release()
print("Silhouette extraction complete.")