import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import pickle
import numpy as np
import argparse
from collections import defaultdict
from scipy.spatial import distance
from models.yolo import YOLOv8Tracker
from models.transreid import load_transreid_model
from utils.pose import extract_skeleton_batch
from utils.gaits import compute_body_ratios, compute_gait_features
from utils.gait.gait_infer import OpenGaitEmbedder
from ultralytics import YOLO as YOLOSeg
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.color import rgb2hsv
from scipy.stats import skew, kurtosis

# Add argument parser for command-line options
parser = argparse.ArgumentParser(description='Create enhanced identity database from video.')
parser.add_argument('--video', default="input/3c.mp4", help='Path to video file')
parser.add_argument('--output', default="identity_database.pkl", help='Output database file')
parser.add_argument('--interactive', action='store_true', help='Use interactive naming')
parser.add_argument('--auto-name', action='store_true', help='Use automatic naming with predefined map')
parser.add_argument('--frames', type=int, default=300, help='Number of frames to process')
parser.add_argument('--skip', type=int, default=2, help='Frame skip rate')
args = parser.parse_args()

VIDEO_PATH = args.video
YOLO_WEIGHTS = "weights/yolo12x.pt"
YOLOSEG_WEIGHTS = "weights/yolo12x-seg.pt"
TRANSREID_WEIGHTS = "weights/transreid_vitbase.pth"
MMPOSE_WEIGHTS = "weights/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth"
GAIT_WEIGHTS = "checkpoints/GaitBase_DA-60000.pt"
OUTPUT_DB = args.output
DEVICE = "mps"  # or "cuda" or "cpu"

# Define predefined name mapping
NAME_MAPPING = {
    1: "Prachit",
    2: "Ashutosh",
    3: "Ojasv",
    4: "Nayan"
}

print(f"Initializing models for enhanced feature extraction...")

# Initialize models
yolo = YOLOv8Tracker(YOLO_WEIGHTS, device=DEVICE, conf_threshold=0.25)
transreid = load_transreid_model(TRANSREID_WEIGHTS, DEVICE)
gait_embedder = OpenGaitEmbedder(weights_path=GAIT_WEIGHTS, device=DEVICE)
yoloseg = YOLOSeg(YOLOSEG_WEIGHTS)

print("Initializing tracking with dual gait feature extraction...")

# Extended feature tracking
track_features = defaultdict(lambda: {
    "appearance": [],
    "raw_skeletons": [],      # Store raw skeleton points
    "body_ratios": [],        # Derived body ratios
    "heights": [],
    "contexts": [],
    "silhouettes": [],        # For OpenGait processing
    "skeleton_sequences": [], # Store continuous skeleton sequences for skeleton gait
    "crops": [],
    "color_hists": [],        # Color histograms 
    "hog_features": [],       # Texture features
    "motion_patterns": [],    # Motion data
    "track_confidence": [],   # Detection confidence scores
    "full_body_visible": [],  # Whether full body is visible
    "industrial_pose_features": [],  # Industrial pose features
    "industrial_color_features": []  # Industrial-specific color features
})

# Motion pattern tracker
motion_trackers = {}

print(f"Processing video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
frame_count = 0

# Enhanced features for industrial environments
def compute_normalized_pose_features(keypoints):
    """
    Compute pose features that are invariant to distance from camera
    and suitable for industrial environments.
    """
    if not keypoints or len(keypoints) < 17:
        return {}
        
    # Filter valid keypoints (confidence > threshold)
    valid_keypoints = [k for k in keypoints if k[2] > 0.2]
    if len(valid_keypoints) < 10:  # Need enough points for meaningful analysis
        return {}
    
    # Get a scale reference (torso height or similar)
    torso_length = None
    if all(keypoints[i][2] > 0.2 for i in [5, 11]):  # shoulder to hip
        torso_length = distance.euclidean(keypoints[5][:2], keypoints[11][:2])
    elif all(keypoints[i][2] > 0.2 for i in [6, 12]):  # shoulder to hip (other side)
        torso_length = distance.euclidean(keypoints[6][:2], keypoints[12][:2])
        
    if not torso_length:
        return {}
    
    features = {}
    
    # 1. Normalized bone lengths (invariant to distance)
    bone_connections = [
        (5, 7, "upper_arm_left_norm"),
        (7, 9, "lower_arm_left_norm"),
        (6, 8, "upper_arm_right_norm"),
        (8, 10, "lower_arm_right_norm"),
        (11, 13, "upper_leg_left_norm"),
        (13, 15, "lower_leg_left_norm"),
        (12, 14, "upper_leg_right_norm"),
        (14, 16, "lower_leg_right_norm"),
        (5, 6, "shoulder_width_norm"),
        (11, 12, "hip_width_norm"),
        (0, 1, "head_width_norm"),
    ]
    
    for p1, p2, name in bone_connections:
        if keypoints[p1][2] > 0.2 and keypoints[p2][2] > 0.2:
            dist = distance.euclidean(keypoints[p1][:2], keypoints[p2][:2])
            # Normalize by torso length (makes it scale-invariant)
            features[name] = dist / torso_length
    
    # 2. Body proportion ratios (highly stable across distances)
    if "upper_leg_left_norm" in features and "lower_leg_left_norm" in features:
        features["leg_proportion_left"] = features["upper_leg_left_norm"] / features["lower_leg_left_norm"]
    
    if "upper_leg_right_norm" in features and "lower_leg_right_norm" in features:
        features["leg_proportion_right"] = features["upper_leg_right_norm"] / features["lower_leg_right_norm"]
    
    if "upper_arm_left_norm" in features and "lower_arm_left_norm" in features:
        features["arm_proportion_left"] = features["upper_arm_left_norm"] / features["lower_arm_left_norm"]
    
    if "upper_arm_right_norm" in features and "lower_arm_right_norm" in features:
        features["arm_proportion_right"] = features["upper_arm_right_norm"] / features["lower_arm_right_norm"]
        
    # 3. Angular features (invariant to scale and translation)
    # Shoulder angle
    if all(keypoints[i][2] > 0.2 for i in [5, 6]):
        shoulder_vector = np.array(keypoints[6][:2]) - np.array(keypoints[5][:2])
        features["shoulder_angle"] = np.arctan2(shoulder_vector[1], shoulder_vector[0])
    
    # Hip angle
    if all(keypoints[i][2] > 0.2 for i in [11, 12]):
        hip_vector = np.array(keypoints[12][:2]) - np.array(keypoints[11][:2])
        features["hip_angle"] = np.arctan2(hip_vector[1], hip_vector[0])
    
    # 4. Torso shape features - useful for identifying industrial clothing
    if all(keypoints[i][2] > 0.2 for i in [5, 6, 11, 12]):
        shoulder_width = distance.euclidean(keypoints[5][:2], keypoints[6][:2])
        hip_width = distance.euclidean(keypoints[11][:2], keypoints[12][:2])
        features["torso_shape"] = shoulder_width / hip_width  # Distinguishes between uniform types
    
    # 5. Industrial posture indicators
    # Head position relative to shoulders (safety helmet can affect this)
    if all(keypoints[i][2] > 0.2 for i in [0, 5, 6]):
        nose = np.array(keypoints[0][:2])
        shoulder_center = (np.array(keypoints[5][:2]) + np.array(keypoints[6][:2])) / 2
        features["head_forward_position"] = (nose[0] - shoulder_center[0]) / torso_length
        features["head_height_position"] = (shoulder_center[1] - nose[1]) / torso_length
    
    # 6. Triangle features - good for capturing posture invariant to distance
    if all(keypoints[i][2] > 0.2 for i in [5, 6, 11]):
        # Triangle: left shoulder, right shoulder, left hip
        a = distance.euclidean(keypoints[5][:2], keypoints[6][:2])
        b = distance.euclidean(keypoints[6][:2], keypoints[11][:2])
        c = distance.euclidean(keypoints[11][:2], keypoints[5][:2])
        s = (a + b + c) / 2
        try:
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))  # Heron's formula
            features["upper_torso_triangle"] = area / (torso_length ** 2)  # Normalized area
        except:
            pass  # Invalid triangle
    
    return features

# Function to extract color features focused on industrial workwear
def extract_industrial_color_features(crop):
    """Extract color features specific to industrial clothing and PPE."""
    if crop is None or crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return None
        
    # Resize for consistency
    resized = cv2.resize(crop, (64, 128))
    
    # Split the image into regions (head, torso, legs)
    head_region = resized[0:30, :]
    torso_region = resized[30:80, :]
    legs_region = resized[80:, :]
    
    features = {}
    
    # Check for safety colors (high-vis yellow, orange, etc.)
    for region_name, region in [("head", head_region), 
                               ("torso", torso_region), 
                               ("legs", legs_region)]:
        
        # Convert to HSV for better color identification
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Safety yellow-green detection (common in vests)
        yellow_green_mask = cv2.inRange(hsv, (25, 100, 100), (65, 255, 255))
        features[f"{region_name}_safety_yellow"] = np.count_nonzero(yellow_green_mask) / region.size
        
        # Safety orange detection
        orange_mask = cv2.inRange(hsv, (5, 100, 100), (25, 255, 255))
        features[f"{region_name}_safety_orange"] = np.count_nonzero(orange_mask) / region.size
        
        # Blue workwear detection (common in industrial uniforms)
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        features[f"{region_name}_workwear_blue"] = np.count_nonzero(blue_mask) / region.size
        
        # Create color histogram for each region (using fewer bins for robustness)
        color_hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        color_hist = cv2.normalize(color_hist, color_hist).flatten()
        features[f"{region_name}_color_hist"] = color_hist
    
    return features

while frame_count < args.frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx += 1
    if frame_idx % args.skip != 0:  # Skip frames for efficiency
        continue
    
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processing frame {frame_count}...")
    
    # Person detection and tracking
    detections = yolo.process_frame(frame)
    if not detections:
        continue
    
    ids, bboxes, confs = zip(*detections)
    
    # Extract full skeleton information
    keypoints_list = extract_skeleton_batch(frame, list(bboxes), weights=MMPOSE_WEIGHTS, device=DEVICE)
    
    # Silhouette extraction using YOLOv8-seg for gait analysis
    seg_results = yoloseg(frame)
    seg_masks = {}
    for r in seg_results:
        if not hasattr(r, 'masks') or r.masks is None or r.masks.data is None:
            continue
        for j, mask in enumerate(r.masks.data):
            if int(r.boxes.cls[j]) != 0:
                continue  # Only person class
            # Try to get track id if available, else use detection index
            track_id = int(r.boxes.id[j]) if hasattr(r.boxes, 'id') and r.boxes.id is not None else j
            mask_np = (mask.cpu().numpy() > 0.5).astype(np.float32)
            seg_masks[track_id] = mask_np
    
    for tid, bbox, conf, keypoints in zip(ids, bboxes, confs, keypoints_list):
        # Extract person crop
        x1, y1, x2, y2 = int(max(0, bbox[0])), int(max(0, bbox[1])), \
                         int(min(frame.shape[1], bbox[2])), int(min(frame.shape[0], bbox[3]))
        crop = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
        
        if crop is None or crop.size == 0:
            continue
        
        # Store the crop image
        track_features[tid]["crops"].append(crop)
        
        # 1. Appearance features from TransReID
        feat = transreid.extract_features(crop)
        track_features[tid]["appearance"].append(feat.cpu().numpy())
        
        # 2. Raw skeleton points - store the full keypoint array
        is_full_body = all(kp[2] > 0.2 for kp in keypoints) if keypoints else False
        track_features[tid]["raw_skeletons"].append(keypoints)
        track_features[tid]["full_body_visible"].append(is_full_body)
        
        # Also maintain continuous sequences for skeleton-based gait
        if "skeleton_sequences" not in track_features[tid]:
            track_features[tid]["skeleton_sequences"] = []
            
        # Check if we have a new sequence or continuing existing
        if len(track_features[tid]["skeleton_sequences"]) == 0 or len(track_features[tid]["skeleton_sequences"][-1]) >= 30:
            # Start a new sequence if we reached max length or first sequence
            track_features[tid]["skeleton_sequences"].append([keypoints])
        else:
            # Continue existing sequence
            track_features[tid]["skeleton_sequences"][-1].append(keypoints)
        
        # Extract normalized pose features for industrial contexts
        industrial_pose_features = compute_normalized_pose_features(keypoints)
        track_features[tid]["industrial_pose_features"].append(industrial_pose_features)
        
        # 3. Body ratios - derived measurements
        ratios = compute_body_ratios(keypoints) if keypoints else {}
        track_features[tid]["body_ratios"].append(ratios)
        
        # 4. Height information
        height = bbox[3] - bbox[1]
        track_features[tid]["heights"].append(height)
        
        # 5. Context information
        context = {
            "frame_idx": frame_idx, 
            "conf": conf,
            "scene_position": ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2),
            "frame_size": frame.shape[:2]
        }
        track_features[tid]["contexts"].append(context)
        
        # 6. Color histogram features (using multiple color spaces)
        if crop is not None and crop.size > 0:
            # RGB histogram
            color_hist_rgb = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            color_hist_rgb = cv2.normalize(color_hist_rgb, color_hist_rgb).flatten()
            
            # HSV histogram - better for color distinction
            crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            color_hist_hsv = cv2.calcHist([crop_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            color_hist_hsv = cv2.normalize(color_hist_hsv, color_hist_hsv).flatten()
            
            # Combined color feature
            color_hist = np.concatenate([color_hist_rgb, color_hist_hsv])
            track_features[tid]["color_hists"].append(color_hist)
            
            # 7. Texture features (HOG)
            try:
                if crop.shape[0] >= 64 and crop.shape[1] >= 64:
                    resized_crop = cv2.resize(crop, (64, 128))
                    gray_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2GRAY)
                    hog_feat = hog(gray_crop, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=False)
                    track_features[tid]["hog_features"].append(hog_feat)
                else:
                    track_features[tid]["hog_features"].append(None)
            except Exception as e:
                print(f"HOG extraction error: {e}")
                track_features[tid]["hog_features"].append(None)
        
        # Extract industrial-specific color features (safety gear, uniforms)
        industrial_color = extract_industrial_color_features(crop)
        if industrial_color:
            track_features[tid]["industrial_color_features"].append(industrial_color)
        
        # 8. Motion patterns
        if tid not in motion_trackers:
            motion_trackers[tid] = {"positions": [], "velocities": []}
        
        position = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
        motion_trackers[tid]["positions"].append(position)
        
        if len(motion_trackers[tid]["positions"]) >= 2:
            prev_pos = motion_trackers[tid]["positions"][-2]
            velocity = (position[0] - prev_pos[0], position[1] - prev_pos[1])
            motion_trackers[tid]["velocities"].append(velocity)
            
            # Compute motion features if we have enough history
            if len(motion_trackers[tid]["velocities"]) >= 3:
                velocities = motion_trackers[tid]["velocities"][-5:] if len(motion_trackers[tid]["velocities"]) >= 5 else motion_trackers[tid]["velocities"]
                vx = [v[0] for v in velocities]
                vy = [v[1] for v in velocities]
                
                motion_features = {
                    "velocity_mean_x": np.mean(vx),
                    "velocity_mean_y": np.mean(vy),
                    "velocity_std_x": np.std(vx),
                    "velocity_std_y": np.std(vy),
                    "velocity_skew_x": skew(vx) if len(vx) >= 3 else 0,
                    "velocity_skew_y": skew(vy) if len(vy) >= 3 else 0,
                }
                track_features[tid]["motion_patterns"].append(motion_features)
            else:
                track_features[tid]["motion_patterns"].append(None)
        else:
            track_features[tid]["motion_patterns"].append(None)
        
        # 9. Track confidence
        track_features[tid]["track_confidence"].append(conf)
        
        # 10. Silhouette for gait (from seg_masks if available)
        if tid in seg_masks:
            track_features[tid]["silhouettes"].append(seg_masks[tid])

cap.release()
print(f"Video processing complete. Processed {frame_count} frames.")

# Determine naming approach
track_id_to_name = {}

# Option 1: Automatic naming with predefined mapping
if args.auto_name:
    print("Using predefined name mapping...")
    for tid in track_features.keys():
        # Use name from mapping or default name
        if tid in NAME_MAPPING:
            track_id_to_name[tid] = NAME_MAPPING[tid]
            print(f"Assigned name '{NAME_MAPPING[tid]}' to track ID {tid}")
        else:
            track_id_to_name[tid] = f"Person_{tid}"
            print(f"No predefined name for track ID {tid}, using default: 'Person_{tid}'")

# Option 2: Interactive naming with visual inspection
elif args.interactive:
    print("Starting interactive naming...")
    for tid, feats in track_features.items():
        # Try to get a good representative image for the track
        if feats["crops"]:
            # Find the crop with highest detection confidence
            best_idx = np.argmax(feats["track_confidence"]) if feats["track_confidence"] else 0
            best_crop = feats["crops"][best_idx] if best_idx < len(feats["crops"]) else feats["crops"][0]
            
            # Display the best crop and all available skeleton points
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(best_crop, cv2.COLOR_BGR2RGB))
            plt.title(f"Track ID: {tid}")
            plt.axis('off')
            
            # Show skeleton visualization if available
            if feats["raw_skeletons"] and best_idx < len(feats["raw_skeletons"]) and feats["raw_skeletons"][best_idx]:
                plt.subplot(1, 2, 2)
                # Create a blank image for skeleton visualization
                blank = np.ones((256, 256, 3), dtype=np.uint8) * 255
                kpts = feats["raw_skeletons"][best_idx]
                
                # Scale keypoints to fit the blank image
                if kpts:
                    # Get bounding box of keypoints
                    valid_kpts = [(int(k[0]), int(k[1])) for k in kpts if k[2] > 0]
                    if valid_kpts:
                        min_x = min([k[0] for k in valid_kpts])
                        max_x = max([k[0] for k in valid_kpts])
                        min_y = min([k[1] for k in valid_kpts])
                        max_y = max([k[1] for k in valid_kpts])
                        
                        # Scale and center keypoints
                        scale = 200 / max(max_x - min_x, max_y - min_y, 1)
                        offset_x = 128 - (min_x + max_x) * scale / 2
                        offset_y = 128 - (min_y + max_y) * scale / 2
                        
                        # Draw connections (simplified skeleton connections)
                        connections = [
                            (0, 1), (1, 2), (2, 3), (3, 4),  # Head to right arm
                            (0, 5), (5, 6), (6, 7),          # Head to left arm
                            (0, 8), (8, 9), (9, 10),         # Head to right leg
                            (0, 11), (11, 12), (12, 13),     # Head to left leg
                        ]
                        
                        for connection in connections:
                            if kpts[connection[0]][2] > 0.2 and kpts[connection[1]][2] > 0.2:
                                pt1 = (int(kpts[connection[0]][0] * scale + offset_x),
                                      int(kpts[connection[0]][1] * scale + offset_y))
                                pt2 = (int(kpts[connection[1]][0] * scale + offset_x),
                                      int(kpts[connection[1]][1] * scale + offset_y))
                                cv2.line(blank, pt1, pt2, (0, 0, 255), 2)
                        
                        # Draw keypoints
                        for i, kpt in enumerate(kpts):
                            if kpt[2] > 0.2:  # Only draw if confidence is high enough
                                x, y = int(kpt[0] * scale + offset_x), int(kpt[1] * scale + offset_y)
                                cv2.circle(blank, (x, y), 3, (0, 255, 0), -1)
                                cv2.putText(blank, str(i), (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                
                plt.imshow(blank)
                plt.title("Skeleton")
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # Suggest the predefined name if available
        suggested_name = NAME_MAPPING.get(tid, f"Person_{tid}")
        name = input(f"Enter name for track ID {tid} (suggested: {suggested_name}): ")
        
        # Use the input name if provided, otherwise use the suggested name
        track_id_to_name[tid] = name.strip() if name.strip() else suggested_name

# Option 3: Default - use predefined mapping without interactive prompts
else:
    print("Using predefined name mapping (default)...")
    for tid in track_features.keys():
        # Use name from mapping or default name
        if tid in NAME_MAPPING:
            track_id_to_name[tid] = NAME_MAPPING[tid]
        else:
            track_id_to_name[tid] = f"Person_{tid}"

print("Aggregating features and building database...")

# Aggregate features for each track
identity_db = {}
for tid, feats in track_features.items():
    # Only include tracks with sufficient data
    if not feats["appearance"] or len(feats["appearance"]) < 3:
        print(f"Skipping track {tid} - insufficient data")
        continue
    
    print(f"Processing track {tid} ({track_id_to_name[tid]})...")
    
    # 1. Appearance: mean of top confident detections
    if feats["appearance"]:
        app_feats = np.stack([feat for i, feat in enumerate(feats["appearance"]) 
                             if feats["track_confidence"][i] > 0.5]) if feats["track_confidence"] else np.stack(feats["appearance"])
        mean_app = np.mean(app_feats, axis=0)
    else:
        mean_app = None
    
    # 2. OpenGait embeddings from silhouettes
    opengait_embedding = None
    if feats["silhouettes"] and len(feats["silhouettes"]) >= 10:
        try:
            print(f"  Processing OpenGait for track {tid} with {len(feats['silhouettes'])} silhouettes...")
            sils = np.stack(feats["silhouettes"][-30:] if len(feats["silhouettes"]) > 30 else feats["silhouettes"], axis=0)
            opengait_embedding = gait_embedder.extract(sils)
            print(f"  Successfully extracted OpenGait embedding of shape: {opengait_embedding.shape if hasattr(opengait_embedding, 'shape') else 'N/A'}")
        except Exception as e:
            print(f"  Error extracting OpenGait embedding: {e}")
    
    # 3. Skeleton-based gait features
    skeleton_gait_features = {}
    for seq_idx, sequence in enumerate(feats.get("skeleton_sequences", [])):
        if len(sequence) >= 10:  # Need enough frames for gait
            try:
                print(f"  Processing skeleton gait for track {tid}, sequence {seq_idx} with {len(sequence)} frames...")
                gait_dict, _ = compute_gait_features(sequence)
                
                # Merge with existing features or create new
                if not skeleton_gait_features:
                    skeleton_gait_features = gait_dict
                else:
                    # Average values for overlapping keys
                    for key, val in gait_dict.items():
                        if key in skeleton_gait_features:
                            skeleton_gait_features[key] = (skeleton_gait_features[key] + val) / 2
                        else:
                            skeleton_gait_features[key] = val
                            
                print(f"  Successfully extracted {len(gait_dict)} skeleton gait features")
            except Exception as e:
                print(f"  Error extracting skeleton gait features: {e}")
    
    # 4. Raw skeleton keypoints - select high confidence keypoints
    if feats["raw_skeletons"]:
        best_skeleton_idx = [i for i, full in enumerate(feats["full_body_visible"]) if full]
        if best_skeleton_idx:
            # Pick the best full-body skeleton
            best_skeleton = feats["raw_skeletons"][best_skeleton_idx[len(best_skeleton_idx)//2]]
        else:
            # If no full-body skeleton, pick the one with highest confidence
            best_skeleton = feats["raw_skeletons"][np.argmax(feats["track_confidence"])]
        print(f"  Selected best skeleton with {len([k for k in best_skeleton if k[2] > 0.2])} visible keypoints")
    else:
        best_skeleton = None
    
    # 5. Body ratios - mean of valid measurements
    valid_ratios = [r for r in feats["body_ratios"] if r]
    if valid_ratios:
        mean_ratios = {
            k: float(np.mean([r[k] for r in valid_ratios if k in r])) 
            for k in set().union(*valid_ratios)
        }
    else:
        mean_ratios = {}
    
    # 6. Color histograms - mean
    if feats["color_hists"]:
        mean_color_hist = np.mean(np.stack(feats["color_hists"]), axis=0)
    else:
        mean_color_hist = None
    
    # 7. HOG features - mean of valid features
    valid_hog = [h for h in feats["hog_features"] if h is not None]
    if valid_hog:
        mean_hog = np.mean(np.stack(valid_hog), axis=0)
    else:
        mean_hog = None
    
    # 8. Motion patterns
    valid_motion = [m for m in feats["motion_patterns"] if m is not None]
    if valid_motion and len(valid_motion) >= 3:
        mean_motion = {
            k: np.mean([m[k] for m in valid_motion]) 
            for k in valid_motion[0].keys()
        }
    else:
        mean_motion = None
    
    # 9. Height - median is more robust than mean for height
    median_height = float(np.median(feats["heights"])) if feats["heights"] else None
    
    # 10. Context - last known
    last_context = feats["contexts"][-1] if feats["contexts"] else {}
    
    # Process industrial pose features
    avg_industrial_pose = {}
    if feats.get("industrial_pose_features"):
        valid_features = [f for f in feats["industrial_pose_features"] if f]
        if valid_features:
            # Combine all dictionaries
            all_keys = set().union(*valid_features)
            for key in all_keys:
                values = [f[key] for f in valid_features if key in f]
                if values:
                    avg_industrial_pose[key] = sum(values) / len(values)
    
    # Process industrial color features
    avg_industrial_color = {}
    if feats.get("industrial_color_features"):
        valid_color = [f for f in feats["industrial_color_features"] if f]
        if valid_color:
            # Find all keys excluding histograms which need special handling
            non_hist_keys = set()
            for f in valid_color:
                non_hist_keys.update([k for k in f.keys() if not k.endswith('_color_hist')])
            
            # Average the scalar values
            for key in non_hist_keys:
                values = [f[key] for f in valid_color if key in f]
                if values:
                    avg_industrial_color[key] = sum(values) / len(values)
            
            # Handle histograms separately
            hist_keys = set([k for f in valid_color for k in f.keys() if k.endswith('_color_hist')])
            for key in hist_keys:
                histograms = [f[key] for f in valid_color if key in f and f[key] is not None]
                if histograms:
                    avg_industrial_color[key] = np.mean(np.stack(histograms), axis=0)
    
    # Construct the final identity entry
    identity_db[tid] = {
        "name": track_id_to_name.get(tid, f"Person_{tid}"),
        "appearance": mean_app,
        "opengait": opengait_embedding,                 # Store OpenGait embedding
        "skeleton_gait": skeleton_gait_features,        # Store skeleton gait features
        "best_skeleton": best_skeleton,
        "industrial_pose": avg_industrial_pose,         # Normalized pose features for industrial context
        "industrial_color": avg_industrial_color,       # Industrial clothing/PPE color features
        "body_ratios": mean_ratios,
        "height": median_height,
        "color_hist": mean_color_hist,
        "hog_features": mean_hog,
        "motion_pattern": mean_motion,
        "context": last_context,
        "feature_quality": {
            "appearance_samples": len(feats["appearance"]),
            "skeleton_samples": len(feats["raw_skeletons"]),
            "opengait_samples": len(feats["silhouettes"]), 
            "skeleton_sequences": len(feats.get("skeleton_sequences", [])),
            "avg_confidence": np.mean(feats["track_confidence"]) if feats["track_confidence"] else 0
        }
    }
    
    print(f"  Added '{track_id_to_name.get(tid)}' to identity database with:")
    print(f"    - {len(feats['appearance'])} appearance samples")
    print(f"    - {'Yes' if opengait_embedding is not None else 'No'} OpenGait embedding")
    print(f"    - {len(avg_industrial_pose) if avg_industrial_pose else 0} industrial pose features")
    print(f"    - {'Yes' if avg_industrial_color else 'No'} industrial color features")

# Save database
with open(OUTPUT_DB, "wb") as f:
    pickle.dump(identity_db, f)

print(f"Identity database saved to {OUTPUT_DB} with {len(identity_db)} identities:")
for tid, data in identity_db.items():
    name = data.get('name', f'Person_{tid}')
    has_opengait = "✅" if data.get('opengait') is not None else "❌"
    has_skeleton_gait = "✅" if data.get('skeleton_gait') and len(data.get('skeleton_gait')) > 0 else "❌"
    has_industrial_pose = "✅" if data.get('industrial_pose') else "❌"
    has_industrial_color = "✅" if data.get('industrial_color') else "❌"
    print(f"  ID {tid}: {name} - OpenGait: {has_opengait}, Skeleton Gait: {has_skeleton_gait}, Industrial Pose: {has_industrial_pose}, Industrial Color: {has_industrial_color}")
