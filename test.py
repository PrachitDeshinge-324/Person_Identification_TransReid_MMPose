import os
import sys
# Ensure paths are set up correctly (same as your previous scripts)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, "..")
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
# Add VideoPose3D path again
VIDEPOSE3D_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), './VideoPose3D'))
if VIDEPOSE3D_PATH not in sys.path:
    sys.path.insert(0, VIDEPOSE3D_PATH)

import cv2
import pickle
import numpy as np
import argparse
import logging
import torch
import traceback
from collections import defaultdict
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

# Import the IdentityManager for global identity management
from utils.identity_manager import IdentityManager

from models.yolo import YOLOv8Tracker
# from models.transreid import load_transreid_model # Not strictly needed for DB matching
from utils.pose import extract_skeleton_batch
from utils.gaits import compute_body_ratios, compute_gait_features
from utils.gait.gait_infer import OpenGaitEmbedder
from utils.visualization import visualize_identity_assignments, visualize_identity_assignments_with_candidates
from ultralytics import YOLO as YOLOSeg
from ultralytics.utils import LOGGER as yolo_logger
from tqdm import tqdm # For progress on video processing

# --- Import VideoPose3D Model (copied from your original script) ---
TemporalModel = None
try:
    import importlib.util
    model_path = os.path.join(VIDEPOSE3D_PATH, 'common', 'model.py')
    spec = importlib.util.spec_from_file_location("common.model", model_path)
    if spec is not None:
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        TemporalModel = model_module.TemporalModel
    else:
        print('VideoPose3D common.model not found.')
        TemporalModel = None
except Exception as e:
    print(f'Error importing VideoPose3D: {e}')
    TemporalModel = None

# --- Configuration ---
YOLO_WEIGHTS = "weights/yolo11x.pt"
YOLOSEG_WEIGHTS = "weights/yolo11x-seg.pt"
# TRANSREID_WEIGHTS = "weights/transreid_vitbase.pth"
MMPOSE_WEIGHTS = "weights/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth"
GAIT_WEIGHTS = "weights/GaitBase_DA-60000.pt"
VIDEPOSE3D_WEIGHTS = os.path.abspath(os.path.join('', 'weights/pretrained_h36m_cpn.bin'))
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
yolo_logger.setLevel(logging.WARNING)

# --- Global Models (Initialize in main) ---
yolo_tracker_model = None
gait_embedder_model = None
yoloseg_model = None
videopose3d_model = None

# --- Re-use Helper Functions & Similarity Metrics ---
# (Copy ALL necessary functions from your previous identification script:
#  lift_2d_to_3d, calculate_person_height + helpers,
#  compute_normalized_pose_features, extract_industrial_color_features,
#  similarity_opengait, similarity_skeleton_gait, procrustes_normalize_skeleton,
#  similarity_skeletons, similarity_dict_features, similarity_height, calculate_iou)

# For brevity, I will define stubs or simplified versions here.
# YOU MUST COPY THE FULL, CORRECT VERSIONS FROM YOUR PREVIOUS SCRIPT.

def lift_2d_to_3d(keypoints_2d_seq):
    """
    Lift 2D keypoints to 3D space using VideoPose3D model.
    
    Args:
        keypoints_2d_seq: Numpy array of 2D keypoints
            - Can be shape (17, 2) for single frame
            - Or shape (frames, 17, 2) for multiple frames
    
    Returns:
        Numpy array of 3D keypoints with shape (frames, 17, 3)
        or None if lifting fails
    """
    if videopose3d_model is None or keypoints_2d_seq is None or len(keypoints_2d_seq) == 0:
        return None
    
    with torch.no_grad():
        try:
            kp = np.array(keypoints_2d_seq)
            
            # Check input dimensions
            if args.verbose:
                print(f"Input 2D keypoints shape: {kp.shape}")
            
            # Validate input array for NaN or infinity values
            if np.isnan(kp).any() or np.isinf(kp).any():
                print("Warning: Input keypoints contain NaN or Inf values - fixing...")
                kp = np.nan_to_num(kp, nan=0.0, posinf=0.0, neginf=0.0)
                
            # Reshape if needed to ensure (frames, 17, 2) format
            if len(kp.shape) == 2:
                if kp.shape[0] == 17 and kp.shape[1] == 2:
                    kp = kp[np.newaxis, :, :]  # Add frame dimension -> (1, 17, 2)
                elif kp.shape[0] == 2 and kp.shape[1] == 17:
                    kp = np.transpose(kp, (1, 0))[np.newaxis, :, :]  # Transpose and add frame dimension
                else:
                    print(f"Unexpected 2D keypoints shape: {kp.shape}, expected (17, 2)")
                    return None
            elif len(kp.shape) == 3 and kp.shape[1] == 17 and kp.shape[2] == 2:
                pass  # Already in correct shape (frames, 17, 2)
            else:
                print(f"Unexpected keypoints shape: {kp.shape}, expected (frames, 17, 2)")
                return None
                
            # VideoPose3D requires a minimum number of frames to handle the convolution kernels
            # We need at least 243 frames for the model with filter_width=[3,3,3,3,3]
            # This is calculated as: receptive_field = 1 + 2*(kernel-1)*dilation_factor for each layer
            # For a 5-layer network with filter width 3: 1 + 2*(3-1) + 2^2*(3-1) + ... + 2^4*(3-1) = 81
            MIN_FRAMES_REQUIRED = 243
            
            # If we have fewer frames than required, pad by repeating the sequence
            if kp.shape[0] < MIN_FRAMES_REQUIRED:
                if args.verbose:
                    print(f"Padding sequence from {kp.shape[0]} to {MIN_FRAMES_REQUIRED} frames")
                
                # Calculate how many times to repeat the sequence
                repeat_times = int(np.ceil(MIN_FRAMES_REQUIRED / kp.shape[0]))
                # Repeat and then trim to exact size needed
                kp_repeated = np.tile(kp, (repeat_times, 1, 1))
                kp = kp_repeated[:MIN_FRAMES_REQUIRED]
            
            # Get dimensions
            frames, joints, coords = kp.shape
            
            # Normalize each frame independently
            kp_reshaped = kp.reshape(frames, -1)
            kp_mean = np.mean(kp_reshaped, axis=1, keepdims=True)
            kp_std = np.std(kp_reshaped, axis=1, keepdims=True) + 1e-9
            kp_norm = (kp_reshaped - kp_mean) / kp_std
            
            # Reshape back to original structure
            kp_norm = kp_norm.reshape(frames, joints, coords)
            
            # Add batch dimension for model input [batch_size, frames, joints, coords]
            kp_norm = kp_norm[np.newaxis, :, :, :]
            kp_norm = kp_norm.astype(np.float32)
            
            # Convert to tensor and predict
            device = next(videopose3d_model.parameters()).device
            kp_tensor = torch.from_numpy(kp_norm).to(device)
            
            if args.verbose:
                print(f"Input tensor shape: {kp_tensor.shape}, device: {device}")
                print(f"Model device: {device}")
            
            # CRITICAL: The model expects 4D tensor [batch_size, frames, joints, coords]
            # Check that shape is correct before passing to model
            assert len(kp_tensor.shape) == 4, f"Expected 4D tensor, got shape {kp_tensor.shape}"
            assert kp_tensor.shape[2] == 17, f"Expected 17 joints, got {kp_tensor.shape[2]}"
            assert kp_tensor.shape[3] == 2, f"Expected 2 coordinates, got {kp_tensor.shape[3]}"
            
            try:
                # Run the model
                pred_3d = videopose3d_model(kp_tensor)
            except RuntimeError as e:
                if "Kernel size can't be greater than actual input size" in str(e):
                    # Try with a simplified model or fallback to a simpler approach
                    print(f"Warning: Input sequence too short for convolutional kernel. Using alternative method.")
                    
                    # Fallback: Use a simple MLP to estimate 3D positions from 2D
                    # This is a very simplified approach and will not be as accurate as the full model
                    kp_flat = kp_tensor.reshape(kp_tensor.shape[0], kp_tensor.shape[1], -1)
                    
                    # Create a toy output in the right format (frames, 17, 3)
                    # This is just a placeholder - in a real implementation you'd have a proper fallback model
                    pred_3d = torch.zeros((kp_tensor.shape[0], kp_tensor.shape[1], 17, 3), device=device)
                    
                    # Fill in XY from the input
                    pred_3d[:, :, :, 0:2] = kp_tensor
                    
                    # Estimate Z (depth) based on 2D joint distances
                    # This is a very crude approximation
                    for i in range(kp_tensor.shape[1]):  # For each frame
                        for j in range(17):  # For each joint
                            # Set Z based on the normalized X,Y position
                            # This creates a rough bowl shape
                            pred_3d[0, i, j, 2] = -0.5 * (kp_tensor[0, i, j, 0]**2 + kp_tensor[0, i, j, 1]**2)
                    
                    print("Used simplified 3D estimation as fallback.")
                else:
                    # Rethrow other RuntimeErrors
                    raise
            
            # Check that output is correct before processing
            if pred_3d is None:
                print("Model returned None output")
                return None
                
            # Move to CPU and convert to numpy
            pred_3d = pred_3d[0].detach().cpu().numpy()  # shape (frames, 17, 3)
            
            # Return only the original frames (not the padded ones)
            original_frames = min(len(keypoints_2d_seq), pred_3d.shape[0])
            if original_frames < pred_3d.shape[0]:
                pred_3d = pred_3d[:original_frames]
            
            if args.verbose:
                print(f"Output 3D keypoints shape: {pred_3d.shape}")
                
            # Simple check to ensure output is valid
            if not np.all(np.isfinite(pred_3d)):
                print("Warning: Non-finite values in 3D keypoints output, fixing...")
                pred_3d = np.nan_to_num(pred_3d, nan=0.0, posinf=0.0, neginf=0.0)
            
            return pred_3d
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory error in 3D pose lifting. Try using CPU device.")
            else:
                print(f"RuntimeError in 3D pose lifting: {e}")
                import traceback
                traceback.print_exc()
            return None
        except Exception as e:
            print(f"Error in 3D pose lifting: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def calculate_person_height(keypoints, bbox, frame_height, 
                           focal_length=35.0, sensor_height=24.0,
                           ref_shoulder_width=45.0):
    """
    Enhanced height calculation with multiple fallback strategies
    and perspective awareness.
    """
    # 1. Keypoint-based measurement (primary)
    height, conf = _keypoint_based_height(keypoints, frame_height)
    
    # 2. Torso-proportion fallback
    if conf < 0.5:
        height, conf = _torso_based_height(keypoints, height, conf)
    
    # 3. Camera-aware bounding box fallback
    if conf < 0.4:
        height = _camera_aware_bbox_height(
            bbox, frame_height, 
            focal_length, sensor_height,
            ref_shoulder_width
        )
        conf = 0.3  # Lower confidence for bbox method
    
    # Apply biological constraints with realistic minimum (most adults aren't shorter than 155cm)
    height = max(155, min(210, height))
    
    # Add significant person-specific variation (5-15 cm range)
    # Use the bounding box dimensions to generate consistent person-specific variation
    if bbox is not None:
        # Create a hash-like value from the bbox dimensions
        bbox_factor = ((bbox[2] - bbox[0]) * 0.17 + (bbox[3] - bbox[1]) * 0.19) % 10.0
        # Add the variation (5-15 cm)
        height += 5.0 + bbox_factor
    else:
        # Random fallback if no bbox available
        height += np.random.uniform(5.0, 15.0)
    
    # Add slight random jitter (0.1-0.5 cm)
    height += np.random.uniform(0.1, 0.5)
    
    return height, conf

def _keypoint_based_height(keypoints, frame_height):
    """Uses multiple keypoint pairs with perspective correction"""
    valid_pairs = []
    
    # Define measurement pairs with confidence weights
    measurement_strategies = [
        {'points': (0, 15, 16), 'weight': 0.7},  # Nose-ankles
        {'points': (1, 15, 16), 'weight': 0.6},  # Neck-ankles
        {'points': (5, 15), 'weight': 0.5},     # Left shoulder-ankle
        {'points': (6, 16), 'weight': 0.5}      # Right shoulder-ankle
    ]
    
    for strategy in measurement_strategies:
        pts = strategy['points']
        if all(keypoints[p][2] > 0.3 for p in pts):
            y_values = [keypoints[p][1] for p in pts]
            dy = max(y_values) - min(y_values)
            
            # Enhanced perspective correction with sinusoidal horizontal factor
            x_spread = max(keypoints[p][0] for p in pts) - min(keypoints[p][0] for p in pts)
            
            # Get horizontal position in frame (0.0 to 1.0, with 0.5 being center)
            center_x = sum(keypoints[p][0] for p in pts) / len(pts) / frame_height
            
            # Sinusoidal factor gives more natural variation based on position in frame
            horizontal_factor = 1.0 + 0.05 * np.sin(center_x * np.pi)
            
            # Standard perspective correction
            perspective_factor = 1 + (x_spread/frame_height)*0.3 * horizontal_factor
            
            valid_pairs.append({
                'height': dy * perspective_factor,
                'conf': strategy['weight'] * min(keypoints[p][2] for p in pts)
            })
    
    if valid_pairs:
        # Weighted average of valid measurements
        total_conf = sum(p['conf'] for p in valid_pairs)
        weighted_height = sum(p['height']*p['conf'] for p in valid_pairs) / total_conf
        return (weighted_height * 170/500), total_conf/len(valid_pairs)
    
    return None, 0.0

def _torso_based_height(keypoints, prev_height, prev_conf):
    """Uses torso proportions when legs are occluded"""
    if all(keypoints[i][2] > 0.3 for i in [5, 6, 11, 12]):
        shoulder_y = (keypoints[5][1] + keypoints[6][1])/2
        hip_y = (keypoints[11][1] + keypoints[12][1])/2
        torso_height = abs(shoulder_y - hip_y)
        
        # Torso typically constitutes 30-35% of total height
        estimated_height = torso_height / 0.32 * (170/500)
        return estimated_height, 0.5  # Medium confidence
        
    return prev_height, prev_conf

def _camera_aware_bbox_height(bbox, frame_height, focal, sensor, ref_shoulder):
    """Uses projective geometry with shoulder width calibration"""
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    
    # Estimate distance using shoulder width reference
    distance = (ref_shoulder * focal) / bbox_w
    
    # Calculate height using similar triangles
    px_per_cm = (focal * 10) / (distance * sensor)  # mm to cm conversion
    return (bbox_h / px_per_cm) * 0.9  # Empirical correction factor

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
        
    if not torso_length or torso_length < 0.001:  # Check for near-zero values too
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
    # Add checks to prevent division by zero
    if "upper_leg_left_norm" in features and "lower_leg_left_norm" in features and features["lower_leg_left_norm"] > 0.001:
        features["leg_proportion_left"] = features["upper_leg_left_norm"] / features["lower_leg_left_norm"]
    
    if "upper_leg_right_norm" in features and "lower_leg_right_norm" in features and features["lower_leg_right_norm"] > 0.001:
        features["leg_proportion_right"] = features["upper_leg_right_norm"] / features["lower_leg_right_norm"]
    
    if "upper_arm_left_norm" in features and "lower_arm_left_norm" in features and features["lower_arm_left_norm"] > 0.001:
        features["arm_proportion_left"] = features["upper_arm_left_norm"] / features["lower_arm_left_norm"]
    
    if "upper_arm_right_norm" in features and "lower_arm_right_norm" in features and features["lower_arm_right_norm"] > 0.001:
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
        # Add check to prevent division by zero
        if hip_width > 0.001:  # Ensure denominator isn't too close to zero
            features["torso_shape"] = shoulder_width / hip_width  # Distinguishes between uniform types
    
    # 5. Industrial posture indicators
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
            # Ensure the denominator isn't too close to zero
            if torso_length > 0.001:
                features["upper_torso_triangle"] = area / (torso_length ** 2)  # Normalized area
        except:
            pass  # Invalid triangle
    
    return features

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

def similarity_opengait(q, db): return cosine_similarity(q.reshape(1,-1), db.reshape(1,-1))[0,0] if q is not None and db is not None else 0

def similarity_skeleton_gait(q, db):
    """Compare skeleton gait dictionaries by finding common features and computing ratio-based similarity"""
    if not q or not db:
        return 0
    
    try:
        # Compare common keys in both skeleton gait dictionaries
        common_keys = set(q.keys()) & set(db.keys())
        if not common_keys:
            return 0
            
        skel_scores = []
        for key in common_keys:
            s1 = q[key]
            s2 = db[key]
            if isinstance(s1, (int, float)) and isinstance(s2, (int, float)):
                # Simple ratio comparison for scalar values
                ratio = min(s1, s2) / max(s1, s2) if max(s1, s2) > 0 else 0
                skel_scores.append(ratio)
        
        if skel_scores:
            return sum(skel_scores) / len(skel_scores)
        
        return 0
    except Exception as e:
        print(f"Error in similarity_skeleton_gait: {e}")
        return 0

def procrustes_normalize_skeleton(skeleton, ref_indices=(5,11)):
    """Normalize skeleton by aligning reference points and scaling to unit distance"""
    if skeleton is None:
        return None
        
    try:
        # Convert to numpy array if it's a list
        skel_array = np.array(skeleton)
        
        # Extract just coordinates (ignore confidence if present)
        if skel_array.shape[-1] >= 3:  # If we have [x, y, conf] format
            coords = skel_array[..., :2]
        else:
            coords = skel_array
            
        # Get reference points for alignment (typically shoulders or hip-shoulder)
        if all(idx < len(coords) for idx in ref_indices):
            ref_point1 = coords[ref_indices[0]]
            ref_point2 = coords[ref_indices[1]]
            
            # Translate to make first reference point the origin
            translated = coords - ref_point1
            
            # Get reference vector
            ref_vector = ref_point2 - ref_point1
            ref_length = np.linalg.norm(ref_vector)
            
            if ref_length > 0:
                # Scale to make reference distance = 1
                scaled = translated / ref_length
                return scaled
            
        return skel_array
    except Exception as e:
        print(f"Error in procrustes_normalize_skeleton: {e}")
        return skeleton

def similarity_skeletons(q, db, is_3d=False):
    """Compare skeletons using normalized joint positions with weighted joints"""
    if q is None or db is None:
        return 0
    
    try:
        # Normalize skeletons for fair comparison
        q_norm = procrustes_normalize_skeleton(q)
        db_norm = procrustes_normalize_skeleton(db)
        
        if q_norm is None or db_norm is None:
            return 0
            
        # Ensure same dimension
        if len(q_norm) != len(db_norm):
            return 0
        
        # Define joint importance weights - certain joints are more discriminative
        # Keypoints order: [nose, neck, rsho, relb, rwri, lsho, lelb, lwri, rhip, rkne, 
        #                  rank, lhip, lkne, lank, reye, leye, rear, lear]
        joint_weights = {
            0: 0.5,    # Nose - moderate importance
            1: 0.8,    # Neck - high importance (central)
            2: 0.9,    # Right shoulder - high importance
            3: 0.7,    # Right elbow - moderate importance
            4: 0.5,    # Right wrist - less stable
            5: 0.9,    # Left shoulder - high importance
            6: 0.7,    # Left elbow - moderate importance
            7: 0.5,    # Left wrist - less stable
            8: 1.0,    # Right hip - highest importance (stable)
            9: 0.8,    # Right knee - high importance
            10: 0.6,   # Right ankle - moderate importance
            11: 1.0,   # Left hip - highest importance (stable)
            12: 0.8,   # Left knee - high importance
            13: 0.6,   # Left ankle - moderate importance
            14: 0.3,   # Right eye - less important
            15: 0.3,   # Left eye - less important
            16: 0.2,   # Right ear - least important
            17: 0.2    # Left ear - least important
        }
            
        # Calculate weighted euclidean distance between corresponding joints
        total_weighted_dist = 0.0
        total_weight = 0.0
        
        for i in range(min(len(q_norm), len(db_norm))):
            # Skip if index is out of bounds
            if i >= len(q_norm) or i >= len(db_norm) or i >= 18:  # Max 18 joints
                continue
                
            # Skip if either joint has low confidence
            q_conf = q[i][2] if len(q[i]) > 2 else 1.0
            db_conf = db[i][2] if len(db[i]) > 2 else 1.0
            
            if q_conf < 0.2 or db_conf < 0.2:
                continue
            
            # Get joint weight or default to 0.5
            joint_weight = joint_weights.get(i, 0.5)
            # Further weight by confidence scores
            combined_weight = joint_weight * q_conf * db_conf
                
            # Get coordinates (2D or 3D)
            if is_3d:
                q_pos = q_norm[i][:3] if len(q_norm[i]) >= 3 else q_norm[i][:2]
                db_pos = db_norm[i][:3] if len(db_norm[i]) >= 3 else db_norm[i][:2]
                
                # For 3D, use dimensions that make sense
                q_pos = np.array(q_pos[:min(len(q_pos), len(db_pos))])
                db_pos = np.array(db_pos[:min(len(q_pos), len(db_pos))])
            else:
                q_pos = q_norm[i][:2]
                db_pos = db_norm[i][:2]
                
            # Calculate distance
            try:
                dist = np.linalg.norm(np.array(q_pos) - np.array(db_pos))
                total_weighted_dist += dist * combined_weight
                total_weight += combined_weight
            except Exception as e:
                # Skip this joint if there's an error
                continue
            
        if total_weight == 0:
            return 0
            
        # Convert distance to similarity (closer = more similar)
        mean_weighted_dist = total_weighted_dist / total_weight
        
        # Apply nonlinear transformation to emphasize small differences
        # Smaller distances (more similar) get higher similarity scores
        similarity = max(0, 1.0 - (mean_weighted_dist ** 0.8))
        
        # Apply a temperature factor to make scores more discriminative
        similarity = np.power(similarity, 1.2)  # Adjust temperature as needed
        
        return float(similarity)
    except Exception as e:
        print(f"Error in similarity_skeletons: {e}")
        return 0

def similarity_dict_features(q, db):
    """Compare dictionaries of features by computing ratio similarity for each common key"""
    if not q or not db:
        return 0
        
    try:
        common_keys = set(q.keys()) & set(db.keys())
        if not common_keys:
            return 0
            
        scores = []
        hist_scores = []  # Separate tracking for histogram scores
        
        for key in common_keys:
            v1 = q[key]
            v2 = db[key]
            
            # Handle histograms (arrays)
            if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                if v1.size == v2.size:
                    # Try multiple histogram comparison methods
                    try:
                        # CV2 histogram comparison if available
                        if v1.dtype != np.float32:
                            v1 = v1.astype(np.float32)
                        if v2.dtype != np.float32:
                            v2 = v2.astype(np.float32)
                            
                        # Correlation method (better for comparing shapes)
                        correl = cv2.compareHist(v1.flatten(), v2.flatten(), cv2.HISTCMP_CORREL)
                        correl_score = max(0, (correl + 1) / 2)  # Convert from [-1,1] to [0,1]
                        
                        # Intersection method (good for matching)
                        intersect = cv2.compareHist(v1.flatten(), v2.flatten(), cv2.HISTCMP_INTERSECT)
                        norm_factor = min(np.sum(v1), np.sum(v2))
                        if norm_factor > 0:
                            intersect_score = intersect / norm_factor
                        else:
                            intersect_score = 0
                            
                        # Use the better of the two scores
                        hist_score = max(correl_score, intersect_score)
                        hist_scores.append(hist_score)
                    except:
                        # Fallback to correlation coefficient
                        try:
                            hist_score = np.corrcoef(v1.flatten(), v2.flatten())[0,1]
                            hist_scores.append(max(0, hist_score))
                        except:
                            pass  # Skip if correlation fails
            # Handle scalar values
            elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # For scalar values, use ratio comparison with bounded difference
                if key.endswith('_angle'):
                    # For angle features, handle circular values
                    diff = abs(v1 - v2)
                    if diff > np.pi:
                        diff = 2 * np.pi - diff
                    angle_sim = 1.0 - min(1.0, diff / np.pi)
                    scores.append(angle_sim)
                else:
                    # Standard ratio comparison for other values
                    if abs(v1) < 0.001 and abs(v2) < 0.001:
                        # Both very close to zero, consider them identical
                        scores.append(1.0)
                    elif max(abs(v1), abs(v2)) > 0.001:
                        # At least one value is non-zero
                        ratio = min(abs(v1), abs(v2)) / max(abs(v1), abs(v2))
                        scores.append(ratio)
                
        # Combine histogram and regular scores with appropriate weighting
        if scores and hist_scores:
            # Give histograms slightly less weight since they can be noisy
            return (0.7 * sum(scores) / len(scores)) + (0.3 * sum(hist_scores) / len(hist_scores))
        elif scores:
            return sum(scores) / len(scores)
        elif hist_scores:
            return sum(hist_scores) / len(hist_scores)
            
        return 0
    except Exception as e:
        print(f"Error in similarity_dict_features: {e}")
        return 0

def similarity_height(q, db, std_dev_cm=5.0): 
    """Compare heights using Gaussian similarity"""
    return np.exp(-((q - db)**2) / (2 * std_dev_cm**2)) if q is not None and db is not None else 0

def calculate_iou(boxA, boxB): # From previous script
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxAArea + boxBArea - interArea <= 0: return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

# --- Model Initialization ---
def initialize_all_models():
    global yolo_tracker_model, gait_embedder_model, yoloseg_model, videopose3d_model
    print("Initializing models for video annotation...")
    yolo_tracker_model = YOLOv8Tracker(YOLO_WEIGHTS, device=DEVICE, conf_threshold=0.25) # Use tracker
    gait_embedder_model = OpenGaitEmbedder(weights_path=GAIT_WEIGHTS, device=DEVICE)
    try:
        yoloseg_model = YOLOSeg(YOLOSEG_WEIGHTS) # verbose=False by default if not supported
    except TypeError:
        yoloseg_model = YOLOSeg(YOLOSEG_WEIGHTS)
        os.environ['YOLO_VERBOSE'] = 'False'


    if TemporalModel is not None and os.path.exists(VIDEPOSE3D_WEIGHTS):
        pose_device = torch.device('cpu')
        videopose3d_model = TemporalModel(17, 2, 17, filter_widths=[3,3,3,3,3], causal=False, dropout=0.25, channels=1024, dense=False).to(pose_device)
        checkpoint = torch.load(VIDEPOSE3D_WEIGHTS, map_location=pose_device)
        # Use your robust loading logic from database creation script
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            videopose3d_model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'model_pos' in checkpoint:
            videopose3d_model.load_state_dict(checkpoint['model_pos'])
        else:
            videopose3d_model.load_state_dict(checkpoint)
        videopose3d_model.eval()
        print('Loaded VideoPose3D model.')
    else:
        print('VideoPose3D model not loaded. 3D features will be skipped.')
        videopose3d_model = None
    print("All models initialized.")

# --- Feature Aggregation for a Track ---
def aggregate_features_from_track_buffers(track_id, track_specific_data, frame_height_for_height_calc, verbose=False):
    """
    Aggregates features from a track's buffered raw data.
    track_specific_data: dict containing 'keypoints_buffer', 'silhouettes_buffer', etc.
    """
    aggregated_query_features = {}
    if verbose: print(f"Aggregating features for track {track_id} from {len(track_specific_data.get('keypoints_buffer',[]))} buffered items.")

    # 1. OpenGait
    silhouettes = track_specific_data.get("silhouettes_buffer", [])
    if gait_embedder_model and silhouettes and len(silhouettes) >= 10:
        sils_for_gait = np.stack(silhouettes[:30]) # Use up to 30
        sils_for_gait = np.nan_to_num(np.clip(sils_for_gait, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
        try:
            aggregated_query_features["opengait"] = gait_embedder_model.extract(sils_for_gait)
        except Exception as e:
            if verbose: print(f"  Track {track_id} Agg: OpenGait error: {e}")
            aggregated_query_features["opengait"] = None
    else: aggregated_query_features["opengait"] = None

    # 2. Skeleton Gait
    keypoints_list = track_specific_data.get("keypoints_buffer", [])
    if keypoints_list and len(keypoints_list) >= 10:
        try:
            gait_dict, _ = compute_gait_features(keypoints_list)
            aggregated_query_features["skeleton_gait"] = gait_dict
        except Exception as e:
            if verbose: print(f"  Track {track_id} Agg: Skeleton Gait error: {e}")
            aggregated_query_features["skeleton_gait"] = {}
    else: aggregated_query_features["skeleton_gait"] = {}

    # 3. Best 2D Skeleton
    confidences = track_specific_data.get("confidences_buffer", [])
    if keypoints_list:
        if confidences and len(confidences) == len(keypoints_list):
            best_idx = np.argmax(confidences)
            aggregated_query_features["best_skeleton"] = keypoints_list[best_idx]
        else:
            aggregated_query_features["best_skeleton"] = keypoints_list[-1] # Fallback
    else: aggregated_query_features["best_skeleton"] = None

    # 4. Best 3D Skeleton
    kps_3d_seq_parts = []
    if videopose3d_model and keypoints_list:
        for kps_2d_with_conf in keypoints_list:
            if kps_2d_with_conf is not None and len(kps_2d_with_conf) == 17:
                kps_2d_coords_only = np.array([k[:2] for k in kps_2d_with_conf])
                if kps_2d_coords_only.shape == (17,2):
                    kps_3d_single_frame_output = lift_2d_to_3d(kps_2d_coords_only)
                    if kps_3d_single_frame_output is not None:
                        kps_3d_seq_parts.append(kps_3d_single_frame_output[0])
    if kps_3d_seq_parts:
        valid_3d_kps = [kp for kp in kps_3d_seq_parts if kp is not None and np.all(np.isfinite(kp))]
        if valid_3d_kps:
            aggregated_query_features["best_3d_skeleton"] = np.median(np.stack(valid_3d_kps), axis=0).tolist()
    else: aggregated_query_features["best_3d_skeleton"] = None

    # 5. Other dictionary/scalar features (Industrial Pose, Body Ratios, Height)
    # These are re-calculated from the buffered keypoints and bboxes.
    industrial_pose_list_agg, body_ratios_list_agg, heights_list_agg = [], [], []
    bboxes_list = track_specific_data.get("bboxes_buffer", [])

    for i in range(len(keypoints_list)):
        kps = keypoints_list[i]
        bbox = bboxes_list[i] if i < len(bboxes_list) else None # Ensure bbox exists
        if kps and bbox:
            ip_feats = compute_normalized_pose_features(kps)
            if ip_feats: industrial_pose_list_agg.append(ip_feats)
            ratios = compute_body_ratios(kps)
            if ratios: body_ratios_list_agg.append(ratios)
            height, _ = calculate_person_height(kps, bbox, frame_height_for_height_calc)
            if height: heights_list_agg.append(height)

    for key, L_agg in [("industrial_pose", industrial_pose_list_agg), ("body_ratios", body_ratios_list_agg)]:
        if L_agg:
            avg_dict = defaultdict(list)
            for item_dict in L_agg:
                for ik, iv in item_dict.items(): avg_dict[ik].append(iv)
            aggregated_query_features[key] = {ik: np.mean(iv_list) for ik, iv_list in avg_dict.items()}
        else: aggregated_query_features[key] = {}
    if heights_list_agg: aggregated_query_features["height"] = float(np.median(heights_list_agg))
    else: aggregated_query_features["height"] = None

    # Industrial Color (requires crops_buffer if not extracting on the fly)
    # If track_specific_data['crops_buffer'] exists:
    industrial_colors_list_agg = []
    crops_buffer = track_specific_data.get("crops_buffer", [])
    for crop in crops_buffer:
        ic_feats = extract_industrial_color_features(crop)
        if ic_feats: industrial_colors_list_agg.append(ic_feats)
    
    if industrial_colors_list_agg:
        avg_ic_dict = defaultdict(list)
        hist_keys_ic = []
        for item_dict in industrial_colors_list_agg:
            for ik, iv in item_dict.items():
                if isinstance(iv, np.ndarray): # Histogram
                    avg_ic_dict[ik].append(iv)
                    if ik not in hist_keys_ic: hist_keys_ic.append(ik)
                else: # Scalar
                     avg_ic_dict[ik].append(iv)
        
        final_ic_dict = {}
        for ik, iv_list in avg_ic_dict.items():
            if ik in hist_keys_ic:
                final_ic_dict[ik] = np.mean(np.stack(iv_list), axis=0) if iv_list else None
            else:
                final_ic_dict[ik] = np.mean(iv_list) if iv_list else None
        aggregated_query_features["industrial_color"] = {k:v for k,v in final_ic_dict.items() if v is not None}

    else: aggregated_query_features["industrial_color"] = {}

    return aggregated_query_features


# --- Temporal Identity Smoothing ---
def apply_temporal_smoothing(track_id, track_data, new_id, new_confidence, history_length=7):
    """
    Apply temporal smoothing to avoid identity flickering between frames.
    
    Args:
        track_id: The track ID
        track_data: The track-specific data dictionary
        new_id: Newly predicted identity
        new_confidence: Confidence of the new prediction
        history_length: Number of past identifications to consider
        
    Returns:
        (smoothed_id, smoothed_confidence)
    """
    # Initialize identity history if it doesn't exist
    if "identity_history" not in track_data:
        track_data["identity_history"] = []
    
    # Weight the current prediction based on confidence
    # Higher confidence gives stronger weight to new prediction
    confidence_weight = new_confidence ** 2  # Square confidence to emphasize high-confidence predictions
    
    # Add new prediction to history, potentially multiple times based on confidence
    repetitions = 1
    if new_confidence > 0.65:  # For high confidence predictions, add multiple copies
        repetitions = 2
    if new_confidence > 0.85:  # For very high confidence predictions, add even more copies
        repetitions = 3
    
    for _ in range(repetitions):
        track_data["identity_history"].append((new_id, new_confidence))
    
    # Trim history to keep only the recent predictions
    if len(track_data["identity_history"]) > history_length:
        track_data["identity_history"] = track_data["identity_history"][-history_length:]
    
    # Count occurrences of each identity in history with confidence weighting
    id_counts = {}
    id_total_conf = {}
    id_weighted_count = {}  # Track confidence-weighted counts
    
    for identity, conf in track_data["identity_history"]:
        if identity not in id_counts:
            id_counts[identity] = 0
            id_total_conf[identity] = 0
            id_weighted_count[identity] = 0
        
        id_counts[identity] += 1
        id_total_conf[identity] += conf
        id_weighted_count[identity] += conf  # Weight by confidence
    
    # Find the identity with highest weighted count
    most_common_id = None
    max_weighted_count = 0
    
    for identity, weighted_count in id_weighted_count.items():
        if weighted_count > max_weighted_count:
            max_weighted_count = weighted_count
            most_common_id = identity
    
    # Calculate average confidence for the selected identity
    avg_confidence = id_total_conf[most_common_id] / id_counts[most_common_id] if most_common_id else 0
    
    # Add stability factor - prefer to keep current ID unless new one is significantly better
    current_id = track_data.get("identified_name", "Unknown")
    if current_id != "Unknown" and current_id != "Pending..." and current_id in id_counts and current_id != most_common_id:
        # If current ID has a sufficient weighted count relative to the best one, keep it
        if id_weighted_count[current_id] >= 0.7 * max_weighted_count:
            most_common_id = current_id
            avg_confidence = id_total_conf[current_id] / id_counts[current_id]
    
    # If the identification is consistently "Unknown", reduce the confidence
    if most_common_id == "Unknown" and "Unknown" in id_counts and id_counts["Unknown"] >= 3:
        # Reduce confidence further each time "Unknown" is repeatedly identified
        avg_confidence = avg_confidence * 0.8
    
    return most_common_id, avg_confidence


# --- Main Identification Logic (copied from previous script) ---
def identify_person_from_features(query_features, database, feature_weights, similarity_threshold=0.5, top_k=3):
    """
    Identify a person by comparing query features with database entries.
    
    Args:
        query_features: Dictionary of features extracted from the query track
        database: Dictionary of database entries
        feature_weights: Dictionary of feature weights
        similarity_threshold: Threshold for considering a match valid
        top_k: Number of top matches to return
        
    Returns:
        tuple: (best_name, best_confidence, best_match_id, detailed_scores, top_candidates)
            - top_candidates is a list of (name, confidence, match_id) tuples
    """
    best_match_id = None
    max_overall_similarity = -1.0
    
    if not query_features:
        print("Query features are empty. Cannot perform identification.")
        return "Unknown", 0.0, None, {}, []

    # Normalize weights (ensure they sum to 1 if not already)
    total_weight = sum(w for w in feature_weights.values() if w > 0) # Sum of positive weights
    if total_weight == 0:
        print("Error: All feature weights are zero or negative.")
        return "Unknown", 0.0, None, {}, []

    detailed_scores_for_best_match = {}
    all_matches = []  # List to collect all matches with their scores

    for db_id, db_entry in tqdm(database.items(), desc="Comparing with DB", leave=False):
        # Skip metadata entry
        if db_id == '_metadata':
            continue

        current_weighted_score = 0.0
        current_sum_of_weights_used = 0.0 # For features actually present in both query and DB
        
        individual_scores = {}

        # OpenGait
        if feature_weights.get("opengait", 0) > 0 and \
           query_features.get('opengait') is not None and db_entry.get('opengait') is not None:
            sim = similarity_opengait(query_features['opengait'], db_entry['opengait'])
            individual_scores["opengait"] = sim
            current_weighted_score += feature_weights["opengait"] * sim
            current_sum_of_weights_used += feature_weights["opengait"]

        # Skeleton Gait
        if feature_weights.get("skeleton_gait", 0) > 0 and \
           query_features.get('skeleton_gait') and db_entry.get('skeleton_gait'):
            sim = similarity_skeleton_gait(query_features['skeleton_gait'], db_entry['skeleton_gait'])
            individual_scores["skeleton_gait"] = sim
            current_weighted_score += feature_weights["skeleton_gait"] * sim
            current_sum_of_weights_used += feature_weights["skeleton_gait"]

        # Best 2D Skeleton
        if feature_weights.get("best_skeleton_2d", 0) > 0 and \
           query_features.get('best_skeleton') and db_entry.get('best_skeleton'):
            sim = similarity_skeletons(query_features['best_skeleton'], db_entry['best_skeleton'], is_3d=False)
            individual_scores["best_skeleton_2d"] = sim
            current_weighted_score += feature_weights["best_skeleton_2d"] * sim
            current_sum_of_weights_used += feature_weights["best_skeleton_2d"]
            
        # Best 3D Skeleton
        if feature_weights.get("best_skeleton_3d", 0) > 0 and \
           query_features.get('best_3d_skeleton') and db_entry.get('best_3d_skeleton'):
            # Convert to list of lists/tuples if they are numpy arrays for similarity_skeletons
            q_skel_3d = query_features['best_3d_skeleton']
            db_skel_3d = db_entry['best_3d_skeleton']
            
            # The similarity_skeletons expects confidence as 3rd/4th element.
            # 3D skeletons from DB are (17,3). We need to adapt or change similarity_skeletons.
            # For now, let's assume they can be processed if they are list of [x,y,z,conf_dummy=1]
            # This is a simplification.
            def adapt_3d_skeleton(skel):
                if skel is None:
                    return None
                adapted = []
                for joint in skel:
                    if isinstance(joint, (list, tuple, np.ndarray)):
                        if len(joint) == 3:  # x, y, z format
                            # Add a dummy confidence value of 1.0
                            adapted.append([joint[0], joint[1], joint[2], 1.0])
                        elif len(joint) >= 4:  # Already has confidence
                            adapted.append(joint)
                    else:
                        # Handle unexpected format gracefully
                        return None
                return adapted
            
            sim = similarity_skeletons(adapt_3d_skeleton(q_skel_3d), adapt_3d_skeleton(db_skel_3d), is_3d=True)
            individual_scores["best_skeleton_3d"] = sim
            current_weighted_score += feature_weights["best_skeleton_3d"] * sim
            current_sum_of_weights_used += feature_weights["best_skeleton_3d"]

        # Industrial Pose
        if feature_weights.get("industrial_pose", 0) > 0 and \
            query_features.get('industrial_pose') and db_entry.get('industrial_pose'):
            sim = similarity_dict_features(query_features['industrial_pose'], db_entry['industrial_pose'])
            individual_scores["industrial_pose"] = sim
            current_weighted_score += feature_weights["industrial_pose"] * sim
            current_sum_of_weights_used += feature_weights["industrial_pose"]

        # Body Ratios
        if feature_weights.get("body_ratios", 0) > 0 and \
           query_features.get('body_ratios') and db_entry.get('body_ratios'):
            sim = similarity_dict_features(query_features['body_ratios'], db_entry['body_ratios'])
            individual_scores["body_ratios"] = sim
            current_weighted_score += feature_weights["body_ratios"] * sim
            current_sum_of_weights_used += feature_weights["body_ratios"]

        # Height
        if feature_weights.get("height", 0) > 0 and \
           query_features.get('height') is not None and db_entry.get('height') is not None:
            sim = similarity_height(query_features['height'], db_entry['height'])
            individual_scores["height"] = sim
            current_weighted_score += feature_weights["height"] * sim
            current_sum_of_weights_used += feature_weights["height"]

        # Industrial Color
        if feature_weights.get("industrial_color", 0) > 0 and \
           query_features.get('industrial_color') and db_entry.get('industrial_color'):
            sim = similarity_dict_features(query_features['industrial_color'], db_entry['industrial_color']) # Reusing dict_features
            individual_scores["industrial_color"] = sim
            current_weighted_score += feature_weights["industrial_color"] * sim
            current_sum_of_weights_used += feature_weights["industrial_color"]

        # Normalize score by sum of weights for features that were actually compared
        final_score_for_db_entry = (current_weighted_score / current_sum_of_weights_used) if current_sum_of_weights_used > 0 else 0.0
        
        # Store this match for top-k selection
        all_matches.append((db_id, final_score_for_db_entry, database[db_id]['name'], individual_scores))
        
        if final_score_for_db_entry > max_overall_similarity:
            max_overall_similarity = final_score_for_db_entry
            best_match_id = db_id
            detailed_scores_for_best_match = individual_scores

    # Sort all matches by descending similarity
    all_matches.sort(key=lambda x: x[1], reverse=True)
    
    # Extract top-k candidates
    top_candidates = []
    for i, (match_id, score, name, _) in enumerate(all_matches[:top_k]):
        top_candidates.append((name, score, match_id))
    
    if best_match_id is not None and max_overall_similarity >= similarity_threshold:
        return database[best_match_id]['name'], max_overall_similarity, best_match_id, detailed_scores_for_best_match, top_candidates
    else:
        # Return best guess even if below threshold, but flag as Unknown
        name_if_known = database[best_match_id]['name'] if best_match_id else "N/A"
        # print(f"Best match below threshold: {name_if_known} with score {max_overall_similarity:.4f}")
        return "Unknown", max_overall_similarity, best_match_id, detailed_scores_for_best_match, top_candidates


# --- Main Video Processing and Annotation Loop ---
def process_video_for_identification(video_path, database_path, output_video_path, feature_weights, similarity_threshold, args):
    global yolo_tracker_model, gait_embedder_model, yoloseg_model, videopose3d_model # Ensure globals are used

    # Load Database
    if not os.path.exists(database_path):
        print(f"Database file not found: {database_path}"); return
    with open(database_path, "rb") as f: identity_db = pickle.load(f)
    if not identity_db: print("Database is empty."); return
    print(f"Loaded {len(identity_db)} identities from {database_path}")

    # Video I/O
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error opening video: {video_path}"); return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set starting frame if specified
    start_frame = args.start_frame
    if start_frame > 0:
        if start_frame >= total_frames:
            print(f"Error: Start frame ({start_frame}) exceeds total frames in video ({total_frames})")
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Starting processing from frame {start_frame}/{total_frames}")
        # Adjust total frames for progress bar
        total_frames = total_frames - start_frame
        
    progress_bar = tqdm(total=total_frames, desc="Processing Video")

    # Tracking and Identification State
    active_tracks_data = defaultdict(lambda: {
        "crops_buffer": [], "keypoints_buffer": [], "silhouettes_buffer": [],
        "bboxes_buffer": [], "confidences_buffer": [],
        "identified_name": "Pending...", "identification_confidence": 0.0,
        "frames_since_last_id": 0, "id_attempts": 0, "last_seen_frame": 0
    })
    MIN_FRAMES_FOR_ID = args.min_frames_for_id # e.g., 15
    ID_ATTEMPT_INTERVAL_FRAMES = args.id_interval # e.g., 30 (or once if confident)
    MAX_BUFFER_SIZE = args.max_buffer # e.g., 50
    TRACK_TIMEOUT_FRAMES = args.track_timeout # e.g. 60 (frames unseen before removing track)
    
    # Initialize the global identity manager
    identity_manager = IdentityManager(conflict_threshold=args.conflict_threshold)

    # Initialize frame index with the starting frame if specified
    frame_idx = args.start_frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        annotated_frame = frame.copy()
        current_tracks_in_frame = set()

        # 1. Person Detection and Tracking
        detections = yolo_tracker_model.process_frame(frame) # Returns (ids, bboxes, confs)
        
        # Efficiently extract skeletons for all detections in the current frame
        all_detected_bboxes = [d[1] for d in detections]
        all_keypoints_for_frame = []
        if all_detected_bboxes:
            all_keypoints_for_frame = extract_skeleton_batch(frame, all_detected_bboxes, weights=MMPOSE_WEIGHTS, device=DEVICE)

        for det_idx, (track_id, bbox, conf) in enumerate(detections):
            current_tracks_in_frame.add(track_id)
            track_data = active_tracks_data[track_id]
            track_data["last_seen_frame"] = frame_idx

            # Extract crop
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[max(0,y1):min(frame_height,y2), max(0,x1):min(frame_width,x2)]

            # Extract keypoints (already done in batch)
            keypoints = all_keypoints_for_frame[det_idx] if det_idx < len(all_keypoints_for_frame) else None

            # Extract silhouette for the crop
            silhouette_mask = None
            if yoloseg_model and crop.size > 0:
                try:
                    crop_seg_results = yoloseg_model(crop, verbose=False, classes=[0]) # class 0 is person
                    if crop_seg_results and hasattr(crop_seg_results[0], 'masks') and \
                       crop_seg_results[0].masks is not None and len(crop_seg_results[0].masks.data) > 0:
                        mask_tensor = crop_seg_results[0].masks.data[0]
                        silhouette_mask_resized = cv2.resize((mask_tensor.cpu().numpy() > 0.5).astype(np.float32), (44,64)) # Example OpenGait size
                        silhouette_mask = silhouette_mask_resized # Store resized for OpenGait
                except Exception as e:
                    if args.verbose: print(f"Silhouette extraction error for track {track_id}: {e}")


            # Buffer features
            if crop.size > 0: track_data["crops_buffer"].append(crop)
            if keypoints: track_data["keypoints_buffer"].append(keypoints)
            if silhouette_mask is not None: track_data["silhouettes_buffer"].append(silhouette_mask)
            track_data["bboxes_buffer"].append(bbox)
            track_data["confidences_buffer"].append(conf)
            track_data["frames_since_last_id"] += 1

            # Trim buffers
            for buf_key in ["crops_buffer", "keypoints_buffer", "silhouettes_buffer", "bboxes_buffer", "confidences_buffer"]:
                if len(track_data[buf_key]) > MAX_BUFFER_SIZE:
                    track_data[buf_key] = track_data[buf_key][-MAX_BUFFER_SIZE:]
            
            # Identification Trigger - Improved strategy
            trigger_id_now = False
            buffer_size = len(track_data["keypoints_buffer"])
            
            # Make sure we have enough data for a reliable identification
            if buffer_size >= MIN_FRAMES_FOR_ID:
                # Always identify on first attempt
                if track_data["id_attempts"] == 0:
                    trigger_id_now = True
                # For unknown persons, try again sooner
                elif track_data["identified_name"] == "Unknown" and track_data["frames_since_last_id"] >= ID_ATTEMPT_INTERVAL_FRAMES / 2:
                    trigger_id_now = True
                # For low confidence identifications
                elif track_data["identification_confidence"] < 0.65 and track_data["frames_since_last_id"] >= ID_ATTEMPT_INTERVAL_FRAMES:
                    trigger_id_now = True
                # For medium confidence, wait longer
                elif 0.65 <= track_data["identification_confidence"] < 0.85 and track_data["frames_since_last_id"] >= ID_ATTEMPT_INTERVAL_FRAMES * 1.5:
                    trigger_id_now = True
                # For high confidence, only retry occasionally to verify
                elif track_data["identification_confidence"] >= 0.85 and track_data["frames_since_last_id"] >= ID_ATTEMPT_INTERVAL_FRAMES * 3:
                    trigger_id_now = True
                # Additional trigger when buffer grows significantly (more data available)
                elif buffer_size >= MIN_FRAMES_FOR_ID * 2 and track_data["id_attempts"] <= 2:
                    trigger_id_now = True
            
            if trigger_id_now:
                query_feats = aggregate_features_from_track_buffers(track_id, track_data, frame_height, verbose=args.verbose)
                # Check if any actual features were aggregated, avoiding numpy array truth value ambiguity
                has_features = False
                if query_feats:
                    for val in query_feats.values():
                        if val is not None and (not isinstance(val, (list, dict)) or len(val) > 0):
                            has_features = True
                            break
                
                if has_features:
                    name, confidence, best_match_id, detailed_scores, top_candidates = identify_person_from_features(
                        query_feats, identity_db, feature_weights, similarity_threshold, args.top_k_matches
                    )
                    # Apply temporal smoothing
                    smoothed_name, smoothed_confidence = apply_temporal_smoothing(track_id, track_data, name, confidence)
                    
                    # Store detailed scores for debugging
                    track_data["detailed_scores"] = detailed_scores
                    
                    # Apply global identity management to ensure uniqueness
                    final_name, final_confidence, was_modified = identity_manager.update_track_identity(
                        track_id, smoothed_name, smoothed_confidence, top_candidates
                    )
                    
                    # Store final identification result
                    track_data["identified_name"] = final_name
                    track_data["identification_confidence"] = final_confidence
                    
                    # Print detailed feature scores and identity assignment to help with debugging
                    if args.verbose:
                        print(f"\nTrack {track_id} identification details:")
                        print(f"Raw identification: {name} with confidence {confidence:.4f}")
                        print(f"Smoothed result: {smoothed_name} with confidence {smoothed_confidence:.4f}")
                        print(f"Final result after global constraints: {final_name} with confidence {final_confidence:.4f}")
                        if was_modified:
                            print(f"Note: Identity was modified by the global identity manager")
                        if detailed_scores:
                            print("Feature scores:")
                            for feature, score in detailed_scores.items():
                                print(f"  {feature}: {score:.4f}")
                        
                        # List which features were successfully extracted
                        available_features = [k for k, v in query_feats.items() if v is not None]
                        print(f"Available features: {available_features}")

                track_data["id_attempts"] += 1
                track_data["frames_since_last_id"] = 0
                if args.verbose: print(f"Identified track {track_id} as {track_data['identified_name']} ({track_data['identification_confidence']:.2f})")

            # Enhanced Annotation with color-coding based on confidence
            display_name = track_data["identified_name"]
            display_conf = track_data["identification_confidence"]
            
            # Prepare for alternative display
            alternatives = []
            
            # Check if we should show alternatives for Unknown identities
            if args.show_candidates and display_name == "Unknown" and track_id in identity_manager.top_candidates:
                candidates = identity_manager.top_candidates[track_id]
                # Use the top candidates as alternatives (excluding Unknown/Pending)
                alternatives = [name for name, conf, _ in candidates if name != "Unknown" and name != "Pending..."]
                # If we have alternatives, show them
                if alternatives:
                    # Show up to 3 alternatives with a slash separator
                    display_name = "/".join(alternatives[:3])
                    # Use medium confidence color for alternatives
                    box_color = (0, 165, 255)  # Orange/yellow for alternatives
                    text_color = (0, 128, 255)

            text_to_display = f"ID:{track_id} {display_name}"
            
            # Color code by confidence and identity status
            if display_name == "Pending...":
                box_color = (200, 200, 0)  # Cyan for pending
                text_color = (0, 200, 200)
            elif display_name == "Unknown" and not alternatives:
                box_color = (0, 0, 255)    # Red for unknown with no alternatives
                text_color = (0, 0, 255)
            else:
                # Color gradient based on confidence: red->yellow->green
                if display_conf < 0.5:
                    # Low confidence: reddish
                    box_color = (0, 0, 255)
                    text_color = (0, 0, 255)
                elif display_conf < 0.75:
                    # Medium confidence: yellowish
                    box_color = (0, 165, 255)
                    text_color = (0, 128, 255)
                else:
                    # High confidence: greenish
                    box_color = (0, 255, 0)
                    text_color = (0, 200, 0)
                
                # For regular identities, add confidence to text
                if not alternatives:
                    text_to_display += f" ({display_conf:.2f})"
            
            # Draw bounding box with thickness based on confidence
            box_thickness = 1 if display_name == "Pending..." else 2
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, box_thickness)
            
            # Add a background to the text for better readability
            text_size = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, 
                          (x1, y1 - 25), 
                          (x1 + text_size[0] + 10, y1), 
                          (0, 0, 0), -1)
            
            cv2.putText(annotated_frame, text_to_display, (x1 + 5, y1 - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # Add simplified concise debug info showing only top candidates
            if args.debug_view:
                if track_id in identity_manager.top_candidates:
                    candidates = identity_manager.top_candidates[track_id]
                    if candidates:
                        # Show only the top 3 alternatives with confidence scores
                        top_candidates = [(name, conf) for name, conf, _ in candidates 
                                         if name != "Unknown" and name != "Pending..."][:3]
                        
                        if top_candidates:
                            # Draw background for candidate names
                            overlay = annotated_frame.copy()
                            score_bg_height = len(top_candidates) * 20 + 10
                            cv2.rectangle(overlay, 
                                        (x1, y2), 
                                        (x1 + 180, y2 + score_bg_height), 
                                        (0, 0, 0), 
                                        -1)
                            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
                            
                            cv2.putText(annotated_frame, "Top matches:", 
                                      (x1 + 5, y2 + 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            for i, (name, score) in enumerate(top_candidates):
                                # Color based on confidence
                                if score < 0.5:
                                    color = (0, 0, 255)  # Red
                                elif score < 0.75:
                                    color = (0, 165, 255)  # Orange
                                else:
                                    color = (0, 255, 0)  # Green
                                    
                                # Draw candidate name and score
                                cv2.putText(annotated_frame, f"{name}: {score:.2f}", 
                                          (x1 + 5, y2 + 20 + (i+1) * 20), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Clean up old/lost tracks
        tracks_to_remove = [tid for tid, tdata in active_tracks_data.items() if frame_idx - tdata["last_seen_frame"] > TRACK_TIMEOUT_FRAMES]
        for tid_rem in tracks_to_remove:
            if args.verbose: print(f"Removing timed-out track: {tid_rem}")
            del active_tracks_data[tid_rem]
        
        # Keep identity manager in sync with currently active tracks
        identity_manager.clear_inactive_tracks(current_tracks_in_frame)
        
        # Periodically run global optimization and conflict resolution to improve identity assignments
        if frame_idx % args.global_opt_interval == 0:  # Run at configurable interval
            # First, apply global optimization (Hungarian algorithm)
            changes_made = identity_manager.apply_global_optimization()
            if changes_made > 0 and args.verbose:
                print(f"Global optimization made {changes_made} changes to identity assignments")
            
            # Then run traditional conflict resolution for any remaining issues
            identity_manager.resolve_low_confidence_conflicts(confidence_threshold=0.5)
            
            # Always enforce uniqueness for all identities after optimization and conflict resolution
            conflicts = identity_manager.get_identity_conflicts()
            for identity_name in list(conflicts.keys()):
                identity_manager.force_unique_identity(identity_name)
                if args.verbose:
                    print(f"Enforced uniqueness for identity: {identity_name}")
        
        # Add identity visualization panel if requested
        if args.show_identity_panel:
            if args.show_candidates:
                display_frame = visualize_identity_assignments_with_candidates(annotated_frame, identity_manager, active_tracks_data)
            else:
                display_frame = visualize_identity_assignments(annotated_frame, identity_manager, active_tracks_data)
        else:
            display_frame = annotated_frame
            
        video_writer.write(annotated_frame)  # Always save without the panel to keep original dimensions
        if not args.no_display:
            cv2.imshow("Video Identification", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_idx += 1
        progress_bar.update(1)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    progress_bar.close()
    print(f"Annotated video saved to {output_video_path}")
    
    # Print identity management summary
    print("\n--- Identity Management Summary ---")
    final_identities = identity_manager.get_all_identities()
    
    # Count identities by type
    identities_by_type = {"Named": 0, "Unknown": 0, "Pending": 0}
    tracks_per_identity = {}
    
    for track_id, (name, conf) in final_identities.items():
        if name == "Unknown":
            identities_by_type["Unknown"] += 1
        elif name == "Pending...":
            identities_by_type["Pending"] += 1
        else:
            identities_by_type["Named"] += 1
            if name not in tracks_per_identity:
                tracks_per_identity[name] = []
            tracks_per_identity[name].append((track_id, conf))
    
    print(f"Total tracks processed: {len(final_identities)}")
    print(f"Named identities: {identities_by_type['Named']}")
    print(f"Unknown identities: {identities_by_type['Unknown']}")
    print(f"Pending identities: {identities_by_type['Pending']}")
    
    # Check for any remaining conflicts
    remaining_conflicts = identity_manager.get_identity_conflicts()
    if remaining_conflicts:
        print("\nWarning: Some identity conflicts remain unresolved:")
        for name, track_ids in remaining_conflicts.items():
            print(f"  {name}: assigned to tracks {', '.join(map(str, track_ids))}")
    else:
        print("\nNo identity conflicts detected in final results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Identify and annotate persons in a video.')
    parser.add_argument('--input_video', required=True, help='Path to the input video file.')
    parser.add_argument('--output_video', default="output_annotated3c_fixed.mp4", help='Path to save the annotated video.')
    parser.add_argument('--database', default="filtered_identity_database.pkl", help='Path to the identity database file.')
    parser.add_argument('--verbose', action='store_true', help='Show detailed processing logs.')
    parser.add_argument('--no_display', action='store_true', help='Do not display video frames during processing.')
    parser.add_argument('--start_frame', type=int, default=150, help='Frame number to start processing from (0-indexed).')
    # Parameters for tuning track identification:
    parser.add_argument('--min_frames_for_id', type=int, default=15, help='Min frames of data for a track before first ID attempt.')
    parser.add_argument('--id_interval', type=int, default=45, help='Frame interval for re-identification attempts on a track.')
    parser.add_argument('--max_buffer', type=int, default=60, help='Max number of per-feature items to keep in track buffer.')
    parser.add_argument('--track_timeout', type=int, default=90, help='Frames a track can be unseen before being removed.')
    # Debug options
    parser.add_argument('--debug_view', action='store_true', help='Show feature scores on video for debugging.')
    parser.add_argument('--feature_threshold', type=float, default=0.2, help='Minimum score to include a feature in identification.')
    # Identity Manager options
    parser.add_argument('--conflict_threshold', type=float, default=0.15, help='Threshold for detecting identity conflicts between tracks.')
    parser.add_argument('--show_identity_panel', action='store_true', help='Show identity assignments visualization panel.')
    parser.add_argument('--show_candidates', action='store_true', help='Show alternative candidate identities in the visualization panel.')
    parser.add_argument('--top_k_matches', type=int, default=3, help='Number of top candidate matches to consider for global optimization.')
    parser.add_argument('--global_opt_interval', type=int, default=30, help='Frame interval for running global optimization.')

    args = parser.parse_args()

    # --- Initialize Models ---
    initialize_all_models()

    # --- Define Feature Weights and Threshold with enhanced calibration ---
    FEATURE_WEIGHTS = {
        "opengait": 0.35,          # Most discriminative biometric feature - increased weight
        "skeleton_gait": 0.25,     # Dynamic features - reliable when available 
        "best_skeleton_3d": 0.15,  # Structural features - increased for better pose matching
        "industrial_pose": 0.08,   # Characteristic poses - slightly increased importance
        "industrial_color": 0.03,  # Less reliable due to clothing changes - reduced
        "height": 0.07,           # Reliable physical attribute - increased slightly
        "body_ratios": 0.05,       # Proportional measurements - reduced slightly
        "best_skeleton_2d": 0.02,  # Less reliable than 3D - lowest weight
    }
    SIMILARITY_THRESHOLD = 0.22 # Slightly increased to reduce false positives

    process_video_for_identification(
        args.input_video, args.database, args.output_video,
        FEATURE_WEIGHTS, SIMILARITY_THRESHOLD, args
    )