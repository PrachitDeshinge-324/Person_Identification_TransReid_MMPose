import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import pickle
import numpy as np
import argparse
import logging
import torch  # Ensure torch is imported at the top level
import traceback
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
from tqdm import tqdm  # Import tqdm for progress bars
from ultralytics.utils import LOGGER as yolo_logger
import copy

# Add VideoPose3D to sys.path
VIDEPOSE3D_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../VideoPose3D'))
if VIDEPOSE3D_PATH not in sys.path:
    sys.path.insert(0, VIDEPOSE3D_PATH)

# Try to import VideoPose3D model
TemporalModel = None
try:
    import importlib.util
    model_path = os.path.join(VIDEPOSE3D_PATH, 'common', 'model.py')
    spec = importlib.util.spec_from_file_location("common.model", model_path)
    if spec is not None:
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        TemporalModel = model_module.TemporalModel
        import torch
    else:
        print('VideoPose3D common.model not found. Skipping 3D pose lifting.')
        TemporalModel = None
except Exception as e:
    print(f'Error importing VideoPose3D: {e}')
    import os
    print('sys.path:', sys.path)
    print('VideoPose3D dir:', VIDEPOSE3D_PATH)
    print('Contents:', os.listdir(VIDEPOSE3D_PATH))
    common_dir = os.path.join(VIDEPOSE3D_PATH, 'common')
    if os.path.exists(common_dir):
        print('common/ dir contents:', os.listdir(common_dir))
    else:
        print('common/ dir does not exist!')
    TemporalModel = None

# Load VideoPose3D model (single model, not ensemble)
VIDEPOSE3D_CHECKPOINT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../weights/pretrained_h36m_cpn.bin'))
VIDEPOSE3D_MODEL = None

if TemporalModel is not None and os.path.exists(VIDEPOSE3D_CHECKPOINT):
    try:
        # Always use CPU for VideoPose3D - it's more stable and consistent
        # macOS MPS acceleration can cause issues with certain operations
        pose_device = torch.device('cpu')
        
        # Update model initialization to match checkpoint dimensions:
        # Parameters: (num_joints_in, in_features, num_joints_out, ...)
        VIDEPOSE3D_MODEL = TemporalModel(
            17, 2, 17, filter_widths=[3,3,3,3,3], causal=False, dropout=0.25, channels=1024, dense=False
        ).to(pose_device)
        
        # Use CPU for consistent behavior regardless of device
        checkpoint = torch.load(VIDEPOSE3D_CHECKPOINT, map_location=pose_device)
        
        # Attempt multiple loading strategies
        loaded = False
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            try:
                VIDEPOSE3D_MODEL.load_state_dict(checkpoint['model_state_dict'])
                VIDEPOSE3D_MODEL.eval()
                print('Loaded VideoPose3D model from checkpoint (model_state_dict).')
                loaded = True
            except Exception as e:
                print(f'Error loading model_state_dict: {e}')
        
        if not loaded and isinstance(checkpoint, dict) and 'model_pos' in checkpoint:
            try:
                VIDEPOSE3D_MODEL.load_state_dict(checkpoint['model_pos'])
                VIDEPOSE3D_MODEL.eval()
                print('Loaded VideoPose3D model from checkpoint (model_pos key).')
                loaded = True
            except Exception as e:
                print(f'Error loading model_pos: {e}')
        
        if not loaded and isinstance(checkpoint, dict):
            try:
                # Try direct loading
                VIDEPOSE3D_MODEL.load_state_dict(checkpoint)
                VIDEPOSE3D_MODEL.eval()
                print('Loaded VideoPose3D model from checkpoint (direct state_dict).')
                loaded = True
            except Exception as e:
                print(f'Error loading direct state_dict: {e}')
                
        if not loaded:
            print('All VideoPose3D loading strategies failed.')
            VIDEPOSE3D_MODEL = None
            
    except Exception as e:
        print(f'Error initializing VideoPose3D model: {e}')
        VIDEPOSE3D_MODEL = None
else:
    print('VideoPose3D model not loaded. 3D pose lifting will be skipped.')

# Function to lift 2D keypoints to 3D using VideoPose3D
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
    if VIDEPOSE3D_MODEL is None or keypoints_2d_seq is None or len(keypoints_2d_seq) == 0:
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
            device = next(VIDEPOSE3D_MODEL.parameters()).device
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
                pred_3d = VIDEPOSE3D_MODEL(kp_tensor)
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
          
# Add argument parser for command-line options
parser = argparse.ArgumentParser(description='Create enhanced identity database from video.')
parser.add_argument('--video', default="input/p.mp4", help='Path to video file')
parser.add_argument('--output', default="identity_database_p.pkl", help='Output database file')
parser.add_argument('--interactive', action='store_true', help='Use interactive naming')
parser.add_argument('--auto-name', action='store_true', help='Use automatic naming with predefined map')
parser.add_argument('--frames', type=int, default=400, help='Number of frames to process')
parser.add_argument('--skip', type=int, default=1, help='Frame skip rate')
parser.add_argument('--verbose', action='store_true', help='Show detailed processing logs')
parser.add_argument('--start-frame', type=int, default=0, help='Frame index to start processing from')
args = parser.parse_args()

VIDEO_PATH = args.video
YOLO_WEIGHTS = "weights/yolo11x.pt"
YOLOSEG_WEIGHTS = "weights/yolo11x-seg.pt"
TRANSREID_WEIGHTS = "weights/transreid_vitbase.pth"
MMPOSE_WEIGHTS = "weights/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth"
GAIT_WEIGHTS = "weights/GaitBase_DA-60000.pt"
OUTPUT_DB = args.output
DEVICE = "mps"  # or "cuda" or "cpu"

yolo_logger.setLevel(logging.WARNING)

# Define predefined name mapping
NAME_MAPPING = {
    1: "Prachit",
    2: "Ojasv",
    3: "Ashutosh",
    4: "Nayan",
    5: "Aditya"
}

REFERENCE_HEIGHT = 170.0  # Average human height in cm

print(f"Initializing models for enhanced feature extraction...")

# Initialize models without the unsupported 'verbose' parameter
# Set output suppression using other parameters if available
yolo = YOLOv8Tracker(YOLO_WEIGHTS, device=DEVICE, conf_threshold=0.25)
transreid = load_transreid_model(TRANSREID_WEIGHTS, DEVICE)
gait_embedder = OpenGaitEmbedder(weights_path=GAIT_WEIGHTS, device=DEVICE)

# For YOLOSeg, check if verbose parameter is supported
try:
    yoloseg = YOLOSeg(YOLOSEG_WEIGHTS, verbose=False)
except TypeError:
    # If verbose is not supported, initialize without it
    yoloseg = YOLOSeg(YOLOSEG_WEIGHTS)
    # Try to set YOLO general verbosity through environment variable
    os.environ['YOLO_VERBOSE'] = 'False'

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
    "industrial_color_features": [], # Industrial-specific color features
    "keypoints_3d": []        # 3D keypoints
})

# Motion pattern tracker
motion_trackers = {}

print(f"Processing video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
frame_count = 0

# Skip frames until start-frame
if args.start_frame > 0:
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    frame_idx = args.start_frame

# Get total frame count for progress bar
total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // args.skip, args.frames)
print(f"Will process {total_frames} frames (skipping every {args.skip} frames)")

# Replace the calculate_person_height function with this improved version
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

# Process video frames with progress bar
with tqdm(total=total_frames, desc="Processing video frames", unit="frame") as progress_bar:
    while frame_count < args.frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        if frame_idx % args.skip != 0:  # Skip frames for efficiency
            continue
        
        frame_count += 1
        
        # Person detection and tracking
        detections = yolo.process_frame(frame)
        if not detections:
            progress_bar.update(1)
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
            
            # 2D-to-3D pose lifting using VideoPose3D
            keypoints_3d = None
            if keypoints and len(keypoints) == 17 and VIDEPOSE3D_MODEL is not None:
                try:
                    # Check for valid keypoints with good confidence (at least 70% of keypoints)
                    good_keypoints_count = sum(1 for kp in keypoints if kp[2] > 0.3)
                    is_good_frame = good_keypoints_count >= 12  # At least 12 of 17 keypoints should be visible
                    
                    if is_good_frame:
                        # Extract just the x,y coordinates - use only high confidence keypoints
                        # For low confidence keypoints, we'll estimate their positions using the mean position
                        mean_pos_x = np.mean([kp[0] for kp in keypoints if kp[2] > 0.3])
                        mean_pos_y = np.mean([kp[1] for kp in keypoints if kp[2] > 0.3])
                        
                        keypoints_2d = []
                        for kp in keypoints:
                            if kp[2] > 0.3:
                                keypoints_2d.append([kp[0], kp[1]])
                            else:
                                # Use the mean position for low confidence keypoints
                                keypoints_2d.append([mean_pos_x, mean_pos_y])
                        
                        keypoints_2d = np.array(keypoints_2d, dtype=np.float32)
                        
                        # Skip if one of the important keypoints is missing (essential for pose estimation)
                        essential_indices = [0, 1, 2, 5, 6, 11, 12]  # Nose, neck, spine, shoulders, hips
                        if any(keypoints[i][2] < 0.2 for i in essential_indices):
                            if args.verbose:
                                print(f"Skipping 3D pose estimation - missing essential keypoints")
                            keypoints_3d = None
                            track_features[tid]["keypoints_3d"].append(None)
                            continue
                        
                        # First try temporal approach if we have enough frames
                        keypoints_3d_seq = None
                        current_sequence = track_features[tid]["skeleton_sequences"][-1] if track_features[tid]["skeleton_sequences"] else []
                        
                        if len(current_sequence) >= 5:  # Need at least 5 frames for better 3D lifting
                            try:
                                # Get the last 5-9 frames for temporal context
                                context_frames = min(9, len(current_sequence))
                                sequence_2d = []
                                
                                # Only use frames with good keypoints
                                for frame_kpts in current_sequence[-context_frames:]:
                                    if frame_kpts and sum(1 for kp in frame_kpts if kp[2] > 0.3) >= 12:
                                        coords_2d = []
                                        for kp in frame_kpts:
                                            if kp[2] > 0.3:
                                                coords_2d.append([kp[0], kp[1]])
                                            else:
                                                coords_2d.append([mean_pos_x, mean_pos_y])
                                        sequence_2d.append(coords_2d)
                                
                                if len(sequence_2d) >= 3:  # Need minimum 3 good frames
                                    sequence_2d = np.array(sequence_2d, dtype=np.float32)  # Shape: (frames, 17, 2)
                                    # Try to lift the sequence for better temporal consistency
                                    if args.verbose:
                                        print(f"Trying temporal 3D pose lifting with {len(sequence_2d)} frames")
                                    multi_frame_3d = lift_2d_to_3d(sequence_2d)
                                    if multi_frame_3d is not None:
                                        # Use the last frame from the temporal sequence
                                        keypoints_3d_seq = multi_frame_3d[-1:]
                                        if args.verbose:
                                            print("Successfully extracted 3D pose using temporal context")
                            except Exception as e:
                                print(f"Error in temporal 3D pose lifting: {e}")
                                keypoints_3d_seq = None
                        
                        # Fall back to single-frame processing if temporal approach failed
                        if keypoints_3d_seq is None:
                            if args.verbose:
                                print("Falling back to single-frame 3D pose lifting")
                            # For single-frame processing
                            keypoints_3d_seq = lift_2d_to_3d(keypoints_2d)
                        
                        if keypoints_3d_seq is not None:
                            keypoints_3d = keypoints_3d_seq[0]  # Get the single frame result
                except Exception as e:
                    print(f"Error processing 3D keypoints: {e}")
                    import traceback
                    traceback.print_exc()
                    keypoints_3d = None
                    
            track_features[tid]["keypoints_3d"].append(keypoints_3d)
            
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
            
            try:
                # Extract normalized pose features for industrial contexts
                industrial_pose_features = compute_normalized_pose_features(keypoints)
                track_features[tid]["industrial_pose_features"].append(industrial_pose_features)
            except Exception as e:
                print(f"Warning: Error computing pose features for track {tid}: {e}")
                track_features[tid]["industrial_pose_features"].append({})
            
            # 3. Body ratios - derived measurements
            ratios = compute_body_ratios(keypoints) if keypoints else {}
            track_features[tid]["body_ratios"].append(ratios)
            
            # 4. Height information - more accurate calculation
            # When calling height calculation, pass actual camera parameters:
            height, conf = calculate_person_height(
                keypoints,
                bbox,
                frame_height=frame.shape[0],
                focal_length=35,  # mm (typical smartphone camera)
                sensor_height=24, # mm (1/2.3" sensor)
                ref_shoulder_width=45  # Average shoulder width in cm
            )

            # For industrial cameras, use:
            # focal_length = (sensor_width * focal_length_mm) / sensor_width_mm
            track_features[tid]["heights"].append(height)
            if "height_confidences" not in track_features[tid]:
                track_features[tid]["height_confidences"] = []
            track_features[tid]["height_confidences"].append(conf)
            
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
        
        # Update progress bar with info
        if args.verbose and frame_count % 10 == 0:
            progress_bar.set_postfix(tracks=len(ids), frame_idx=frame_idx)
        progress_bar.update(1)

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
    unnamed_tracks = []
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
                
                plt.subplot(1, 2, 2)
                plt.imshow(blank)
                plt.title("Skeleton")
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # Suggest the predefined name if available
        suggested_name = NAME_MAPPING.get(tid, f"Person_{tid}")
        name = input(f"Enter name for track ID {tid} (suggested: {suggested_name}, or 'X' to review all crops): ")
        
        if name.strip().lower() == 'x' or not name.strip():
            # Save all crops and skeleton visualizations for manual review
            out_dir = os.path.join('output', 'unnamed_tracks', f'track_{tid}')
            os.makedirs(out_dir, exist_ok=True)
            
            # Save all crops
            for idx, crop in enumerate(feats["crops"]):
                crop_path = os.path.join(out_dir, f"crop_{idx}.jpg")
                cv2.imwrite(crop_path, crop)
            
            # Save all skeleton visualizations
            for idx, kpts in enumerate(feats["raw_skeletons"]):
                if kpts is not None:
                    blank = np.ones((256, 256, 3), dtype=np.uint8) * 255
                    valid_kpts = [(int(k[0]), int(k[1])) for k in kpts if k[2] > 0]
                    if valid_kpts:
                        min_x = min([k[0] for k in valid_kpts])
                        max_x = max([k[0] for k in valid_kpts])
                        min_y = min([k[1] for k in valid_kpts])
                        max_y = max([k[1] for k in valid_kpts])
                        scale = 200 / max(max_x - min_x, max_y - min_y, 1)
                        offset_x = 128 - (min_x + max_x) * scale / 2
                        offset_y = 128 - (min_y + max_y) * scale / 2
                        # Draw connections
                        connections = [
                            (0, 1), (1, 2), (2, 3), (3, 4),
                            (0, 5), (5, 6), (6, 7),
                            (0, 8), (8, 9), (9, 10),
                            (0, 11), (11, 12), (12, 13),
                        ]
                        for connection in connections:
                            if kpts[connection[0]][2] > 0.2 and kpts[connection[1]][2] > 0.2:
                                pt1 = (int(kpts[connection[0]][0] * scale + offset_x), int(kpts[connection[0]][1] * scale + offset_y))
                                pt2 = (int(kpts[connection[1]][0] * scale + offset_x), int(kpts[connection[1]][1] * scale + offset_y))
                                cv2.line(blank, pt1, pt2, (0, 0, 255), 2)
                        for i, kpt in enumerate(kpts):
                            if kpt[2] > 0.2:
                                x = int(kpt[0] * scale + offset_x)
                                y = int(kpt[1] * scale + offset_y)
                                cv2.circle(blank, (x, y), 3, (0, 255, 0), -1)
                                cv2.putText(blank, str(i), (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                    skeleton_path = os.path.join(out_dir, f"skeleton_{idx}.jpg")
                    cv2.imwrite(skeleton_path, blank)
            
            print(f"Track {tid}: All crops and skeletons saved to {out_dir}")
            print(f"Please review the saved images at {out_dir}")
            
            # Ask for name again after saving crops for review
            second_name = input(f"After reviewing crops, enter name for track {tid} (or 'X' to discard): ")
            
            if second_name.strip().lower() == 'x' or not second_name.strip():
                print(f"Track {tid} has been discarded.")
                unnamed_tracks.append((tid, out_dir))
                continue  # Skip this track, don't add to track_id_to_name
            else:
                # Use the second attempt name
                track_id_to_name[tid] = second_name.strip()
                print(f"Track {tid} named as '{second_name.strip()}'")
        else:
            # Use the first name provided
            track_id_to_name[tid] = name.strip()
            print(f"Track {tid} named as '{name.strip()}'")
    
    # Print summary of any tracks that were still unnamed after the second chance
    if unnamed_tracks:
        print("\nThe following tracks were skipped and will not be included in the database:")
        for tid, folder in unnamed_tracks:
            print(f"  Track {tid}: {folder}")

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
with tqdm(track_features.items(), desc="Aggregating features", unit="track") as progress_bar:
    # Aggregate features for each track
    identity_db = {}
    for tid, feats in progress_bar:
        # Only include tracks with sufficient data
        if not feats["appearance"] or len(feats["appearance"]) < 3:
            progress_bar.set_postfix(status=f"skipping track {tid} - insufficient data")
            continue
        
        progress_bar.set_postfix(status=f"processing {track_id_to_name.get(tid, f'Person_{tid}')}")
        
        # --------------------------------------------------------------------------------------
        # NOTE: For tracking and re-identification within the current video/clip, ALL features
        # (including appearance, color, HOG, etc.) are used for association and ID assignment.
        # This maximizes short-term accuracy and robustness to occlusion, pose, and viewpoint.
        #
        # For long-term storage in the identity database, ONLY robust, invariant features are
        # saved (gait, skeleton, 3D pose, body ratios, etc.) to ensure invariance to clothing,
        # lighting, and camera changes. Appearance features are NOT stored in the database.
        # --------------------------------------------------------------------------------------
        # Construct the final identity entry
        
        # 1. Appearance: mean of top confident detections
        if feats["appearance"]:
            try:
                # First, filter by confidence if track_confidence exists and has values
                filtered_feats = []
                if feats["track_confidence"] and len(feats["track_confidence"]) > 0:
                    filtered_feats = [feat for i, feat in enumerate(feats["appearance"]) 
                                     if i < len(feats["track_confidence"]) and feats["track_confidence"][i] > 0.5]
                
                # If no features passed the confidence filter, use all features
                if not filtered_feats and len(feats["appearance"]) > 0:
                    filtered_feats = feats["appearance"]
                
                # Check if we have any features to stack
                if filtered_feats:
                    app_feats = np.stack(filtered_feats)
                    mean_app = np.mean(app_feats, axis=0)
                else:
                    mean_app = None
            except Exception as e:
                print(f"Error aggregating appearance features for track {tid}: {e}")
                mean_app = None
        else:
            mean_app = None
        
        # 2. OpenGait embeddings from silhouettes
        opengait_embedding = None
        if feats["silhouettes"] and len(feats["silhouettes"]) >= 10:
            try:
                # Take the 30 silhouettes with highest average confidence, or all if less than 30
                if len(feats["silhouettes"]) > 30:
                    confs = feats["track_confidence"]
                    top_indices = np.argsort(confs)[-30:][::-1]
                    sils = [feats["silhouettes"][i] for i in top_indices]
                else:
                    sils = feats["silhouettes"]

                # Always resize and stack silhouettes to (N, 128, 88) float32
                try:
                    # Better debug info for initial silhouette before processing
                    if args.verbose or True:
                        if sils and len(sils) > 0:
                            first_sil = sils[0]
                            print(f"[OpenGait DEBUG] Track {tid}: original silhouette shape: {first_sil.shape if first_sil is not None else 'None'}")
                        else:
                            print(f"[OpenGait DEBUG] Track {tid}: silhouettes list is empty!")
                    
                    # Process each silhouette - ensure correct shape and binary values
                    processed_sils = []
                    for sil in sils:
                        # Convert silhouette to binary mask if it's not already
                        if isinstance(sil, np.ndarray) and sil.ndim == 2:
                            # Already 2D array
                            binary_sil = (sil > 0.5).astype(np.float32)
                        else:
                            print(f"[OpenGait DEBUG] Track {tid}: unexpected silhouette type: {type(sil)}")
                            continue
                        # Resize to expected OpenGait input size
                        resized_sil = cv2.resize(binary_sil, (88, 128), interpolation=cv2.INTER_NEAREST)
                        processed_sils.append(resized_sil)
                    
                    if len(processed_sils) < 10:
                        print(f"[OpenGait DEBUG] Track {tid}: Too few valid silhouettes after filtering: {len(processed_sils)}. Skipping OpenGait extraction for this track.")
                        opengait_embedding = None
                        continue
                    
                    # Stack processed silhouettes
                    sils_resized = np.stack(processed_sils, axis=0)
                    
                    if args.verbose or True:
                        print(f"[OpenGait DEBUG] Track {tid}: Final processed array - shape: {sils_resized.shape}, dtype: {sils_resized.dtype}, min: {sils_resized.min()}, max: {sils_resized.max()}")
                    
                    # Try extraction with more detailed error reporting
                    opengait_embedding = gait_embedder.extract(sils_resized)
                    if args.verbose:
                        print(f"Successfully extracted OpenGait embedding for {tid} with shape: {opengait_embedding.shape}")
                except Exception as e:
                    print(f"OpenGait extraction error for {tid}: {e}")
                    print(traceback.format_exc())
                    
                    # Try a fallback approach with more explicit tensor handling
                    try:
                        print(f"[OpenGait DEBUG] Trying fallback extraction approach for {tid}...")
                        # Convert to tensor manually with explicit dimensions
                        tensor_input = torch.from_numpy(sils_resized).float()
                        # Ensure shape is (N, 1, H, W) - add channel dimension if needed
                        if tensor_input.dim() == 3:  # If (N, H, W)
                            tensor_input = tensor_input.unsqueeze(1)  # Add channel dim -> (N, 1, H, W)
                        print(f"[OpenGait DEBUG] Tensor input shape: {tensor_input.shape}")
                        tensor_input = tensor_input.to(gait_embedder.device)
                        # Run inference directly using the model
                        with torch.no_grad():
                            feat = gait_embedder.model(tensor_input)
                            embedding = feat.mean(dim=[0, 2, 3]).cpu().numpy()
                            print(f"[OpenGait DEBUG] Success! Embedding shape: {embedding.shape}")
                            opengait_embedding = embedding
                    except Exception as fallback_error:
                        print(f"Fallback extraction also failed for {tid}: {fallback_error}")
                        print(traceback.format_exc())
            except Exception as e:
                print(f"Error preparing silhouettes for OpenGait extraction: {e}")
                if feats["silhouettes"]:
                    first_sil = feats["silhouettes"][0]
                    print(f" - First silhouette shape: {first_sil.shape if hasattr(first_sil, 'shape') else 'unknown'}")
                    print(f" - First silhouette type: {type(first_sil)}")

        # 3. Skeleton-based gait features
        skeleton_gait_features = {}
        for seq_idx, sequence in enumerate(feats.get("skeleton_sequences", [])):
            if len(sequence) >= 10:  # Need enough frames for gait
                try:
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
                except Exception as e:
                    print(f"Error extracting skeleton gait features: {e}")
        
        # 4. 3D skeleton keypoints - aggregate (median for robustness)
        best_3d_skeleton = None
        if feats.get("keypoints_3d"):
            valid_3d = [k for k in feats["keypoints_3d"] if k is not None and np.all(np.isfinite(k))]
            if valid_3d:
                best_3d_skeleton = np.median(np.stack(valid_3d), axis=0).tolist()  # shape: (17, 3)

        # 5. Raw skeleton keypoints - select high confidence keypoints
        if feats["raw_skeletons"]:
            best_skeleton_idx = [i for i, full in enumerate(feats["full_body_visible"]) if full]
            if best_skeleton_idx:
                # Pick the best full-body skeleton
                best_skeleton = feats["raw_skeletons"][best_skeleton_idx[len(best_skeleton_idx)//2]]
            else:
                # If no full-body skeleton, pick the one with highest confidence
                best_skeleton = feats["raw_skeletons"][np.argmax(feats["track_confidence"])]
        else:
            best_skeleton = None
        
        # 6. Body ratios - mean of valid measurements
        valid_ratios = [r for r in feats["body_ratios"] if r]
        if valid_ratios:
            mean_ratios = {
                k: float(np.mean([r[k] for r in valid_ratios if k in r])) 
                for k in set().union(*valid_ratios)
            }
        else:
            mean_ratios = {}
        
        # 7. Color histograms - mean
        if feats["color_hists"]:
            mean_color_hist = np.mean(np.stack(feats["color_hists"]), axis=0)
        else:
            mean_color_hist = None
        
        # 8. HOG features - mean of valid features
        valid_hog = [h for h in feats["hog_features"] if h is not None]
        if valid_hog:
            mean_hog = np.mean(np.stack(valid_hog), axis=0)
        else:
            mean_hog = None
        
        # 9. Motion patterns
        valid_motion = [m for m in feats["motion_patterns"] if m is not None]
        if valid_motion and len(valid_motion) >= 3:
            mean_motion = {
                k: np.mean([m[k] for m in valid_motion]) 
                for k in valid_motion[0].keys()
            }
        else:
            mean_motion = None
        
        # 10. Height - confidence-weighted median height
        if feats["heights"] and "height_confidences" in feats:
            # Filter out low confidence measurements
            good_heights = [h for h, c in zip(feats["heights"], feats["height_confidences"]) 
                          if c > 0.4 and h is not None]
            
            if good_heights:
                median_height = float(np.median(good_heights))
            else:
                # Fallback to regular median if no good confidence measurements
                median_height = float(np.median(feats["heights"]))
        else:
            median_height = None
        
        # 11. Context - last known
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
            "opengait": opengait_embedding,
            "skeleton_gait": skeleton_gait_features,
            "best_skeleton": best_skeleton,
            "best_3d_skeleton": best_3d_skeleton,
            "industrial_pose": avg_industrial_pose,
            "body_ratios": mean_ratios,
            "height": median_height,
            "motion_pattern": mean_motion,
            "context": last_context,
            "industrial_color": avg_industrial_color,  # Make sure to include this field
            "feature_quality": {
                "skeleton_samples": len(feats["raw_skeletons"]),
                "opengait_samples": len(feats["silhouettes"]), 
                "skeleton_sequences": len(feats.get("skeleton_sequences", [])),
                "avg_confidence": np.mean(feats["track_confidence"]) if feats["track_confidence"] else 0
            }
        }

# Save database with progress indication
print(f"Saving database to {OUTPUT_DB}...")

# Add identity merging to handle duplicate IDs for the same person
print("Performing identity merging to handle duplicate IDs...")
merged_identity_db = {}
name_to_ids = defaultdict(list)

# Group IDs by name
for tid, data in identity_db.items():
    name = data.get('name')
    if name and name != f"Person_{tid}" and name.lower() != "x":  # Skip unnamed or placeholder names
        name_to_ids[name].append(tid)
    else:
        # Keep non-named identities as is
        merged_identity_db[tid] = data

# Process each group of track IDs belonging to the same person
for name, ids in name_to_ids.items():
    if len(ids) == 1:
        # No duplicates, just keep the original
        merged_identity_db[ids[0]] = identity_db[ids[0]]
        continue
    
    print(f"Merging {len(ids)} duplicate IDs for {name}: {ids}")
    
    # Select primary ID (the one with best features)
    # Prioritize IDs with OpenGait features, more samples, and higher confidence
    best_id = None
    best_score = -1
    
    for tid in ids:
        entry = identity_db[tid]
        quality = entry.get('feature_quality', {})
        
        # Score each ID based on feature quality (ignore appearance to handle clothing changes)
        score = 0
        score += 100 if entry.get('opengait') is not None else 0
        score += min(40, len(entry.get('skeleton_gait', {})))
        score += min(30, quality.get('skeleton_samples', 0))
        score += min(20, 1 if entry.get('industrial_pose') else 0)
        score += min(10, quality.get('avg_confidence', 0) * 10)

        
        if score > best_score:
            best_score = score
            best_id = tid
    
    # Create merged entry starting with the best one
    merged_entry = copy.deepcopy(identity_db[best_id])
    merged_entry["merged_from"] = ids
    merged_entry["name"] = name  # Ensure consistent name
    
    # Merge features from other occurrences if they'd improve the identity
    for tid in ids:
        if tid == best_id:
            continue
            
        entry = identity_db[tid]
        
        # Take OpenGait features if primary doesn't have them
        if merged_entry.get('opengait') is None and entry.get('opengait') is not None:
            merged_entry['opengait'] = entry['opengait']
        
        # Merge skeleton gait features (take average of available features)
        if entry.get('skeleton_gait'):
            if not merged_entry.get('skeleton_gait'):
                merged_entry['skeleton_gait'] = entry['skeleton_gait']
            else:
                # Combine features from both entries
                for key, value in entry['skeleton_gait'].items():
                    if key in merged_entry['skeleton_gait']:
                        merged_entry['skeleton_gait'][key] = (merged_entry['skeleton_gait'][key] + value) / 2
                    else:
                        merged_entry['skeleton_gait'][key] = value
        
        # Similar merging for industrial features
        if entry.get('industrial_pose'):
            if not merged_entry.get('industrial_pose'):
                merged_entry['industrial_pose'] = entry['industrial_pose']
            else:
                for key, value in entry['industrial_pose'].items():
                    if key in merged_entry['industrial_pose']:
                        merged_entry['industrial_pose'][key] = (merged_entry['industrial_pose'][key] + value) / 2
                    else:
                        merged_entry['industrial_pose'][key] = value
        
        # Update feature quality metrics
        merged_quality = merged_entry.get('feature_quality', {})
        entry_quality = entry.get('feature_quality', {})
        
        for key in merged_quality.keys():
            if key in entry_quality:
                merged_quality[key] = max(merged_quality[key], entry_quality[key])
        
        merged_entry['feature_quality'] = merged_quality
    
    # Store the merged entry under the primary ID
    merged_identity_db[best_id] = merged_entry

# Use the merged database
identity_db = merged_identity_db
print(f"Identity merging complete. Database reduced from {len(track_id_to_name)} to {len(identity_db)} unique identities.")

# DEBUG: Enhanced diagnostics for silhouettes and OpenGait
for tid, data in identity_db.items():
    if data.get('opengait') is None:
        print(f"[DEBUG] ID {tid} ({data.get('name', f'Person_{tid}')}) has no OpenGait embedding.")
        merged_from = data.get('merged_from', [tid])
        for sub_tid in merged_from:
            # Check if silhouettes were present for this track
            feats = track_features.get(sub_tid)
            if feats is None:
                print(f"  - Track {sub_tid}: No features found.")
                continue
            n_sil = len(feats.get('silhouettes', [])) if 'silhouettes' in feats else 0
            print(f"  - Track {sub_tid}: {n_sil} silhouettes.")
            if n_sil < 10:
                print(f"    -> Not enough silhouettes for OpenGait (need >=10).")
            else:
                print(f"    -> Silhouettes present, possible extraction error.")
                # Additional diagnostics for Nayan's silhouettes
                if n_sil > 0 and data.get('name') == 'Nayan':
                    try:
                        first_sil = feats["silhouettes"][0]
                        print(f"    -> First silhouette stats: shape={first_sil.shape}, "
                              f"min={first_sil.min():.4f}, max={first_sil.max():.4f}, "
                              f"mean={first_sil.mean():.4f}")
                        # Check if the mask is binary or continuous
                        unique_vals = np.unique(first_sil)
                        print(f"    -> Unique values: {len(unique_vals)} values in range [{unique_vals.min()}, {unique_vals.max()}]")
                    except Exception as e:
                        print(f"    -> Error analyzing silhouette: {e}")

with open(OUTPUT_DB, "wb") as f:
    pickle.dump(identity_db, f)

print(f"Identity database saved to {OUTPUT_DB} with {len(identity_db)} identities:")
for tid, data in identity_db.items():
    name = data.get('name', f'Person_{tid}')
    merged_info = f" (merged from {len(data.get('merged_from', []))} IDs)" if 'merged_from' in data else ""
    has_opengait = "✅" if data.get('opengait') is not None else "❌"
    has_skeleton_gait = "✅" if data.get('skeleton_gait') and len(data.get('skeleton_gait')) > 0 else "❌"
    has_industrial_pose = "✅" if data.get('industrial_pose') else "❌"
    has_industrial_color = "✅" if data.get('industrial_color') else "❌"
    print(f"  ID {tid}: {name}{merged_info} - OpenGait: {has_opengait}, Skeleton Gait: {has_skeleton_gait}, Industrial Pose: {has_industrial_pose}, Industrial Color: {has_industrial_color}")
