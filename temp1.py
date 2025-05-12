# %%
# Import all necessary modules
import os
import sys
import cv2
import numpy as np
import torch
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the VideoPose3D directory to the Python path
videopose3d_path = os.path.abspath("./VideoPose3D")
if videopose3d_path not in sys.path:
    sys.path.append(videopose3d_path)

# Now import the necessary modules from VideoPose3D
from VideoPose3D.common.model import TemporalModel
from VideoPose3D.common.camera import normalize_screen_coordinates

# Load the trained VideoPose3D model
checkpoint_path = "./weight/pretrained_h36m_cpn.bin"
# Initialize model with the same parameters as the pre-trained model
model = TemporalModel(
    num_joints_in=17,  # Number of input joints (COCO format)
    in_features=2,     # 2D keypoints (x, y)
    num_joints_out=17, # Number of output joints
    filter_widths=[3, 3, 3, 3, 3],  # Must match pre-trained model
    causal=False,
    dropout=0.25,
    channels=1024
)

# Load model weights
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
if 'model_pos' in checkpoint:
    model.load_state_dict(checkpoint['model_pos'])
else:
    raise KeyError("The checkpoint does not contain the expected 'model_pos' key.")

model.eval()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def detect_2d_keypoints(frame, target_bbox=None, target_descriptor=None, is_first_frame=False):
    """
    Detect 2D keypoints using MediaPipe Pose.
    Focus on a single person, preferably the one closest to the target_bbox.
    
    Args:
        frame: The video frame to process
        target_bbox: Optional bounding box of the person to track [x_min, y_min, x_max, y_max]
        target_descriptor: Optional descriptor of the person we're tracking
        is_first_frame: Whether this is the first frame (helps with initialization)
    
    Returns:
        keypoints: The detected 2D keypoints for the selected person
        descriptor: Updated descriptor for the tracked person (for next frame)
    """
    # Default empty descriptor
    current_descriptor = None
    
    # Try standard MediaPipe pose detection first
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose_detector:
        
        results = pose_detector.process(frame_rgb)
    
    # If we have a detection and no specific target, or this is the first frame,
    # use this detection as our primary person
    if results.pose_landmarks and (target_bbox is None or is_first_frame):
        height, width = frame.shape[:2]
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            x = landmark.x * width
            y = landmark.y * height
            keypoints.append([x, y])
        
        keypoints = np.array(keypoints)[:17]
        
        # Create descriptor for this person
        x_min = np.min(keypoints[:, 0])
        x_max = np.max(keypoints[:, 0])
        y_min = np.min(keypoints[:, 1])
        y_max = np.max(keypoints[:, 1])
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        height = y_max - y_min
        width = x_max - x_min
        aspect_ratio = height / max(width, 1)
        
        current_descriptor = {
            'center': (center_x, center_y),
            'bbox': [x_min, y_min, x_max, y_max],
            'aspect_ratio': aspect_ratio,
            'pose_size': height * width,
            'last_seen': 0  # Frame counter since last seen
        }
        
        return keypoints, current_descriptor
    
    # If we're tracking a specific person or multiple people might be in frame
    if target_bbox is not None or not results.pose_landmarks:
        # Try to detect multiple poses
        poses = detect_multiple_poses(frame)
        
        if not poses:
            # If no poses detected, return empty keypoints but maintain descriptor
            return np.zeros((17, 2)), target_descriptor
        
        # If we have a target bbox, find the best matching pose
        if target_bbox:
            best_match = None
            best_score = 0
            
            for pose in poses:
                # Calculate IoU with target bbox
                iou = calculate_iou(pose['bbox'], target_bbox)
                
                # Calculate distance between centers
                target_center_x = (target_bbox[0] + target_bbox[2]) / 2
                target_center_y = (target_bbox[1] + target_bbox[3]) / 2
                dist_x = pose['center'][0] - target_center_x
                dist_y = pose['center'][1] - target_center_y
                center_dist = np.sqrt(dist_x**2 + dist_y**2)
                
                # Normalize distance by frame dimensions
                height, width = frame.shape[:2]
                normalized_dist = center_dist / np.sqrt(width**2 + height**2)
                
                # Combined score: IoU is more important, but center distance helps when IoU fails
                score = iou * 0.7 + (1 - normalized_dist) * 0.3
                
                # If we have a target descriptor, also factor in aspect ratio similarity
                if target_descriptor:
                    aspect_ratio_diff = abs(pose['aspect_ratio'] - target_descriptor['aspect_ratio']) / \
                                        max(pose['aspect_ratio'], target_descriptor['aspect_ratio'])
                    size_ratio = min(pose['pose_size'], target_descriptor['pose_size']) / \
                                max(pose['pose_size'], target_descriptor['pose_size'])
                    
                    descriptor_score = (1 - aspect_ratio_diff) * 0.5 + size_ratio * 0.5
                    score = score * 0.7 + descriptor_score * 0.3
                
                if score > best_score:
                    best_score = score
                    best_match = pose
            
            # If we found a good match, use it
            if best_match and best_score > 0.3:
                current_descriptor = best_match
                current_descriptor['last_seen'] = 0  # Reset counter
                current_descriptor['match_score'] = best_score
                return best_match['keypoints'], current_descriptor
        
        # If no target or no good match, just take the biggest pose (most central in the frame)
        if poses:
            # Sort by size (area of bounding box)
            poses.sort(key=lambda p: p['pose_size'], reverse=True)
            
            # Take the largest person
            current_descriptor = poses[0]
            current_descriptor['last_seen'] = 0
            return poses[0]['keypoints'], current_descriptor
    
    # If we get here, we couldn't find a good match; return empty keypoints but maintain descriptor
    if results.pose_landmarks:
        height, width = frame.shape[:2]
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            x = landmark.x * width
            y = landmark.y * height
            keypoints.append([x, y])
        
        keypoints = np.array(keypoints)[:17]
        return keypoints, target_descriptor
    
    return np.zeros((17, 2)), target_descriptor

def detect_multiple_poses(frame):
    """
    Detect multiple people in a frame using MediaPipe Pose.
    This is a utility function to get multiple pose detections.
    
    Args:
        frame: The video frame to process
        
    Returns:
        A list of detected poses (if any)
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    poses = []
    
    # We'll use multiple instances of the detector with different parameters to 
    # try to get multiple poses
    for model_complexity in [1, 2]:
        with mp_pose.Pose(
            static_image_mode=True,  # We want the full detection, not tracking
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            enable_segmentation=False) as pose_detector:
            
            results = pose_detector.process(frame_rgb)
            
            if results.pose_landmarks:
                height, width = frame.shape[:2]
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    x = landmark.x * width
                    y = landmark.y * height
                    keypoints.append([x, y])
                
                # Convert to array and take only the first 17 points (COCO format compatibility)
                keypoints = np.array(keypoints)[:17]
                
                # Calculate the bounding box
                x_min = np.min(keypoints[:, 0])
                x_max = np.max(keypoints[:, 0])
                y_min = np.min(keypoints[:, 1])
                y_max = np.max(keypoints[:, 1])
                
                # Calculate the center of the person
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                
                # Calculate a simple feature descriptor: height/width ratio
                height = y_max - y_min
                width = x_max - x_min
                aspect_ratio = height / max(width, 1)  # Avoid division by zero
                
                # Get a rough estimate of the person's "signature"
                # This could be improved with more sophisticated features
                feature_descriptor = {
                    'center': (center_x, center_y),
                    'bbox': [x_min, y_min, x_max, y_max],
                    'keypoints': keypoints,
                    'aspect_ratio': aspect_ratio,
                    'pose_size': height * width
                }
                
                # Add this pose if it doesn't overlap significantly with existing ones
                is_new_pose = True
                for existing_pose in poses:
                    iou = calculate_iou(feature_descriptor['bbox'], existing_pose['bbox'])
                    if iou > 0.5:  # High overlap means it's probably the same person
                        is_new_pose = False
                        break
                
                if is_new_pose:
                    poses.append(feature_descriptor)
    
    return poses

def visualize_3d_keypoints_on_frame(frame, keypoints_3d):
    """
    Overlay 3D keypoints visualization on a 2D frame.
    Focus on a single skeleton with enhanced depth visualization.
    
    Args:
        frame: The video frame to overlay visualization on
        keypoints_3d: 3D keypoints from VideoPose3D
    """
    # Define connections between joints for visualization (COCO format)
    connections = [
        (0, 1), (1, 2), (2, 3),  # Right arm
        (0, 4), (4, 5), (5, 6),  # Left arm
        (0, 7),                  # Spine
        (7, 8), (8, 9), (9, 10),  # Right leg
        (7, 11), (11, 12), (12, 13),  # Left leg
        (0, 14), (14, 15), (15, 16)  # Head
    ]
    
    # Project 3D keypoints to 2D for visualization
    keypoints_2d = keypoints_3d[:, :2]  # Just use x, y for 2D position
    z_values = keypoints_3d[:, 2]  # Z coordinates for depth
    
    # Scale to frame coordinates
    height, width = frame.shape[:2]
    keypoints_2d[:, 0] = (keypoints_2d[:, 0] + 0.5) * width
    keypoints_2d[:, 1] = (keypoints_2d[:, 1] + 0.5) * height
    
    # Calculate confidence based on keypoint visibility
    # We consider a joint visible if its coordinates are valid (not NaN)
    visible_joints = ~np.isnan(keypoints_2d).any(axis=1)
    joints_visibility_count = np.sum(visible_joints)
    
    # Only proceed if we have enough visible keypoints
    min_visible_joints = 8
    if joints_visibility_count >= min_visible_joints:
        # Normalize z-values for color-coding depth
        valid_z = z_values[~np.isnan(z_values)]
        if len(valid_z) > 0:
            z_min, z_max = valid_z.min(), valid_z.max()
            z_range = max(0.001, z_max - z_min)  # Avoid division by zero
            
            # Draw connections with depth-based coloring
            for connection in connections:
                start_idx, end_idx = connection
                if (start_idx < len(keypoints_2d) and end_idx < len(keypoints_2d) and
                    not np.isnan(keypoints_2d[start_idx]).any() and 
                    not np.isnan(keypoints_2d[end_idx]).any()):
                    
                    # Get the average depth of the connection for coloring
                    start_z = z_values[start_idx]
                    end_z = z_values[end_idx]
                    avg_z = (start_z + end_z) / 2
                    
                    # Color based on depth (closer = red, further = blue)
                    depth_ratio = (avg_z - z_min) / z_range if z_range > 0 else 0.5
                    depth_ratio = min(1.0, max(0.0, depth_ratio))  # Clamp to [0, 1]
                    
                    color = (
                        int(255 * (1 - depth_ratio)),  # Blue (far)
                        50,  # Green (constant)
                        int(255 * depth_ratio)  # Red (close)
                    )
                    
                    # Draw thicker lines for connections closer to the camera
                    thickness = max(1, min(4, int(4 * (1 - depth_ratio) + 1)))
                    
                    # Draw the connection
                    start_point = (int(keypoints_2d[start_idx][0]), int(keypoints_2d[start_idx][1]))
                    end_point = (int(keypoints_2d[end_idx][0]), int(keypoints_2d[end_idx][1]))
                    cv2.line(frame, start_point, end_point, color, thickness)
            
            # Draw keypoints with depth-based coloring
            for i, (x, y) in enumerate(keypoints_2d):
                if not np.isnan(x) and not np.isnan(y) and not np.isnan(z_values[i]):
                    depth_ratio = (z_values[i] - z_min) / z_range if z_range > 0 else 0.5
                    depth_ratio = min(1.0, max(0.0, depth_ratio))
                    
                    color = (
                        int(255 * (1 - depth_ratio)),  # Blue (far)
                        50,  # Green (constant)
                        int(255 * depth_ratio)  # Red (close)
                    )
                    
                    # Larger points for joints closer to the camera
                    radius = max(3, min(8, int(8 * (1 - depth_ratio) + 3)))
                    cv2.circle(frame, (int(x), int(y)), radius, color, -1)
                    
            # Draw a depth color bar
            bar_width = 20
            bar_height = height // 3
            bar_x = width - bar_width - 10
            bar_y = 10
            
            # Draw the background for the color bar
            cv2.rectangle(frame, (bar_x-2, bar_y-2), (bar_x + bar_width + 2, bar_y + bar_height + 2), 
                         (255, 255, 255), 1)
            
            # Draw the color gradient
            for i in range(bar_height):
                depth_ratio = 1 - (i / bar_height)
                color = (
                    int(255 * (1 - depth_ratio)),  # Blue (far) 
                    50,  # Green (constant)
                    int(255 * depth_ratio)  # Red (close)
                )
                cv2.line(frame, (bar_x, bar_y + i), (bar_x + bar_width, bar_y + i), color, 1)
            
            # Add labels for the color bar
            cv2.putText(frame, "NEAR", (bar_x - 42, bar_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 50, 50), 1)
            cv2.putText(frame, "FAR", (bar_x - 35, bar_y + bar_height - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 255), 1)
            
            # Add Z-value range
            cv2.putText(frame, f"Z: [{z_min:.2f}, {z_max:.2f}]", (bar_x - 80, bar_y + bar_height + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add skeleton stats overlay
        info_y = 30
        cv2.putText(frame, "3D Pose Estimation", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        info_y += 25
        
        # Add visibility metric
        cv2.putText(frame, f"Visible joints: {joints_visibility_count}/17", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info_y += 20
        
        # Add confidence score (based on joint visibility percentage)
        confidence = (joints_visibility_count / 17) * 100
        confidence_color = (0, 255, 0) if confidence > 70 else ((0, 255, 255) if confidence > 50 else (0, 0, 255))
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, confidence_color, 1)
        
    else:
        # If not enough keypoints are visible, show a message
        cv2.putText(frame, "No clear skeleton detected", (width//2 - 120, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Visible joints: {joints_visibility_count}/17", (width//2 - 100, height//2 + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

def process_video(video_path, output_path, receptive_field=243, frame_skip=5):
    """
    Process a video file to estimate 3D poses using VideoPose3D.
    Will focus on tracking a single person through the video.

    Args:
        video_path: Path to the input video file.
        output_path: Path to save the output video with 3D pose visualization.
        receptive_field: Number of frames needed for the model's receptive field.
        frame_skip: Process every Nth frame to speed up computation.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps//frame_skip, (width, height))

    # Initialize a buffer to store 2D keypoints and frames
    all_frames = []
    all_keypoints_2d = []
    
    # Variables for tracking the main person
    primary_person_bbox = None
    primary_person_descriptor = None
    max_lost_frames = 30  # Number of frames we'll maintain tracking with no clear detection
    lost_frame_counter = 0
    
    print("Reading video frames...")
    
    # First, read frames and detect keypoints (with skipping)
    frame_count = 0
    processed_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Skip frames to speed up processing
        if frame_count % frame_skip != 0:
            continue
            
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Read {processed_count} frames (from {frame_count} total)")
            
        # Store the frame
        all_frames.append(frame.copy())
        
        # On first frame, initialize tracking
        is_first_frame = (processed_count == 1)
        
        # Detect 2D keypoints with improved tracking
        keypoints_2d, descriptor = detect_2d_keypoints(
            frame, 
            target_bbox=primary_person_bbox,
            target_descriptor=primary_person_descriptor,
            is_first_frame=is_first_frame
        )
        
        # Check if we have valid keypoints
        if np.count_nonzero(keypoints_2d) > 0:
            # We have a good detection
            lost_frame_counter = 0
            primary_person_descriptor = descriptor
            primary_person_bbox = descriptor['bbox']
        else:
            # No good detection, increment lost counter
            lost_frame_counter += 1
            
            # If we've lost tracking for too long, reset
            if lost_frame_counter > max_lost_frames:
                # Try to find any person in the frame
                poses = detect_multiple_poses(frame)
                if poses:
                    # Start tracking the largest person
                    poses.sort(key=lambda p: p['pose_size'], reverse=True)
                    primary_person_descriptor = poses[0]
                    primary_person_bbox = poses[0]['bbox']
                    keypoints_2d = poses[0]['keypoints']
                    lost_frame_counter = 0
                    print(f"Reset tracking at frame {processed_count} after losing target")
        
        # Store keypoints for 3D pose estimation
        all_keypoints_2d.append(keypoints_2d)
    
    total_frames = len(all_frames)
    print(f"Completed reading {total_frames} processed frames out of {frame_count} total frames")
    
    if total_frames == 0:
        print("No frames were read from the video")
        return
        
    # If we don't have enough frames for the receptive field, we need to pad
    if total_frames < receptive_field:
        print(f"Video has {total_frames} frames, padding to {receptive_field} frames")
        pad_frames = receptive_field - total_frames
        pad_start = pad_frames // 2
        pad_end = pad_frames - pad_start
        
        # Pad frames and keypoints
        all_frames = [all_frames[0]] * pad_start + all_frames + [all_frames[-1]] * pad_end
        all_keypoints_2d = [all_keypoints_2d[0]] * pad_start + all_keypoints_2d + [all_keypoints_2d[-1]] * pad_end
    
    print("Processing frames for 3D pose estimation...")
    
    # Now process frames
    for i in range(total_frames):
        if i % 10 == 0:
            print(f"Processing frame {i}/{total_frames}")
        
        # Calculate the window indices
        start_idx = max(0, i - receptive_field // 2)
        end_idx = min(len(all_keypoints_2d), start_idx + receptive_field)
        # If we don't have enough frames at the end, take from the beginning
        if end_idx - start_idx < receptive_field:
            start_idx = end_idx - receptive_field
            
        # Get the keypoints for the current window
        keypoints_window = all_keypoints_2d[start_idx:end_idx]
        
        # Prepare 2D keypoints for input
        input_keypoints = np.array(keypoints_window)
        input_keypoints = normalize_screen_coordinates(input_keypoints, w=width, h=height)
        input_keypoints = input_keypoints[np.newaxis, :, :, :]  # Add batch dimension
        keypoints_2d_tensor = torch.from_numpy(input_keypoints).float()

        # Perform 3D pose estimation
        with torch.no_grad():
            keypoints_3d = model(keypoints_2d_tensor)  # Shape: (1, frames, num_joints, 3)

        # Get the 3D keypoints for the center frame (corresponding to the current frame)
        center_offset = i - start_idx
        if center_offset >= keypoints_3d.shape[1]:
            center_offset = keypoints_3d.shape[1] - 1
        current_frame_3d = keypoints_3d[0, center_offset].cpu().numpy()

        # Visualize 3D keypoints on the frame
        frame = all_frames[i].copy()
        visualize_3d_keypoints_on_frame(frame, current_frame_3d)
        
        # Draw the tracked bounding box if available
        if primary_person_bbox is not None:
            x_min, y_min, x_max, y_max = [int(coord) for coord in primary_person_bbox]
            
            # Draw a highlight box around the tracked person
            # Use a thicker box with double lines for clarity
            cv2.rectangle(frame, (x_min-1, y_min-1), (x_max+1, y_max+1), (0, 255, 255), 2)
            cv2.rectangle(frame, (x_min-4, y_min-4), (x_max+4, y_max+4), (255, 255, 0), 1)
            
            # Add tracking status text
            tracking_status = "Tracked Person"
            if lost_frame_counter > 0:
                tracking_status = f"Tracking (lost for {lost_frame_counter} frames)"
                
            cv2.putText(frame, tracking_status, (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Write the processed frame to the output video
        out.write(frame)

    print(f"Completed processing {total_frames} frames")
    cap.release()
    out.release()

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Each box is [x_min, y_min, x_max, y_max]
    """
    # Determine intersection coordinates
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Compute intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Compute area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# Calculate the receptive field size needed for the model
# For filter_widths=[3, 3, 3, 3, 3], the receptive field is 243 frames
# This is because each layer has a dilation of 3^(i-1), so the total receptive field is:
# 3 + 2*(3*3) + 2*(3*3*3) + 2*(3*3*3*3) + 2*(3*3*3*3*3) = 3 + 18 + 54 + 162 + 486 = 723/3 = 241 + padding = 243

# Example usage
video_path = "./input/3c.mp4"  # Update with your input video path
output_path = "./output/processed_video.mp4"  # Update with your desired output path
# Use a receptive field that matches the model's architecture
# Skip frames to make processing faster (every 5th frame)
process_video(video_path, output_path, receptive_field=243, frame_skip=5)



