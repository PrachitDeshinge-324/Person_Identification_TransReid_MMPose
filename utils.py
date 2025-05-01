import cv2
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from functools import lru_cache

# Try loading MMPose, handle potential errors gracefully
try:
    from mmpose.apis import inference_topdown
    from mmpose.structures import PoseDataSample, merge_data_samples
    from models import get_best_device
except ImportError as e:
    print(f"[WARN] MMPose or its dependencies not found. Pose estimation disabled. Error: {e}")


def get_pose_model(config_path, checkpoint_path, device):
    """
    Load and cache the MMPose model for a given config, checkpoint, and device.
    """
    if device is None:
        from models import get_best_device
        device = get_best_device()
    try:
        from mmpose.apis import init_model
        pose_model = init_model(config_path, checkpoint_path, device=device)
        return pose_model
    except Exception as e:
        print(f"[WARN] Failed to load MMPose model: {e}")
        return None


_pose_model_cache = {}


def get_unique_color(track_id):
    """
    Generate a unique color for each track ID using HSV color space

    Args:
        track_id: Unique tracking ID

    Returns:
        BGR color tuple
    """
    # Use golden ratio to generate well-distributed colors
    golden_ratio = 0.618033988749895
    h = (track_id * golden_ratio) % 1.0

    # Convert HSV to RGB then to BGR
    r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.95)
    return (int(b*255), int(g*255), int(r*255))


def draw_skeleton(frame, keypoints, color=(0,255,255), thickness=2, confidence_threshold=0.3):
    """
    Draw skeleton keypoints and connections on the frame.
    Supports COCO 17-keypoint format (RTMPose-L output).
    Uses confidence threshold to filter visible points/lines.
    """
    if keypoints is None or len(keypoints) == 0:
        return frame

    coco_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms
        (5, 11), (6, 12), (11, 12), # Torso
        (11, 13), (13, 15), (12, 14), (14, 16) # Legs
    ]

    if len(keypoints) < 17:
        print(f"[WARN] draw_skeleton received only {len(keypoints)} keypoints, expected 17 for COCO.")
        for idx, (x, y, v) in enumerate(keypoints):
           if v > confidence_threshold:
               cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        return frame

    for idx, (x, y, v) in enumerate(keypoints):
        if v > confidence_threshold:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)

    for i, j in coco_connections:
        if i < len(keypoints) and j < len(keypoints) and \
           keypoints[i][2] > confidence_threshold and keypoints[j][2] > confidence_threshold:
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(frame, pt1, pt2, color, thickness)

    return frame


def draw_tracking_results(frame, tracks, inactive_tracks=None, frame_count=None, skeleton_history=None, debug_info=None):
    """
    Draw bounding boxes and IDs on the frame with enhanced visualization

    Args:
        frame: Original video frame
        tracks: Dictionary of track_id -> (bbox, features, [confidence])
        inactive_tracks: Dictionary of inactive tracks (optional)
        frame_count: Current frame count (optional)
        skeleton_history: Dictionary of track_id -> list of keypoints (optional)
        debug_info: Dictionary of track_id -> list of debug strings (optional)

    Returns:
        Annotated frame
    """
    frame_copy = frame.copy()

    track_count = len(tracks)
    inactive_count = 0 if inactive_tracks is None else len(inactive_tracks)

    if frame_count is not None:
        cv2.putText(frame_copy, f"Frame: {frame_count}",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame_copy, f"Frame: {frame_count}",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame_copy, f"Active tracks: {track_count}",
               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame_copy, f"Active tracks: {track_count}",
               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    if inactive_count > 0:
        cv2.putText(frame_copy, f"Inactive tracks: {inactive_count}",
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame_copy, f"Inactive tracks: {inactive_count}",
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

    if inactive_tracks is not None:
        for track_id, track_data in inactive_tracks.items():
            if len(track_data) >= 4:
                 bbox, _, last_seen, conf = track_data[:4]
            elif len(track_data) >= 3:
                 bbox, _, last_seen = track_data[:3]
                 conf = 0.0
            else:
                print(f"[WARN] Unexpected inactive track data format for ID {track_id}: {track_data}")
                continue

            if bbox is None or len(bbox) != 4:
                print(f"[WARN] Invalid bbox for inactive track ID {track_id}: {bbox}")
                continue

            x1, y1, x2, y2 = [int(c) for c in bbox]

            color = get_unique_color(track_id)

            dash_len = 5
            for i in range(x1, x2, dash_len * 2):
                 cv2.line(frame_copy, (i, y1), (min(i + dash_len, x2), y1), color, 1)
            for i in range(x1, x2, dash_len * 2):
                 cv2.line(frame_copy, (i, y2), (min(i + dash_len, x2), y2), color, 1)
            for i in range(y1, y2, dash_len * 2):
                 cv2.line(frame_copy, (x1, i), (x1, min(i + dash_len, y2)), color, 1)
            for i in range(y1, y2, dash_len * 2):
                 cv2.line(frame_copy, (x2, i), (x2, min(i + dash_len, y2)), color, 1)

            frames_missing = frame_count - last_seen if frame_count is not None and last_seen is not None else "?"
            text = f"ID: {track_id} (lost: {frames_missing})"
            font_size = 0.4
            font_thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)

            cv2.rectangle(frame_copy,
                         (x1, y1 - text_h - 5),
                         (x1 + text_w + 5, y1),
                         color, -1)

            cv2.putText(frame_copy, text,
                       (x1 + 2, y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)

    for track_id, data in tracks.items():
        if len(data) >= 3:
            bbox, _, conf = data
        else:
            bbox, _ = data
            conf = 1.0

        if bbox is None or len(bbox) != 4:
             print(f"[WARN] Invalid bbox for active track ID {track_id}: {bbox}")
             continue

        x1, y1, x2, y2 = [int(c) for c in bbox]

        color = get_unique_color(track_id)

        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

        text = f"ID: {track_id} ({conf:.2f})"
        font_size = 0.6
        font_thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)

        cv2.rectangle(frame_copy,
                     (x1, y1 - text_h - 10),
                     (x1 + text_w + 10, y1),
                     color, -1)

        cv2.putText(frame_copy, text,
                   (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)

        skeleton_missing = True
        if skeleton_history is not None and track_id in skeleton_history and skeleton_history[track_id]:
            raw_keypoints = skeleton_history[track_id][-1]
            if raw_keypoints:
                draw_skeleton(frame_copy, raw_keypoints, color=color)
                skeleton_missing = False

        if skeleton_missing and debug_info and track_id in debug_info:
             pass

        if debug_info and track_id in debug_info:
            lines = debug_info[track_id]
            for i, line in enumerate(lines):
                (line_w, line_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_y = y1 + 20 + i*18
                cv2.rectangle(frame_copy, (x1, text_y - line_h - 1), (x1 + line_w + 2, text_y + 2), (0,0,0), -1)
                cv2.putText(frame_copy, line, (x1 + 1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)

    return frame_copy


def visualize_reid_matches(query_image, gallery_images, similarities, top_k=5):
    """
    Visualize top-k matches from re-identification

    Args:
        query_image: Query person image
        gallery_images: List of gallery images
        similarities: List of similarity scores
        top_k: Number of top matches to show

    Returns:
        Visualization image
    """
    if not gallery_images or not similarities:
         print("[WARN] No gallery images or similarities to visualize.")
         h, w = query_image.shape[:2] if query_image is not None else (100,100)
         return np.zeros((h, w * 2, 3), dtype=np.uint8)

    sim_array = np.array(similarities)
    sorted_indices = np.argsort(-sim_array)

    valid_top_k = min(top_k, len(gallery_images))
    top_indices = sorted_indices[:valid_top_k]

    top_matches = [gallery_images[i] for i in top_indices]
    top_scores = [sim_array[i] for i in top_indices]

    fig, axes = plt.subplots(1, valid_top_k + 1, figsize=(max(6, 2 * (valid_top_k + 1)), 3))

    if valid_top_k == 0:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    if query_image is not None:
        axes[0].imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Query")
    else:
        axes[0].set_title("No Query")
    axes[0].axis('off')

    for i, (img, score) in enumerate(zip(top_matches, top_scores)):
        ax_idx = i + 1
        if ax_idx < len(axes):
            if img is not None:
                axes[ax_idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axes[ax_idx].set_title(f"Match {i+1}\nScore: {score:.2f}")
            else:
                axes[ax_idx].set_title(f"Match {i+1}\n(No Img)")
            axes[ax_idx].axis('off')

    for j in range(valid_top_k + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    fig.canvas.draw()

    vis_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_img = vis_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return vis_img


def extract_skeleton_keypoints(image, bbox=None, conf=1.0, weights=None, device=None):
    """
    Extract skeleton keypoints using MMPose RTMPose.
    Now takes weights and device as arguments.
    """
    MMPOSE_CONFIG = 'rtmpose-l_8xb256-420e_humanart-256x192.py'
    checkpoint = weights or 'weights/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth'
    cache_key = (checkpoint, device)
    if cache_key not in _pose_model_cache:
        _pose_model_cache[cache_key] = get_pose_model(MMPOSE_CONFIG, checkpoint, device)
    pose_model = _pose_model_cache[cache_key]
    if pose_model is None:
        return None
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return None

    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        bbox_xyxy = [
            max(0.0, float(x1)),
            max(0.0, float(y1)),
            min(float(w), float(x2)),
            min(float(h), float(y2))
        ]
        person_bboxes = np.array([bbox_xyxy], dtype=np.float32)
        print(f"[POSE_LOG] Running inference on bbox: {bbox_xyxy} within image (w={w}, h={h})")
    else:
        person_bboxes = np.array([[0., 0., float(w), float(h)]], dtype=np.float32)
        print(f"[POSE_LOG] Running inference on full image (w={w}, h={h})")

    try:
        results = inference_topdown(pose_model, image, bboxes=person_bboxes, bbox_format='xyxy')
        print(f"[POSE_LOG] Inference completed. Results obtained: {'Yes' if results else 'No'}")

        if not results:
            return None

        instance = results[0]

        if not hasattr(instance, 'pred_instances'):
            print("[POSE_LOG] No 'pred_instances' found in result.")
            return None

        pred_instances = instance.pred_instances

        if hasattr(pred_instances, 'keypoints') and len(pred_instances.keypoints) > 0 and \
           hasattr(pred_instances, 'keypoint_scores') and len(pred_instances.keypoint_scores) > 0:

            keypoints = pred_instances.keypoints[0]
            scores = pred_instances.keypoint_scores[0]

            if keypoints.shape[0] != scores.shape[0] or keypoints.ndim != 2 or scores.ndim != 1:
                print(f"[POSE_LOG] Mismatched keypoint/score shapes: KP={keypoints.shape}, Scores={scores.shape}")
                return None

            output_keypoints = [(float(kp[0]), float(kp[1]), float(s)) for kp, s in zip(keypoints, scores)]
            print(f"[POSE_LOG] Successfully extracted {len(output_keypoints)} keypoints (relative to full frame).")
            return output_keypoints
        else:
            print("[POSE_LOG] Instance found, but no keypoints/scores data.")
            return None

    except Exception as e:
        print(f"[ERROR] Error during MMPose inference or processing: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_body_ratios(keypoints):
    """
    Compute body ratios from skeleton keypoints (COCO 17 format).
    Returns a dict of ratios.
    """
    if keypoints is None or len(keypoints) < 17:
        return {}

    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    mid_shoulder = ((keypoints[LEFT_SHOULDER][0] + keypoints[RIGHT_SHOULDER][0]) / 2,
                    (keypoints[LEFT_SHOULDER][1] + keypoints[RIGHT_SHOULDER][1]) / 2)
    mid_hip = ((keypoints[LEFT_HIP][0] + keypoints[RIGHT_HIP][0]) / 2,
               (keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1]) / 2)

    def dist(kp_idx_a, kp_idx_b):
        if keypoints[kp_idx_a][2] > 0.1 and keypoints[kp_idx_b][2] > 0.1:
            return np.linalg.norm(np.array(keypoints[kp_idx_a][:2]) - np.array(keypoints[kp_idx_b][:2]))
        return 0.0

    def dist_point_idx(point_a, kp_idx_b):
         if keypoints[kp_idx_b][2] > 0.1:
             return np.linalg.norm(np.array(point_a) - np.array(keypoints[kp_idx_b][:2]))
         return 0.0

    ratios = {}
    ratios['shoulder_width'] = dist(LEFT_SHOULDER, RIGHT_SHOULDER)
    ratios['hip_width'] = dist(LEFT_HIP, RIGHT_HIP)
    ratios['torso_height'] = dist_point_idx(mid_shoulder, LEFT_HIP) + dist_point_idx(mid_shoulder, RIGHT_HIP) / 2

    ratios['left_arm_len'] = dist(LEFT_SHOULDER, LEFT_WRIST)
    ratios['right_arm_len'] = dist(RIGHT_SHOULDER, RIGHT_WRIST)
    ratios['left_leg_len'] = dist(LEFT_HIP, LEFT_ANKLE)
    ratios['right_leg_len'] = dist(RIGHT_HIP, RIGHT_ANKLE)

    denominator = ratios['torso_height'] + 1e-6
    ratios['arm_to_torso'] = (ratios['left_arm_len'] + ratios['right_arm_len']) / (2 * denominator)
    ratios['leg_to_torso'] = (ratios['left_leg_len'] + ratios['right_leg_len']) / (2 * denominator)
    ratios['shoulder_to_hip_width'] = ratios['shoulder_width'] / (ratios['hip_width'] + 1e-6)

    valid_ratios = {k: v for k, v in ratios.items() if v > 1e-5}
    return valid_ratios


def compute_gait_features(keypoints_sequence):
    """
    Compute gait features from a sequence of keypoints (COCO 17 format),
    and also return the body ratios of the last frame for robust, scale-invariant analysis.
    """
    if not keypoints_sequence or len(keypoints_sequence) < 5:
         return {}, {}

    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    LEFT_HIP = 11
    RIGHT_HIP = 12

    stride_lengths = []
    step_widths = []
    hip_oscillations_y = []

    valid_sequence = [kp for kp in keypoints_sequence if kp is not None and len(kp) >= 17]

    if len(valid_sequence) < 5:
         return {}, {}

    for i in range(1, len(valid_sequence)):
        prev_kp = valid_sequence[i-1]
        curr_kp = valid_sequence[i]

        if prev_kp[LEFT_ANKLE][2] > 0.2 and curr_kp[LEFT_ANKLE][2] > 0.2 and \
           prev_kp[RIGHT_ANKLE][2] > 0.2 and curr_kp[RIGHT_ANKLE][2] > 0.2:

            left_stride = np.linalg.norm(np.array(curr_kp[LEFT_ANKLE][:2]) - np.array(prev_kp[LEFT_ANKLE][:2]))
            right_stride = np.linalg.norm(np.array(curr_kp[RIGHT_ANKLE][:2]) - np.array(prev_kp[RIGHT_ANKLE][:2]))
            stride_lengths.append((left_stride + right_stride) / 2)

            step_width = np.linalg.norm(np.array(curr_kp[LEFT_ANKLE][:2]) - np.array(curr_kp[RIGHT_ANKLE][:2]))
            step_widths.append(step_width)

        if prev_kp[LEFT_HIP][2] > 0.2 and curr_kp[LEFT_HIP][2] > 0.2 and \
           prev_kp[RIGHT_HIP][2] > 0.2 and curr_kp[RIGHT_HIP][2] > 0.2:
            prev_mid_hip_y = (prev_kp[LEFT_HIP][1] + prev_kp[RIGHT_HIP][1]) / 2
            curr_mid_hip_y = (curr_kp[LEFT_HIP][1] + curr_kp[RIGHT_HIP][1]) / 2
            hip_oscillations_y.append(abs(curr_mid_hip_y - prev_mid_hip_y))

    gait = {
        'mean_stride_len': float(np.mean(stride_lengths)) if stride_lengths else 0,
        'std_stride_len': float(np.std(stride_lengths)) if len(stride_lengths) > 1 else 0,
        'mean_step_width': float(np.mean(step_widths)) if step_widths else 0,
        'std_step_width': float(np.std(step_widths)) if len(step_widths) > 1 else 0,
        'mean_hip_osc_y': float(np.mean(hip_oscillations_y)) if hip_oscillations_y else 0,
    }
    # Add body ratios from the last valid frame
    last_kp = valid_sequence[-1]
    body_ratios = compute_body_ratios(last_kp)
    return gait, body_ratios