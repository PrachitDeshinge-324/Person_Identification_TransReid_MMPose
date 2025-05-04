import numpy as np

def smooth_keypoints_sequence(keypoints_seq, window=5):
    """
    Apply moving average smoothing to a sequence of keypoints.
    Args:
        keypoints_seq: list of [N x 3] keypoints (x, y, conf)
        window: window size for smoothing
    Returns:
        Smoothed sequence (same shape)
    """
    if not keypoints_seq or len(keypoints_seq) < 2:
        return keypoints_seq
    arr = np.array([
        kp if kp is not None else np.zeros((17, 3))
        for kp in keypoints_seq
    ])
    smoothed = []
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        window_arr = arr[start:i+1]
        mean_xy = np.mean(window_arr[..., :2], axis=0)
        mean_conf = np.mean(window_arr[..., 2], axis=0)
        smoothed.append([(float(x), float(y), float(c)) for (x, y), c in zip(mean_xy, mean_conf)])
    return smoothed

def normalize_keypoints_camera_view(keypoints):
    """
    Normalize keypoints for camera view by torso orientation and scale.
    - Translate so mid-hip is at (0,0)
    - Rotate so torso (mid-hip to mid-shoulder) is vertical
    - Scale so torso length is 1
    Args:
        keypoints: list of (x, y, conf) for 17 COCO keypoints
    Returns:
        Normalized keypoints (same shape)
    """
    if keypoints is None or len(keypoints) < 17:
        return keypoints
    # Mid-hip and mid-shoulder
    lhip, rhip = keypoints[11], keypoints[12]
    lsho, rsho = keypoints[5], keypoints[6]
    if lhip[2] < 0.1 or rhip[2] < 0.1 or lsho[2] < 0.1 or rsho[2] < 0.1:
        return keypoints
    mid_hip = np.array([(lhip[0] + rhip[0]) / 2, (lhip[1] + rhip[1]) / 2])
    mid_sho = np.array([(lsho[0] + rsho[0]) / 2, (lsho[1] + rsho[1]) / 2])
    # Translate
    pts = np.array([[x, y] for x, y, c in keypoints])
    pts -= mid_hip
    # Rotate
    torso_vec = mid_sho - mid_hip
    angle = np.arctan2(torso_vec[0], torso_vec[1])  # angle to vertical
    rot = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
    pts = pts @ rot.T
    # Scale
    torso_len = np.linalg.norm(mid_sho - mid_hip)
    if torso_len > 1e-3:
        pts /= torso_len
    # Rebuild keypoints with confidence
    normed = [(float(x), float(y), float(c)) for (x, y), (_, _, c) in zip(pts, keypoints)]
    return normed

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

def compute_joint_angles(keypoints):
    """
    Compute main lower-body joint angles (hip, knee, ankle) for both sides.
    Returns dict of angles in degrees.
    """
    if keypoints is None or len(keypoints) < 17:
        return {}
    def angle(a, b, c):
        # angle at b between points a-b-c
        a = np.array(a[:2]); b = np.array(b[:2]); c = np.array(c[:2])
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    # COCO: 11-LHip, 12-RHip, 13-LKnee, 14-RKnee, 15-LAnkle, 16-RAnkle
    angles = {}
    # Left leg
    if keypoints[11][2] > 0.1 and keypoints[13][2] > 0.1 and keypoints[15][2] > 0.1:
        angles['left_knee'] = angle(keypoints[11], keypoints[13], keypoints[15])
    if keypoints[5][2] > 0.1 and keypoints[11][2] > 0.1 and keypoints[13][2] > 0.1:
        angles['left_hip'] = angle(keypoints[5], keypoints[11], keypoints[13])
    # Right leg
    if keypoints[12][2] > 0.1 and keypoints[14][2] > 0.1 and keypoints[16][2] > 0.1:
        angles['right_knee'] = angle(keypoints[12], keypoints[14], keypoints[16])
    if keypoints[6][2] > 0.1 and keypoints[12][2] > 0.1 and keypoints[14][2] > 0.1:
        angles['right_hip'] = angle(keypoints[6], keypoints[12], keypoints[14])
    return angles

def compute_gait_features(keypoints_sequence):
    """
    Compute gait features from a sequence of keypoints (COCO 17 format),
    and also return the body ratios of the last frame for robust, scale-invariant analysis.
    Now includes joint angle time-series and smoothing.
    """
    if not keypoints_sequence or len(keypoints_sequence) < 5:
         return {}, {}
    # Smoothing
    smoothed_seq = smooth_keypoints_sequence(keypoints_sequence, window=5)
    # Normalize for camera view
    normed_seq = [normalize_keypoints_camera_view(kp) for kp in smoothed_seq]
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    LEFT_HIP = 11
    RIGHT_HIP = 12
    stride_lengths = []
    step_widths = []
    hip_oscillations_y = []
    left_knee_angles = []
    right_knee_angles = []
    left_hip_angles = []
    right_hip_angles = []
    valid_sequence = [kp for kp in normed_seq if kp is not None and len(kp) >= 17]
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
        # Joint angles
        angles = compute_joint_angles(curr_kp)
        if 'left_knee' in angles:
            left_knee_angles.append(angles['left_knee'])
        if 'right_knee' in angles:
            right_knee_angles.append(angles['right_knee'])
        if 'left_hip' in angles:
            left_hip_angles.append(angles['left_hip'])
        if 'right_hip' in angles:
            right_hip_angles.append(angles['right_hip'])
    gait = {
        'mean_stride_len': float(np.mean(stride_lengths)) if stride_lengths else 0,
        'std_stride_len': float(np.std(stride_lengths)) if len(stride_lengths) > 1 else 0,
        'mean_step_width': float(np.mean(step_widths)) if step_widths else 0,
        'std_step_width': float(np.std(step_widths)) if len(step_widths) > 1 else 0,
        'mean_hip_osc_y': float(np.mean(hip_oscillations_y)) if hip_oscillations_y else 0,
        'mean_left_knee_angle': float(np.mean(left_knee_angles)) if left_knee_angles else 0,
        'mean_right_knee_angle': float(np.mean(right_knee_angles)) if right_knee_angles else 0,
        'mean_left_hip_angle': float(np.mean(left_hip_angles)) if left_hip_angles else 0,
        'mean_right_hip_angle': float(np.mean(right_hip_angles)) if right_hip_angles else 0,
    }
    # Add body ratios from the last valid frame
    last_kp = valid_sequence[-1]
    body_ratios = compute_body_ratios(last_kp)
    return gait, body_ratios