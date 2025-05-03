import numpy as np

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