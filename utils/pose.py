import logging
from functools import lru_cache
import numpy as np
import cv2
try:
    from mmpose.apis import inference_topdown
    from mmpose.structures import PoseDataSample, merge_data_samples
except ImportError as e:
    logging.getLogger(__name__).warning("MMPose or its dependencies not found. Pose estimation disabled. Error: %s", e)

@lru_cache(maxsize=None)
def get_pose_model(config_path: str, checkpoint_path: str, device: str):
    """
    Load and cache the MMPose model for a given config, checkpoint, and device.
    """
    if device is None:
        from models.device import get_best_device
        device = get_best_device()
    try:
        from mmpose.apis import init_model
        pose_model = init_model(config_path, checkpoint_path, device=device)
        return pose_model
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to load MMPose model: %s", e)
        return None

def extract_skeleton_keypoints(image, bbox=None, conf=1.0, weights=None, device=None):
    """
    Extract skeleton keypoints using MMPose RTMPose.
    Now takes weights and device as arguments.
    """
    MMPOSE_CONFIG = 'configs/rtmpose-l_8xb256-420e_humanart-256x192.py'
    checkpoint = weights or 'weights/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth'
    pose_model = get_pose_model(MMPOSE_CONFIG, checkpoint, device)
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
        logging.getLogger(__name__).debug("Running inference on bbox: %s within image (w=%d, h=%d)", bbox_xyxy, w, h)
    else:
        person_bboxes = np.array([[0., 0., float(w), float(h)]], dtype=np.float32)
        logging.getLogger(__name__).debug("Running inference on full image (w=%d, h=%d)", w, h)
    try:
        results = inference_topdown(pose_model, image, bboxes=person_bboxes, bbox_format='xyxy')
        logging.getLogger(__name__).debug("Inference completed. Results obtained: %s", bool(results))
        if not results:
            return None
        instance = results[0]
        if not hasattr(instance, 'pred_instances'):
            logging.getLogger(__name__).debug("No 'pred_instances' found in result.")
            return None
        pred_instances = instance.pred_instances
        if hasattr(pred_instances, 'keypoints') and len(pred_instances.keypoints) > 0 and \
           hasattr(pred_instances, 'keypoint_scores') and len(pred_instances.keypoint_scores) > 0:
            keypoints = pred_instances.keypoints[0]
            scores = pred_instances.keypoint_scores[0]
            if keypoints.shape[0] != scores.shape[0] or keypoints.ndim != 2 or scores.ndim != 1:
                logging.getLogger(__name__).debug("Mismatched keypoint/score shapes: KP=%s, Scores=%s", keypoints.shape, scores.shape)
                return None
            output_keypoints = [(float(kp[0]), float(kp[1]), float(s)) for kp, s in zip(keypoints, scores)]
            logging.getLogger(__name__).debug("Successfully extracted %d keypoints.", len(output_keypoints))
            return output_keypoints
        else:
            logging.getLogger(__name__).debug("Instance found, but no keypoints/scores data.")
            return None
    except Exception as e:
        logging.getLogger(__name__).error("Error during MMPose inference or processing: %s", e, exc_info=True)
        return None

def extract_skeleton_batch(image, bboxes, weights=None, device=None):
    """
    Batch-process multiple bboxes in a single MMPose call.
    Returns list of keypoints (or None) matching input order.
    """
    if not bboxes:
        return []
    MMPOSE_CONFIG = 'configs/rtmpose-l_8xb256-420e_humanart-256x192.py'
    checkpoint = weights or 'weights/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth'
    pose_model = get_pose_model(MMPOSE_CONFIG, checkpoint, device)
    if pose_model is None or image is None or image.size == 0:
        return [None] * len(bboxes)
    h, w = image.shape[:2]
    arr = []
    for x1,y1,x2,y2 in bboxes:
        arr.append([
            max(0.0, float(x1)), max(0.0, float(y1)),
            min(float(w), float(x2)), min(float(h), float(y2))
        ])
    person_bboxes = np.array(arr, dtype=np.float32)
    try:
        results = inference_topdown(pose_model, image, bboxes=person_bboxes, bbox_format='xyxy')
    except Exception as e:
        logging.getLogger(__name__).error("Batch MMPose inference failed: %s", e, exc_info=True)
        return [None] * len(bboxes)
    keypoints_list = []
    for inst in results:
        if not hasattr(inst, 'pred_instances'):
            keypoints_list.append(None)
            continue
        pi = inst.pred_instances
        if not hasattr(pi, 'keypoints') or len(pi.keypoints)==0:
            keypoints_list.append(None)
            continue
        kp = pi.keypoints[0]; sc = pi.keypoint_scores[0]
        pts = []
        for (x,y),s in zip(kp, sc):
            pts.append((float(x), float(y), float(s)))
        keypoints_list.append(pts if pts else None)
    return keypoints_list