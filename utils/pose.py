import logging
from functools import lru_cache
import numpy as np
import cv2
from ultralytics import YOLO

# Cache YOLOv8-pose model
@lru_cache(maxsize=None)
def get_yolopose_model(weights_path: str = 'weights/yolo11x-pose.pt', device: str = None):
    try:
        model = YOLO(weights_path)
        if device:
            model.to(device)
        return model
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to load YOLOv8-pose model: %s", e)
        return None

def extract_skeleton_keypoints_yolopose(image, bbox=None, conf=1.0, weights=None, device=None):
    """
    Extract skeleton keypoints using YOLOv8-pose model.
    """
    weights_path = weights or 'weights/yolo11x-pose.pt'
    model = get_yolopose_model(weights_path, device)
    if model is None or image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
        crop = image[y1:y2, x1:x2]
        results = model(crop)
        if not results or not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
            return None
        kps = results[0].keypoints.xy.cpu().numpy()[0]  # shape (17,2)
        scores = results[0].keypoints.conf.cpu().numpy()[0]  # shape (17,)
        # Map keypoints back to original image coordinates
        kps[:, 0] += x1
        kps[:, 1] += y1
        return [(float(x), float(y), float(s)) for (x, y), s in zip(kps, scores)]
    else:
        results = model(image)
        if not results or not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
            return None
        kps = results[0].keypoints.xy.cpu().numpy()[0]
        scores = results[0].keypoints.conf.cpu().numpy()[0]
        return [(float(x), float(y), float(s)) for (x, y), s in zip(kps, scores)]

def extract_skeleton_batch_yolopose(image, bboxes, weights=None, device=None):
    """
    Batch-process multiple bboxes using YOLOv8-pose model.
    Returns list of keypoints (or None) matching input order.
    """
    if not bboxes:
        return []
    weights_path = weights or 'weights/yolo11x-pose.pt'
    model = get_yolopose_model(weights_path, device)
    if model is None or image is None or image.size == 0:
        return [None] * len(bboxes)
    h, w = image.shape[:2]
    keypoints_list = []
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
        crop = image[y1:y2, x1:x2]
        results = model(crop)
        if not results or not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
            keypoints_list.append(None)
            continue
        kps_data = results[0].keypoints.xy
        scores_data = results[0].keypoints.conf
        if kps_data is None or scores_data is None:
            keypoints_list.append(None)
            continue
        kps = kps_data.cpu().numpy()[0]
        scores = scores_data.cpu().numpy()[0]
        # Map keypoints back to original image coordinates
        kps[:, 0] += x1
        kps[:, 1] += y1
        keypoints_list.append([(float(x), float(y), float(s)) for (x, y), s in zip(kps, scores)])
    return keypoints_list