import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow/Mediapipe logs

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import torch
import numpy as np
import logging
from models.yolo import YOLOv8Tracker
from models.device import get_best_device
from models.kalman import KalmanBoxTracker
from models.transreid import load_transreid_model
from models.bytetrack import ByteTrack
from utils.visualization import draw_tracking_results, get_unique_color
from utils.pose import extract_skeleton_keypoints_yolopose, extract_skeleton_batch_yolopose
from utils.gaits import compute_body_ratios, compute_gait_features
from scipy.spatial.distance import cdist
from ultralytics import YOLO
from utils.gait.gait_infer import OpenGaitEmbedder
from ultralytics.utils import LOGGER as yolo_logger

# Set Ultralytics logger level to WARNING
yolo_logger.setLevel(logging.WARNING)

# initialize root logger: writes to logs/person_tracker.log in logs folder
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join('logs', 'person_tracker.log'),
    filemode='a',
    format='%(asctime)s %(name)s [%(levelname)s] %(message)s'
)

logger = logging.getLogger(__name__)

def load_gait_embeddings(gait_feature_dir):
    """
    Load gait embeddings for all tracks from OpenGait output directory.
    Returns: dict {track_id: embedding (np.ndarray)}
    """
    embeddings = {}
    for track_folder in os.listdir(gait_feature_dir):
        track_path = os.path.join(gait_feature_dir, track_folder)
        if not os.path.isdir(track_path):
            continue
        for root, dirs, files in os.walk(track_path):
            for file in files:
                if file.endswith('.npy'):
                    emb = np.load(os.path.join(root, file))
                    embeddings[track_folder] = emb
    return embeddings


def track_to_track_gait_matching(gait_feature_dir, threshold=0.7):
    """
    Compare gait embeddings for all tracks and print similarity scores.
    Optionally, flag pairs above a similarity threshold.
    """
    embeddings = load_gait_embeddings(gait_feature_dir)
    track_ids = list(embeddings.keys())
    for i, tid1 in enumerate(track_ids):
        for j, tid2 in enumerate(track_ids):
            if i >= j:
                continue  # avoid duplicate/self-comparison
            emb1 = embeddings[tid1]
            emb2 = embeddings[tid2]
            sim = 1 - cdist([emb1], [emb2], metric='cosine')[0, 0]
            print(f"Similarity between {tid1} and {tid2}: {sim:.3f}")
            if sim > threshold:
                print(f"  -> Likely same person (sim > {threshold})")

class PersonTracker:
    def __init__(self, yolo_weights_path, transreid_weights_path, 
                 device=None, conf_threshold=0.3, reid_threshold=0.7,
                 appearance_weight=0.4, gait_weight=0.15, body_weight=0.15, color_weight=0.2, height_weight=0.1, context_weight=0.05, debug_visualize=False,
                 tracker_type='kalman', realtime=False):
        """Initialize the person tracker with robust tracking capabilities"""
        # Use the best available device if none specified
        self.device = device if device is not None else get_best_device()
        logger.info("Using device: %s", self.device)
        
        # Initialize YOLOv8 for detection and tracking
        self.yolo_tracker = YOLOv8Tracker(yolo_weights_path, device=self.device, 
                                          conf_threshold=conf_threshold)
        
        # Initialize TransReID for re-identification (cached)
        self.transreid = load_transreid_model(transreid_weights_path, self.device)
        
        # Tracking parameters
        self.next_id = 1
        self.tracks = {}  # Active tracks: id -> track info
        self.feature_database = {}  # Feature history: id -> features
        self.reid_threshold = reid_threshold
        self.frame_count = 0
        
        # Enhanced tracking parameters
        self.kalman_trackers = {}  # id -> KalmanBoxTracker
        self.inactive_tracks = {}  # id -> (last_bbox, feature, last_seen_frame, confidence)
        self.max_age = 90  # Maximum frames to keep inactive tracks
        self.min_hits = 5  # Minimum hits before track is confirmed
        self.motion_weight = 1.0  # Weight for motion-based matching
        self.appearance_weight = appearance_weight
        self.gait_weight = gait_weight
        self.body_weight = body_weight
        self.color_weight = color_weight
        self.height_weight = height_weight
        self.context_weight = context_weight
        self.debug_visualize = debug_visualize
        
        # Feature history
        self.feature_history = {}  # id -> list of historical features
        self.max_history = 10  # Maximum features to store per track
        
        # Add storage for skeleton and gait features
        self.skeleton_history = {}  # id -> list of keypoints
        self.body_ratio_history = {}  # id -> list of body ratios
        self.gait_history = {}  # id -> list of gait features
        
        # Add identity gallery
        self.identity_gallery = {}  # id -> dict with appearance, gait, body, context
        
        # Skeleton detection stats
        self.skeleton_missing_count = 0
        self.skeleton_total_count = 0
        self.last_debug_info = {}
        
        # Tracker type and real-time mode
        self.tracker_type = tracker_type
        self.realtime = realtime
        if self.tracker_type == 'bytetrack':
            self.bytetrack = ByteTrack(track_thresh=conf_threshold)
        else:
            self.bytetrack = None

        # YOLOv8-seg and OpenGait integration
        self.seg_model = YOLO('weights/yolo11x-seg.pt')
        self.gait_embedder = OpenGaitEmbedder(device=self.device)
        self.silhouette_history = {}  # track_id -> list of silhouettes
    
    def _extract_color_histogram(self, image):
        """Extract a normalized color histogram from the person crop (HSV, 16 bins per channel)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 16], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def _compare_histograms(self, hist1, hist2):
        if hist1 is None or hist2 is None:
            return 0.0
        # Use correlation as similarity (1.0 = identical)
        return float(cv2.compareHist(hist1.astype('float32'), hist2.astype('float32'), cv2.HISTCMP_CORREL))

    def _calculate_motion_distance(self, bbox1, bbox2):
        """Calculate motion/position distance between two bounding boxes"""
        # Add validation to prevent errors with invalid bounding boxes
        if not isinstance(bbox1, (list, tuple, np.ndarray)) or not isinstance(bbox2, (list, tuple, np.ndarray)):
            return 1.0  # Return maximum distance if invalid format
        
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 1.0  # Return maximum distance if invalid format
        
        # Extract centers
        try:
            c1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
            c2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
            
            # Calculate Euclidean distance between centers
            dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            
            # Calculate size difference
            w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
            w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
            size1 = w1 * h1
            size2 = w2 * h2
            size_diff = abs(size1 - size2) / max(size1, size2)
            
            # Normalize by diagonal size of the image (assuming 1920x1080 if not provided)
            norm_factor = np.sqrt(1920**2 + 1080**2)
            
            # Combine position and size difference
            combined_dist = (dist / norm_factor) * 0.8 + size_diff * 0.2
            
            # Return as a score between 0 and 1 (0 is perfect match)
            return min(combined_dist, 1.0)
        except (IndexError, TypeError, ValueError) as e:
            logger.error("Error in motion distance calculation: %s", e)
            logger.error("bbox1: %s, bbox2: %s", bbox1, bbox2)
            return 1.0  # Return maximum distance on error
    
    def _update_feature_history(self, track_id, feature):
        """Update feature history for a track"""
        if track_id not in self.feature_history:
            self.feature_history[track_id] = []
            
        # Add new feature
        self.feature_history[track_id].append(feature)
        
        # Keep only the most recent features
        if len(self.feature_history[track_id]) > self.max_history:
            self.feature_history[track_id].pop(0)
    
    def _get_averaged_feature(self, track_id):
        """Get time-weighted averaged feature for a track"""
        if track_id not in self.feature_history or not self.feature_history[track_id]:
            return None
            
        features = self.feature_history[track_id]
        if len(features) == 1:
            return features[0]
            
        # Apply time-based weighting (newer features have higher weight)
        weights = np.linspace(0.5, 1.0, len(features))
        weights = weights / np.sum(weights)  # Normalize to sum to 1
        
        # Weighted average
        avg_feature = torch.zeros_like(features[0])
        for i, feat in enumerate(features):
            avg_feature += feat * weights[i]
            
        # Normalize
        avg_feature = torch.nn.functional.normalize(avg_feature, p=2, dim=0)
        
        return avg_feature
    
    def _calculate_appearance_similarity(self, feature1, feature2):
        """Calculate appearance similarity score between features"""
        similarity = torch.nn.functional.cosine_similarity(
            feature1.unsqueeze(0), feature2.unsqueeze(0)
        ).item()
        return similarity
    
    def _calculate_body_similarity(self, ratios1, ratios2):
        """
        Calculate similarity between two sets of body ratios.
        Returns a value between 0 and 1 (1 = identical).
        """
        if not ratios1 or not ratios2:
            return 0.0
        keys = set(ratios1.keys()) & set(ratios2.keys())
        if not keys:
            return 0.0
        v1 = np.array([ratios1[k] for k in keys])
        v2 = np.array([ratios2[k] for k in keys])
        # Use negative normalized L2 distance as similarity
        dist = np.linalg.norm(v1 - v2) / (np.linalg.norm(v1) + np.linalg.norm(v2) + 1e-6)
        return 1.0 - dist

    def _calculate_gait_similarity(self, gait1, gait2, ratios1=None, ratios2=None):
        """
        Calculate similarity between two gait features, combining OpenGait embedding and skeleton gait dict.
        Always combine both if available.
        """
        # OpenGait cosine similarity (if both are numpy arrays)
        cosine_sim = 0.0
        if isinstance(gait1, np.ndarray) and isinstance(gait2, np.ndarray) and gait1.size > 0 and gait2.size > 0:
            # Cosine similarity can be in [-1, 1], but for embeddings it should be [0, 1]
            cosine_sim = 1 - cdist([gait1], [gait2], metric='cosine')[0, 0]
            # Clamp to [0, 1] for safety
            cosine_sim = max(0.0, min(1.0, cosine_sim))

        # Skeleton gait similarity (if both are dicts)
        skel_score = 0.0
        if isinstance(gait1, dict) and isinstance(gait2, dict) and gait1 and gait2:
            gait_keys = set(gait1.keys()) & set(gait2.keys())
            if gait_keys:
                gait_v1 = np.array([gait1[k] for k in gait_keys])
                gait_v2 = np.array([gait2[k] for k in gait_keys])
                gait_dist = np.linalg.norm(gait_v1 - gait_v2) / (np.linalg.norm(gait_v1) + np.linalg.norm(gait_v2) + 1e-6)
                skel_score = 1.0 - gait_dist
                skel_score = max(0.0, min(1.0, skel_score))

        # Optionally, combine with body ratios if available
        ratio_score = 0.0
        if ratios1 and ratios2 and isinstance(ratios1, dict) and isinstance(ratios2, dict):
            ratio_keys = set(ratios1.keys()) & set(ratios2.keys())
            if ratio_keys:
                r1 = np.array([ratios1[k] for k in ratio_keys])
                r2 = np.array([ratios2[k] for k in ratio_keys])
                ratio_dist = np.linalg.norm(r1 - r2) / (np.linalg.norm(r1) + np.linalg.norm(r2) + 1e-6)
                ratio_score = 1.0 - ratio_dist
                ratio_score = max(0.0, min(1.0, ratio_score))

        # Print cosine_sim for debug, include track info if available
        track_info = ""
        if hasattr(self, "current_track_id_for_gait_sim") and hasattr(self, "current_gallery_track_id_for_gait_sim"):
            track_info = f" (detection track {self.current_track_id_for_gait_sim} vs gallery track {self.current_gallery_track_id_for_gait_sim})"
        elif hasattr(self, "current_track_id_for_gait_sim"):
            track_info = f" (track {self.current_track_id_for_gait_sim})"
        elif hasattr(self, "current_gallery_track_id_for_gait_sim"):
            track_info = f" (gallery track {self.current_gallery_track_id_for_gait_sim})"
        # print(f"Frame {getattr(self, 'frame_count', 'N/A')}{track_info} Gait cosine_sim: {cosine_sim:.4f} skel_score: {skel_score:.4f} ratio_score: {ratio_score:.4f}")

        # Combine: 60% OpenGait, 25% skeleton gait, 15% body ratios
        return 0.6 * cosine_sim + 0.25 * skel_score + 0.15 * ratio_score

    def _update_identity_gallery(self, track_id, color_hist=None, height=None):
        """Update the identity gallery with the latest features for a track."""
        # Get averaged features
        appearance = self._get_averaged_feature(track_id)
        gait = self.gait_history[track_id][-1] if track_id in self.gait_history and self.gait_history[track_id] else None
        body = self.body_ratio_history[track_id][-1] if track_id in self.body_ratio_history and self.body_ratio_history[track_id] else None
        context = {
            'last_seen': self.frame_count,
            'height': height,
        }
        self.identity_gallery[track_id] = {
            'appearance': appearance,
            'gait': gait,
            'body': body,
            'color_hist': color_hist,
            'context': context
        }

    def _match_to_gallery(self, feature, gait, body, color_hist, height, context):
        """Match a new detection to the identity gallery using all cues."""
        best_id = None
        best_score = float('-inf')  # Allow negative scores for best match
        debug_scores = []
        for id_, entry in self.identity_gallery.items():
            # Skip comparing to self if possible (i.e., don't compare detection to its own gallery entry)
            if hasattr(self, "current_track_id_for_gait_sim") and self.current_track_id_for_gait_sim == id_:
                continue
            score = 0
            # Appearance
            app_score = self._calculate_appearance_similarity(feature, entry['appearance']) if entry['appearance'] is not None and feature is not None else 0.0
            score += self.appearance_weight * app_score
            # Gait
            self.current_gallery_track_id_for_gait_sim = id_
            gait_score = self._calculate_gait_similarity(gait, entry['gait'], body, entry['body']) if entry['gait'] is not None and gait is not None else 0.0
            score += self.gait_weight * gait_score
            # Body
            body_score = self._calculate_body_similarity(body, entry['body']) if entry['body'] is not None and body is not None else 0.0
            score += self.body_weight * body_score
            # Color histogram
            color_score = self._compare_histograms(color_hist, entry['color_hist']) if entry.get('color_hist') is not None and color_hist is not None else 0.0
            score += self.color_weight * color_score
            # Height (normalized difference)
            height_score = 0.0
            if 'height' in entry['context'] and height is not None and entry['context']['height'] is not None:
                h1, h2 = float(height), float(entry['context']['height'])
                height_score = max(0, 1 - abs(h1 - h2) / max(h1, h2, 1))
            score += self.height_weight * height_score
            # Context (e.g., time proximity)
            context_score = 0.0
            if 'last_seen' in entry['context']:
                time_diff = abs(self.frame_count - entry['context']['last_seen'])
                context_score = max(0, 1 - time_diff / 100)
            score += self.context_weight * context_score
            debug_scores.append((id_, score, app_score, gait_score, body_score, color_score, height_score, context_score))
            # Only update best_id if score is higher than previous best
            if score > best_score:
                best_score = score
                best_id = id_
        debug_scores_sorted = sorted(debug_scores, key=lambda x: x[1], reverse=True)
        best_id, best_score = (debug_scores_sorted[0][0], debug_scores_sorted[0][1]) if debug_scores_sorted else (None, None)
        second_id, second_score = (debug_scores_sorted[1][0], debug_scores_sorted[1][1]) if len(debug_scores_sorted) > 1 else (None, None)
        return best_id, best_score, second_id, second_score

    def process_frame(self, frame):
        """Process a frame with enhanced tracking logic"""
        self.frame_count += 1

        # Real-time mode: optionally skip frames for speed
        if self.realtime and self.frame_count % 2 == 1:
            return self.tracks
        
        # Get detections from YOLOv8
        yolo_results = self.yolo_tracker.process_frame(frame)

        # If using ByteTrack, update and get tracks
        if self.tracker_type == 'bytetrack':
            dets = [(bbox, conf) for tid, bbox, conf in yolo_results]
            bytetrack_results = self.bytetrack.update(dets)
            # Assign new track_ids from ByteTrack
            yolo_results = [(tid, bbox, conf) for tid, bbox, conf in bytetrack_results]
        
        # Predict new locations of existing tracks
        predicted_tracks = {}
        for track_id, tracker in self.kalman_trackers.items():
            try:
                # Get prediction with better error handling
                prediction = tracker.predict()
                
                # Handle different types of predictions
                if isinstance(prediction, (list, tuple, np.ndarray)):
                    # Convert to list if it's a numpy array
                    if isinstance(prediction, np.ndarray):
                        prediction = prediction.tolist()
                    
                    # If it's a nested list, extract the inner list
                    if len(prediction) == 1 and isinstance(prediction[0], (list, tuple, np.ndarray)):
                        prediction = prediction[0]
                    
                    # Make sure we have exactly 4 values for the bbox
                    if len(prediction) == 4:
                        predicted_tracks[track_id] = prediction
                    else:
                        logger.warning("Invalid prediction length for track %d: %d", track_id, len(prediction))
                else:
                    logger.warning("Unexpected prediction type for track %d: %s", track_id, type(prediction))
            except Exception as e:
                logger.error("Error predicting track %d: %s", track_id, str(e))
        
        # Current active tracks
        current_tracks = {}
        
        # Debug info for visualization
        debug_info = {}
        
        # --- Silhouette extraction using YOLOv8-seg ---
        seg_results = self.seg_model(frame)
        seg_masks = {}
        for r in seg_results:
            if not hasattr(r, 'masks') or r.masks is None or r.masks.data is None:
                continue
            for j, mask in enumerate(r.masks.data):
                if int(r.boxes.cls[j]) != 0:
                    continue  # Only person class
                track_id = int(r.boxes.id[j]) if hasattr(r.boxes, 'id') and r.boxes.id is not None else j
                sil = (mask.cpu().numpy() > 0.5).astype(np.float32)
                # Print silhouette info
                # print(f"Frame {self.frame_count} Silhouette for track {track_id}: shape={sil.shape}")
                seg_masks[track_id] = sil
                if track_id not in self.silhouette_history:
                    self.silhouette_history[track_id] = []
                self.silhouette_history[track_id].append(sil)
                if len(self.silhouette_history[track_id]) > 30:
                    self.silhouette_history[track_id].pop(0)
        
        # --- INSERT BATCH SKELETON CALL HERE ---
        yolo_results = self.yolo_tracker.process_frame(frame)
        if not yolo_results or len(yolo_results) == 0:
            self.tracks = {}
            self.last_debug_info = {}
            return {}
        # collect IDs, bboxes, confidences
        ids, bboxes, confs = zip(*[(tid, bbox, conf) for tid, bbox, conf in yolo_results])
        # run one YOLOv8-pose call for all bboxes
        keypoints_list = extract_skeleton_batch_yolopose(frame, list(bboxes), device=self.device)

        # build enriched detections
        detections = []
        for tid, bbox, conf, keypts in zip(ids, bboxes, confs, keypoints_list):
            # Print silhouette history info for all tracks
            # for track_id, sil_hist in self.silhouette_history.items():
            #     print(f"Frame {self.frame_count} Track {track_id} silhouette history length: {len(sil_hist)}")
            body_ratios = compute_body_ratios(keypts) if keypts else {}
            person_crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            feature = self.transreid.extract_features(person_crop)
            color_hist = self._extract_color_histogram(person_crop)
            height = bbox[3] - bbox[1]
            debug_lines = []
            if self.debug_visualize:
                debug_lines.append(f"Skeleton: {'NO' if not keypts else 'YES'}")
            debug_info[tid] = debug_lines
            gait_embedding = None
            detections.append((tid, bbox, conf, keypts, body_ratios, feature, color_hist, height, debug_info, gait_embedding))
        # After detections, extract gait embeddings for all tracks with enough silhouettes
        for track_id, sil_hist in self.silhouette_history.items():
            if len(sil_hist) >= 10:
                try:
                    silhouettes = np.stack(sil_hist[-10:], axis=0)
                    gait_embedding = self.gait_embedder.extract(silhouettes)
                    if track_id not in self.gait_history:
                        self.gait_history[track_id] = []
                    self.gait_history[track_id].append(gait_embedding)
                except Exception as e:
                    logger.warning(f"OpenGait embedding extraction failed for track {track_id}: {e}")
        
        # Step 1: First associate detections with high confidence tracks based on both motion and appearance
        detections_to_process = []
        for data in detections:
            track_id, bbox, conf, keypoints, body_ratios, feature, color_hist, height, debug_info, _ = data
            
            # Skip tiny detections that are likely false positives
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w < 20 or h < 20:
                continue
            
            # Skeleton detection monitoring
            self.skeleton_total_count += 1
            if not keypoints:
                self.skeleton_missing_count += 1
            
            # For overlay: show if skeleton is missing
            debug_info[track_id] = debug_info[track_id]
            
            # If YOLOv8 returned a valid track_id, use it
            if track_id != -1 and track_id in self.kalman_trackers:
                # Update Kalman tracker
                self.kalman_trackers[track_id].update(bbox)
                self._update_feature_history(track_id, feature)
                current_tracks[track_id] = (bbox, feature, conf)
                
                # Update skeleton and body ratio history
                if track_id not in self.skeleton_history:
                    self.skeleton_history[track_id] = []
                if track_id not in self.body_ratio_history:
                    self.body_ratio_history[track_id] = []
                self.skeleton_history[track_id].append(keypoints)
                self.body_ratio_history[track_id].append(body_ratios)
                
                # Update gait history with embedding
                if track_id in self.gait_history and self.gait_history[track_id]:
                    gait_embedding = self.gait_history[track_id][-1]
                else:
                    gait_embedding = None
                
                if gait_embedding is not None:
                    if track_id not in self.gait_history:
                        self.gait_history[track_id] = []
                    self.gait_history[track_id].append(gait_embedding)
                
                # Update identity gallery
                self._update_identity_gallery(track_id, color_hist=color_hist, height=height)
            else:
                # Process later
                detections_to_process.append((bbox, feature, conf, keypoints, body_ratios, color_hist, height, gait_embedding))
        
        # Step 2: Associate remaining detections with tracks
        for bbox, feature, conf, keypoints, body_ratios, color_hist, height, _ in detections_to_process:
            best_match_id = None
            best_match_score = 0
            
            # Calculate scores for all existing tracks
            for track_id in list(predicted_tracks.keys()) + list(self.inactive_tracks.keys()):
                if track_id in current_tracks:
                    continue  # Already matched
                    
                # Get track data
                if track_id in predicted_tracks:
                    track_bbox = predicted_tracks[track_id]
                    track_feature = self._get_averaged_feature(track_id)
                    
                    if track_feature is None:
                        continue
                        
                    # Calculate motion and appearance scores
                    motion_score = 1.0 - self._calculate_motion_distance(bbox, track_bbox)
                    appearance_score = self._calculate_appearance_similarity(feature, track_feature)
                    
                    # Calculate body and gait similarity
                    body_score = 0.0
                    gait_score = 0.0
                    if track_id in self.body_ratio_history and self.body_ratio_history[track_id]:
                        prev_ratios = self.body_ratio_history[track_id][-1]
                        body_score = self._calculate_body_similarity(body_ratios, prev_ratios)
                    if track_id in self.gait_history and self.gait_history[track_id]:
                        prev_gait = self.gait_history[track_id][-1]
                        prev_ratios = self.body_ratio_history[track_id][-1] if track_id in self.body_ratio_history and self.body_ratio_history[track_id] else None
                        gait_score = self._calculate_gait_similarity(prev_gait, prev_gait, body_ratios, prev_ratios)
                    
                    # Compare color histograms
                    color_score = 0.0
                    if track_id in self.identity_gallery and 'color_hist' in self.identity_gallery[track_id]:
                        prev_color_hist = self.identity_gallery[track_id]['color_hist']
                        color_score = self._compare_histograms(color_hist, prev_color_hist)
                    
                    # Compare height
                    height_score = 0.0
                    if track_id in self.identity_gallery and 'height' in self.identity_gallery[track_id]['context']:
                        prev_height = self.identity_gallery[track_id]['context']['height']
                        h1, h2 = float(height), float(prev_height)
                        height_score = max(0, 1 - abs(h1 - h2) / max(h1, h2, 1))
                    
                    # Time factor (reduce score for long-inactive tracks)
                    time_factor = 5.0
                    
                    # Final score combines motion, appearance, body, gait, color, and height with time factor
                    final_score = (
                        self.motion_weight * motion_score +
                        self.appearance_weight * appearance_score +
                        self.body_weight * body_score +
                        self.gait_weight * gait_score +
                        self.color_weight * color_score +
                        self.height_weight * height_score
                    ) * time_factor
                elif track_id in self.inactive_tracks:
                    inactive_bbox, inactive_feature, last_seen, inactive_conf = self.inactive_tracks[track_id]
                    
                    # Calculate scores
                    motion_score = 1.0 - self._calculate_motion_distance(bbox, inactive_bbox)
                    appearance_score = self._calculate_appearance_similarity(feature, inactive_feature)
                    
                    # Calculate body and gait similarity
                    body_score = 0.0
                    gait_score = 0.0
                    if track_id in self.body_ratio_history and self.body_ratio_history[track_id]:
                        prev_ratios = self.body_ratio_history[track_id][-1]
                        body_score = self._calculate_body_similarity(body_ratios, prev_ratios)
                    if track_id in self.gait_history and self.gait_history[track_id]:
                        prev_gait = self.gait_history[track_id][-1]
                        prev_ratios = self.body_ratio_history[track_id][-1] if track_id in self.body_ratio_history and self.body_ratio_history[track_id] else None
                        gait_score = self._calculate_gait_similarity(prev_gait, prev_gait, body_ratios, prev_ratios)
                    
                    # Compare color histograms
                    color_score = 0.0
                    if track_id in self.identity_gallery and 'color_hist' in self.identity_gallery[track_id]:
                        prev_color_hist = self.identity_gallery[track_id]['color_hist']
                        color_score = self._compare_histograms(color_hist, prev_color_hist)
                    
                    # Compare height
                    height_score = 0.0
                    if track_id in self.identity_gallery and 'height' in self.identity_gallery[track_id]['context']:
                        prev_height = self.identity_gallery[track_id]['context']['height']
                        h1, h2 = float(height), float(prev_height)
                        height_score = max(0, 1 - abs(h1 - h2) / max(h1, h2, 1))
                    
                    # Time penalty (reduce score for tracks inactive for many frames)
                    time_factor = max(0.5, 1.0 - (self.frame_count - last_seen) / self.max_age)
                    
                    # Final score combines motion, appearance, body, gait, color, and height with time factor
                    final_score = (
                        self.motion_weight * motion_score +
                        self.appearance_weight * appearance_score +
                        self.body_weight * body_score +
                        self.gait_weight * gait_score +
                        self.color_weight * color_score +
                        self.height_weight * height_score
                    ) * time_factor
                else:
                    continue
                
                # Update best match if score is high enough
                if final_score > self.reid_threshold and final_score > best_match_score:
                    best_match_score = final_score
                    best_match_id = track_id
            
            # Process the match or create new track
            if best_match_id is not None:
                # Update existing track
                track_id = best_match_id
                
                # Update or create Kalman tracker
                if track_id in self.kalman_trackers:
                    self.kalman_trackers[track_id].update(bbox)
                else:
                    self.kalman_trackers[track_id] = KalmanBoxTracker(bbox)
                
                # Update feature history
                self._update_feature_history(track_id, feature)
                
                # Remove from inactive tracks if it was there
                if track_id in self.inactive_tracks:
                    del self.inactive_tracks[track_id]
                
                # Add to current tracks
                current_tracks[track_id] = (bbox, feature, conf)
                
                # Update skeleton and body ratio history
                if track_id not in self.skeleton_history:
                    self.skeleton_history[track_id] = []
                if track_id not in self.body_ratio_history:
                    self.body_ratio_history[track_id] = []
                self.skeleton_history[track_id].append(keypoints)
                self.body_ratio_history[track_id].append(body_ratios)
                
                # Update gait history with embedding
                if track_id in self.gait_history and self.gait_history[track_id]:
                    gait_embedding = self.gait_history[track_id][-1]
                else:
                    gait_embedding = None
                
                if gait_embedding is not None:
                    if track_id not in self.gait_history:
                        self.gait_history[track_id] = []
                    self.gait_history[track_id].append(gait_embedding)
                
                # Update identity gallery
                self._update_identity_gallery(track_id, color_hist=color_hist, height=height)
            else:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                
                # Initialize Kalman tracker
                self.kalman_trackers[track_id] = KalmanBoxTracker(bbox)
                
                # Start feature history
                self._update_feature_history(track_id, feature)
                
                # Add to current tracks
                current_tracks[track_id] = (bbox, feature, conf)
                
                # Initialize skeleton and body ratio history
                self.skeleton_history[track_id] = [keypoints]
                self.body_ratio_history[track_id] = [body_ratios]
                
                # Update gait history with embedding
                if track_id in self.gait_history and self.gait_history[track_id]:
                    gait_embedding = self.gait_history[track_id][-1]
                else:
                    gait_embedding = None
                
                if gait_embedding is not None:
                    self.gait_history[track_id] = [gait_embedding]
                
                # Update identity gallery
                self._update_identity_gallery(track_id, color_hist=color_hist, height=height)
        
        # Step 3: Update inactive tracks
        for track_id in list(self.kalman_trackers.keys()):
            if track_id not in current_tracks:
                # Track not found in current frame
                if self.kalman_trackers[track_id].time_since_update < self.max_age:
                    # Store in inactive tracks
                    last_bbox = self.kalman_trackers[track_id].last_bbox
                    avg_feature = self._get_averaged_feature(track_id)
                    if track_id in self.tracks and len(self.tracks[track_id]) >= 3:
                        last_conf = self.tracks[track_id][2]
                    else:
                        last_conf = 0.5
                    
                    self.inactive_tracks[track_id] = (last_bbox, avg_feature, self.frame_count, last_conf)
                else:
                    # Track too old, remove it
                    del self.kalman_trackers[track_id]
                    if track_id in self.feature_history:
                        del self.feature_history[track_id]
        
        # Remove old inactive tracks
        for track_id in list(self.inactive_tracks.keys()):
            _, _, last_seen, _ = self.inactive_tracks[track_id]
            if self.frame_count - last_seen > self.max_age:
                del self.inactive_tracks[track_id]
                if track_id in self.feature_history:
                    del self.feature_history[track_id]
        
        # Compute gait features for each track
        for track_id in current_tracks:
            if track_id in self.skeleton_history:
                gait, ratios = compute_gait_features(self.skeleton_history[track_id])
                if track_id not in self.gait_history:
                    self.gait_history[track_id] = []
                self.gait_history[track_id].append(gait)
                # Also update body_ratio_history with ratios from gait
                if track_id not in self.body_ratio_history:
                    self.body_ratio_history[track_id] = []
                self.body_ratio_history[track_id].append(ratios)
        
        # Optionally print skeleton detection stats every 100 frames
        if self.frame_count % 100 == 0 and self.skeleton_total_count > 0:
            miss_rate = 100.0 * self.skeleton_missing_count / self.skeleton_total_count
            logger.info("[Skeleton Stats] Missing: %d/%d (%.1f%%)",
                        self.skeleton_missing_count,
                        self.skeleton_total_count,
                        miss_rate)
        
        # Update tracks
        self.tracks = current_tracks
        self.last_debug_info = debug_info
        
        # Print tracking stats every 30 frames
        if self.frame_count % 30 == 0:
            logger.info("Frame %d: Tracking %d active persons, %d inactive tracks, %d total trackers",
                        self.frame_count, len(current_tracks), len(self.inactive_tracks), len(self.kalman_trackers))
        
        return current_tracks
    
    def visualize(self, frame, tracks=None):
        """Visualize tracking results with enhanced display"""
        if tracks is None:
            tracks = self.tracks
        
        return draw_tracking_results(
            frame, 
            tracks, 
            inactive_tracks=self.inactive_tracks, 
            frame_count=self.frame_count,
            skeleton_history=self.skeleton_history,
            debug_info=self.last_debug_info
        )

def process_video(video_path, output_path, yolo_weights, transreid_weights, 
                 conf_threshold=0.3, reid_threshold=0.7, device=None,
                 appearance_weight=0.4, gait_weight=0.15, body_weight=0.15, color_weight=0.2, height_weight=0.1, context_weight=0.05, debug_visualize=False, mmpose_weights=None, show_window=False, tracker_type='kalman', realtime=False):
    """Process a video file for person tracking."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video file %s", video_path)
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info("Video: %dx%d at %.2f FPS, %d frames", width, height, fps, total_frames)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize tracker with auto device selection
    tracker = PersonTracker(yolo_weights, transreid_weights, device=device,
                           conf_threshold=conf_threshold, reid_threshold=reid_threshold,
                           appearance_weight=appearance_weight, gait_weight=gait_weight, body_weight=body_weight, color_weight=color_weight, height_weight=height_weight, context_weight=context_weight, debug_visualize=debug_visualize, tracker_type=tracker_type, realtime=realtime)
    
    frame_idx = 0
    while cap.isOpened() and frame_idx < 5000:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        if frame_idx % 10 == 0:  # Only print every 10 frames to reduce console output
            logger.debug("Processing frame %d/%d (%.1f%%)",
                         frame_idx, total_frames, frame_idx/total_frames*100)
        
        # Process frame
        tracks = tracker.process_frame(frame)
        
        # Visualize results
        output_frame = tracker.visualize(frame, tracks)
        
        # Write to output video
        out.write(output_frame)
        
        # Display (optional)
        if show_window:
            cv2.imshow('Person Tracking', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    out.release()
    if show_window:
        cv2.destroyAllWindows()
    
    logger.info("Tracking complete. Output saved to %s", output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Person tracking with YOLOv8 and TransReID')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output video')
    parser.add_argument('--yolo_weights', type=str, default='yolov8m.pt', help='Path to YOLOv8 weights')
    parser.add_argument('--transreid_weights', type=str, required=True, help='Path to TransReID weights')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold for detections')
    parser.add_argument('--reid_threshold', type=float, default=0.7, help='Similarity threshold for re-identification')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda, mps, cpu). If not specified, best available device will be used.')
    parser.add_argument('--appearance_weight', type=float, default=0.4, help='Weight for appearance similarity')
    parser.add_argument('--gait_weight', type=float, default=0.15, help='Weight for gait similarity')
    parser.add_argument('--body_weight', type=float, default=0.15, help='Weight for body similarity')
    parser.add_argument('--color_weight', type=float, default=0.2, help='Weight for color similarity')
    parser.add_argument('--height_weight', type=float, default=0.1, help='Weight for height similarity')
    parser.add_argument('--context_weight', type=float, default=0.05, help='Weight for context similarity')
    parser.add_argument('--debug_visualize', action='store_true', help='Enable debug visualization of matching scores')
    parser.add_argument('--show_window', action='store_true', help='Show OpenCV window with imshow (for debugging)')
    parser.add_argument('--tracker', type=str, default='kalman', choices=['kalman', 'bytetrack'], help='Tracking algorithm to use (kalman or bytetrack)')
    parser.add_argument('--realtime', action='store_true', help='Enable real-time mode with frame skipping')
    
    args = parser.parse_args()
    
    process_video(args.video, args.output, args.yolo_weights, args.transreid_weights,
                 conf_threshold=args.conf, reid_threshold=args.reid_threshold, device=args.device,
                 appearance_weight=args.appearance_weight, gait_weight=args.gait_weight, body_weight=args.body_weight, color_weight=args.color_weight, height_weight=args.height_weight, context_weight=args.context_weight, debug_visualize=args.debug_visualize, show_window=args.show_window, tracker_type=args.tracker, realtime=args.realtime)
