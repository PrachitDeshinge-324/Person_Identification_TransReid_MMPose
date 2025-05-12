import numpy as np
from scipy.spatial import distance
import pickle
import logging

logger = logging.getLogger(__name__)

def load_identity_database(db_path):
    """
    Load the identity database from a pickle file.
    """
    with open(db_path, "rb") as f:
        return pickle.load(f)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    if vec1 is None or vec2 is None:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)

def dict_similarity(dict1, dict2):
    """Calculate similarity between two dictionaries of features."""
    if not dict1 or not dict2:
        return 0.0
    
    # Find common keys
    common_keys = set(dict1.keys()) & set(dict2.keys())
    if not common_keys:
        return 0.0
    
    # Calculate similarity for each common key
    similarities = []
    for key in common_keys:
        if isinstance(dict1[key], (int, float)) and isinstance(dict2[key], (int, float)):
            # For numeric values, use normalized absolute difference
            max_val = max(abs(dict1[key]), abs(dict2[key]))
            if max_val > 0:
                sim = 1 - abs(dict1[key] - dict2[key]) / max_val
            else:
                sim = 1.0
            similarities.append(sim)
        elif isinstance(dict1[key], np.ndarray) and isinstance(dict2[key], np.ndarray):
            # For arrays, use cosine similarity
            similarities.append(cosine_similarity(dict1[key], dict2[key]))
    
    if similarities:
        return np.mean(similarities)
    return 0.0

def height_similarity(h1, h2, tolerance=15.0):
    """Calculate similarity based on height difference."""
    if h1 is None or h2 is None:
        return 0.0
    
    # Convert to float to ensure proper calculation
    h1, h2 = float(h1), float(h2)
    
    # Use a Gaussian-like function to calculate similarity based on height difference
    diff = abs(h1 - h2)
    return np.exp(-0.5 * (diff / tolerance) ** 2)

def identify_person(query_features, identity_db, feature_weights=None, threshold=0.5, verbose=False):
    """
    Identify a person using multiple feature types from the database.
    
    Args:
        query_features: Dict with features of the detected person
                        (appearance, opengait, skeleton_gait, body_ratios, etc.)
        identity_db: Dict mapping person IDs to their feature data
        feature_weights: Dict of weights for different feature types
        threshold: Minimum similarity score to consider a match
        verbose: Whether to print detailed matching information
    
    Returns:
        Tuple of (best_match_id, best_score, match_details)
    """
    if feature_weights is None:
        # Default weights prioritizing most discriminative features
        feature_weights = {
            "opengait": 1.0,           # Most reliable for long-term identification
            "skeleton_gait": 0.8,      # Very reliable when available
            "industrial_pose": 0.7,    # Good for consistent posture patterns
            "body_ratios": 0.7,        # Fairly stable across different sessions
            "height": 0.5,             # Useful but less reliable alone
            "best_skeleton": 0.4,      # Helpful when other features unavailable
            "best_3d_skeleton": 0.6,   # Better than 2D skeleton
            "industrial_color": 0.3,   # Can help but changes with clothing
            "motion_pattern": 0.3      # Useful for movement style but varies
        }
    
    best_id = None
    best_score = 0
    match_details = {}
    all_match_scores = {}
    
    # Normalize query feature dicts if they contain numpy arrays
    if query_features.get("industrial_pose") and isinstance(query_features["industrial_pose"], dict):
        # Ensure all numeric arrays are normalized
        for k, v in query_features["industrial_pose"].items():
            if isinstance(v, np.ndarray) and np.linalg.norm(v) > 0:
                query_features["industrial_pose"][k] = v / np.linalg.norm(v)
    
    for person_id, db_entry in identity_db.items():
        name = db_entry.get('name', f'Person_{person_id}')
        individual_scores = {}
        weights_used = {}
        total_weight = 0
        
        # 1. OpenGait comparison (highest weight)
        if "opengait" in query_features and query_features["opengait"] is not None and \
           "opengait" in db_entry and db_entry["opengait"] is not None:
            weight = feature_weights.get("opengait", 1.0)
            sim = cosine_similarity(query_features["opengait"], db_entry["opengait"])
            individual_scores["opengait"] = sim
            weights_used["opengait"] = weight
            total_weight += weight
        
        # 2. Skeleton gait comparison
        if "skeleton_gait" in query_features and query_features["skeleton_gait"] and \
           "skeleton_gait" in db_entry and db_entry["skeleton_gait"]:
            weight = feature_weights.get("skeleton_gait", 0.8)
            sim = dict_similarity(query_features["skeleton_gait"], db_entry["skeleton_gait"])
            individual_scores["skeleton_gait"] = sim
            weights_used["skeleton_gait"] = weight
            total_weight += weight
        
        # 3. Industrial pose features
        if "industrial_pose" in query_features and query_features["industrial_pose"] and \
           "industrial_pose" in db_entry and db_entry["industrial_pose"]:
            weight = feature_weights.get("industrial_pose", 0.7)
            sim = dict_similarity(query_features["industrial_pose"], db_entry["industrial_pose"])
            individual_scores["industrial_pose"] = sim
            weights_used["industrial_pose"] = weight
            total_weight += weight
        
        # 4. Body ratios comparison
        if "body_ratios" in query_features and query_features["body_ratios"] and \
           "body_ratios" in db_entry and db_entry["body_ratios"]:
            weight = feature_weights.get("body_ratios", 0.7)
            sim = dict_similarity(query_features["body_ratios"], db_entry["body_ratios"])
            individual_scores["body_ratios"] = sim
            weights_used["body_ratios"] = weight
            total_weight += weight
        
        # 5. Height comparison
        if "height" in query_features and query_features["height"] is not None and \
           "height" in db_entry and db_entry["height"] is not None:
            weight = feature_weights.get("height", 0.5)
            sim = height_similarity(query_features["height"], db_entry["height"])
            individual_scores["height"] = sim
            weights_used["height"] = weight
            total_weight += weight

        # 6. Skeleton comparison (coordinate-based with normalization)
        if "best_skeleton" in query_features and query_features["best_skeleton"] is not None and \
           "best_skeleton" in db_entry and db_entry["best_skeleton"] is not None:
            weight = feature_weights.get("best_skeleton", 0.4)
            # Normalize skeletons to center and scale before comparing
            try:
                # Calculate centroids
                query_pts = np.array([kp[:2] for kp in query_features["best_skeleton"] if kp[2] > 0.2])
                db_pts = np.array([kp[:2] for kp in db_entry["best_skeleton"] if kp[2] > 0.2])
                
                if len(query_pts) >= 5 and len(db_pts) >= 5:  # Need enough points
                    query_centroid = np.mean(query_pts, axis=0)
                    db_centroid = np.mean(db_pts, axis=0)
                    
                    # Center both skeletons
                    query_centered = query_pts - query_centroid
                    db_centered = db_pts - db_centroid
                    
                    # Scale to unit norm
                    query_norm = np.linalg.norm(query_centered)
                    db_norm = np.linalg.norm(db_centered)
                    
                    if query_norm > 0 and db_norm > 0:
                        query_normalized = query_centered / query_norm
                        db_normalized = db_centered / db_norm
                        
                        # Calculate Procrustes distance (shape similarity)
                        sim = 1 - np.mean(np.sqrt(np.sum((query_normalized - db_normalized)**2, axis=1)))
                        sim = max(0, min(1, sim))  # Clip to [0,1]
                        
                        individual_scores["best_skeleton"] = sim
                        weights_used["best_skeleton"] = weight
                        total_weight += weight
            except Exception as e:
                logger.warning(f"Error comparing skeletons: {e}")

        # 7. 3D Skeleton comparison
        if "best_3d_skeleton" in query_features and query_features["best_3d_skeleton"] is not None and \
           "best_3d_skeleton" in db_entry and db_entry["best_3d_skeleton"] is not None:
            weight = feature_weights.get("best_3d_skeleton", 0.6)
            try:
                # Calculate centroids and normalize 3D points
                query_pts = np.array([kp for kp in query_features["best_3d_skeleton"]])
                db_pts = np.array([kp for kp in db_entry["best_3d_skeleton"]])
                
                if len(query_pts) >= 5 and len(db_pts) >= 5:  # Need enough points
                    query_centroid = np.mean(query_pts, axis=0)
                    db_centroid = np.mean(db_pts, axis=0)
                    
                    # Center both skeletons
                    query_centered = query_pts - query_centroid
                    db_centered = db_pts - db_centroid
                    
                    # Scale to unit norm
                    query_norm = np.linalg.norm(query_centered)
                    db_norm = np.linalg.norm(db_centered)
                    
                    if query_norm > 0 and db_norm > 0:
                        query_normalized = query_centered / query_norm
                        db_normalized = db_centered / db_norm
                        
                        # Calculate 3D shape similarity
                        sim = 1 - np.mean(np.sqrt(np.sum((query_normalized - db_normalized)**2, axis=1)))
                        sim = max(0, min(1, sim))  # Clip to [0,1]
                        
                        individual_scores["best_3d_skeleton"] = sim
                        weights_used["best_3d_skeleton"] = weight
                        total_weight += weight
            except Exception as e:
                logger.warning(f"Error comparing 3D skeletons: {e}")
                
        # 8. Industrial color comparison
        if "industrial_color" in query_features and query_features["industrial_color"] and \
           "industrial_color" in db_entry and db_entry["industrial_color"]:
            weight = feature_weights.get("industrial_color", 0.3)
            sim = dict_similarity(query_features["industrial_color"], db_entry["industrial_color"])
            individual_scores["industrial_color"] = sim
            weights_used["industrial_color"] = weight
            total_weight += weight
            
        # 9. Motion pattern comparison
        if "motion_pattern" in query_features and query_features["motion_pattern"] and \
           "motion_pattern" in db_entry and db_entry["motion_pattern"]:
            weight = feature_weights.get("motion_pattern", 0.3)
            sim = dict_similarity(query_features["motion_pattern"], db_entry["motion_pattern"])
            individual_scores["motion_pattern"] = sim
            weights_used["motion_pattern"] = weight
            total_weight += weight
        
        # Calculate weighted average of all available scores
        if total_weight > 0:
            weighted_sum = sum(individual_scores[feat] * weights_used[feat] for feat in individual_scores)
            final_score = weighted_sum / total_weight
            
            if verbose:
                logger.info(f"ID {person_id} ({name}): Score {final_score:.3f}, Details: {individual_scores}")
            
            # Store all match scores
            all_match_scores[person_id] = {
                "name": name,
                "score": final_score,
                "details": individual_scores
            }
            
            # Update best match if score is higher
            if final_score > best_score:
                best_score = final_score
                best_id = person_id
                match_details = individual_scores
    
    # Only consider matches above threshold
    if best_score >= threshold:
        return best_id, best_score, match_details, all_match_scores
    else:
        return None, best_score, match_details, all_match_scores
