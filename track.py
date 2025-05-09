import cv2
import pickle
from person_tracker import PersonTracker
import numpy as np
import torch
from utils.gaits import compute_gait_features  # Import to compute skeleton-based gait
from tqdm import tqdm  # Import tqdm for progress bar

VIDEO_PATH = "input/3c.mp4"
IDENTITY_DB_PATH = "identity_database.pkl"
YOLO_WEIGHTS = "weights/yolo11x.pt"
TRANSREID_WEIGHTS = "weights/transreid_vitbase.pth"
MMPOSE_WEIGHTS = "weights/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth"

# Load identity database
with open(IDENTITY_DB_PATH, "rb") as f:
    identity_db = pickle.load(f)

# Define person name mapping
name_mapping = {
    1: "Prachit",
    2: "Ashutosh", 
    3: "Ojasv",
    4: "Nayan"
}

# Add a history tracking mechanism for more stable identification
identification_history = {}
HISTORY_WINDOW = 5  # Consider last 5 frames for smoothing predictions

# Feature weights for matching - adjust these based on your specific scenario
FEATURE_WEIGHTS = {
    'appearance': 0.35,       # ReID appearance features
    'opengait': 0.15,         # OpenGait features
    'skeleton_gait': 0.15,    # Skeleton-based gait features
    'industrial_pose': 0.15,  # Industrial pose features
    'industrial_gait': 0.10,  # Industrial gait features 
    'industrial_color': 0.05, # Industrial uniform/PPE features
    'body_ratios': 0.05,      # Body proportion features
}

# Temperature parameter for similarity scaling (higher = more discriminative scores)
SIMILARITY_TEMPERATURE = 3.0  # Default = 1.0, higher values make differences more pronounced

# Helper function to safely convert tensor to numpy
def to_numpy(tensor):
    """Safely convert a tensor to numpy array, handling device transfer if needed."""
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, (int, float)):
        return tensor
    if hasattr(tensor, 'is_cuda') and tensor.is_cuda:
        return tensor.cpu().numpy()
    if hasattr(tensor, 'device') and tensor.device.type in ['mps', 'cuda']:
        return tensor.cpu().numpy()
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy() if tensor.requires_grad else tensor.numpy()
    return tensor

# Initialize tracker without the unsupported 'verbose' parameter
tracker = PersonTracker(
    YOLO_WEIGHTS, TRANSREID_WEIGHTS,
    mmpose_weights=MMPOSE_WEIGHTS,
    debug_visualize=False,
    reid_threshold=0.55
    # 'verbose' parameter removed as it's not supported
)
tracker.identity_gallery = identity_db  # Use the loaded DB for matching

# Enhanced matching function to include industrial features
def compute_match_with_gait_features(appearance, opengait, skeleton_gait, body_ratios, 
                                   industrial_pose, industrial_gait, industrial_color,
                                   color_hist, height, context, db_entry):
    scores = []
    weights = []
    features_used = []  # Track which features were successfully used
    
    # Appearance score - safely convert tensors to numpy arrays
    if appearance is not None and 'appearance' in db_entry and db_entry['appearance'] is not None:
        app_np = to_numpy(appearance)
        db_app_np = to_numpy(db_entry['appearance'])
        
        if app_np is not None and db_app_np is not None:
            try:
                app_score = np.dot(app_np, db_app_np) / (np.linalg.norm(app_np) * np.linalg.norm(db_app_np))
                
                # Apply temperature to make scores more discriminative
                app_score = np.power(max(0, app_score), SIMILARITY_TEMPERATURE)
                
                scores.append(float(app_score))
                weights.append(FEATURE_WEIGHTS['appearance'])
                features_used.append('appearance')
            except Exception as e:
                print(f"Error computing appearance score: {e}")
    
    # OpenGait score - safely convert tensors
    if opengait is not None and 'opengait' in db_entry and db_entry['opengait'] is not None:
        try:
            opengait_np = to_numpy(opengait)
            db_gait_np = to_numpy(db_entry['opengait'])
            
            if opengait_np is not None and db_gait_np is not None:
                gait_score = np.dot(opengait_np, db_gait_np) / (np.linalg.norm(opengait_np) * np.linalg.norm(db_gait_np))
                gait_score = np.power(max(0, gait_score), SIMILARITY_TEMPERATURE)
                scores.append(float(gait_score))
                weights.append(FEATURE_WEIGHTS['opengait'])
                features_used.append('opengait')
        except Exception as e:
            print(f"Error computing OpenGait score: {e}")
            
    # Fallback to legacy 'gait' key if 'opengait' isn't available
    elif opengait is not None and 'gait' in db_entry and db_entry['gait'] is not None:
        try:
            opengait_np = to_numpy(opengait)
            db_gait_np = to_numpy(db_entry['gait'])
            
            # Skip calculation if either value is a dictionary instead of a vector
            if (isinstance(opengait_np, dict) or isinstance(db_gait_np, dict)):
                pass
            elif opengait_np is not None and db_gait_np is not None:
                gait_score = np.dot(opengait_np, db_gait_np) / (np.linalg.norm(opengait_np) * np.linalg.norm(db_gait_np))
                gait_score = np.power(max(0, gait_score), SIMILARITY_TEMPERATURE)
                scores.append(float(gait_score))
                weights.append(FEATURE_WEIGHTS['opengait'])
                features_used.append('legacy_gait')
        except Exception as e:
            # Handle error without printing trace for dictionary type errors which we now handle
            if "unsupported operand type(s) for *: 'dict' and" not in str(e):
                print(f"Error computing legacy gait score: {e}")
    
    # Skeleton gait score - this is a dictionary of features in the database
    if skeleton_gait is not None and 'skeleton_gait' in db_entry and db_entry['skeleton_gait'] is not None:
        try:
            # Compare common keys in both skeleton gait dictionaries
            common_keys = set(skeleton_gait.keys()) & set(db_entry['skeleton_gait'].keys())
            if common_keys:
                skel_scores = []
                for key in common_keys:
                    s1 = skeleton_gait[key]
                    s2 = db_entry['skeleton_gait'][key]
                    if isinstance(s1, (int, float)) and isinstance(s2, (int, float)):
                        # Simple ratio comparison for scalar values
                        ratio = min(s1, s2) / max(s1, s2) if max(s1, s2) > 0 else 0
                        skel_scores.append(ratio)
                if skel_scores:
                    avg_skeleton_score = sum(skel_scores) / len(skel_scores)
                    scores.append(avg_skeleton_score)
                    weights.append(FEATURE_WEIGHTS['skeleton_gait'])
                    features_used.append('skeleton_gait')
        except:
            pass  # Skip if comparison fails
    
    # Body ratios score
    if body_ratios and 'body_ratios' in db_entry and db_entry['body_ratios']:
        try:
            common_keys = set(body_ratios.keys()) & set(db_entry['body_ratios'].keys())
            if common_keys:
                body_scores = []
                for key in common_keys:
                    r1 = body_ratios[key]
                    r2 = db_entry['body_ratios'][key]
                    ratio = min(r1, r2) / max(r1, r2) if max(r1, r2) > 0 else 0
                    body_scores.append(ratio)
                if body_scores:
                    avg_body_score = sum(body_scores) / len(body_scores)
                    scores.append(avg_body_score)
                    weights.append(FEATURE_WEIGHTS['body_ratios'])
                    features_used.append('body_ratios')
        except:
            pass
    
    # Industrial pose features
    if industrial_pose and 'industrial_pose' in db_entry and db_entry['industrial_pose']:
        try:
            common_keys = set(industrial_pose.keys()) & set(db_entry['industrial_pose'].keys())
            if common_keys:
                pose_scores = []
                for key in common_keys:
                    p1 = industrial_pose[key]
                    p2 = db_entry['industrial_pose'][key]
                    if isinstance(p1, (int, float)) and isinstance(p2, (int, float)):
                        # Use ratio for comparison (stable across distances)
                        ratio = min(p1, p2) / max(p1, p2) if max(p1, p2) > 0 else 0
                        pose_scores.append(ratio)
                if pose_scores:
                    avg_pose_score = sum(pose_scores) / len(pose_scores)
                    scores.append(avg_pose_score)
                    weights.append(FEATURE_WEIGHTS['industrial_pose'])
                    features_used.append('industrial_pose')
        except:
            pass
    
    # Industrial gait features
    if industrial_gait and 'industrial_gait' in db_entry and db_entry['industrial_gait']:
        try:
            common_keys = set(industrial_gait.keys()) & set(db_entry['industrial_gait'].keys())
            if common_keys:
                gait_scores = []
                for key in common_keys:
                    g1 = industrial_gait[key]
                    g2 = db_entry['industrial_gait'][key]
                    if isinstance(g1, (int, float)) and isinstance(g2, (int, float)):
                        ratio = min(g1, g2) / max(g1, g2) if max(g1, g2) > 0 else 0
                        gait_scores.append(ratio)
                if gait_scores:
                    avg_gait_score = sum(gait_scores) / len(gait_scores)
                    scores.append(avg_gait_score)
                    weights.append(FEATURE_WEIGHTS['industrial_gait'])
                    features_used.append('industrial_gait')
        except:
            pass
    
    # Industrial color features (focus on safety colors, workwear)
    if industrial_color and 'industrial_color' in db_entry and db_entry['industrial_color']:
        try:
            # Get scalar features (percentage of safety colors, etc.)
            scalar_keys = [k for k in set(industrial_color.keys()) & set(db_entry['industrial_color'].keys()) 
                         if not k.endswith('_color_hist')]
            
            if scalar_keys:
                color_scores = []
                for key in scalar_keys:
                    c1 = industrial_color[key]
                    c2 = db_entry['industrial_color'][key]
                    if isinstance(c1, (int, float)) and isinstance(c2, (int, float)):
                        diff = 1.0 - abs(c1 - c2) / max(max(c1, c2), 0.01)
                        color_scores.append(diff)
                
                if color_scores:
                    avg_color_score = sum(color_scores) / len(color_scores)
                    scores.append(avg_color_score)
                    weights.append(FEATURE_WEIGHTS['industrial_color'])
                    features_used.append('industrial_color')
                
            # Process histograms separately if needed
            hist_keys = [k for k in set(industrial_color.keys()) & set(db_entry['industrial_color'].keys()) 
                       if k.endswith('_color_hist')]
            
            if hist_keys:
                hist_scores = []
                for key in hist_keys:
                    h1 = industrial_color[key]
                    h2 = db_entry['industrial_color'][key]
                    if isinstance(h1, np.ndarray) and isinstance(h2, np.ndarray):
                        hist_score = cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL)
                        hist_scores.append(max(0, hist_score))  # Correlation from -1 to 1, we use 0 to 1
                
                if hist_scores:
                    avg_hist_score = sum(hist_scores) / len(hist_scores)
                    scores.append(avg_hist_score)
                    weights.append(FEATURE_WEIGHTS['industrial_color'] * 0.5)  # Lower weight for histograms
                    features_used.append('industrial_color_hist')
        except Exception as e:
            print(f"Error comparing industrial color: {e}")
    
    # Color histogram score
    if color_hist is not None and 'color_hist' in db_entry and db_entry['color_hist'] is not None:
        try:
            # Calculate histogram intersection
            hist_score = cv2.compareHist(color_hist, db_entry['color_hist'], cv2.HISTCMP_INTERSECT)
            # Normalize to 0-1
            hist_score = hist_score / sum(color_hist)
            scores.append(float(hist_score))
            weights.append(FEATURE_WEIGHTS['color_hist'])
            features_used.append('color_hist')
        except:
            pass
    
    # Compute weighted average score
    if scores and weights:
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        return weighted_score, features_used
    return 0.0, []

# Main tracking loop
frame_count = 0
cap = cv2.VideoCapture(VIDEO_PATH)

# Get total frame count for progress bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
max_frames = min(total_frames, 100)  # Process max 100 frames or total frames, whichever is smaller

# Use tqdm for progress bar
with tqdm(total=max_frames, desc="Processing frames", unit="frame") as progress_bar:
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        tracks = tracker.process_frame(frame)
        
        # For each track, get the assigned identity (if any)
        for track_id, track_info in tracks.items():
            # Get features from tracker histories
            appearance = tracker.feature_history.get(track_id, [None])[-1]
            opengait = tracker.gait_history.get(track_id, [None])[-1]  # OpenGait features
            skeleton_data = tracker.skeleton_history.get(track_id, [])  # Raw skeleton data for the last few frames
            body_ratios = tracker.body_ratio_history.get(track_id, [None])[-1]
            
            # Compute skeleton-based gait features if we have enough skeleton data
            skeleton_gait = None
            if len(skeleton_data) >= 8:  # Need enough frames for gait analysis
                try:
                    # Extract last 10 frames of skeleton data
                    recent_skeletons = skeleton_data[-10:]
                    gait_dict, _ = compute_gait_features(recent_skeletons)
                    skeleton_gait = gait_dict
                except Exception as e:
                    print(f"Error computing skeleton gait: {e}")
            
            # Fetch color_hist, height, context from track_info or histories if available
            color_hist = track_info.get('color_hist') if isinstance(track_info, dict) else None
            height = track_info.get('height') if isinstance(track_info, dict) else None
            context = track_info.get('context') if isinstance(track_info, dict) else None
            
            # Get industrial features from track_info or history
            industrial_pose = track_info.get('industrial_pose') if isinstance(track_info, dict) else None
            industrial_gait = track_info.get('industrial_gait') if isinstance(track_info, dict) else None
            industrial_color = track_info.get('industrial_color') if isinstance(track_info, dict) else None
            
            # Use our enhanced matching with industrial features
            best_score = 0
            best_id = None
            best_features = []
            second_best_score = 0
            second_best_id = None
            second_best_features = []
            
            # Evaluate all candidates using our industrial-enhanced matching
            for db_id, db_entry in identity_db.items():
                custom_score, features_used = compute_match_with_gait_features(
                    appearance, opengait, skeleton_gait, body_ratios, 
                    industrial_pose, industrial_gait, industrial_color,
                    color_hist, height, context, db_entry
                )
                
                if custom_score > best_score:
                    second_best_score = best_score
                    second_best_id = best_id
                    second_best_features = best_features
                    best_score = custom_score
                    best_id = db_id
                    best_features = features_used
                elif custom_score > second_best_score:
                    second_best_score = custom_score
                    second_best_id = db_id
                    second_best_features = features_used
            
            matched_id = best_id
            score = best_score
            second_id = second_best_id
            second_score = second_best_score

            if matched_id is not None:
                # Track history of identifications for this track
                if track_id not in identification_history:
                    identification_history[track_id] = []
                    
                # Add current match to history
                identification_history[track_id].append((matched_id, score))
                
                # Only keep recent history
                if len(identification_history[track_id]) > HISTORY_WINDOW:
                    identification_history[track_id].pop(0)
                
                # Use majority voting for more stable identification
                if len(identification_history[track_id]) >= 3:  # Only after collecting some history
                    id_counts = {}
                    for hist_id, hist_score in identification_history[track_id]:
                        id_counts[hist_id] = id_counts.get(hist_id, 0) + 1
                    
                    # Get the most frequent ID in recent history
                    stable_id = max(id_counts.items(), key=lambda x: x[1])[0]
                    
                    # Only override if the current match is significantly weaker
                    if id_counts[stable_id] >= len(identification_history[track_id]) * 0.6:  # 60% majority
                        if matched_id != stable_id and score < 0.55:  # Current match isn't very confident
                            print(f"Stabilizing: {matched_id} -> {stable_id} (based on history)")
                            matched_id = stable_id
                
                # Get the name from the database or name mapping
                if matched_id in identity_db and "name" in identity_db[matched_id]:
                    matched_name = identity_db[matched_id]["name"]
                else:
                    matched_name = name_mapping.get(matched_id, str(matched_id))
                
                # Enhance the output to show which features contributed
                features_str = f" using {', '.join(best_features)}" if best_features else ""
                print(f"Track {track_id}: Best match is {matched_name} (score={score:.2f}){features_str}", end='')
                
                # Log confusion cases where scores are very close
                if second_id is not None and (score - second_score) < 0.03:
                    second_name = ""
                    if second_id in identity_db and "name" in identity_db[second_id]:
                        second_name = identity_db[second_id]["name"]
                    else:
                        second_name = name_mapping.get(second_id, str(second_id))
                    print(f" | Close match! 2nd best: {second_name} (score={second_score:.2f})")
                elif second_id is not None:
                    second_name = ""
                    if second_id in identity_db and "name" in identity_db[second_id]:
                        second_name = identity_db[second_id]["name"]
                    else:
                        second_name = name_mapping.get(second_id, str(second_id))
                    print(f" | 2nd best: {second_name} (score={second_score:.2f})")
                else:
                    print()
            else:
                # Only print "no good match" when debugging is needed
                # print(f"Track {track_id}: No good match found (best score={score:.2f})")
                pass
            
            # Debug the feature similarity between the current track and potential matches
            if frame_count % 20 == 0 and matched_id is not None:  # Periodically check
                print(f"  Feature analysis for Track {track_id}:")
                for db_id in [1, 2, 3, 4]:
                    if db_id in identity_db and appearance is not None and "appearance" in identity_db[db_id]:
                        try:
                            # Calculate cosine similarity with safe conversion
                            app_np = to_numpy(appearance)
                            db_feat_np = to_numpy(identity_db[db_id]["appearance"])
                            
                            if app_np is not None and db_feat_np is not None:
                                similarity = np.dot(app_np, db_feat_np) / (np.linalg.norm(app_np) * np.linalg.norm(db_feat_np))
                                similarity = np.power(max(0, similarity), SIMILARITY_TEMPERATURE)
                                print(f"    Similarity to {name_mapping.get(db_id, str(db_id))}: {similarity:.4f}")
                        except Exception as e:
                            print(f"    Error computing similarity to {db_id}: {e}")

        # Update the visualization to display names instead of IDs and scores
        for track_id, track_info in tracks.items():
            # Ensure track_info is a dictionary
            if isinstance(track_info, tuple):
                track_info = {
                    'bbox': track_info[0],
                    'matched_id': track_info[1]
                }

            matched_id = track_info.get('matched_id')
            if matched_id is not None:
                # Get the name from the database or name mapping
                if matched_id in identity_db and "name" in identity_db[matched_id]:
                    matched_name = identity_db[matched_id]["name"]
                else:
                    matched_name = name_mapping.get(matched_id, str(matched_id))

                # Ensure only the name is displayed
                if isinstance(matched_name, dict):
                    matched_name = matched_name.get("name", "Unknown")
                matched_name = str(matched_name)  # Convert to string if not already
                x, y, w, h = map(int, track_info['bbox'])  # Assuming bbox is in track_info
                cv2.putText(frame, matched_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Optionally visualize
        vis_frame = tracker.visualize(frame, tracks)
        cv2.imshow("Identification", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        # Update progress bar
        progress_bar.update(1)

cap.release()
cv2.destroyAllWindows()