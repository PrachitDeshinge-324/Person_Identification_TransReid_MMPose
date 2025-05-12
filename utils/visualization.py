import colorsys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

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
        logging.getLogger(__name__).warning("draw_skeleton received only %d keypoints, expected 17 for COCO.", len(keypoints))
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

def visualize_identity_assignments_with_candidates(frame, identity_manager, active_tracks_data):
    """
    Creates a visualization panel showing current identity assignments with candidate options.
    
    Args:
        frame: The current video frame
        identity_manager: The global IdentityManager instance
        active_tracks_data: Dict of track data
    
    Returns:
        Modified frame with an identity assignment panel
    """
    # Create a sidebar panel for identity assignments
    frame_height, frame_width = frame.shape[:2]
    sidebar_width = 350  # Wider to show candidates
    panel_height = frame_height
    
    # Create a black sidebar
    sidebar = np.zeros((panel_height, sidebar_width, 3), dtype=np.uint8)
    
    # Get all current identity assignments
    identity_assignments = identity_manager.get_all_identities()
    
    # Group tracks by assigned identity
    identities_to_tracks = {}
    for track_id, (name, conf) in identity_assignments.items():
        if name != "Unknown" and name != "Pending...":
            if name not in identities_to_tracks:
                identities_to_tracks[name] = []
            identities_to_tracks[name].append((track_id, conf))
    
    # Draw header
    cv2.putText(sidebar, "Identity Assignments", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.line(sidebar, (10, 40), (sidebar_width-10, 40), (100, 100, 100), 1)
    
    y_pos = 70
    
    # Draw identity assignments, highlighting conflicts
    for name, tracks in sorted(identities_to_tracks.items()):
        # Determine color based on whether there's a conflict (multiple tracks)
        if len(tracks) > 1:
            # Conflict - red
            color = (0, 0, 255)
            status = "CONFLICT"
        else:
            # No conflict - green
            color = (0, 255, 0)
            status = "OK"
        
        # Draw identity name
        cv2.putText(sidebar, f"{name} ({status})", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_pos += 25
        
        # Draw associated tracks
        for track_id, conf in tracks:
            cv2.putText(sidebar, f"  Track {track_id}: {conf:.2f}", (20, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 250), 1)
            y_pos += 20
            
            # Show candidate alternatives from top matches
            if hasattr(identity_manager, 'top_candidates') and track_id in identity_manager.top_candidates:
                candidates = identity_manager.top_candidates[track_id]
                for i, (cand_name, cand_conf, _) in enumerate(candidates):
                    if cand_name != name:  # Only show alternatives
                        cand_color = (100, 100, 100) if cand_conf < conf else (100, 180, 250)
                        cv2.putText(sidebar, f"    Alt {i+1}: {cand_name} ({cand_conf:.2f})", 
                                   (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cand_color, 1)
                        y_pos += 15
            
            y_pos += 5
        
        y_pos += 10
    
    # Add statistics
    y_pos += 20
    cv2.putText(sidebar, "Statistics:", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    y_pos += 25
    
    # Count unique identities
    unique_ids = len(identities_to_tracks)
    cv2.putText(sidebar, f"Unique IDs: {unique_ids}", (20, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 250), 1)
    y_pos += 20
    
    # Count active tracks
    active_tracks = len(active_tracks_data)
    cv2.putText(sidebar, f"Active tracks: {active_tracks}", (20, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 250), 1)
    y_pos += 20
    
    # Count conflicts
    conflicts = sum(1 for tracks in identities_to_tracks.values() if len(tracks) > 1)
    cv2.putText(sidebar, f"ID conflicts: {conflicts}", (20, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 250), 1)
    
    # Add global optimization status
    y_pos += 30
    cv2.putText(sidebar, "Global Optimization:", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    y_pos += 25
    
    # Show optimization status
    candidates_count = len(identity_manager.top_candidates) if hasattr(identity_manager, 'top_candidates') else 0
    cv2.putText(sidebar, f"Tracks with candidates: {candidates_count}", (20, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 250), 1)
    
    # Combine the original frame with the sidebar
    combined = np.zeros((frame_height, frame_width + sidebar_width, 3), dtype=np.uint8)
    combined[:, :frame_width, :] = frame
    combined[:, frame_width:, :] = sidebar
    
    return combined

def visualize_identity_assignments(frame, identity_manager, active_tracks_data):
    """
    Creates a visualization panel showing current identity assignments.
    
    Args:
        frame: The current video frame
        identity_manager: The global IdentityManager instance
        active_tracks_data: Dict of track data
    Returns:
        Modified frame with an identity assignment panel
    """
    # Create a sidebar panel for identity assignments
    frame_height, frame_width = frame.shape[:2]
    sidebar_width = 280
    panel_height = frame_height
    
    # Create a black sidebar
    sidebar = np.zeros((panel_height, sidebar_width, 3), dtype=np.uint8)
    
    # Get all current identity assignments
    identity_assignments = identity_manager.get_all_identities()
    
    # Group tracks by assigned identity
    identities_to_tracks = {}
    for track_id, (name, conf) in identity_assignments.items():
        if name != "Unknown" and name != "Pending...":
            if name not in identities_to_tracks:
                identities_to_tracks[name] = []
            identities_to_tracks[name].append((track_id, conf))
    
    # Draw header
    cv2.putText(sidebar, "Identity Assignments", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.line(sidebar, (10, 40), (sidebar_width-10, 40), (100, 100, 100), 1)
    
    y_pos = 70
    
    # Draw identity assignments, highlighting conflicts
    for name, tracks in sorted(identities_to_tracks.items()):
        # Determine color based on whether there's a conflict (multiple tracks)
        if len(tracks) > 1:
            # Conflict - red
            color = (0, 0, 255)
            status = "CONFLICT"
        else:
            # No conflict - green
            color = (0, 255, 0)
            status = "OK"
        
        # Draw identity name
        cv2.putText(sidebar, f"{name} ({status})", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_pos += 25
        
        # Draw associated tracks
        for track_id, conf in tracks:
            cv2.putText(sidebar, f"  Track {track_id}: {conf:.2f}", (20, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 250), 1)
            y_pos += 20
        
        y_pos += 10
    
    # Add statistics
    y_pos += 20
    cv2.putText(sidebar, "Statistics:", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    y_pos += 25
    
    # Count unique identities
    unique_ids = len(identities_to_tracks)
    cv2.putText(sidebar, f"Unique IDs: {unique_ids}", (20, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 250), 1)
    y_pos += 20
    
    # Count active tracks
    active_tracks = len(active_tracks_data)
    cv2.putText(sidebar, f"Active tracks: {active_tracks}", (20, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 250), 1)
    y_pos += 20
    
    # Count conflicts
    conflicts = sum(1 for tracks in identities_to_tracks.values() if len(tracks) > 1)
    cv2.putText(sidebar, f"ID conflicts: {conflicts}", (20, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 250), 1)
    
    # Combine the original frame with the sidebar
    combined = np.zeros((frame_height, frame_width + sidebar_width, 3), dtype=np.uint8)
    combined[:, :frame_width, :] = frame
    combined[:, frame_width:, :] = sidebar
    
    return combined