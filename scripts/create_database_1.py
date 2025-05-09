import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import pickle
import numpy as np
import argparse
import logging
from collections import defaultdict
from tqdm import tqdm  # Import tqdm for progress bars
from ultralytics.utils import LOGGER as yolo_logger
from person_tracker import PersonTracker  # Import robust tracker

# Add argument parser for command-line options
parser = argparse.ArgumentParser(description='Create enhanced identity database from video.')
parser.add_argument('--video', default="input/My Movie.mp4", help='Path to video file')
parser.add_argument('--output', default="identity_database.pkl", help='Output database file')
parser.add_argument('--interactive', action='store_true', help='Use interactive naming')
parser.add_argument('--auto-name', action='store_true', help='Use automatic naming with predefined map')
parser.add_argument('--frames', type=int, default=3000, help='Number of frames to process')
parser.add_argument('--skip', type=int, default=1, help='Frame skip rate')
parser.add_argument('--verbose', action='store_true', help='Show detailed processing logs')
args = parser.parse_args()

VIDEO_PATH = args.video
YOLO_WEIGHTS = "weights/yolo11x.pt"
TRANSREID_WEIGHTS = "weights/transreid_vitbase.pth"
MMPOSE_WEIGHTS = "weights/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth"
OUTPUT_DB = args.output
DEVICE = "mps"  # or "cuda" or "cpu"

yolo_logger.setLevel(logging.WARNING)

# Define predefined name mapping
NAME_MAPPING = {
    1: "Prachit",
    2: "Ashutosh",
    3: "Ojasv",
    4: "Nayan"
}

print(f"Initializing models for enhanced feature extraction...")

# Initialize PersonTracker for robust tracking
tracker = PersonTracker(
    yolo_weights_path=YOLO_WEIGHTS,
    transreid_weights_path=TRANSREID_WEIGHTS,
    device=DEVICE,
    conf_threshold=0.3,
    reid_threshold=0.5,
    mmpose_weights=MMPOSE_WEIGHTS,
    tracker_type='kalman',
    debug_visualize=args.verbose
)

# Dictionary to store features per track
track_features = defaultdict(lambda: {
    "appearance": [],
    "opengait": [],         # OpenGait embeddings (GaitBase)
    "skeleton_gait": [],    # Skeleton-based gait features
    "body_ratios": [],
    "heights": [],
    "contexts": [],
    "color_hists": [],
    "raw_skeletons": [],
    "silhouettes": [],
})

print(f"Processing video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
frame_count = 0

total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // args.skip, args.frames)
print(f"Will process {total_frames} frames (skipping every {args.skip} frames)")

with tqdm(total=total_frames, desc="Processing video frames", unit="frame") as progress_bar:
    while frame_count < args.frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % args.skip != 0:
            continue
        frame_count += 1
        # Use robust tracker
        tracks = tracker.process_frame(frame)
        for tid, track in tracks.items():
            # Aggregate features for each track
            if 'appearance' in track:
                track_features[tid]["appearance"].append(track['appearance'])
            if 'gait' in track and track['gait'] is not None:
                # If gait is a numpy array, treat as OpenGait embedding
                if isinstance(track['gait'], np.ndarray):
                    track_features[tid]["opengait"].append(track['gait'])
                # If gait is a dict, treat as skeleton gait
                elif isinstance(track['gait'], dict):
                    track_features[tid]["skeleton_gait"].append(track['gait'])
            if 'body' in track:
                track_features[tid]["body_ratios"].append(track['body'])
            if 'color_hist' in track:
                track_features[tid]["color_hists"].append(track['color_hist'])
            if 'height' in track:
                track_features[tid]["heights"].append(track['height'])
            if 'context' in track:
                track_features[tid]["contexts"].append(track['context'])
            if 'skeleton' in track:
                track_features[tid]["raw_skeletons"].append(track['skeleton'])
            if 'silhouette' in track:
                track_features[tid]["silhouettes"].append(track['silhouette'])
        progress_bar.update(1)

cap.release()
print(f"Video processing complete. Processed {frame_count} frames.")

# --- Confirm identity for each track before saving ---
import matplotlib.pyplot as plt
track_id_to_name = {}

if args.auto_name:
    print("Using predefined name mapping...")
    for tid in track_features.keys():
        track_id_to_name[tid] = NAME_MAPPING.get(tid, f"Person_{tid}")
        print(f"Assigned name '{track_id_to_name[tid]}' to track ID {tid}")
elif args.interactive:
    print("Starting interactive naming...")
    for tid, feats in track_features.items():
        # Try to get a good representative image for the track
        crop = None
        if feats["raw_skeletons"] and feats["raw_skeletons"][0] and feats["silhouettes"]:
            # Show the first silhouette if available
            sil = feats["silhouettes"][0]
            plt.figure(figsize=(6,3))
            plt.subplot(1,2,1)
            plt.imshow(sil, cmap='gray')
            plt.title(f"Silhouette (Track {tid})")
            plt.axis('off')
        if feats["appearance"]:
            # Show the first appearance feature as an image if possible
            # (Assume you have crops stored, else skip)
            pass
        plt.suptitle(f"Track ID: {tid}")
        plt.show()
        suggested_name = NAME_MAPPING.get(tid, f"Person_{tid}")
        name = input(f"Enter name for track ID {tid} (suggested: {suggested_name}): ")
        track_id_to_name[tid] = name.strip() if name.strip() else suggested_name
else:
    print("Using predefined name mapping (default)...")
    for tid in track_features.keys():
        track_id_to_name[tid] = NAME_MAPPING.get(tid, f"Person_{tid}")

# Attach the name mapping to the database
for tid in track_features:
    track_features[tid]["name"] = track_id_to_name.get(tid, f"Person_{tid}")

# Save the aggregated track features as the database
with open(OUTPUT_DB, "wb") as f:
    pickle.dump(dict(track_features), f)

print(f"Identity database saved to {OUTPUT_DB} with {len(track_features)} tracks. (OpenGait and skeleton gait features are stored separately)")
