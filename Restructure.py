import os
import shutil

sil_dir = "output/silhouettes"
for track in os.listdir(sil_dir):
    track_path = os.path.join(sil_dir, track)
    if not os.path.isdir(track_path):
        continue
    # Create dummy CASIA-B structure: 001/nm-01/
    seq_dir = os.path.join(track_path, "001", "nm-01")
    os.makedirs(seq_dir, exist_ok=True)
    # Move all PNGs into the new folder
    for fname in os.listdir(track_path):
        if fname.endswith(".png"):
            src = os.path.join(track_path, fname)
            dst = os.path.join(seq_dir, fname)
            shutil.move(src, dst)
print("Restructuring complete.")