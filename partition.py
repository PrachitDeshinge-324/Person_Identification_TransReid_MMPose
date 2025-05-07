import os
import json

sil_dir = "output/silhouettes"
tracks = [d for d in os.listdir(sil_dir) if d.startswith("track_")]
partition = {"TRAIN_SET": [], "TEST_SET": tracks}
with open(os.path.join(sil_dir, "partition.json"), "w") as f:
    json.dump(partition, f)
print("partition.json created with tracks:", tracks)