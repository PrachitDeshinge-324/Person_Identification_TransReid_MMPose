#!/bin/bash
# Run script for Person_New project with MMPose and all required arguments

# Optional: activate virtual environment if exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Install dependencies
# pip install -r requirements.txt

# Download MMPose config and checkpoint if not present (edit paths as needed)
# if [ ! -f "rtmpose-l_8xb256-420e_humanart-256x192.py" ]; then
#   echo "Downloading RTMPose config..."
#   curl -O https://github.com/open-mmlab/mmpose/raw/main/configs/body_2d_keypoint/rtmpose/rtmpose-s-imagenet-pt-body7-256x192.py
# fi
# if [ ! -f "./weights/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth" ]; then
#   echo "Downloading RTMPose checkpoint..."
#   curl -O https://github.com/open-mmlab/mmpose/raw/main/configs/body_2d_keypoint/rtmpose/rtmpose-s-imagenet-pt-body7-256x192.pth
# fi

# Run the main script with all required arguments
python person_tracker.py \
  --video 1.mp4 \
  --output output.mp4 \
  --yolo_weights weights/yolov12m.pt \
  --transreid_weights weights/transreid_vitbase.pth \
  --mmpose_weights weights/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth \
  --conf 0.3 \
  --reid_threshold 0.7 \
  --device mps \
  --appearance_weight 0.4 \
  --gait_weight 0.5 \
  --body_weight 0.15 \
  --color_weight 0.15 \
  --height_weight 0.1 \
  --context_weight 0.05 \
  --show_window 
  # --debug_visualize  # Uncomment to enable debug visualization
