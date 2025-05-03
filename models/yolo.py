import os
from ultralytics import YOLO
from .device import get_best_device

class YOLOv8Tracker:
    def __init__(self, weights_path, device=None, conf_threshold=0.3):
        """
        Initialize YOLOv8 tracker
        
        Args:
            weights_path: Path to pre-trained weights (or 'yolov8m.pt' for default)
            device: Device to run the model on
            conf_threshold: Confidence threshold for detections
        """
        self.device = device if device is not None else get_best_device()
        self.conf_threshold = conf_threshold
        
        # Initialize YOLOv8 model
        if os.path.exists(weights_path):
            # Load from local file
            self.model = YOLO(weights_path)
        else:
            # Load default YOLOv8m model
            self.model = YOLO('yolov8m.pt')
        
        # Set device
        self.model.to(self.device)
        
        # For tracking
        self.track_history = {}
    
    def process_frame(self, frame):
        """
        Process a frame through YOLOv8 to get detections and tracks
        
        Args:
            frame: BGR image (OpenCV format)
        
        Returns:
            List of (track_id, bbox, conf) tuples
        """
        # Run YOLOv8 tracking on the frame
        try:
            # First attempt: use basic tracking without specifying a tracker
            results = self.model.track(
                frame,
                persist=True,          # Enable tracking persistence
                classes=0,             # Class 0 is person in COCO
                conf=self.conf_threshold,  # Confidence threshold
                iou=0.5                # IOU threshold for NMS
                # Don't specify tracker to use default
            )
        except Exception as e:
            print(f"Warning: Tracking failed with error: {str(e)}")
            print("Falling back to detection only...")
            # Fallback to detection without tracking
            results = self.model(
                frame,
                classes=0,             # Class 0 is person in COCO
                conf=self.conf_threshold,  # Confidence threshold
                iou=0.5                # IOU threshold for NMS
            )
        
        detections = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            # Check if tracking info is available
            if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().numpy()
                
                # Create (track_id, bbox, conf) tuples
                for i, ((x1, y1, x2, y2), conf) in enumerate(zip(boxes, confs)):
                    track_id = int(track_ids[i])
                    detections.append((track_id, (x1, y1, x2, y2), conf))
            else:
                # If tracking failed or not enabled, use detection index as placeholder
                for i, ((x1, y1, x2, y2), conf) in enumerate(zip(boxes, confs)):
                    detections.append((-1, (x1, y1, x2, y2), conf))  # -1 indicates no tracking ID
        
        return detections