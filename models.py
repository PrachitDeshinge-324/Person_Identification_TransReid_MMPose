import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.hub import load_state_dict_from_url
import numpy as np
from PIL import Image
import cv2
import os
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

def get_best_device():
    """
    Returns the best available device: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

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

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box.
        bbox is in the format [x1, y1, x2, y2]
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],      # state transition matrix
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],  
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([    # measurement function
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])

        self.kf.R[2:,2:] *= 10.   # measurement uncertainty
        self.kf.P[4:,4:] *= 1000. # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[4:,4:] *= 0.01  # process uncertainty
        
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_bbox = bbox
        
    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
        self.last_bbox = bbox
        
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        try:
            if((self.kf.x[6]+self.kf.x[2])<=0):
                self.kf.x[6] *= 0.0
                
            self.kf.predict()
            self.age += 1
            if(self.time_since_update>0):
                self.hit_streak = 0
            self.time_since_update += 1
            
            # Get bbox prediction from Kalman filter
            bbox = self._convert_x_to_bbox(self.kf.x)
            
            # Ensure it's a proper numpy array
            if not isinstance(bbox, np.ndarray):
                bbox = np.array(bbox)
                
            # Ensure the shape is correct
            if bbox.size != 4:
                print(f"Warning: Kalman prediction has wrong size: {bbox.size}")
                # Return the last bbox if prediction is invalid
                return self.last_bbox
                
            # Append to history and ensure we're returning a list of 4 floats
            self.history.append(bbox)
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        except Exception as e:
            print(f"Error in Kalman prediction: {str(e)}")
            # Fall back to last known bbox if there's an error
            return self.last_bbox
    
    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self._convert_x_to_bbox(self.kf.x)
    
    def _convert_bbox_to_z(self, bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the center of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h    # scale is area
        r = w / float(h) if h > 0 else 1.0
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _convert_x_to_bbox(self, x):
        """
        Takes a bounding box in the center form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([
            x[0] - w/2.,
            x[1] - h/2.,
            x[0] + w/2.,
            x[1] + h/2.
        ]).reshape((1,4))[0]

class TransReIDModel:
    def __init__(self, weights_path, device=None):
        """
        Initialize TransReID model for person re-identification
        
        Args:
            weights_path: Path to pre-trained weights
            device: Device to run the model on
        """
        self.device = device if device is not None else get_best_device()
        
        # TransReID feature dimension (commonly 768 for ViT-base)
        self.feature_dim = 768
        
        # Initialize model
        self.model = self._load_model(weights_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transforms
        self.transforms = transforms.Compose([
            transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, weights_path):
        """Load the TransReID model or create a placeholder if weights not available"""
        # Check if weights file exists
        if not os.path.exists(weights_path):
            print(f"Warning: TransReID weights file not found at {weights_path}")
            print("Creating a placeholder model for demonstration purposes...")
            return self._create_placeholder_model()
            
        try:
            # Attempt to load the state dict
            print(f"Loading TransReID weights from {weights_path}...")
            state_dict = torch.load(weights_path, map_location=self.device)
            
            # Analyze the state dict to determine model architecture
            if isinstance(state_dict, dict):
                # If it's a dict with 'state_dict' key, it's likely a checkpoint
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # Try to determine the model architecture from keys
                model_keys = list(state_dict.keys())
                if len(model_keys) > 0:
                    print(f"Model has {len(model_keys)} parameters")
                    sample_keys = model_keys[:3]
                    print(f"Sample keys: {sample_keys}")
                    
                    # Detect ViT architecture based on key patterns
                    if any('patch_embed' in k for k in model_keys):
                        print("Detected Vision Transformer architecture")
                        if any('base.1.' in k for k in model_keys):
                            self.feature_dim = 768  # ViT-base typically has 768 features
                        elif any('large.1.' in k for k in model_keys):
                            self.feature_dim = 1024  # ViT-large typically has 1024 features
                    
                    # Here we would try to load the actual TransReID model
                    # Since we don't have the actual implementation, we'll use a placeholder
                    print("Using a placeholder model for demonstration")
                    return self._create_placeholder_model()
            
            # If we can't determine the structure, use a placeholder
            print("Could not determine model structure from weights file")
            print("Using a placeholder model for demonstration")
            return self._create_placeholder_model()
            
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            print("Using a placeholder model for demonstration")
            return self._create_placeholder_model()
    
    def _create_placeholder_model(self):
        """Create a placeholder model that generates feature vectors"""
        print(f"Creating placeholder model with feature dimension {self.feature_dim}")
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, self.feature_dim)
        )
        return model
    
    def extract_features(self, person_crop):
        """
        Extract feature embeddings from a person crop
        
        Args:
            person_crop: Cropped image of a person (BGR format)
        
        Returns:
            Feature embedding tensor
        """
        # Convert BGR to RGB
        if person_crop.size == 0:
            return torch.zeros(self.feature_dim).to(self.device)
            
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)
        
        # Preprocess
        input_tensor = self.transforms(pil_image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(input_tensor)
        
        # Normalize feature vector
        normalized_features = nn.functional.normalize(features, p=2, dim=1)
        
        return normalized_features.squeeze()
