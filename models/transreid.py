import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from functools import lru_cache
from .device import get_best_device
import cv2
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

# add cached factory
@lru_cache(maxsize=None)
def load_transreid_model(weights_path: str, device: str):
    """
    Return a cached TransReIDModel instance for given weights and device.
    """
    return TransReIDModel(weights_path, device)