import os
import torch
import numpy as np
from .backbones import Plain, ResGCN, ResNet, GCN, UNet

# Example: update these paths as needed
DEFAULT_CFG = os.path.join(os.path.dirname(__file__), '../../gaitbase_da_casiab.yaml')
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), '../../checkpoints/GaitBase_DA-60000.pt')

class OpenGaitEmbedder:
    def __init__(self, cfg_path=DEFAULT_CFG, weights_path=DEFAULT_WEIGHTS, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: Load config and build model according to config
        # For demo, we use Plain backbone. Replace with actual model as needed.
        self.model = Plain(in_channels=1, out_channels=64)  # Example, adjust as per config
        self.model.eval()
        self.model.to(self.device)
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device), strict=False)
        else:
            raise FileNotFoundError(f"Weights not found: {weights_path}")

    def extract(self, silhouettes):
        """
        silhouettes: numpy array or list of shape (N, H, W), values in [0, 255] or [0, 1]
        Returns: gait embedding as numpy array
        """
        if isinstance(silhouettes, list):
            silhouettes = np.stack(silhouettes, axis=0)
        if silhouettes.max() > 1:
            silhouettes = silhouettes / 255.0
        silhouettes = torch.from_numpy(silhouettes).float().unsqueeze(1)  # (N, 1, H, W)
        silhouettes = silhouettes.to(self.device)
        with torch.no_grad():
            feat = self.model(silhouettes)
            # Global average pooling
            embedding = feat.mean(dim=[0, 2, 3]).cpu().numpy()
        return embedding

# Usage example:
# embedder = OpenGaitEmbedder()
# embedding = embedder.extract(list_of_silhouette_images)