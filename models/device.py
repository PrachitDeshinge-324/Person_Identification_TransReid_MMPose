import torch

# Device selection utility

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