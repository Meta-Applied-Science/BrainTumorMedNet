"""
Utility for setting random seeds to ensure reproducibility.
"""

import random
import numpy as np
import torch


def set_seeds(seed: int = 42) -> None:
    """
    Sets seed for Python, NumPy and PyTorch (CPU & CUDA).

    Args:
        seed: Integer seed value (default is 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
