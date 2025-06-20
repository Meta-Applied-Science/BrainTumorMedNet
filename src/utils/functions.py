import random
import numpy as np
import torch

import torch
import torchvision
import torchvision.transforms as transforms

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

# === XAI Task ===
# Prepare data transformation pipeline

def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x


def inv_transform(mean:list, std:list) -> transforms.Compose:
    transform  = [
        torchvision.transforms.Lambda(nhwc_to_nchw),
        torchvision.transforms.Normalize(
            mean=(-1 * np.array(mean) / np.array(std)).tolist(),
            std=(1 / np.array(std)).tolist(),
        ),
        torchvision.transforms.Lambda(nchw_to_nhwc),]
    
    return torchvision.transforms.Compose(transform)


def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    return tensor * std + mean

def load_model_checkpoint(model: torch.nn.Module, 
                          checkpoint_path: str, 
                          device: torch.device) -> torch.nn.Module:
    
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Remap keys cho classifier head nếu cần
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("heads.") and not k.startswith("heads.head."):
            new_key = "heads.head." + k[len("heads."):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    # Load state dict đã được remap vào model
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model


# Type
# pretrained imageNet
# Patch size    
