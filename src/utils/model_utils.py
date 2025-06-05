import torch
import torch.nn as nn
import torchvision.models as tv_models

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from typing import List
import numpy as np

import random

def load_vit_model_config(model_config: str, num_classes: int):
    """
    Load and customize a Vision Transformer model from torchvision or timm.

    Args:
        device:        torch.device to place the model on.
        model_config:  Dot-separated string identifying source and pretrained weights,
                       e.g. "torchvision.ViT_B_16_Weights.IMAGENET1K_V1"
                       or "timm.vit_base_patch16_224".
        num_classes:   Number of output classes for the classification head.

    Returns:
        model:             The ViT model with its head replaced for `num_classes`.
        model_transforms:  The corresponding torchvision/timm transform pipeline.

    Raises:
        ValueError: If `model_config` is malformed or the source is unknown.
    """
    parts = model_config.split(".")
    if len(parts) < 2:
        raise ValueError("Wrong model_config!")
    
    source = parts[0].lower()  
    
    if source == "torchvision":
        weights_class_name = parts[1]
        model_name = weights_class_name.lower().replace("_weights", "")
        if len(parts) == 3:
            weights_class = getattr(tv_models, weights_class_name)
            weights = getattr(weights_class, parts[2])
            
            model_transforms = weights.transforms()

        else:
            weights = None
        model = getattr(tv_models, model_name)(weights=weights)
    
    elif source == "timm":
        model_name = parts[1]+"."+parts[2]
        print("model name:",model_name)
        pretrained = (len(parts) >= 3)
        model = timm.create_model(model_name, pretrained=pretrained)

        # Get data configuration for the model
        config = resolve_data_config({}, model=model)

        # Create transform based on config
        model_transforms = create_transform(**config)
    else:
        raise ValueError("MODEL_SOURCE shouldbe 'torchvision' or 'timm'.")
    
    # model = model.to(device)
    
    if source == "torchvision":
        in_features = model.heads.head.in_features
        # model.heads.head = nn.Linear(in_features, num_classes).to(device)      
        model.heads.head = nn.Linear(in_features, num_classes)

    elif source == "timm":
        # check for ViT orig
        if isinstance(model.head, nn.Identity):
            print("ViT original head")
            in_features = model.num_features
            # model.head = nn.Linear(in_features, num_classes).to(device)
            model.head = nn.Linear(in_features, num_classes)

        else:
            in_features = model.head.in_features
            # model.head = nn.Linear(in_features, num_classes).to(device)
            model.head = nn.Linear(in_features, num_classes)

    return model,model_transforms

def load_cnn_model_config(device: torch.device, model_config: str, num_classes: int):
    """
    Load and customize a CNN model from torchvision or timm.

    Args:
        device:        torch.device to place the model on.
        model_config:  Dot-separated string identifying source and pretrained weights,
                       e.g. "torchvision.ResNet50_Weights.IMAGENET1K_V1"
                       or "timm.resnet50".
        num_classes:   Number of output classes for the classification head.

    Returns:
        model:             The CNN model with its final FC/classifier layer reset to `num_classes`.
        model_transforms:  The corresponding torchvision/timm transform pipeline, or None if unavailable.

    Raises:
        ValueError: If `model_config` is malformed or the source is unknown.
                   Also if the model’s classifier structure cannot be identified.
    """
    parts = model_config.split(".")
    if len(parts) < 2:
        raise ValueError("Incorrect model_config!")
    
    source = parts[0].lower()
    
    # ---- torchvision branch ----
    if source == "torchvision":
        weights = None
        model_transforms = None
        if len(parts) >= 3:
            weights_class_name = parts[1]
            model_name = weights_class_name.lower().replace("_weights", "")
            weights_class = getattr(tv_models, weights_class_name)
            weights = getattr(weights_class, parts[2])
            model_transforms = weights.transforms()
        else:
            model_name = parts[1].lower()
        model = getattr(tv_models, model_name)(weights=weights)

        # Update classifier/head
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                if isinstance(model.classifier[-1], nn.Linear):
                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(in_features, num_classes)
                else:
                    raise ValueError("Unsupported classifier structure!")
            elif isinstance(model.classifier, nn.Linear):
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
            else:
                raise ValueError("Unknown classifier type for this torchvision model!")
        else:
            raise ValueError("Unable to determine the structure of this torchvision model!")

        try:
            model_transforms = model_transforms
        except:
            model_transforms = None

    # ---- timm branch ----
    elif source == "timm":
        if len(parts) >= 3:
            model_name = parts[1]
            pretrained = True
        else:
            model_name = parts[1]
            pretrained = False
        
        model = timm.create_model(model_name, pretrained=pretrained)
        
        try:
            config = resolve_data_config({}, model=model)
            model_transforms = create_transform(**config)
        except Exception:
            model_transforms = None

        # Update classifier/head
        if hasattr(model, "reset_classifier"):
            model.reset_classifier(num_classes)
        elif hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                if isinstance(model.classifier[-1], nn.Linear):
                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(in_features, num_classes)
                else:
                    raise ValueError("Unsupported classifier structure!")
            elif isinstance(model.classifier, nn.Linear):
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
            else:
                raise ValueError("Unknown classifier type for this timm model!")
        else:
            raise ValueError("Unable to determine the structure of this timm model!")

    else:
        raise ValueError("MODEL_SOURCE must be 'torchvision' or 'timm'.")
    
    model = model.to(device)
    return model, model_transforms

def model_info_retrieval(model: nn.Module) -> List:
    """
    Extract architecture details from a ViT or CNN model instance.

    Returns a list containing:
        [model_type_str,
         patch_size (for ViT) or None,
         embedding_dimension or None,
         number_of_layers (ViT) or None,
         number_of_heads (ViT) or None,
         total_parameter_count,
         framework_name ("torchvision", "timm", or "Unknown")]

    If the model is not recognized as a ViT, returns
        ["CNN-based Model", None, None, None, None, total_params, "Unknown"].

    Always guarantees returning seven elements.
    """
    try:
        is_torchvision_vit = hasattr(model, "conv_proj") and hasattr(model, "encoder")
        is_timm_vit = hasattr(model, "patch_embed") and hasattr(model, "blocks")

        total_params = sum(p.numel() for p in model.parameters())

        if not (is_torchvision_vit or is_timm_vit):
            return ["CNN-based Model", None, None, None, None, total_params, "Unknown"]

        if is_torchvision_vit:
            embed_dim = model.conv_proj.out_channels
            patch_size = model.conv_proj.kernel_size[0]
            num_layers = len(model.encoder.layers)
            num_heads = model.encoder.layers[0].self_attention.num_heads
            framework = "torchvision"
        else:
            embed_dim = model.patch_embed.proj.out_channels
            patch_size = model.patch_embed.proj.kernel_size[0]
            num_layers = len(model.blocks)
            num_heads = model.blocks[0].attn.num_heads
            framework = "timm"

        vit_type = (
            "ViT-Base" if embed_dim == 768
            else "ViT-Large" if embed_dim == 1024
            else f"ViT-Custom (embed_dim={embed_dim})"
        )

        return [vit_type, patch_size, embed_dim, num_layers, num_heads, total_params, framework]

    except Exception:
        total_params = sum(p.numel() for p in model.parameters())
        return ["Unknown Model", None, None, None, None, total_params, "Unknown"]


