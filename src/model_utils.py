import torch
import torch.nn as nn
import torchvision.models as tv_models

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import numpy as np

import random

def load_vit_model(device: torch.device,
                   model_config: str,
                   num_classes: int):

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

def load_cnn_model(device: torch.device, model_config: str, num_classes: int):
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

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)