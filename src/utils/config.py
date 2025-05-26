MODEL_CONFIG_MAP = {
    "B16": {
        "1K": "torchvision.ViT_B_16_Weights.IMAGENET1K_V1",
        "21K": "timm.vit_base_patch16_224.augreg_in21k.PRETRAINED",
    },
    "L16": {
        "1K": "torchvision.ViT_L_16_Weights.IMAGENET1K_V1",
        "21K": "timm.vit_large_patch16_224.augreg_in21k.PRETRAINED",
    },
    "B32": {
        "1K": "torchvision.ViT_B_32_Weights.IMAGENET1K_V1",
        "21K": "timm.vit_base_patch32_224.augreg_in21k.PRETRAINED",
    },
    "L32": {
        "1K": "torchvision.ViT_L_32_Weights.IMAGENET1K_V1",
        "21K": "timm.vit_large_patch32_224.orig_in21k.PRETRAINED",
    },
}

OPTIMIZER_CONFIG_MAP = {
    "adadelta": {
        "1": {"lr": 0.1, "ne": 15, "mbs": 16},
        "2": {"lr": 0.1, "ne": 20, "mbs": 32},
        "3": {"lr": 0.05, "ne": 15, "mbs": 32},
        "4": {"lr": 0.05, "ne": 15, "mbs": 32},
    },
    "adam": {
        "1": {"lr": 1e-4, "ne": 25, "mbs": 16},
        "2": {"lr": 1e-4, "ne": 20, "mbs": 32},
        "3": {"lr": 5e-5, "ne": 15, "mbs": 32},
        "4": {"lr": 5e-5, "ne": 1, "mbs": 8}, # For test
    },
    "rmsprop": {
        "1": {"lr": 1e-4, "ne": 25, "mbs": 16},
        "2": {"lr": 1e-4, "ne": 20, "mbs": 32},
        "3": {"lr": 5e-5, "ne": 15, "mbs": 32},
    },
}
