
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

VIT_MODEL_CONFIG_MAP = {
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
    "B8": {
        "dino": "timm.vit_base_patch8_224.dino.PRETRAINED",
        "21K": "timm.vit_base_patch8_224.augreg_in21k.PRETRAINED", 
    },
    "S8": {
        "dino": "timm.vit_small_patch8_224.dino.PRETRAINED",
    },
    "deit3": {
        "S16": "timm.deit3_small_patch16_224.fb_in22k_ft_in1k.PRETRAINED",
        "M16": "timm.deit3_medium_patch16_224.fb_in22k_ft_in1k.PRETRAINED",
        "B16": "timm.deit3_base_patch16_224.fb_in22k_ft_in1k.PRETRAINED",
        "L16": "timm.deit3_large_patch16_224.fb_in22k_ft_in1k.PRETRAINED",
    },
    "swin": {
        "Bp4w7": "timm.swin_base_patch4_window7_224.ms_in22k.PRETRAINED",
        "Lp4w7": "timm.swin_large_patch4_window7_224.ms_in22k.PRETRAINED",
    }
}

CNN_MODEL_CONFIG_MAP = {
    "VGG16": "torchvision.VGG16_Weights.IMAGENET1K_V1",
    "ResNet50": "torchvision.ResNet50_Weights.IMAGENET1K_V1",
    "GoogLeNet": "torchvision.GoogLeNet_Weights.IMAGENET1K_V1",
    "MobileNetV2": "torchvision.MobileNet_V2_Weights.IMAGENET1K_V1",
    "DenseNet121": "torchvision.DenseNet121_Weights.IMAGENET1K_V1",
    "ConvNeXt_Large": "torchvision.ConvNeXt_Large_Weights.IMAGENET1K_V1",
    "EfficientNetB0":"torchvision.EfficientNet_B0_Weights.IMAGENET1K_V1",
}

OTHER_MODEL_CONFIG_MAP = {} # In the Future, we can define the other model that not belong to both cnn and vit

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

DEFAULT_SIZE = (248, 248)
CROP_SIZE = 224

TRANSFORMS_CONFIG_MAP = {
    "vit": lambda vit_transform: transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        vit_transform,
    ]),
    "cnn": transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(DEFAULT_SIZE, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
}


DATASET_CONFIG_MAP = {
    "Br35H": {
        "path": "/home/citiai-cygnus/AlexServer/phase2/BrainTumorMedNet/dts/Br35H",
        "dataloader-mode": "binary_class",
        "num_classes": 2,
        "train-val-test_ratio": (0.7, 0.1, 0.2),
    },
    "BT_Large_4c": {
        "path": "/home/citiai-cygnus/AlexServer/phase2/BrainTumorMedNet/dts/BT_Large_4c",
        "dataloader-mode": "split_folder",
        "num_classes": 4,
        "train-val-test_ratio": (0.7, 0.1, 0.2),
    },
    "Figshare": {
        "path": "/home/citiai-cygnus/AlexServer/phase2/BrainTumorMedNet/dts/Figshare",
        "dataloader-mode": "flat",
        "num_classes": 3,
        "train-val-test_ratio": (0.7, 0.1, 0.2),
    },
    "Figshare_x10": {
        "path": "/home/citiai-cygnus/AlexServer/phase2/BrainTumorMedNet/dts/Figshare_x10",
        "dataloader-mode": "flat",
        "num_classes": 3,
        "train-val-test_ratio": (0.7, 0.1, 0.2),
    },
    "MRI-Scan": {
        "path": "/home/citiai-cygnus/AlexServer/phase2/BrainTumorMedNet/dts/MRI_Scan",
        "dataloader-mode": "split_folder",
        "num_classes": 4,
        "train-val-test_ratio": (0.7, 0.1, 0.2),
    },
    "MRI-Scan_2": {
        "path": "/home/citiai-cygnus/AlexServer/phase2/BrainTumorMedNet/dts/MRI_Scan_2",
        "dataloader-mode": "split_folder",
        "num_classes": 4,
        "train-val-test_ratio": (0.7, 0.1, 0.2),
    }
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
        "5": {"lr": 1e-4, "ne": 25, "mbs": 8} # For CNNs
    },
    "rmsprop": {
        "1": {"lr": 1e-4, "ne": 25, "mbs": 16},
        "2": {"lr": 1e-4, "ne": 20, "mbs": 32},
        "3": {"lr": 5e-5, "ne": 15, "mbs": 32},
    },
}

METRICS_SUPPORTED = [
    "accuracy",
    "confusion_counts",
    "confusion_matrix",
    "precision_binary",
    "recall_binary",
    "sensitivity",
    "specificity",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "roc_auc_binary",
    "pr_auc_binary",
    "roc_auc_ovr",
    "pr_auc_ovr",
    "mcc_binary",
    "kappa_binary",
    "full", # all above metrics
]


OPTIMIZER_SUPPORTED = ["adadelta", "adam", "rmsprop"]

