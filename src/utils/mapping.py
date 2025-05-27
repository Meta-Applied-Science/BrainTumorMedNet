from pathlib import Path
from typing import Tuple

from config import (
    VIT_MODEL_CONFIG_MAP,
    CNN_MODEL_CONFIG_MAP,
    DATASET_CONFIG_MAP,
    OPTIMIZER_CONFIG_MAP,
    TRANSFORMS_CONFIG_MAP
)


def get_vit_model_config(patch: str, img_net: str) -> str:
    """
    Retrieve the torchvision ViT weights identifier for a given patch size and ImageNet variant.

    Args:
        patch:    Patch size key, e.g. "B16" or "L32".
        img_net:  ImageNet pretraining dataset key, e.g. "1K" or "21K".

    Returns:
        The full weights string, e.g. "torchvision.ViT_B_16_Weights.IMAGENET1K_V1".

    Raises:
        ValueError: If the patch or img_net combination is not found.
    """
    try:
        return VIT_MODEL_CONFIG_MAP[patch][img_net]
    except KeyError:
        raise ValueError(f"[ERROR] Invalid ViT config: patch={patch}, img_net={img_net}")


def get_cnn_model_config(name: str) -> str:
    """
    Retrieve the torchvision CNN weights identifier for a given model name.

    Args:
        name:  CNN model shorthand, e.g. "ResNet50", "VGG16".

    Returns:
        The full weights string, e.g. "torchvision.ResNet50_Weights.IMAGENET1K_V1".

    Raises:
        ValueError: If the model name is not recognized.
    """
    try:
        return CNN_MODEL_CONFIG_MAP[name]
    except KeyError:
        raise ValueError(f"[ERROR] Unknown CNN model name: {name}")


def get_dataset_path(dataset_name: str, root_dir: Path = Path.cwd().parent) -> Path:
    """
    Construct the absolute path to a dataset based on its configuration.

    Args:
        dataset_name:  Key in DATASET_CONFIG_MAP.
        root_dir:      Base project directory (default = parent of cwd).

    Returns:
        Absolute Path to the dataset directory.

    Raises:
        ValueError: If the dataset key is missing or the config format is invalid.
    """
    try:
        rel_path = DATASET_CONFIG_MAP[dataset_name]["path"]
        return root_dir / rel_path
    except KeyError:
        raise ValueError(f"[ERROR] Unknown dataset name: {dataset_name}")
    except TypeError:
        raise ValueError(f"[ERROR] Invalid dataset config format for: {dataset_name}")


def get_dataloader_mode(dataset_name: str) -> str:
    """
    Get the dataloader mode for a dataset (e.g. "split_folder", "flat", "binary_class").

    Args:
        dataset_name:  Key in DATASET_CONFIG_MAP.

    Returns:
        The mode string used by create_dataloaders.

    Raises:
        ValueError: If the mode key is missing or config format is invalid.
    """
    try:
        return DATASET_CONFIG_MAP[dataset_name]["dataloader-mode"]
    except KeyError:
        raise ValueError(f"[ERROR] No dataloader mode for dataset: {dataset_name}")
    except TypeError:
        raise ValueError(f"[ERROR] Invalid dataset config format for: {dataset_name}")


def get_num_classes(dataset_name: str) -> int:
    """
    Retrieve the number of target classes for a dataset.

    Args:
        dataset_name:  Key in DATASET_CONFIG_MAP.

    Returns:
        Integer number of classes.

    Raises:
        ValueError: If the num_classes key is missing or config format is invalid.
    """
    try:
        return DATASET_CONFIG_MAP[dataset_name]["num_classes"]
    except KeyError:
        raise ValueError(f"[ERROR] No class count defined for dataset: {dataset_name}")
    except TypeError:
        raise ValueError(f"[ERROR] Invalid dataset config format for: {dataset_name}")


def get_optimizer_hparams(optimizer_name: str, case_num: str) -> dict:
    """
    Fetch hyperparameter settings for a given optimizer and case scenario.

    Args:
        optimizer_name:  e.g. "adam", "rmsprop", "adadelta".
        case_num:        Case identifier as string, e.g. "1", "4".

    Returns:
        Dictionary of hyperparameters, e.g. {"lr": 1e-4, "ne": 25, "mbs": 16}.

    Raises:
        ValueError: If the optimizer or case key is missing.
    """
    try:
        return OPTIMIZER_CONFIG_MAP[optimizer_name][case_num]
    except KeyError:
        raise ValueError(
            f"[ERROR] Missing hyperparameters for optimizer='{optimizer_name}', case='{case_num}'"
        )


def get_model_type(model_name: str) -> str:
    """
    Determine whether a given model identifier refers to a ViT or a CNN.

    Args:
        model_name:  Shorthand identifier, e.g. "B16_1K" or "ResNet50".

    Returns:
        "vit" or "cnn".

    Raises:
        ValueError: If the model_name does not match any known config.
    """
    if model_name in CNN_MODEL_CONFIG_MAP:
        return "cnn"
    try:
        patch, img_net = model_name.split("_", maxsplit=1)
        if patch in VIT_MODEL_CONFIG_MAP and img_net in VIT_MODEL_CONFIG_MAP[patch]:
            return "vit"
    except ValueError:
        pass
    raise ValueError(f"Unknown model type for name: {model_name}")


def get_transform_config(model_type: str, vit_transform=None):
    """
    Retrieve the Compose transform pipeline for a given model type.

    Args:
        model_type:     "vit" or "cnn".
        vit_transform:  For "vit", the pretrained ViT-specific transforms.

    Returns:
        torchvision.transforms.Compose object.

    Raises:
        ValueError: If the model_type is unknown or vit_transform is missing for "vit".
    """
    if model_type not in TRANSFORMS_CONFIG_MAP:
        raise ValueError(f"[ERROR] No transform config for model type: {model_type}")

    tfms = TRANSFORMS_CONFIG_MAP[model_type]
    if model_type == "vit":
        if vit_transform is None:
            raise ValueError("[ERROR] vit_transform must be provided for model_type='vit'")
        return tfms(vit_transform)
    else:
        return tfms


def get_split_ratio(dataset_name: str) -> Tuple[float, float, float]:
    """
    Fetch the train/val/test split ratios for a dataset.

    Args:
        dataset_name:  Key in DATASET_CONFIG_MAP.

    Returns:
        Tuple of three floats summing to <= 1.0, e.g. (0.7, 0.1, 0.2).

    Raises:
        ValueError: If the ratio key is missing, not a 3-element list/tuple,
                    or sums to more than 1.0.
    """
    try:
        split = DATASET_CONFIG_MAP[dataset_name]["train-val-test_ratio"]
    except KeyError:
        raise ValueError(f"[ERROR] Missing 'train-val-test_ratio' for dataset: {dataset_name}")

    if not isinstance(split, (tuple, list)) or len(split) != 3:
        raise ValueError(f"[ERROR] train-val-test_ratio must be a 3-element tuple/list for dataset: {dataset_name}")

    total = sum(split)
    if total > 1.0 + 1e-6:
        raise ValueError(f"[ERROR] train-val-test_ratio total exceeds 1.0 for dataset {dataset_name}: {total:.3f}")

    return tuple(split)
