import torch
from torch import nn

from data.data_loaders import create_dataloaders
from model.utils import load_vit_model, load_cnn_model
from training.engine import test_step
from utils.mapping import *

def evaluate_checkpoint_on_dataset(
    *,
    model_name: str,
    checkpoint_path: str,
    dataset_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    device: torch.device | str = None,
) -> tuple[float, dict]:
    """
    Load a saved model checkpoint (ViT or CNN) and evaluate its performance on the test split.

    Args:
        model_name:      Shorthand identifier for the model (e.g., "B16_1K" or "ResNet50").
        checkpoint_path: Filesystem path to the saved .pth checkpoint.
        dataset_name:    Key for the dataset in DATASET_CONFIG_MAP.
        batch_size:      Batch size to use during evaluation.
        num_workers:     Number of subprocesses for data loading.
        seed:            Random seed for deterministic splitting.
        device:          torch.device or string ("cuda"/"cpu"). If None, auto-selects.

    Returns:
        A 2-tuple containing:
          - test_loss (float):    Average CrossEntropyLoss on the test set.
          - full_metrics_dict (dict):
              A dictionary of evaluation metrics as returned by test_step in "full" mode,
              including accuracy, F1 scores, ROC-AUC, confusion matrix, etc.

    Raises:
        ValueError: If `model_name` is not found in the config maps,
                    or if the checkpoint cannot be loaded,
                    or if the dataset split configuration is invalid.
    """
    # Device
    if device is None or isinstance(device, str):
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Dataset config
    dataset_dir   = get_dataset_path(dataset_name)
    mode          = get_dataloader_mode(dataset_name)
    num_classes   = get_num_classes(dataset_name)
    split_ratio   = get_split_ratio(dataset_name)

    # Model config & transforms
    model_type = get_model_type(model_name)
    
    if model_type == "vit":
        patch, img_net = model_name.split("_", 1)
        model_cfg, model_tfms = get_vit_model_config(patch, img_net), None
        model, model_tfms = load_vit_model(device, model_cfg, num_classes)
        data_transforms = get_transform_config("vit", model_tfms)
    else:  # cnn
        model_cfg = get_cnn_model_config(model_name)
        model, _ = load_cnn_model(device, model_cfg, num_classes)
        data_transforms = get_transform_config("cnn")

    # Load weights
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # DataLoader
    _, _, test_loader = create_dataloaders(
        dataset_dir=dataset_dir,
        transform=data_transforms,
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=split_ratio[0],
        val_ratio=split_ratio[1],
        test_ratio=split_ratio[2],
        seed=seed,
        mode=mode,
    )

    # Evaluate with full metrics
    loss_fn = nn.CrossEntropyLoss()
    test_loss, test_full = test_step(
        model,
        test_loader,
        loss_fn,
        device,
        secondary_metric="full",
        topk=(1, 3),
        positive_label=1,
    )

    return test_loss, test_full
