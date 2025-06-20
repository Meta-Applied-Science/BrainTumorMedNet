from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np

from training.engine import train_val, test_step
from utils.model_utils import load_vit_model, load_cnn_model
from utils.functions import set_seeds
from data.data_loaders import create_dataloaders
from utils.mapping import (
    get_vit_model_config,
    get_cnn_model_config,
    get_dataloader_mode,
    get_optimizer_hparams,
    get_dataset_path,
    get_model_type,
    get_num_classes, 
    get_transform_config,
    get_split_ratio
)

def run_experiments(
    *,
    model_name: str,           # e.g., "B16_21K" or "ResNet50"
    case_num: str,
    optimizer_name: str,
    dataset_name: str,         # e.g., "BT_Large_4c"
    experiment_root: str = "result/experiments",
    seed: int = 42,
    device: torch.device | None = None
) -> Dict:
    """
    Orchestrate a single end-to-end experiment: data loading, model setup, training, validation and testing.

    Args:
        model_name:        Shorthand identifier for the model (e.g. "B16_1K" or "ResNet50").
        case_num:          Identifier for hyperparameter set / experiment case.
        optimizer_name:    Which optimizer to use ("adam", "rmsprop", "adadelta").
        dataset_name:      Key for dataset configuration (e.g. "BT_Large_4c").
        experiment_root:   Base directory under which logs and weights are saved.
        seed:              Random seed for reproducibility.
        device:            torch.device to run on; if None, auto-selects CUDA if available.

    Returns:
        A dictionary containing:
            - "history":            Training/validation loss & metric per epoch.
            - "test_metrics":       Test set metrics (accuracy or full metrics dict).
            - "model_config":       The actual model weights string used.
            - "optimizer_hparams":  The hyperparameters used for the optimizer.
            - "weights_path":       Filepath where final model weights were saved.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seeds(seed)

    dataset_dir = get_dataset_path(dataset_name)
    dataloader_type = get_dataloader_mode(dataset_name)
    model_type = get_model_type(model_name)
    num_classes = get_num_classes(dataset_name)

    # Load model + transforms
    if model_type == "vit":
        patch, img_net = model_name.split("_")
        model_cfg = get_vit_model_config(patch, img_net)
        model, model_tfms = load_vit_model(device, model_cfg, num_classes)
        data_tfms = get_transform_config("vit", model_tfms)
    elif model_type == "cnn":
        model_cfg = get_cnn_model_config(model_name)
        model, _ = load_cnn_model(device, model_cfg, num_classes)
        data_tfms = get_transform_config("cnn")
    else:
        raise ValueError(f"Unsupported model_type: {model_type!r}")

    # Optimizer + hparams
    hparams = get_optimizer_hparams(optimizer_name, case_num)
    optimizer = _build_optimizer(model, optimizer_name, hparams["lr"])

    # Dataloaders
    split_ratio = get_split_ratio(dataset_name)

    train_loader, val_loader, test_loader = _create_loaders(
        dataset_dir=dataset_dir,
        transform=data_tfms,
        batch_size=hparams["mbs"],
        seed=seed,
        mode=dataloader_type,
        split_ratio=split_ratio,
    )

    # Experiment name
    exp_name = f"{model_type.upper()}_{model_name}_{dataset_name}_case{case_num}"
    log_path, save_path = _prepare_dirs(experiment_root, optimizer_name, exp_name)

    # Train
    loss_fn = nn.CrossEntropyLoss()
    history = train_val(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=hparams["ne"],
        device=device,
        secondary_metric="accuracy", 
        topk=(1, 3),                 
        positive_label=1,           
        save_path=save_path,
        log=log_path,
    )

    # Test
    test_loss, test_full_metrics = test_step(
        model,
        test_loader,
        loss_fn,
        device,
        secondary_metric="full",
        topk=(1,3),
        positive_label=1,
    )
    # log
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("\nTest loss   : {:.4f}\n".format(test_loss))
        fh.write("Full metrics:\n")
        for name, val in test_full_metrics.items():
            if isinstance(val, list):
                fh.write(f"  {name:20s}: {', '.join(f'{v:.4f}' for v in val)}\n")
            elif torch.is_tensor(val):
                # confusion_matrix
                fh.write(f"  {name:20s}:\n{val}\n")
            else:
                fh.write(f"  {name:20s}: {val:.4f}\n")
        fh.write("=" * 80 + "\n")

    return {
        "history": history,
        "test_loss": test_loss,
        "test_full_metrics": test_full_metrics,
        "model_config": model_cfg,
        "optimizer_hparams": hparams,
        "weights_path": save_path,
    }

def run_stragified_kfold_experiments(
    *,
    model_name: str,
    case_num: str,
    optimizer_name: str,
    dataset_name: str,
    experiment_root: str = "results",
    k_folds: int = 8,
    seed: int = 42,
    device: torch.device | None = None,
    secondary_metric: str = "f1_macro",
    topk: tuple[int, ...] = (1, 3),
    positive_label: int = 1,
    epochs: int | None = None,
) -> dict[str, float]:
    """
    Perform k-fold cross-validation on the train+val portion of a dataset and
    return the validation metric for each fold plus their average.

    Args:
        model_name:       Shorthand model ID, e.g. "B16_1K" or "ResNet50".
        case_num:         Hyperparameter case identifier as string.
        optimizer_name:   "adam", "rmsprop", or "adadelta".
        dataset_name:     Key in DATASET_CONFIG_MAP.
        experiment_root:  Base directory to save logs and weights.
        k_folds:          Number of folds (e.g. 8 to mimic 70/10 splits).
        seed:             RNG seed for reproducibility.
        device:           torch.device, or None to auto-select.
        secondary_metric: Which metric to optimize/record on validation.
        topk:             Tuple for top-k accuracy.
        positive_label:   Positive class index for binary metrics.
        epochs:           Override number of epochs from optimizer config.

    Returns:
        A dict mapping:
          - "fold_1", …, "fold_{k_folds}": validation metric for each fold
          - "avg_val": mean validation metric across all folds
    """
    # Setup device & reproducibility
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(seed)

    # Prepare directories
    root = Path(experiment_root)
    log_root = root / "logs" / optimizer_name / f"{k_folds}fold_{model_name}_case{case_num}"
    wts_root = root / "weights" / optimizer_name / f"{k_folds}fold_{model_name}_case{case_num}"
    log_root.mkdir(parents=True, exist_ok=True)
    wts_root.mkdir(parents=True, exist_ok=True)

    # Dataset and transforms
    dataset_dir = get_dataset_path(dataset_name)
    num_classes = __import__("utils.mapping", fromlist=["get_num_classes"]).get_num_classes(dataset_name)
    model_type = get_model_type(model_name)

    if model_type == "vit":
        patch, img_net = model_name.split("_", 1)
        vit_cfg = get_vit_model_config(patch, img_net)
        _, model_tfms = load_vit_model(device, vit_cfg, num_classes)
        data_tfms = get_transform_config("vit", model_tfms)
    else:
        cnn_cfg = get_cnn_model_config(model_name)
        _, _ = load_cnn_model(device, cnn_cfg, num_classes)
        data_tfms = get_transform_config("cnn")

    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=data_tfms)
    labels = np.array(full_dataset.targets)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    results: dict[str, float] = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        # Subsets & loaders
        train_ds = Subset(full_dataset, train_idx)
        val_ds   = Subset(full_dataset, val_idx)
        batch_size = get_optimizer_hparams(optimizer_name, case_num)["mbs"]
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=os.cpu_count() or 1, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                  num_workers=os.cpu_count() or 1, pin_memory=True)

        # Fresh model + optimizer
        if model_type == "vit":
            model, _ = load_vit_model(device, vit_cfg, num_classes)
        else:
            model, _ = load_cnn_model(device, cnn_cfg, num_classes)

        hparams = get_optimizer_hparams(optimizer_name, case_num)
        if epochs is not None:
            hparams["ne"] = epochs

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
        elif optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=hparams["lr"])
        else:
            optimizer = torch.optim.Adadelta(model.parameters(), lr=hparams["lr"])

        # Paths for this fold
        log_path  = log_root / f"fold{fold}.txt"
        wts_path  = wts_root / f"fold{fold}.pth"

        # Train and validate
        history = train_val(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            epochs=hparams["ne"],
            device=device,
            secondary_metric=secondary_metric,
            topk=topk,
            positive_label=positive_label,
            save_path=str(wts_path),
            log=str(log_path),
        )

        # Record final validation metric
        val_metrics = history[f"val_{secondary_metric}"]
        results[f"fold_{fold}"] = val_metrics[-1]

    # Compute average
    results["avg_val"] = sum(results[f"fold_{i}"] for i in range(1, k_folds+1)) / k_folds
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_dirs(root: str, optim: str, exp_name: str) -> Tuple[str, str]:
    log_dir = Path(root) / "logs" / optim
    weight_dir = Path(root) / "weights" / optim
    log_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir / f"{exp_name}.txt"), str(weight_dir / f"{exp_name}.pth")


def _build_optimizer(model: torch.nn.Module, name: str, lr: float):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "adadelta":
        return torch.optim.Adadelta(model.parameters(), lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    raise ValueError(f"Unsupported optimizer {name!r}")


def _create_loaders(
    dataset_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    seed: int,
    mode: str,
    split_ratio: Tuple[float, float, float],
):
    num_workers = os.cpu_count() or 1
    return create_dataloaders(
        dataset_dir=dataset_dir,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=split_ratio[0],
        val_ratio=split_ratio[1],
        test_ratio=split_ratio[2],
        seed=seed,
        mode=mode,
    )
