from __future__ import annotations

from typing import Dict, List, Tuple, Union, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from training.metrics import *
from utils.model_utils import model_info_retrieval
import torch.nn.functional as F
from training.metrics import (
    accuracy,
    sensitivity,
    specificity,
    precision_binary,
    recall_binary,
    precision_macro,
    recall_macro,
    f1_macro,
    precision_weighted,
    recall_weighted,
    f1_weighted,
    roc_auc_binary,
    pr_auc_binary,
    roc_auc_ovr,
    pr_auc_ovr,
    mcc_binary,
    kappa_binary,
    confusion_matrix,
)

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    secondary_metric: str = "accuracy",
    topk: Sequence[int] | Tuple[int, ...] = (1,),
    positive_label: int = 1,
) -> Union[
    float,
    Tuple[float, List[float]],
    Tuple[float, Dict[str, Union[float, List[float], torch.Tensor]]],
]:
    """
    Run inference on `dataloader`, compute average loss and a chosen metric.

    Args:
        model:           Trained PyTorch model.
        dataloader:      DataLoader for evaluation.
        loss_fn:         Loss function (e.g. CrossEntropyLoss).
        device:          torch.device("cuda") or ("cpu").
        secondary_metric:
                         Which metric to return. Options:
                         "accuracy", "topk",
                         "sensitivity", "specificity",
                         "precision_binary", "recall_binary",
                         "precision_macro", "recall_macro", "f1_macro",
                         "precision_weighted", "recall_weighted", "f1_weighted",
                         "roc_auc", "pr_auc", "roc_auc_ovr", "pr_auc_ovr",
                         "mcc", "kappa", "full".
        topk:            Tuple of k values for top-k accuracy (only if using "topk" or "full").
        positive_label:  Label index considered “positive” for binary metrics.

    Returns:
        If secondary_metric != "full":
            (avg_loss, metric_value)
            - avg_loss: float
            - metric_value: float or List[float] (for "topk")
        If secondary_metric == "full":
            (avg_loss, metrics_dict)
            - metrics_dict: Dict mapping each metric name to its value,
              including confusion_matrix (torch.Tensor).
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits, all_labels = [], []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        bs = x.size(0)

        total_loss += loss.item() * bs
        total_samples += bs

        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    avg_loss = total_loss / total_samples
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    preds_cat = logits_cat.argmax(dim=1)
    probs_cat = F.softmax(logits_cat, dim=1)
    num_classes = logits_cat.size(1)

    # helpers to choose correct ROC/PR for binary vs. multi-class
    def _roc_auc():
        return (roc_auc_binary(probs_cat, labels_cat)
                if num_classes == 2
                else roc_auc_ovr(probs_cat, labels_cat, num_classes))

    def _pr_auc():
        return (pr_auc_binary(probs_cat, labels_cat)
                if num_classes == 2
                else pr_auc_ovr(probs_cat, labels_cat, num_classes))

    # FULL: compute everything
    if secondary_metric == "full":
        return avg_loss, {
            "accuracy":            (preds_cat == labels_cat).float().mean().item(),
            "topk":                accuracy(logits_cat, labels_cat, topk=topk),
            "sensitivity":         sensitivity(preds_cat, labels_cat, positive_label=positive_label),
            "specificity":         specificity(preds_cat, labels_cat, positive_label=positive_label),
            "precision_binary":    precision_binary(preds_cat, labels_cat, positive_label=positive_label),
            "recall_binary":       recall_binary(preds_cat, labels_cat, positive_label=positive_label),
            "precision_macro":     precision_macro(preds_cat, labels_cat, num_classes),
            "recall_macro":        recall_macro(preds_cat, labels_cat, num_classes),
            "f1_macro":            f1_macro(preds_cat, labels_cat, num_classes),
            "precision_weighted":  precision_weighted(preds_cat, labels_cat, num_classes),
            "recall_weighted":     recall_weighted(preds_cat, labels_cat, num_classes),
            "f1_weighted":         f1_weighted(preds_cat, labels_cat, num_classes),
            "roc_auc":             _roc_auc(),
            "pr_auc":              _pr_auc(),
            "roc_auc_ovr":         roc_auc_ovr(probs_cat, labels_cat, num_classes),
            "pr_auc_ovr":          pr_auc_ovr(probs_cat, labels_cat, num_classes),
            "mcc":                 mcc_binary(preds_cat, labels_cat),
            "kappa":               kappa_binary(preds_cat, labels_cat),
            "confusion_matrix":    confusion_matrix(preds_cat, labels_cat, num_classes),
        }

    # single-metric modes
    if secondary_metric == "accuracy":
        return avg_loss, (preds_cat == labels_cat).float().mean().item()
    if secondary_metric == "topk":
        return avg_loss, accuracy(logits_cat, labels_cat, topk=topk)
    if secondary_metric == "sensitivity":
        return avg_loss, sensitivity(preds_cat, labels_cat, positive_label=positive_label)
    if secondary_metric == "specificity":
        return avg_loss, specificity(preds_cat, labels_cat, positive_label=positive_label)
    if secondary_metric == "precision_binary":
        return avg_loss, precision_binary(preds_cat, labels_cat, positive_label=positive_label)
    if secondary_metric == "recall_binary":
        return avg_loss, recall_binary(preds_cat, labels_cat, positive_label=positive_label)
    if secondary_metric == "precision_macro":
        return avg_loss, precision_macro(preds_cat, labels_cat, num_classes)
    if secondary_metric == "recall_macro":
        return avg_loss, recall_macro(preds_cat, labels_cat, num_classes)
    if secondary_metric == "f1_macro":
        return avg_loss, f1_macro(preds_cat, labels_cat, num_classes)
    if secondary_metric == "precision_weighted":
        return avg_loss, precision_weighted(preds_cat, labels_cat, num_classes)
    if secondary_metric == "recall_weighted":
        return avg_loss, recall_weighted(preds_cat, labels_cat, num_classes)
    if secondary_metric == "f1_weighted":
        return avg_loss, f1_weighted(preds_cat, labels_cat, num_classes)
    if secondary_metric == "roc_auc":
        return avg_loss, _roc_auc()
    if secondary_metric == "pr_auc":
        return avg_loss, _pr_auc()
    if secondary_metric == "roc_auc_ovr":
        return avg_loss, roc_auc_ovr(probs_cat, labels_cat, num_classes)
    if secondary_metric == "pr_auc_ovr":
        return avg_loss, pr_auc_ovr(probs_cat, labels_cat, num_classes)
    if secondary_metric == "mcc":
        return avg_loss, mcc_binary(preds_cat, labels_cat)
    if secondary_metric == "kappa":
        return avg_loss, kappa_binary(preds_cat, labels_cat)

    raise ValueError(f"[ERROR] Unsupported secondary_metric: {secondary_metric!r}")

def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Perform a single training epoch over `dataloader`.

    Args:
        model:      The PyTorch model to train.
        dataloader: DataLoader yielding (inputs, labels) for training.
        loss_fn:    Loss function (e.g., nn.CrossEntropyLoss).
        optimizer:  Optimizer for updating model parameters.
        device:     Device on which to run (cuda or cpu).

    Returns:
        A tuple (avg_loss, avg_acc) where:
            avg_loss: Average loss over all batches.
            avg_acc:  Average accuracy (fraction correct) over all batches.
    """

    model.train()
    epoch_loss, epoch_acc = 0.0, 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        epoch_acc += (preds == y).sum().item() / len(y_pred)

    epoch_loss /= len(dataloader)
    epoch_acc /= len(dataloader)
    return epoch_loss, epoch_acc


def test_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    secondary_metric: str = "accuracy",
    topk: Sequence[int] = (1,),
    positive_label: int = 1,
) -> Tuple[float, Union[float, List[float], dict]]:
    """
    Wrapper around `evaluate` for a single pass on validation or test split.

    Args:
        model:           Trained PyTorch model.
        dataloader:      DataLoader for validation/test.
        loss_fn:         Loss function.
        device:          torch.device.
        secondary_metric:
                         Metric to compute (see `evaluate` docs).
        topk:            Top-k tuple for accuracy if needed.
        positive_label:  Positive class index for binary metrics.

    Returns:
        avg_loss:    float
        metric_val:  float, List[float], or dict (if secondary_metric="full").
    """

    return evaluate(
        model,
        dataloader,
        loss_fn,
        device,
        secondary_metric=secondary_metric,
        topk=topk,
        positive_label=positive_label,
    )

def train_val(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
    *,
    secondary_metric: str = "accuracy",
    topk: Sequence[int] = (1,),
    positive_label: int = 1,
    save_path: str = "model.pth",
    log: str = "log.txt",
) -> Dict[str, List[float]]:
    """
    Train a model with validation after each epoch, log progress, and save best weights.

    Args:
        model:             PyTorch model to train.
        train_dataloader:  DataLoader for training data.
        val_dataloader:    DataLoader for validation data.
        optimizer:         Torch optimizer.
        loss_fn:           Loss function.
        epochs:            Number of epochs.
        device:            torch.device.
        secondary_metric:  Which metric to compute on validation.
        topk:              Top-k tuple for accuracy.
        positive_label:    Positive class index for binary metrics.
        save_path:         File path to save final model weights.
        log:               File path to append training logs.

    Returns:
        Dictionary containing:
            "train_loss":       List of training losses,
            "train_acc":        List of training accuracies,
            "val_loss":         List of validation losses,
            f"val_{secondary_metric}": List of chosen validation metric per epoch.
    """
    results: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        f"val_{secondary_metric}": [],
    }
    model.to(device)
    info = model_info_retrieval(model)

    with open(log, "a", encoding="utf-8") as fh:
        fh.write(
            "=" * 87 +
            f"\nModel Info - {info[0]} | Patch: {info[1]} | EmbDim: {info[2]} | "
            f"Layers: {info[3]} | Heads: {info[4]} | Params: {info[5]:,} | Lib: {info[6]}" +
            "\nOptimizer:\n " + str(optimizer) +
            f"\nTrain Batch: {train_dataloader.batch_size} | Val Batch: {val_dataloader.batch_size}\n" +
            "=" * 87
        )

    for epoch in tqdm(range(epochs), desc="Train + Validate"):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        val_loss, val_metric = test_step(
            model,
            val_dataloader,
            loss_fn,
            device,
            secondary_metric=secondary_metric,
            topk=topk,
            positive_label=positive_label,
        )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val {secondary_metric}: {val_metric:.4f}"
        )
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(
                f"\nEpoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val {secondary_metric}: {val_metric:.4f}"
            )
            fh.write("\n" + "=" * 87)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results[f"val_{secondary_metric}"].append(val_metric)

    torch.save(model.state_dict(), save_path)
    print(f"\nModel weights saved to: {save_path}")

    return results
