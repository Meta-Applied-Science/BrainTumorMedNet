from __future__ import annotations

from typing import Sequence, Tuple, List
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc as sklearn_auc,
    matthews_corrcoef,
    cohen_kappa_score,
)

__all__ = [
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
]


def accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    topk: Sequence[int] | Tuple[int, ...] = (1,),
) -> List[float]:
    """Return top-k accuracies for the given batch."""
    if logits.ndim != 2:
        raise ValueError("`logits` must have shape (N, C)")
    maxk = max(topk)
    batch_size = labels.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    return [(correct[:k].reshape(-1).float().sum() / batch_size).item() for k in topk]


def confusion_counts(
    preds: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive_label: int = 1,
) -> Tuple[int, int, int, int]:
    """Return (TP, TN, FP, FN) for binary classification."""
    preds = preds.detach()
    labels = labels.detach()
    TP = ((preds == positive_label) & (labels == positive_label)).sum().item()
    TN = ((preds != positive_label) & (labels != positive_label)).sum().item()
    FP = ((preds == positive_label) & (labels != positive_label)).sum().item()
    FN = ((preds != positive_label) & (labels == positive_label)).sum().item()
    return TP, TN, FP, FN


def confusion_matrix(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Multi-class confusion matrix of shape (num_classes, num_classes),
    where entry (i, j) counts samples with true label i predicted as j.
    """
    preds = preds.detach().view(-1)
    labels = labels.detach().view(-1)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(labels, preds):
        cm[t.long(), p.long()] += 1
    return cm


def precision_binary(
    preds: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive_label: int = 1,
) -> float:
    TP, _, FP, _ = confusion_counts(preds, labels, positive_label=positive_label)
    return TP / (TP + FP + 1e-8)


def recall_binary(
    preds: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive_label: int = 1,
) -> float:
    TP, _, _, FN = confusion_counts(preds, labels, positive_label=positive_label)
    return TP / (TP + FN + 1e-8)


sensitivity = recall_binary


def specificity(
    preds: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive_label: int = 1,
) -> float:
    _, TN, FP, _ = confusion_counts(preds, labels, positive_label=positive_label)
    return TN / (TN + FP + 1e-8)


def precision_macro(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """Macro-averaged precision for multi-class."""
    scores = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        scores.append(tp / (tp + fp + 1e-8))
    return sum(scores) / num_classes


def recall_macro(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """Macro-averaged recall for multi-class."""
    scores = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        scores.append(tp / (tp + fn + 1e-8))
    return sum(scores) / num_classes


def f1_macro(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """Macro-averaged F1-score for multi-class."""
    scores = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        scores.append(2 * prec * rec / (prec + rec + 1e-8))
    return sum(scores) / num_classes


def precision_weighted(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """Weighted-averaged precision for multi-class."""
    precisions = []
    supports = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        support = (labels == c).sum().item()
        precisions.append(tp / (tp + fp + 1e-8))
        supports.append(support)
    total = sum(supports)
    return sum(p * s for p, s in zip(precisions, supports)) / total if total > 0 else 0.0


def recall_weighted(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """Weighted-averaged recall for multi-class."""
    recalls = []
    supports = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        support = (labels == c).sum().item()
        recalls.append(tp / (tp + fn + 1e-8))
        supports.append(support)
    total = sum(supports)
    return sum(r * s for r, s in zip(recalls, supports)) / total if total > 0 else 0.0


def f1_weighted(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """Weighted-averaged F1-score for multi-class."""
    f1s = []
    supports = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        support = (labels == c).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
        supports.append(support)
    total = sum(supports)
    return sum(f * s for f, s in zip(f1s, supports)) / total if total > 0 else 0.0


def roc_auc_binary(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Binary ROC AUC (uses probability of class 1)."""
    p1 = probs[:, 1].cpu().numpy()
    y  = labels.cpu().numpy()
    return roc_auc_score(y, p1)


def pr_auc_binary(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Binary Precision-Recall AUC (uses probability of class 1)."""
    p1 = probs[:, 1].cpu().numpy()
    y  = labels.cpu().numpy()
    prec, rec, _ = precision_recall_curve(y, p1)
    return sklearn_auc(rec, prec)


def roc_auc_ovr(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """Macro-average ROC AUC one-vs-rest for multi-class."""
    y = labels.cpu().numpy()
    P = probs.cpu().numpy()
    aucs = []
    for c in range(num_classes):
        y_c = (y == c).astype(int)
        p_c = P[:, c]
        try:
            aucs.append(roc_auc_score(y_c, p_c))
        except ValueError:
            pass
    return sum(aucs) / len(aucs) if aucs else 0.0


def pr_auc_ovr(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """Macro-average PR AUC one-vs-rest for multi-class."""
    y = labels.cpu().numpy()
    P = probs.cpu().numpy()
    aucs = []
    for c in range(num_classes):
        y_c = (y == c).astype(int)
        p_c = P[:, c]
        try:
            prec, rec, _ = precision_recall_curve(y_c, p_c)
            aucs.append(sklearn_auc(rec, prec))
        except ValueError:
            pass
    return sum(aucs) / len(aucs) if aucs else 0.0


def mcc_binary(
    preds: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Matthews Correlation Coefficient for binary."""
    return matthews_corrcoef(labels.cpu().numpy(), preds.cpu().numpy())


def kappa_binary(
    preds: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Cohen's Kappa for binary."""
    return cohen_kappa_score(labels.cpu().numpy(), preds.cpu().numpy())
