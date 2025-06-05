"""
Utility functions for visualizing model training and predictions.
"""

import torch
import matplotlib.pyplot as plt
from typing import List
import torchvision


def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
) -> None:
    """
    Plots training and test data with optional predictions.

    Args:
        train_data: Input features for training.
        train_labels: Ground truth labels for training data.
        test_data: Input features for testing.
        test_labels: Ground truth labels for test data.
        predictions: Optional predicted labels for test data.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()


def plot_loss_curves(results: dict, mode: str = "val") -> None:
    """
    Plots training and validation/test loss and accuracy curves.

    Args:
        results (dict): Dictionary with loss and accuracy metrics.
        mode (str): One of "val" or "test" indicating which set to compare.
    """
    loss = results["train_loss"]
    test_loss = results["val_loss"] if mode == "val" else results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["val_acc"] if mode == "val" else results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, test_loss, label=f"{mode.capitalize()} Loss")
    plt.xlabel("Epochs")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="Train Accuracy")
    plt.plot(epochs, test_accuracy, label=f"{mode.capitalize()} Accuracy")
    plt.xlabel("Epochs")
    plt.title("Accuracy")
    plt.legend()

    plt.show()


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    """
    Makes a prediction on an image and displays it with predicted label.

    Args:
        model: Trained PyTorch model.
        image_path: Path to input image.
        class_names: List of class names.
        transform: Optional torchvision transform.
        device: Device to run inference on.
    """
    image = torchvision.io.read_image(str(image_path)).type(torch.float32) / 255.0
    if transform:
        image = transform(image)

    model.to(device)
    model.eval()
    with torch.inference_mode():
        image = image.unsqueeze(dim=0).to(device)
        preds = model(image)
        probs = torch.softmax(preds, dim=1)
        label = torch.argmax(probs, dim=1)

    plt.imshow(image.squeeze().permute(1, 2, 0).cpu())
    title = f"Pred: {class_names[label.item()] if class_names else label.item()} | Prob: {probs.max().item():.3f}"
    plt.title(title)
    plt.axis(False)
    plt.show()
