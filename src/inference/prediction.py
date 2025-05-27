"""
Inference utilities for making predictions with trained PyTorch models.
"""

from typing import List, Tuple, Union
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    transform: transforms.Compose,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Tuple[int, float]:
    """
    Predicts the class index and probability for a single image.

    Returns:
        Tuple of (predicted class index, probability).
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.to(device)
    model.eval()
    with torch.inference_mode():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs.max().item()

    return pred_idx, confidence


def predict_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform: transforms.Compose = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> None:
    """
    Predicts and visualizes the result on an image.

    Args:
        model: Trained model.
        image_path: Path to input image.
        class_names: Optional list of class names.
        transform: Image transform (should match training transform).
        device: Device to run inference on.
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    pred_idx, prob = predict_image(model, image_path, transform, device)

    img = Image.open(image_path).convert("RGB")
    plt.imshow(img)
    title = f"Pred: {class_names[pred_idx] if class_names else pred_idx} | Prob: {prob:.3f}"
    plt.title(title)
    plt.axis(False)
    plt.show()


def predict_on_folder(
    model: torch.nn.Module,
    folder_path: str,
    transform: transforms.Compose,
    class_names: List[str] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    file_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
) -> List[Tuple[str, str, float]]:
    """
    Runs predictions on all images in a folder.

    Returns:
        List of tuples: (filename, predicted class, probability)
    """
    results = []
    model.to(device)
    model.eval()

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(file_extensions):
            continue
        path = os.path.join(folder_path, filename)
        pred_idx, prob = predict_image(model, path, transform, device)
        class_name = class_names[pred_idx] if class_names else str(pred_idx)
        results.append((filename, class_name, prob))

    return results
