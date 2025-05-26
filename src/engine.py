"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Callable

import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import InterpolationMode

from torch import nn

import timm
import torchvision.models as models
from torchinfo import summary

import os
import sys

from model_utils import load_vit_model, load_cnn_model, set_seeds
from data_loader import create_dataloaders

from torch.utils.data import DataLoader,Subset
from sklearn.model_selection import train_test_split
import numpy as np

import pickle
import itertools
from tqdm import tqdm

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results

def model_info_retrieval(model):
    try:
        is_torchvision_vit = hasattr(model, 'conv_proj') and hasattr(model, 'encoder')
        is_timm_vit = hasattr(model, 'patch_embed') and hasattr(model, 'blocks')
        
        if not (is_torchvision_vit or is_timm_vit):
            total_params = sum(p.numel() for p in model.parameters())
            return ["CNN-based Model", None, None, None, None, total_params, "Unknown"]
        
        if is_torchvision_vit:
            total_params = sum(p.numel() for p in model.parameters())
            embed_dim = model.conv_proj.out_channels
            patch_size = model.conv_proj.kernel_size[0]
            num_layers = len(model.encoder.layers)
            num_heads = model.encoder.layers[0].self_attention.num_heads
            framework = "torchvision"
            if embed_dim == 768:
                vit_type = "ViT-Base"
            elif embed_dim == 1024:
                vit_type = "ViT-Large"
            else:
                vit_type = f"ViT-Custom (embed_dim={embed_dim})"
            return [vit_type, patch_size, embed_dim, num_layers, num_heads, total_params, framework]

        if is_timm_vit:
            total_params = sum(p.numel() for p in model.parameters())
            embed_dim = model.patch_embed.proj.out_channels
            patch_size = model.patch_embed.proj.kernel_size[0]
            num_layers = len(model.blocks)
            num_heads = model.blocks[0].attn.num_heads
            framework = "timm"
            if embed_dim == 768:
                vit_type = "ViT-Base"
            elif embed_dim == 1024:
                vit_type = "ViT-Large"
            else:
                vit_type = f"ViT-Custom (embed_dim={embed_dim})"
            return [vit_type, patch_size, embed_dim, num_layers, num_heads, total_params, framework]
        
    except Exception as e:
        total_params = sum(p.numel() for p in model.parameters())
        return ["Unknown Model", None, None, None, None, total_params, "Unknown"]
    
def trainVal(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          save_path: str = "model.pth",
          log:str ='log.txt') -> Dict[str, List]:
    """Trains and validates a PyTorch model.

    Args:
        model: A PyTorch model to be trained and validated.
        train_dataloader: A DataLoader instance for training.
        val_dataloader: A DataLoader instance for validation.
        optimizer: A PyTorch optimizer to minimize the loss function.
        loss_fn: A PyTorch loss function.
        epochs: Number of epochs to train for.
        device: Target device (e.g., "cuda" or "cpu").

    Returns:
        A dictionary containing training and validation loss/accuracy metrics.
    """
    
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []}
    
    model.to(device)
    model_info = model_info_retrieval(model)

    with open(log,"a",encoding="utf-8") as file:
        file.write(
            "=======================================================================================\n"
            f"\nModel Info - {model_info[0]} | "
            f"Patch Size: {model_info[1]}x{model_info[1]} | "
            f"Embedding Dim: {model_info[2]} | "
            f"Number of Layers: {model_info[3]} | "
            f"Number of Heads: {model_info[4]} | "
            f"Total Parameters: {model_info[5]:,} | "
            f"Lib: {model_info[6]}"
            f"\nOptimizer Info:\n {optimizer}"
            f"\nTrain Batch Size: {train_dataloader.batch_size} | "
            f"Val Batch Size: {val_dataloader.batch_size}"
            "\n=======================================================================================")

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        val_loss, val_acc = test_step(model=model,
                                      dataloader=val_dataloader,
                                      loss_fn=loss_fn,
                                      device=device)

        print(
          f"Epoch: {epoch+1} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f}"
        )

        with open(log,"a",encoding="utf-8") as file:
            file.write(
              f"\n Epoch: {epoch+1} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}"
            )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
    
    with open(log,"a",encoding="utf-8") as file:
        file.write("\n=======================================================================================")
        
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

    return results

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    model.eval() 
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc



def load_model_checkpoint(model: torch.nn.Module, 
                          checkpoint_path: str, 
                          device: torch.device) -> torch.nn.Module:
    # state_dict = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(state_dict)
    # model.to(device)
    # model.eval()
    # return model
    #### GPT FIX
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Remap keys cho classifier head nếu cần
    new_state_dict = {}
    for k, v in state_dict.items():
        # Nếu key bắt đầu bằng "heads." mà không bắt đầu với "heads.head.",
        # chuyển đổi thành "heads.head." + phần còn lại của key.
        if k.startswith("heads.") and not k.startswith("heads.head."):
            new_key = "heads.head." + k[len("heads."):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    # Load state dict đã được remap vào model
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model


def ensemble_test_step(models: List[torch.nn.Module], 
                       dataloader: torch.utils.data.DataLoader, 
                       loss_fn: torch.nn.Module,
                       device: torch.device) -> Tuple[float, float]:

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Metrics for Sensitivity and Specificity
    TP, TN, FP, FN = 0, 0, 0, 0  

    for model in models:
        model.eval()

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            ensemble_softmax = None

            for model in models:
                logits = model(X)
                softmax_out = torch.softmax(logits, dim=1)
                if ensemble_softmax is None:
                    #TODO: To avoid override softmax_out -> affect to some ensemble methods in long-term impact
                    ensemble_softmax = softmax_out.clone()
                else:
                    ensemble_softmax += softmax_out

            ensemble_softmax /= len(models)

            #TODO: updated loss_fn()
            loss = loss_fn(ensemble_softmax, y) 
            total_loss += loss.item()

            preds = ensemble_softmax.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += X.size(0)

            # Compute TP, TN, FP, FN
            TP += ((preds == 1) & (y == 1)).sum().item()
            TN += ((preds == 0) & (y == 0)).sum().item()
            FP += ((preds == 1) & (y == 0)).sum().item()
            FN += ((preds == 0) & (y == 1)).sum().item()

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    return avg_loss, avg_acc, sensitivity, specificity

def improved_ensemble_test_step(
    models_with_transforms: List[Tuple[torch.nn.Module, Callable]], 
    dataloader: torch.utils.data.DataLoader, 
    loss_fn: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float]:

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    TP, TN, FP, FN = 0, 0, 0, 0

    for model, _ in models_with_transforms:
        model.eval()

    with torch.inference_mode():
        for raw_x, y in dataloader:
            y = y.to(device)

            ensemble_softmax = None

            for model, transform in models_with_transforms:
                x = torch.stack([transform(img) for img in raw_x]) 
                x = x.to(device)

                logits = model(x)
                softmax_out = torch.softmax(logits, dim=1)

                if ensemble_softmax is None:
                    ensemble_softmax = softmax_out.clone()
                else:
                    ensemble_softmax += softmax_out

            ensemble_softmax /= len(models_with_transforms)

            loss = loss_fn(ensemble_softmax, y) 
            total_loss += loss.item()

            preds = ensemble_softmax.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

            TP += ((preds == 1) & (y == 1)).sum().item()
            TN += ((preds == 0) & (y == 0)).sum().item()
            FP += ((preds == 1) & (y == 0)).sum().item()
            FN += ((preds == 0) & (y == 1)).sum().item()

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    return avg_loss, avg_acc, sensitivity, specificity


def run_vit_experiment(
    model_patch_size: str = "B32",
    img_net: str = "21K",
    case_num: str = "3",
    optimizer_name: str = "adam",
    seed: int = 42,
    dataset_dir: str = "/home/hoai-linh.dao/Works/BraTS/dts/Figshare_x10",
    num_classes: int = 3,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    experiment_root: str = (
        "result/self-experiments/extended-dataset-Figshare-x10"
    ),
    model_config_map: dict | None = None,
    optimizer_config_map: dict | None = None,
    dataloader_type: str = "flat",
) -> dict:
    
    # print(f"--> Start experiment: {model_patch_size}_{img_net}_case{case_num}_{optimizer_name}")

    if model_patch_size not in model_config_map:
        raise ValueError(
            f"`model_patch_size` must be one of {set(model_config_map)} "
            f"(got {model_patch_size!r})."
        )

    try:
        model_cfg = model_config_map[model_patch_size][img_net]
    except KeyError as exc:
        raise ValueError(
            f"No weights found for patch={model_patch_size!r}, img_net={img_net!r}."
        ) from exc

    experiment_name = f"{model_patch_size}_{img_net}_case{case_num}"

    log_dir = os.path.join(experiment_root, "logs", optimizer_name)
    weight_dir = os.path.join(experiment_root, "weights", optimizer_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{experiment_name}.txt")
    save_path = os.path.join(weight_dir, f"{experiment_name}.pth")


    set_seeds(seed)

    model, model_transforms = load_vit_model(device, model_cfg, num_classes)

    data_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            model_transforms,
        ]
    )

    try:
        hparams = optimizer_config_map[optimizer_name][case_num]
    except KeyError:
        print(
            f"[SKIP] {experiment_name} – missing hyper-parameters for "
            f"{optimizer_name!r} / case {case_num}."
        )
        return {}

    if optimizer_name == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=hparams["lr"])
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=hparams["lr"])
    else: 
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}.")


    num_workers = os.cpu_count() or 1
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_dir=dataset_dir,
        transform=data_transforms,
        batch_size=hparams["mbs"],
        num_workers=num_workers,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        seed=seed,
        mode=dataloader_type
    )

    loss_fn = nn.CrossEntropyLoss()

    results = trainVal(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=hparams["ne"],
        device=device,
        save_path=save_path,
        log=log_path,
    )

    test_results = test_step(model, test_loader, loss_fn, device)

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\nTest metrics: {test_results}\n")
        fh.write("=" * 80 + "\n")

    return {
        "train_val_results": results,
        "test_results": test_results,
        "model_config": model_cfg,
        "optimizer_hparams": hparams,
        "weights_path": save_path,
    }


def run_cnn_experiment(
    model_name: str = "B32",
    case_num: str = "3",
    optimizer_name: str = "adam",
    seed: int = 42,
    dataset_dir: str = "/home/hoai-linh.dao/Works/BraTS/dts/Figshare_x10",
    num_classes: int = 3,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    experiment_root: str = (
        "result/self-experiments/extended-dataset-Figshare-x10"
    ),
    model_config_map: dict | None = None,
    optimizer_config_map: dict | None = None,
    dataloader_type: str = "flat"
) -> dict:
    
    if model_name not in model_config_map:
        raise ValueError(
            f"`model_name` must be one of {set(model_config_map)} "
            f"(got {model_name!r})."
        )

    try:
        model_cfg = model_config_map[model_name]
    except KeyError as exc:
        raise ValueError(
            f"No weights found for name={model_name!r}."
        ) from exc

    experiment_name = f"{model_name}_case{case_num}"

    log_dir = os.path.join(experiment_root, "logs", optimizer_name)
    weight_dir = os.path.join(experiment_root, "weights", optimizer_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{experiment_name}.txt")
    save_path = os.path.join(weight_dir, f"{experiment_name}.pth")


    set_seeds(seed)

    model, model_transforms = load_cnn_model(device, model_cfg, num_classes)

    #### Temporary Hard Code for Transforms
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    DEFAULT_SIZE = (248, 248)
    CROP_SIZE = 224

    DEFAULT_NORMALIZE = transforms.Normalize(mean=MEAN, std=STD)

    data_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(DEFAULT_SIZE, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            DEFAULT_NORMALIZE
        ]
    )
    ####
    try:
        hparams = optimizer_config_map[optimizer_name][case_num]
    except KeyError:
        print(
            f"[SKIP] {experiment_name} – missing hyper-parameters for "
            f"{optimizer_name!r} / case {case_num}."
        )
        return {}

    if optimizer_name == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=hparams["lr"])
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=hparams["lr"])
    else: 
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}.")


    num_workers = os.cpu_count() or 1
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_dir=dataset_dir,
        transform=data_transforms,
        batch_size=hparams["mbs"],
        num_workers=num_workers,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        seed=seed,
        mode=dataloader_type
    )

    loss_fn = nn.CrossEntropyLoss()

    results = trainVal(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=hparams["ne"],
        device=device,
        save_path=save_path,
        log=log_path,
    )

    test_results = test_step(model, test_loader, loss_fn, device)

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\nTest metrics: {test_results}\n")
        fh.write("=" * 80 + "\n")

    return {
        "train_val_results": results,
        "test_results": test_results,
        "model_config": model_cfg,
        "optimizer_hparams": hparams,
        "weights_path": save_path,
    }

