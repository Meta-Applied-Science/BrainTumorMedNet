"""
A series of helper functions used throughout the course.

If a function gets defined once and could be used over and over, it'll go in here.
"""
import os

import zipfile
from pathlib import Path
import requests

import shutil
import random

import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


# === XAI Task ===
# Prepare data transformation pipeline

def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x


def inv_transform(mean:list, std:list) -> transforms.Compose:
    transform  = [
        torchvision.transforms.Lambda(nhwc_to_nchw),
        torchvision.transforms.Normalize(
            mean=(-1 * np.array(mean) / np.array(std)).tolist(),
            std=(1 / np.array(std)).tolist(),
        ),
        torchvision.transforms.Lambda(nchw_to_nhwc),]
    
    return torchvision.transforms.Compose(transform)


def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    return tensor * std + mean


# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
  Plots linear training data and test data and compares predictions.
  """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# Plot loss curves of a model
def plot_loss_curves(results,type):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """


    loss = results["train_loss"]
    test_loss = results['val_loss'] if type == "val" else results['test_loss']

    accuracy = results["train_acc"]
    test_accuracy = results['val_acc'] if type =='val' else results['test_acc']

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label=("val" if type == "val" else "test") + "_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label=("val" if type == "val" else "test") + "_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()


# Pred and plot image function from notebook 04
# See creation: https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function
from typing import List
import torchvision


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
    
    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

def splitDataset(dataset_dir, output_dir):
    split_dirs = {subset: os.path.join(output_dir, subset) for subset in ["train", "val", "test"]}

    for path in split_dirs.values():
        os.makedirs(path, exist_ok=True)
   
    for sub_dir in os.scandir(dataset_dir):
        if not sub_dir.is_dir():  
            continue  

        print("Processing:", sub_dir.name)
        sub_path = sub_dir.path
        all_images = [f for f in os.listdir(sub_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not all_images:
            continue  

        random.shuffle(all_images)

        total = len(all_images)
        train_split, val_split = int(0.7 * total), int(0.1 * total)

        subsets = {
            "train": all_images[:train_split],
            "val": all_images[train_split:train_split + val_split],
            "test": all_images[train_split + val_split:]
        }

        #subset: "train",...
        for subset, images in subsets.items():
            for filename in images:
                src = os.path.join(sub_path, filename)
                
                dst_path= split_dirs[subset]
                dst_dir= os.path.join(dst_path,sub_dir.name)

                # print("dst_dir:",dst_dir)

                os.makedirs(dst_dir, exist_ok=True)
            
                shutil.copy2(src, dst_dir)  

        print(f"{sub_dir.name}: Train={len(subsets['train'])}, Val={len(subsets['val'])}, Test={len(subsets['test'])}")

def merge_folders(source_dirs, destination_root):
    """
    Merge subfolders from a list of source directories into a destination directory.

    - source_dirs: List of source directories (e.g., ["./testing", "./training"]).
    - destination_root: Destination directory to store merged files (e.g., "./merged").
    """
    os.makedirs(destination_root, exist_ok=True)

    print("source_dirs[0]:",source_dirs[0])

    # Loop through all subdirectories in the first source directory (assuming they are the same in others)
    for sub_dir in os.listdir(source_dirs[0]):
        # print("sub_dir:",sub_dir) #no_tumor,..
        if(sub_dir != ".DS_Store"):
            merged_sub_dir = os.path.join(destination_root, sub_dir)
            os.makedirs(merged_sub_dir, exist_ok=True)

            # Loop through each source directory
            for source_dir in source_dirs:
                # print("source_dir:",source_dir)


                image_in_sub_dir = os.path.join(source_dir, sub_dir)

                if os.path.exists(image_in_sub_dir):  # Check if the subdirectory exists
                    for filename in os.listdir(image_in_sub_dir):

                        image_path = os.path.join(image_in_sub_dir, filename)
                        dest_path = os.path.join(merged_sub_dir, filename)

                        # Handle duplicate filenames by appending a number
                        if os.path.exists(dest_path):
                            base, ext = os.path.splitext(filename)
                            # print("base:",base,"\t ext:",ext) #image name - extension: .jpg, .png
                            counter = 1
                            new_dest_path = os.path.join(merged_sub_dir, f"{base}_{counter}{ext}")
                            while os.path.exists(new_dest_path):
                                counter += 1
                                new_dest_path = os.path.join(merged_sub_dir, f"{base}_{counter}{ext}")
                            dest_path = new_dest_path

                        shutil.copy2(image_path, dest_path)

            print("\nMerging",sub_dir,"completed. Merged files are stored in:", destination_root)
        else:
            print("non")
    print("\nNumber of items in each subfolder of the merged directory:")
    for sub_dir in os.listdir(destination_root):
        merged_sub_dir = os.path.join(destination_root, sub_dir)
        if os.path.isdir(merged_sub_dir):
            num_items = len([f for f in os.listdir(merged_sub_dir) if os.path.isfile(os.path.join(merged_sub_dir, f))])
            print(f"{sub_dir}: {num_items} items")

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)