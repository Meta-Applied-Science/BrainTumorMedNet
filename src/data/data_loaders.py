import os
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from typing import Literal

class CustomSubset(torch.utils.data.Dataset):
    """Subset of a dataset that also returns image file paths."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.dataset[real_idx]
        path = self.dataset.imgs[real_idx][0]  # Get image path

        return image, label, path

    def __len__(self):
        return len(self.indices)
    
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        # Original tuple (image, label)
        img, label = super().__getitem__(index)
        # Get the image file path
        path = self.imgs[index][0]

        return img, label, path

def create_dataloaders(
    dataset_dir: str,
    transform,
    batch_size: int,
    num_workers: int,
    seed: int,
    mode: Literal["flat", "split_folder", "binary_class"],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
):
    """
    Create PyTorch DataLoaders for train/validation/test splits from an image dataset.

    Supports three folder layouts:
      - "flat":       Single root folder with class subdirectories.
      - "split_folder":
                      Separate "Training" and "Testing" subfolders under root.
      - "binary_class":
                      Single folder with multiple classes, but only 'yes'/'no' retained.

    Splits the data according to provided ratios and uses stratified sampling to
    preserve class distributions.

    Args:
        dataset_dir:   Path to the dataset root directory.
        transform:     torchvision transforms to apply to each image.
        batch_size:    Number of samples per batch.
        num_workers:   Number of subprocesses for data loading.
        seed:          Random seed for reproducible splits.
        mode:          One of "flat", "split_folder", or "binary_class".
        train_ratio:   Proportion of data for training (default 0.7).
        val_ratio:     Proportion of data for validation (default 0.1).
        test_ratio:    Proportion of data for testing (default 0.2).

    Returns:
        Tuple of (train_loader, val_loader, test_loader), each a torch.utils.data.DataLoader.
        In "split_folder" mode, the test set is taken from the "Testing" folder.
        In "binary_class" mode, only samples labeled 'yes' or 'no' are included.

    Raises:
        ValueError: If `mode` is not one of the supported strings.
    """
    if mode == "flat":
        dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
        indices = list(range(len(dataset)))
        labels = dataset.targets

        train_idx, temp_idx = train_test_split(indices, test_size=(1 - train_ratio),
                                               stratify=labels, random_state=seed)
        temp_labels = [labels[i] for i in temp_idx]
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)

        val_idx, test_idx = train_test_split(temp_idx, test_size=(1 - relative_val_ratio),
                                             stratify=temp_labels, random_state=seed)

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        test_subset = Subset(dataset, test_idx)

    elif mode == "split_folder":
        train_dataset_full = datasets.ImageFolder(root=os.path.join(dataset_dir, "Training"), transform=transform)
        test_dataset = ImageFolderWithPaths(root=os.path.join(dataset_dir, "Testing"), transform=transform)

        indices = list(range(len(train_dataset_full)))
        labels = train_dataset_full.targets
        val_ratio_effective = val_ratio / (train_ratio + val_ratio)
        train_idx, val_idx = train_test_split(indices, test_size=val_ratio_effective,
                                              stratify=labels, random_state=seed)

        train_subset = Subset(train_dataset_full, train_idx)
        val_subset = Subset(train_dataset_full, val_idx)
        test_subset = test_dataset

    elif mode == "binary_class":
        allowed_classes = ['yes', 'no']
        full_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
        class_to_idx = full_dataset.class_to_idx
        allowed_class_indices = [class_to_idx[c] for c in allowed_classes if c in class_to_idx]

        filtered_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label in allowed_class_indices]
        filtered_labels = [full_dataset.targets[i] for i in filtered_indices]

        train_idx, temp_idx = train_test_split(filtered_indices, test_size=(1 - train_ratio),
                                               stratify=filtered_labels, random_state=seed)
        temp_labels = [full_dataset.targets[i] for i in temp_idx]
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)

        val_idx, test_idx = train_test_split(temp_idx, test_size=(1 - relative_val_ratio),
                                             stratify=temp_labels, random_state=seed)

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        test_subset = Subset(full_dataset, test_idx)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use one of: 'flat', 'split_folder', 'binary_class'.")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    test_loader = (
        DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True)
        if isinstance(test_subset, Subset) or isinstance(test_subset, datasets.ImageFolder)
        else None
    )

    return train_loader, val_loader, test_loader