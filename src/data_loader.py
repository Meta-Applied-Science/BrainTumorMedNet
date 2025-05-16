import os
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def create_dataloaders(dataset_dir: str,
                       transform,
                       batch_size: int,
                       num_workers: int,
                       train_ratio: float,
                       val_ratio: float,
                       test_ratio: float,
                       seed: int):


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
    
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
    
    return train_dataloader, val_dataloader, test_dataloader

def create_dataloaders_2(dataset_dir: str,
                       transform,
                       batch_size: int,
                       num_workers: int,
                       val_ratio: float,
                       seed: int):

    train_dataset_full = datasets.ImageFolder(root=os.path.join(dataset_dir, "Training"), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, "Testing"), transform=transform)

    indices = list(range(len(train_dataset_full)))
    labels = train_dataset_full.targets

    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, stratify=labels, random_state=seed)

    train_subset = Subset(train_dataset_full, train_idx)
    val_subset = Subset(train_dataset_full, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def create_dataloaders_3(dataset_dir: str,
                       transform,
                       batch_size: int,
                       num_workers: int,
                       train_ratio: float,
                       val_ratio: float,
                       test_ratio: float,
                       seed: int):


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

    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader