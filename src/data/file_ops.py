import os
import shutil
import random
from typing import Tuple

def split_dataset(
    dataset_dir: str,
    output_dir: str,
    split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2)
) -> None:
    """
    Split a “flat” image dataset (one folder per class) into train/val/test subfolders
    according to specified ratios.

    Args:
        dataset_dir:   Path to the root folder containing one subdirectory per class.
        output_dir:    Path where the new “train”, “val”, and “test” folders will be created.
        split_ratios:  Tuple of (train_ratio, val_ratio, test_ratio). Must sum to <= 1.0.

    Behavior:
      - Creates `output_dir/train`, `output_dir/val`, `output_dir/test` (and class subfolders).
      - For each class subfolder under `dataset_dir`, shuffles its images and assigns:
          * `train_ratio` fraction to train
          * `val_ratio` fraction to val
          * Remaining images (1 - train_ratio - val_ratio) to test
      - Copies files (preserving metadata) into the respective split folders.
      - Prints a summary line per class with counts for train/val/test.

    Raises:
        ValueError: If `split_ratios` does not contain three elements or sums to >1.0.
    """
    train_ratio, val_ratio, test_ratio = split_ratios
    if not (isinstance(split_ratios, (tuple, list)) and len(split_ratios) == 3):
        raise ValueError("split_ratios must be a 3-element tuple or list")
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio > 1.0 + 1e-6:
        raise ValueError(f"Split ratios sum to {total_ratio:.3f}, which exceeds 1.0")

    # Prepare output directories
    split_dirs = {
        "train": os.path.join(output_dir, "train"),
        "val":   os.path.join(output_dir, "val"),
        "test":  os.path.join(output_dir, "test")
    }
    for path in split_dirs.values():
        os.makedirs(path, exist_ok=True)

    # Process each class folder
    for class_entry in os.scandir(dataset_dir):
        if not class_entry.is_dir():
            continue

        img_files = [
            f for f in os.listdir(class_entry.path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not img_files:
            continue

        random.shuffle(img_files)
        total = len(img_files)
        n_train = int(train_ratio * total)
        n_val   = int(val_ratio   * total)
        # test gets the remainder: total - n_train - n_val

        subsets = {
            "train": img_files[:n_train],
            "val":   img_files[n_train:n_train + n_val],
            "test":  img_files[n_train + n_val:]
        }

        # Copy into split folders
        for split, files in subsets.items():
            dest_class_dir = os.path.join(split_dirs[split], class_entry.name)
            os.makedirs(dest_class_dir, exist_ok=True)
            for fname in files:
                src = os.path.join(class_entry.path, fname)
                dst = os.path.join(dest_class_dir, fname)
                shutil.copy2(src, dst)

        print(
            f"{class_entry.name}: "
            f"Train={len(subsets['train'])}, "
            f"Val={len(subsets['val'])}, "
            f"Test={len(subsets['test'])}"
        )


def merge_folders(source_dirs: list[str], destination_root: str) -> None:
    """
    Merge multiple datasets organized by class into a single consolidated dataset.

    Args:
        source_dirs:       List of root directories, each containing identical class subfolders.
        destination_root:  Path where merged class subfolders will be created.

    Behavior:
      - Ensures `destination_root` exists.
      - Iterates over each class name found in the first source directory.
      - For each source directory:
          * If the class folder exists, copies all files into `destination_root/<class_name>`.
          * On filename conflicts, appends `_1`, `_2`, etc. to avoid overwriting.
      - Prints a confirmation line per class after merging.

    Raises:
        None (conflicts are handled by renaming).
    """
    os.makedirs(destination_root, exist_ok=True)

    for class_name in os.listdir(source_dirs[0]):
        if class_name == ".DS_Store":
            continue

        merged_dir = os.path.join(destination_root, class_name)
        os.makedirs(merged_dir, exist_ok=True)

        for source_dir in source_dirs:
            class_path = os.path.join(source_dir, class_name)
            if os.path.exists(class_path):
                for filename in os.listdir(class_path):
                    src = os.path.join(class_path, filename)
                    dst = os.path.join(merged_dir, filename)

                    # Handle name conflicts
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dst):
                        dst = os.path.join(merged_dir, f"{base}_{counter}{ext}")
                        counter += 1

                    shutil.copy2(src, dst)

        print(f"Merged: {class_name} => {merged_dir}")
