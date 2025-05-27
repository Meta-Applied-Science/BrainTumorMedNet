import os
import random
import argparse
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

offline_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=0,
        translate=(4 / 224, 4 / 224)
    ),
    transforms.Lambda(lambda img: F.rotate(
        img,
        angle=random.choice([
            -45, 45, -90, 90, -120, 120, -180, 180,
            -270, 270, -300, 300, -330, 330
        ])
    )),
])

def create_offline_augmented_dataset(
    src_root: str,
    dst_root: str,
    factor: int = 10
):
    assert factor >= 1, "Factor must be >= 1"
    for subdir, _, files in os.walk(src_root):
        rel = os.path.relpath(subdir, src_root)
        dst_sub = os.path.join(dst_root, rel)
        os.makedirs(dst_sub, exist_ok=True)

        for fname in files:
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                continue

            src_path = os.path.join(subdir, fname)
            img = Image.open(src_path).convert('L')  # Convert to grayscale

            base, ext = os.path.splitext(fname)
            img_rgb = img.convert('RGB')
            img_rgb.save(os.path.join(dst_sub, f"{base}_orig{ext}"))

            for i in range(factor - 1):
                aug = offline_aug(img)
                aug_rgb = aug.convert('RGB')
                aug_rgb.save(
                    os.path.join(dst_sub, f"{base}_aug{i + 1}{ext}")
                )

    print(f"Offline augmentation completed!")
    print(f"Output directory: {dst_root}")
    print(f"Total x{factor} images generated per original image.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an offline augmented dataset from original images."
    )
    parser.add_argument(
        "--src", type=str, required=True,
        help="Path to the original (source) dataset."
    )
    parser.add_argument(
        "--dst", type=str, required=True,
        help="Path where the augmented dataset will be saved."
    )
    parser.add_argument(
        "--factor", type=int, default=10,
        help="Total number of images to generate per original image (default: 10)."
    )

    args = parser.parse_args()
    create_offline_augmented_dataset(args.src, args.dst, args.factor)
