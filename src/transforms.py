"""
Data augmentation and transformation pipelines for Bird Classification.
"""

from typing import Tuple, Optional
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import TrivialAugmentWide


def get_train_transforms(
    image_size: int = 224,
    crop_scale: Tuple[float, float] = (0.8, 1.0),
    rotation_degrees: int = 15,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
    horizontal_flip_prob: float = 0.5,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """
    Get training data transformations with augmentation.

    Args:
        image_size: Target image size
        crop_scale: Scale range for random resized crop
        rotation_degrees: Max rotation degrees
        brightness: Brightness jitter factor
        contrast: Contrast jitter factor
        saturation: Saturation jitter factor
        hue: Hue jitter factor
        horizontal_flip_prob: Probability of horizontal flip
        mean: Normalization mean
        std: Normalization std

    Returns:
        Composed transforms
    """
    return transforms.Compose([
        # Spatial transforms
        transforms.RandomResizedCrop(
            size=image_size,
            scale=crop_scale,
            ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
        transforms.RandomRotation(
            degrees=rotation_degrees,
            interpolation=transforms.InterpolationMode.BILINEAR
        ),

        # Color transforms
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        ),

        # Random augmentations
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.1),

        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),

        # Random erasing for regularization
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3))
    ])


def get_val_transforms(
    image_size: int = 224,
    resize_size: int = 256,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """
    Get validation/test data transformations (no augmentation).

    Args:
        image_size: Target image size for center crop
        resize_size: Size to resize before center crop
        mean: Normalization mean
        std: Normalization std

    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(
            size=resize_size,
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(size=image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_test_transforms(
    image_size: int = 224,
    resize_size: int = 256,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """
    Get test data transformations (same as validation).

    Args:
        image_size: Target image size
        resize_size: Size to resize before center crop
        mean: Normalization mean
        std: Normalization std

    Returns:
        Composed transforms
    """
    return get_val_transforms(image_size, resize_size, mean, std)


def get_tta_transforms(
    image_size: int = 224,
    resize_size: int = 256,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> list:
    """
    Get Test-Time Augmentation (TTA) transforms.
    Returns multiple transform pipelines for ensemble prediction.

    Args:
        image_size: Target image size
        resize_size: Size to resize
        mean: Normalization mean
        std: Normalization std

    Returns:
        List of transform pipelines
    """
    base_transforms = [
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    tta_list = [
        # Original center crop
        transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        # Five crop (center + 4 corners)
        transforms.Compose([
            transforms.Resize(resize_size),
            transforms.FiveCrop(image_size),
            transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
            transforms.Lambda(lambda crops: [transforms.Normalize(mean=mean, std=std)(crop) for crop in crops])
        ])
    ]

    return tta_list


def denormalize(
    tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
):
    """
    Denormalize a tensor for visualization.

    Args:
        tensor: Normalized tensor (C, H, W)
        mean: Normalization mean
        std: Normalization std

    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    if tensor.device != mean.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)

    return tensor * std + mean


class Mixup:
    """Mixup data augmentation for training."""

    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha

    def __call__(self, batch_x, batch_y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size).to(batch_x.device)

        mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
        y_a, y_b = batch_y, batch_y[index]

        return mixed_x, y_a, y_b, lam


class CutMix:
    """CutMix data augmentation for training."""

    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter for lambda
            prob: Probability of applying CutMix
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch_x, batch_y):
        """
        Apply CutMix augmentation.

        Args:
            batch_x: Batch of images (B, C, H, W)
            batch_y: Batch of labels (B,)

        Returns:
            mixed_x, y_a, y_b, lam
        """
        if np.random.rand() > self.prob:
            return batch_x, batch_y, batch_y, 1.0

        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size).to(batch_x.device)

        # Get image dimensions
        W, H = batch_x.size(3), batch_x.size(2)

        # Generate random bounding box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply cutmix
        batch_x_clone = batch_x.clone()
        batch_x_clone[:, :, bby1:bby2, bbx1:bbx2] = batch_x[index, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda to exact area ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        return batch_x_clone, batch_y, batch_y[index], lam


def get_sota_train_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """
    SOTA augmentation for fine-grained bird classification.

    Includes:
    - Aggressive random crop
    - TrivialAugmentWide
    - Strong color jitter
    - Random erasing

    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std

    Returns:
        Composed transforms
    """
    return transforms.Compose([
        # Geometric transforms
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.6, 1.0),
            ratio=(0.75, 1.33),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=20,
            interpolation=transforms.InterpolationMode.BILINEAR
        ),

        # TrivialAugment - SOTA automatic augmentation
        TrivialAugmentWide(),

        # Additional strong color augmentation
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.15
            )
        ], p=0.8),

        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),

        # Random erasing for regularization
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])


if __name__ == "__main__":
    # Test transforms
    print("Testing transforms...")

    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    print("Train transforms:")
    print(train_transform)

    print("\nValidation transforms:")
    print(val_transform)

    # Test with a dummy image
    from PIL import Image  # Only needed for testing

    dummy_img = Image.new('RGB', (300, 300), color='red')

    train_tensor = train_transform(dummy_img)
    val_tensor = val_transform(dummy_img)

    print(f"\nTrain tensor shape: {train_tensor.shape}")
    print(f"Val tensor shape: {val_tensor.shape}")
    print(f"Train tensor range: [{train_tensor.min():.3f}, {train_tensor.max():.3f}]")

    print("\nAll transforms working correctly!")
