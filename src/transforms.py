"""
Data augmentation and transformation pipelines for Bird Classification.
"""

from typing import Tuple, Optional
from torchvision import transforms


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
    import torch
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    if tensor.device != mean.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)

    return tensor * std + mean


class Cutout:
    """Cutout data augmentation."""

    def __init__(self, n_holes: int = 1, length: int = 16):
        """
        Args:
            n_holes: Number of holes to cut out
            length: Length (pixels) of each hole
        """
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        import torch
        import numpy as np

        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask


class Mixup:
    """Mixup data augmentation for training."""

    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha

    def __call__(self, batch_x, batch_y):
        import torch
        import numpy as np

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size)

        mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
        y_a, y_b = batch_y, batch_y[index]

        return mixed_x, y_a, y_b, lam


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
    from PIL import Image
    import torch

    dummy_img = Image.new('RGB', (300, 300), color='red')

    train_tensor = train_transform(dummy_img)
    val_tensor = val_transform(dummy_img)

    print(f"\nTrain tensor shape: {train_tensor.shape}")
    print(f"Val tensor shape: {val_tensor.shape}")
    print(f"Train tensor range: [{train_tensor.min():.3f}, {train_tensor.max():.3f}]")

    print("\nAll transforms working correctly!")
