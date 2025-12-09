"""
Dataset and DataLoader for Bird Classification.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch

logger = logging.getLogger(__name__)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .config import Config
from .transforms import get_train_transforms, get_val_transforms, get_test_transforms, get_sota_train_transforms


class BirdDataset(Dataset):
    """
    Dataset class for bird images.

    Supports:
    - Training with labels and attributes
    - Validation with labels
    - Test without labels (for submission)
    """

    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        attributes_csv: Optional[str] = None,
        transform=None,
        mode: str = "train",
        indices: Optional[List[int]] = None
    ):
        """
        Args:
            csv_file: Path to CSV file with image paths and labels
            image_dir: Directory containing images
            attributes_csv: Path to attributes CSV (optional)
            transform: Torchvision transforms
            mode: 'train', 'val', or 'test'
            indices: Optional list of indices to use (for train/val split)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode

        # Load data
        self.data = pd.read_csv(csv_file)

        # Apply indices filter if provided
        if indices is not None:
            self.data = self.data.iloc[indices].reset_index(drop=True)

        # Load attributes if provided
        self.attributes = None
        if attributes_csv and os.path.exists(attributes_csv):
            attr_df = pd.read_csv(attributes_csv)
            # Convert to numpy array (Class_id, 1-312 attributes)
            self.attributes = attr_df.iloc[:, 1:].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
            - 'image': Image tensor
            - 'label': Class label (0-indexed)
            - 'attributes': Attribute vector (if available)
            - 'image_path': Original image path
        """
        row = self.data.iloc[idx]

        # Get image path
        if self.mode == "test":
            # Test mode - use image path from test_images_path.csv format
            if 'image_path' in row:
                img_name = os.path.basename(row['image_path'])
            else:
                img_name = f"{row['id']}.jpg"
        else:
            # Train/val mode
            img_path = row['image_path']
            img_name = os.path.basename(img_path)

        full_path = os.path.join(self.image_dir, img_name)

        # Load image
        try:
            image = Image.open(full_path).convert('RGB')
        except (FileNotFoundError, IOError, UnidentifiedImageError) as e:
            logger.error(f"Error loading image {full_path}: {e}")
            # Return a placeholder
            image = Image.new('RGB', (224, 224), color='gray')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Prepare output
        output = {
            'image': image,
            'image_path': full_path
        }

        # Add label for train/val
        if self.mode != "test" and 'label' in row:
            label = int(row['label']) - 1  # Convert to 0-indexed
            output['label'] = torch.tensor(label, dtype=torch.long)

            # Add attributes if available
            if self.attributes is not None:
                attr_vector = self.attributes[label]  # Get attributes for this class
                output['attributes'] = torch.tensor(attr_vector, dtype=torch.float32)
        else:
            # For test mode, include id for submission
            if 'id' in row:
                output['id'] = row['id']

        return output


def create_dataloaders(
    config: Config,
    train_transform=None,
    val_transform=None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Configuration object
        train_transform: Optional custom train transforms
        val_transform: Optional custom val transforms

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set default transforms
    if train_transform is None:
        if config.use_sota_augmentation:
            train_transform = get_sota_train_transforms(
                image_size=config.image_size,
                mean=config.normalize_mean,
                std=config.normalize_std
            )
        else:
            train_transform = get_train_transforms(
                image_size=config.image_size,
                crop_scale=config.random_crop_scale,
                rotation_degrees=config.random_rotation_degrees,
                brightness=config.color_jitter_brightness,
                contrast=config.color_jitter_contrast,
                saturation=config.color_jitter_saturation,
                mean=config.normalize_mean,
                std=config.normalize_std
            )

    if val_transform is None:
        val_transform = get_val_transforms(
            image_size=config.image_size,
            mean=config.normalize_mean,
            std=config.normalize_std
        )

    # Load full training data to get labels for stratified split
    full_data = pd.read_csv(config.train_csv)
    labels = full_data['label'].values

    # Create stratified train/val split
    train_indices, val_indices = train_test_split(
        range(len(full_data)),
        test_size=1 - config.train_val_split,
        stratify=labels,
        random_state=config.random_seed
    )

    # Create datasets
    train_dataset = BirdDataset(
        csv_file=config.train_csv,
        image_dir=config.train_image_dir,
        attributes_csv=config.attributes_csv,
        transform=train_transform,
        mode="train",
        indices=train_indices
    )

    val_dataset = BirdDataset(
        csv_file=config.train_csv,
        image_dir=config.train_image_dir,
        attributes_csv=config.attributes_csv,
        transform=val_transform,
        mode="val",
        indices=val_indices
    )

    # Test dataset
    test_dataset = BirdDataset(
        csv_file=config.test_csv,
        image_dir=config.test_image_dir,
        attributes_csv=None,
        transform=val_transform,
        mode="test"
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


def get_class_names(class_names_csv: str) -> Dict[int, str]:
    """
    Load class names mapping.

    Args:
        class_names_csv: Path to class names CSV

    Returns:
        Dictionary mapping label (1-indexed) to class name
    """
    df = pd.read_csv(class_names_csv)
    return dict(zip(df['Label'], df['Class_name']))


def load_class_attribute_matrix(attributes_csv: str) -> torch.Tensor:
    """
    Load the class-attribute matrix for consistency loss.

    Args:
        attributes_csv: Path to attributes CSV file (200 classes x 312 attributes)

    Returns:
        Tensor of shape (200, 312) with class-attribute probabilities
    """
    attr_df = pd.read_csv(attributes_csv)
    # Skip the Class_id column, get only attribute columns
    matrix = attr_df.iloc[:, 1:].values.astype(np.float32)
    return torch.tensor(matrix)


def get_class_weights(train_loader: DataLoader, num_classes: int = 200) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset.

    Args:
        train_loader: Training dataloader
        num_classes: Number of classes

    Returns:
        Tensor of class weights
    """
    class_counts = torch.zeros(num_classes)

    for batch in train_loader:
        labels = batch['label']
        for label in labels:
            class_counts[label] += 1

    # Inverse frequency weighting
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes

    return weights


def visualize_batch(
    batch: Dict[str, torch.Tensor],
    class_names: Optional[Dict[int, str]] = None,
    num_samples: int = 8,
    save_path: Optional[str] = None
):
    """
    Visualize a batch of images.

    Args:
        batch: Batch dictionary from dataloader
        class_names: Optional class name mapping
        num_samples: Number of samples to show
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    from .transforms import denormalize

    images = batch['image'][:num_samples]
    labels = batch.get('label', None)

    # Denormalize images
    images = denormalize(images)
    images = images.permute(0, 2, 3, 1).numpy()
    images = np.clip(images, 0, 1)

    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img)
        ax.axis('off')

        if labels is not None:
            label = labels[i].item() + 1  # Convert back to 1-indexed
            if class_names:
                title = f"{label}: {class_names.get(label, 'Unknown')}"
            else:
                title = f"Label: {label}"
            ax.set_title(title, fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Test dataset
    from .config import Config

    config = Config()
    print(f"Creating dataloaders...")

    train_loader, val_loader, test_loader = create_dataloaders(config)

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Test batch
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Label shape: {batch['label'].shape}")

    if 'attributes' in batch:
        print(f"Attributes shape: {batch['attributes'].shape}")

    print("\nDataset test complete!")
