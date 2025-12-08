"""
Configuration module for Bird Classification model.
All hyperparameters and paths are managed centrally here.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import torch


@dataclass
class Config:
    """Configuration class for the Bird Classification model."""

    # ===================
    # Path Configuration
    # ===================
    project_root: str = "/Users/berkbahcetepe/Desktop/AML_Project"
    data_root: str = field(default="")
    train_csv: str = field(default="")
    test_csv: str = field(default="")
    test_path_csv: str = field(default="")
    class_names_csv: str = field(default="")
    attributes_csv: str = field(default="")
    checkpoint_dir: str = field(default="")
    log_dir: str = field(default="")
    submission_dir: str = field(default="")

    # ===================
    # Model Configuration
    # ===================
    num_classes: int = 200
    num_attributes: int = 312
    image_size: int = 224
    in_channels: int = 3

    # CNN Architecture
    base_channels: int = 64
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8)  # 64, 128, 256, 512
    num_blocks_per_stage: Tuple[int, ...] = (2, 2, 2, 2)
    dropout_rate: float = 0.3
    use_se_block: bool = True  # Squeeze-and-Excitation

    # ===================
    # Training Configuration
    # ===================
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 100

    # Optimizer
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD

    # Learning Rate Scheduler
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Loss
    label_smoothing: float = 0.1
    attribute_loss_weight: float = 0.3
    use_class_weights: bool = True

    # Regularization
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 15

    # ===================
    # Data Configuration
    # ===================
    train_val_split: float = 0.8  # 80% train, 20% val
    random_seed: int = 42
    pin_memory: bool = True

    # Augmentation
    use_augmentation: bool = True
    random_crop_scale: Tuple[float, float] = (0.8, 1.0)
    random_rotation_degrees: int = 15
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    horizontal_flip_prob: float = 0.5

    # Normalization (ImageNet stats)
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    # ===================
    # Device Configuration
    # ===================
    device: str = field(default="")
    use_amp: bool = False  # Automatic Mixed Precision

    # ===================
    # Logging Configuration
    # ===================
    log_interval: int = 10  # Log every N batches
    save_interval: int = 5  # Save checkpoint every N epochs
    use_tensorboard: bool = True
    experiment_name: str = "bird_classifier_v1"

    def __post_init__(self):
        """Initialize paths and device after dataclass initialization."""
        # Set up paths
        root = Path(self.project_root)
        data_folder = root / "aml-2025-feathers-in-focus"

        self.data_root = str(data_folder)
        self.train_csv = str(data_folder / "train_images.csv")
        self.test_csv = str(data_folder / "test_images_sample.csv")
        self.test_path_csv = str(data_folder / "test_images_path.csv")
        self.class_names_csv = str(root / "class_names.csv")
        self.attributes_csv = str(root / "attributes.csv")
        self.checkpoint_dir = str(root / "checkpoints")
        self.log_dir = str(root / "logs")
        self.submission_dir = str(root / "submissions")

        # Set up device
        if not self.device:
            self.device = self._get_device()

    def _get_device(self) -> str:
        """Automatically detect the best available device."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    @property
    def train_image_dir(self) -> str:
        """Path to training images directory."""
        return str(Path(self.data_root) / "train_images" / "train_images")

    @property
    def test_image_dir(self) -> str:
        """Path to test images directory."""
        return str(Path(self.data_root) / "test_images" / "test_images")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create config from dictionary."""
        return cls(**config_dict)

    def __str__(self) -> str:
        """String representation of config."""
        lines = ["=" * 50, "Configuration", "=" * 50]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        lines.append("=" * 50)
        return "\n".join(lines)


# Default configuration instance
def get_config(**kwargs) -> Config:
    """Get configuration with optional overrides."""
    return Config(**kwargs)


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print(config)
    print(f"\nDevice: {config.device}")
    print(f"Train images: {config.train_image_dir}")
    print(f"Test images: {config.test_image_dir}")
