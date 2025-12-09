"""
Utility functions for Bird Classification model.
Includes seed setting, logging, device management, and helper functions.
"""

import os
import random
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(preferred: str = "auto") -> torch.device:
    """
    Get the best available device.

    Args:
        preferred: Preferred device ("auto", "mps", "cuda", "cpu")

    Returns:
        torch.device object
    """
    if preferred == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(preferred)


def setup_logging(
    log_dir: str,
    experiment_name: str,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logging for the experiment.

    Args:
        log_dir: Directory to save logs
        experiment_name: Name of the experiment
        level: Logging level

    Returns:
        Logger instance
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{experiment_name}_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")

    return logger


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_parameters(num_params: int) -> str:
    """
    Format parameter count for display.

    Args:
        num_params: Number of parameters

    Returns:
        Formatted string (e.g., "5.2M")
    """
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False
) -> str:
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Dictionary of metrics
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
        is_best: Whether this is the best model so far

    Returns:
        Path to saved checkpoint
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }

    # Save regular checkpoint
    save_path = checkpoint_path / filename
    torch.save(checkpoint, save_path)

    # Save best model separately
    if is_best:
        best_path = checkpoint_path / "best_model.pth"
        torch.save(checkpoint, best_path)

    return str(save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load to

    Returns:
        Dictionary with epoch and metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint['metrics']
    }


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = "Meter"):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()
        return self

    def stop(self):
        self.end_time = time.time()
        return self

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        return self.format_time(self.elapsed)

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max"
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for metric direction
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def print_model_summary(model: nn.Module, input_size: tuple = (1, 3, 224, 224)):
    """
    Print a summary of the model architecture.

    Args:
        model: PyTorch model
        input_size: Input tensor size
    """
    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Total Parameters: {format_parameters(count_parameters(model, trainable_only=False))}")
    print(f"Trainable Parameters: {format_parameters(count_parameters(model, trainable_only=True))}")
    print("=" * 60)
    print("\nModel Architecture:")
    print(model)
    print("=" * 60)


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")

    # Test seed
    set_seed(42)
    print(f"Random number: {random.random()}")

    # Test device
    device = get_device()
    print(f"Device: {device}")

    # Test timer
    with Timer() as t:
        time.sleep(1)
    print(f"Elapsed: {t}")

    # Test average meter
    meter = AverageMeter("Loss")
    meter.update(0.5)
    meter.update(0.3)
    print(meter)

    print("\nAll utilities working correctly!")
