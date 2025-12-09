"""
Trainer class for Bird Classification model.
Handles training loop, validation, checkpointing, and logging.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from .config import Config

logger = logging.getLogger(__name__)
from .utils import (
    AverageMeter, Timer, EarlyStopping,
    save_checkpoint, load_checkpoint, get_lr
)
from .metrics import MetricTracker, accuracy, top_k_accuracy


class Trainer:
    """
    Trainer class for model training and evaluation.

    Features:
    - Training and validation loops
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Progress logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        config: Config,
        device: torch.device
    ):
        """
        Args:
            model: Neural network model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Configuration object
            device: Training device
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.best_epoch = 0

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode='max'
        )

        # Logging
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }

        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.writer = None
        if config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=config.log_dir)
            except ImportError:
                logger.warning("TensorBoard not available, skipping...")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        loss_meter = AverageMeter('Loss')
        acc_meter = AverageMeter('Acc')
        attr_loss_meter = AverageMeter('AttrLoss')

        # Initialize CutMix/MixUp if enabled
        cutmix = None
        mixup = None
        if self.config.use_cutmix:
            from .transforms import CutMix
            cutmix = CutMix(alpha=self.config.cutmix_alpha, prob=self.config.cutmix_prob)
        if self.config.use_mixup:
            from .transforms import Mixup
            mixup = Mixup(alpha=self.config.mixup_alpha)

        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch + 1}/{self.config.epochs} [Train]',
            leave=False
        )

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Apply CutMix or MixUp
            use_mixed = False
            lam = 1.0
            y_a, y_b = labels, labels

            if cutmix is not None or mixup is not None:
                rand_val = np.random.rand()
                if rand_val < 0.5 and cutmix is not None:
                    images, y_a, y_b, lam = cutmix(images, labels)
                    use_mixed = True
                elif mixup is not None:
                    images, y_a, y_b, lam = mixup(images, labels)
                    use_mixed = True

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)

            if use_mixed:
                # Mixed loss for CutMix/MixUp
                targets_a = {'label': y_a}
                targets_b = {'label': y_b}
                if 'attributes' in batch:
                    attrs = batch['attributes'].to(self.device)
                    targets_a['attributes'] = attrs
                    targets_b['attributes'] = attrs

                loss_dict_a = self.criterion(outputs, targets_a, epoch=self.current_epoch)
                loss_dict_b = self.criterion(outputs, targets_b, epoch=self.current_epoch)
                loss = lam * loss_dict_a['total'] + (1 - lam) * loss_dict_b['total']
                loss_dict = {'total': loss}

                # For attr_loss tracking, use weighted average
                if 'attr_loss' in loss_dict_a:
                    loss_dict['attr_loss'] = lam * loss_dict_a['attr_loss'] + (1 - lam) * loss_dict_b['attr_loss']
            else:
                # Standard loss
                targets = {'label': labels}
                if 'attributes' in batch:
                    targets['attributes'] = batch['attributes'].to(self.device)

                loss_dict = self.criterion(outputs, targets, epoch=self.current_epoch)
                loss = loss_dict['total']

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )

            self.optimizer.step()

            # Update metrics (use original labels for accuracy)
            batch_acc = accuracy(outputs['class_logits'], labels)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(batch_acc, images.size(0))

            if 'attr_loss' in loss_dict:
                attr_loss_val = loss_dict['attr_loss']
                if isinstance(attr_loss_val, torch.Tensor):
                    attr_loss_val = attr_loss_val.item()
                attr_loss_meter.update(attr_loss_val, images.size(0))

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}',
                'lr': f'{get_lr(self.optimizer):.6f}'
            })

        return {
            'loss': loss_meter.avg,
            'accuracy': acc_meter.avg,
            'attr_loss': attr_loss_meter.avg
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        loss_meter = AverageMeter('Loss')
        tracker = MetricTracker(self.config.num_classes)

        pbar = tqdm(
            self.val_loader,
            desc=f'Epoch {self.current_epoch + 1}/{self.config.epochs} [Val]',
            leave=False
        )

        for batch in pbar:
            # Move data to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            targets = {'label': labels}
            if 'attributes' in batch:
                targets['attributes'] = batch['attributes'].to(self.device)

            # Forward pass
            outputs = self.model(images)

            # Compute loss (pass epoch for consistency)
            loss_dict = self.criterion(outputs, targets, epoch=self.current_epoch)
            loss_meter.update(loss_dict['total'].item(), images.size(0))

            # Track metrics
            tracker.update(outputs['class_logits'], labels, loss_dict)

            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

        # Compute all metrics
        metrics = tracker.compute()
        metrics['loss'] = loss_meter.avg

        return metrics

    def train(self) -> Dict[str, Any]:
        """
        Full training loop.

        Returns:
            Training history
        """
        logger.info("=" * 60)
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        logger.info("=" * 60)

        timer = Timer().start()

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_timer = Timer().start()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['lr'].append(get_lr(self.optimizer))

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
                self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('LR', get_lr(self.optimizer), epoch)

            # Check if best model
            is_best = val_metrics['accuracy'] > self.best_metric
            if is_best:
                self.best_metric = val_metrics['accuracy']
                self.best_epoch = epoch

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    checkpoint_dir=self.config.checkpoint_dir,
                    filename=f'checkpoint_epoch_{epoch+1}.pth',
                    is_best=is_best
                )

            # Log epoch summary
            epoch_timer.stop()
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs} | Time: {epoch_timer}")
            logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                        f"Top5: {val_metrics.get('top5_accuracy', 0):.4f}")
            logger.info(f"  LR: {get_lr(self.optimizer):.6f}")
            if is_best:
                logger.info("  ** New best model! **")

            # Early stopping
            if self.early_stopping(val_metrics['accuracy']):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        timer.stop()
        logger.info("=" * 60)
        logger.info(f"Training completed in {timer}")
        logger.info(f"Best validation accuracy: {self.best_metric:.4f} at epoch {self.best_epoch + 1}")
        logger.info("=" * 60)

        if self.writer:
            self.writer.close()

        return self.history

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint to resume training."""
        info = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            str(self.device)
        )
        self.current_epoch = info['epoch'] + 1
        self.best_metric = info['metrics'].get('accuracy', 0)
        logger.info(f"Loaded checkpoint from epoch {info['epoch'] + 1}")

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            data_loader: Data loader for evaluation

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        tracker = MetricTracker(self.config.num_classes)

        for batch in tqdm(data_loader, desc='Evaluating'):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            outputs = self.model(images)
            tracker.update(outputs['class_logits'], labels)

        return tracker.compute()


def create_optimizer(
    model: nn.Module,
    config: Config
) -> Optimizer:
    """
    Create optimizer based on config.

    Args:
        model: Neural network model
        config: Configuration object

    Returns:
        Optimizer instance
    """
    if config.optimizer.lower() == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def create_scheduler(
    optimizer: Optimizer,
    config: Config,
    steps_per_epoch: int
) -> Optional[_LRScheduler]:
    """
    Create learning rate scheduler based on config.

    Args:
        optimizer: Optimizer
        config: Configuration object
        steps_per_epoch: Number of batches per epoch

    Returns:
        Scheduler instance or None
    """
    if config.scheduler.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.min_lr
        )
    elif config.scheduler.lower() == 'cosine_warmup':
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=config.warmup_epochs
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=config.epochs - config.warmup_epochs,
            eta_min=config.min_lr
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[config.warmup_epochs]
        )
    elif config.scheduler.lower() == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    elif config.scheduler.lower() == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=config.min_lr
        )
    elif config.scheduler.lower() == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Trainer module loaded successfully!")
    logger.info("Use this module to train your bird classification model.")
