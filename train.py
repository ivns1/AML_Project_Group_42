#!/usr/bin/env python3
"""
Training script for Bird Classification model.

Usage:
    python train.py
    python train.py --epochs 50 --batch_size 64
    python train.py --resume checkpoints/checkpoint_epoch_10.pth
"""

import argparse
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.dataset import create_dataloaders, load_class_attribute_matrix
from src.model import (
    BirdClassifier, LightBirdClassifier, BirdClassifierV2,
    get_model_info, create_model
)
from src.losses import CombinedLoss, CombinedLossV2
from src.trainer import Trainer, create_optimizer, create_scheduler
from src.utils import set_seed, setup_logging, print_model_summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Bird Classification Model')

    # Model
    parser.add_argument('--model', type=str, default='v2',
                        choices=['standard', 'light', 'v2', 'v2_gated'],
                        help='Model type: standard (~5M), light (~2.5M), v2 (shared latent ~4.7M), v2_gated (with attr gates)')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent dimension for v2 models')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')

    # Loss
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    parser.add_argument('--attr_weight', type=float, default=0.3,
                        help='Attribute loss weight (for CombinedLoss)')
    parser.add_argument('--use_enhanced_loss', action='store_true',
                        help='Use CombinedLossV2 with consistency loss (recommended)')
    parser.add_argument('--no_enhanced_loss', action='store_true',
                        help='Disable enhanced loss, use standard CombinedLoss instead')
    parser.add_argument('--consistency_weight', type=float, default=0.1,
                        help='Consistency loss weight')
    parser.add_argument('--attr_weight_initial', type=float, default=0.5,
                        help='Initial attribute weight (for scheduling)')
    parser.add_argument('--attr_weight_final', type=float, default=0.1,
                        help='Final attribute weight (after decay)')

    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine_warmup',
                        choices=['cosine', 'cosine_warmup', 'step', 'plateau', 'none'],
                        help='Learning rate scheduler')

    # Data
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')

    # Augmentation
    parser.add_argument('--sota_aug', action='store_true', default=False,
                        help='Use SOTA augmentation (TrivialAugmentWide + stronger transforms)')
    parser.add_argument('--cutmix', action='store_true', default=False,
                        help='Enable CutMix augmentation')
    parser.add_argument('--mixup', action='store_true', default=False,
                        help='Enable MixUp augmentation')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                        help='CutMix alpha parameter')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='MixUp alpha parameter')

    # Checkpointing
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--experiment_name', type=str, default='bird_classifier',
                        help='Experiment name for logging')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Create config with CLI overrides
    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        attribute_loss_weight=args.attr_weight,
        scheduler=args.scheduler,
        num_workers=args.num_workers,
        image_size=args.image_size,
        save_interval=args.save_interval,
        random_seed=args.seed,
        experiment_name=args.experiment_name,
        # Augmentation settings
        use_sota_augmentation=args.sota_aug,
        use_cutmix=args.cutmix,
        use_mixup=args.mixup,
        cutmix_alpha=args.cutmix_alpha,
        mixup_alpha=args.mixup_alpha
    )

    # Set seed for reproducibility
    set_seed(config.random_seed)

    # Setup logging
    logger = setup_logging(config.log_dir, config.experiment_name)
    logger.info(f"Configuration:\n{config}")

    # Device
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    # Check MPS availability
    if config.device == "mps":
        if torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon GPU) is available and will be used")
        else:
            logger.warning("MPS not available, falling back to CPU")
            device = torch.device("cpu")

    # Log augmentation settings
    if config.use_sota_augmentation:
        logger.info("Using SOTA augmentation (TrivialAugmentWide + strong transforms)")
    if config.use_cutmix:
        logger.info(f"CutMix enabled (alpha={config.cutmix_alpha})")
    if config.use_mixup:
        logger.info(f"MixUp enabled (alpha={config.mixup_alpha})")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    logger.info(f"Train: {len(train_loader.dataset)} samples")
    logger.info(f"Val: {len(val_loader.dataset)} samples")
    logger.info(f"Test: {len(test_loader.dataset)} samples")

    # Create model
    logger.info(f"Creating {args.model} model...")
    if args.model == 'standard':
        model = BirdClassifier(
            num_classes=config.num_classes,
            num_attributes=config.num_attributes,
            dropout_rate=config.dropout_rate,
            use_se=config.use_se_block
        )
    elif args.model == 'light':
        model = LightBirdClassifier(
            num_classes=config.num_classes,
            num_attributes=config.num_attributes,
            dropout_rate=config.dropout_rate
        )
    elif args.model in ['v2', 'v2_gated']:
        model = BirdClassifierV2(
            num_classes=config.num_classes,
            num_attributes=config.num_attributes,
            latent_dim=args.latent_dim,
            use_se=config.use_se_block,
            use_gated_attrs=(args.model == 'v2_gated'),
            dropout_rate=config.dropout_rate
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model_info = get_model_info(model)
    logger.info(f"Model parameters: {model_info['total_params_str']}")

    # Create loss function
    # Use enhanced loss by default unless --no_enhanced_loss is specified
    use_enhanced = args.use_enhanced_loss or not args.no_enhanced_loss
    if use_enhanced:
        # Load class-attribute matrix for consistency loss
        class_attr_matrix = load_class_attribute_matrix(config.attributes_csv)
        logger.info(f"Loaded class-attribute matrix: {class_attr_matrix.shape}")

        criterion = CombinedLossV2(
            class_attribute_matrix=class_attr_matrix,
            num_classes=config.num_classes,
            label_smoothing=config.label_smoothing,
            attr_weight_initial=args.attr_weight_initial,
            attr_weight_final=args.attr_weight_final,
            attr_weight_decay_epochs=config.attr_weight_decay_epochs,
            consistency_weight=args.consistency_weight,
            consistency_temperature=config.consistency_loss_temperature
        )
        logger.info("Using CombinedLossV2 with consistency loss and attribute scheduling")
    else:
        criterion = CombinedLoss(
            num_classes=config.num_classes,
            label_smoothing=config.label_smoothing,
            attribute_weight=config.attribute_loss_weight
        )
        logger.info("Using original CombinedLoss")

    # Create optimizer
    optimizer = create_optimizer(model, config)
    logger.info(f"Optimizer: {config.optimizer}")

    # Create scheduler
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    logger.info(f"Scheduler: {config.scheduler}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Print model summary
    print_model_summary(model)

    # Train
    logger.info("Starting training...")
    history = trainer.train()

    # Final evaluation
    logger.info("Final evaluation on validation set...")
    final_metrics = trainer.evaluate(val_loader)
    logger.info(f"Final validation metrics: {final_metrics}")

    # Save final model
    final_path = Path(config.checkpoint_dir) / "final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'history': history,
        'final_metrics': final_metrics
    }, final_path)
    logger.info(f"Final model saved to: {final_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Accuracy: {trainer.best_metric:.4f}")
    print(f"Best Epoch: {trainer.best_epoch + 1}")
    print(f"Model saved: {final_path}")
    print("=" * 60)

    return trainer.best_metric


if __name__ == "__main__":
    main()
