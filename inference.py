#!/usr/bin/env python3
"""
Inference script for Bird Classification model.
Generates predictions for test set and creates Kaggle submission file.

Usage:
    python inference.py
    python inference.py --checkpoint checkpoints/best_model.pth
    python inference.py --checkpoint checkpoints/best_model.pth --tta
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.model import BirdClassifier, LightBirdClassifier, BirdClassifierV2
from src.transforms import get_val_transforms, get_tta_transforms
from src.utils import set_seed
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class DirectTestDataset(Dataset):
    """Test dataset that loads images directly by ID (1.jpg, 2.jpg, etc.)."""

    def __init__(self, image_dir, num_images, transform):
        self.image_dir = image_dir
        self.num_images = num_images
        self.transform = transform

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # ID is 1-indexed (1 to 4000)
        img_id = idx + 1
        img_path = f"{self.image_dir}/{img_id}.jpg"

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='gray')

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'id': img_id}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate predictions for bird classification')

    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='v2',
                        choices=['standard', 'light', 'v2'],
                        help='Model type: standard, light, or v2')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent dimension for v2 model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--tta', action='store_true',
                        help='Use Test-Time Augmentation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output submission file path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Get predictions for a batch of images.

    Args:
        model: Trained model
        images: Batch of images
        device: Device

    Returns:
        Predicted class indices (1-indexed for submission)
    """
    images = images.to(device)
    outputs = model(images)
    logits = outputs['class_logits']

    # Get predictions (convert to 1-indexed)
    predictions = logits.argmax(dim=1) + 1

    return predictions.cpu()


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    image_path: str,
    tta_transforms: list,
    device: torch.device
) -> int:
    """
    Get prediction with Test-Time Augmentation.

    Args:
        model: Trained model
        image_path: Path to image
        tta_transforms: List of TTA transforms
        device: Device

    Returns:
        Predicted class (1-indexed)
    """
    image = Image.open(image_path).convert('RGB')

    all_logits = []

    for transform in tta_transforms:
        transformed = transform(image)

        # Handle FiveCrop which returns list
        if isinstance(transformed, list):
            for t in transformed:
                t = t.unsqueeze(0).to(device)
                outputs = model(t)
                all_logits.append(outputs['class_logits'])
        else:
            transformed = transformed.unsqueeze(0).to(device)
            outputs = model(transformed)
            all_logits.append(outputs['class_logits'])

    # Average logits
    avg_logits = torch.cat(all_logits, dim=0).mean(dim=0, keepdim=True)
    prediction = avg_logits.argmax(dim=1).item() + 1  # 1-indexed

    return prediction


def main():
    """Main inference function."""
    args = parse_args()
    set_seed(args.seed)

    # Config
    config = Config()

    # Device
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")

    if args.model == 'standard':
        model = BirdClassifier(
            num_classes=config.num_classes,
            num_attributes=config.num_attributes
        )
    elif args.model == 'light':
        model = LightBirdClassifier(
            num_classes=config.num_classes,
            num_attributes=config.num_attributes
        )
    elif args.model == 'v2':
        model = BirdClassifierV2(
            num_classes=config.num_classes,
            num_attributes=config.num_attributes,
            latent_dim=args.latent_dim,
            use_se=True,
            use_gated_attrs=False
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Create test dataset
    print("Loading test data...")
    test_transform = get_val_transforms(
        image_size=config.image_size,
        mean=config.normalize_mean,
        std=config.normalize_std
    )

    # Load test paths
    test_df = pd.read_csv(config.test_csv)
    test_path_df = pd.read_csv(config.test_path_csv)

    # Merge to get image paths
    if 'image_path' not in test_df.columns and len(test_path_df) > 0:
        test_df['image_path'] = test_path_df['image_path']

    print(f"Test samples: {len(test_df)}")

    # Generate predictions
    predictions = []

    if args.tta:
        print("Using Test-Time Augmentation...")
        tta_transforms = get_tta_transforms(
            image_size=config.image_size,
            mean=config.normalize_mean,
            std=config.normalize_std
        )

        for idx in tqdm(range(len(test_df)), desc="Predicting"):
            row = test_df.iloc[idx]
            img_path = row['image_path']
            # Fix path
            img_name = img_path.split('/')[-1]
            full_path = Path(config.test_image_dir) / img_name

            pred = predict_with_tta(model, str(full_path), tta_transforms, device)
            predictions.append({
                'id': row['id'],
                'label': pred
            })
    else:
        print("Standard inference...")

        # FIXED: Use direct ID -> filename mapping
        # Kaggle expects id=1 to predict for 1.jpg, id=2 for 2.jpg, etc.
        # Create dataset with direct ID mapping
        test_dataset = DirectTestDataset(
            image_dir=config.test_image_dir,
            num_images=4000,  # 4000 test images
            transform=test_transform
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=False  # Disabled for MPS compatibility
        )

        all_ids = []
        all_preds = []

        for batch in tqdm(test_loader, desc="Predicting"):
            images = batch['image'].to(device)
            ids = batch['id']

            outputs = model(images)
            preds = outputs['class_logits'].argmax(dim=1) + 1  # 1-indexed

            all_ids.extend(ids.tolist() if torch.is_tensor(ids) else ids)
            all_preds.extend(preds.cpu().tolist())

        predictions = [{'id': id_, 'label': pred}
                       for id_, pred in zip(all_ids, all_preds)]

    # Create submission dataframe
    submission_df = pd.DataFrame(predictions)
    submission_df = submission_df.sort_values('id').reset_index(drop=True)

    # Save submission
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(config.submission_dir) / f"submission_{timestamp}.csv"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    print(f"\nSubmission saved to: {output_path}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nFirst 5 predictions:")
    print(submission_df.head())

    # Print label distribution
    print(f"\nPrediction distribution:")
    print(f"  Unique labels: {submission_df['label'].nunique()}")
    print(f"  Min label: {submission_df['label'].min()}")
    print(f"  Max label: {submission_df['label'].max()}")

    return output_path


if __name__ == "__main__":
    main()
