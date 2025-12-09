# Bird Classification - Feathers in Focus

A custom deep learning pipeline for fine-grained bird species classification, developed for the UvA Applied Machine Learning course.

## Overview

This project implements a custom CNN architecture for classifying 200 bird species using multi-task learning with attribute prediction. The model uses a shared latent representation to jointly predict class labels and 312 visual attributes.

### Key Features

- **Custom CNN Architecture**: BirdClassifierV2 with SE attention blocks (~4.7M parameters)
- **Multi-task Learning**: Joint classification and attribute prediction
- **Consistency Loss**: Enforces alignment between class and attribute predictions
- **Comprehensive Augmentation**: CutMix, MixUp, TrivialAugmentWide support
- **Test-Time Augmentation (TTA)**: Ensemble predictions for improved accuracy

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd AML_Project

# Install dependencies
pip install torch torchvision pandas numpy tqdm pillow tensorboard
```

## Project Structure

```
AML_Project/
├── src/                    # Source code modules
│   ├── config.py          # Configuration management
│   ├── dataset.py         # Dataset and dataloader
│   ├── losses.py          # Loss functions (multi-task)
│   ├── metrics.py         # Evaluation metrics
│   ├── model.py           # Model architectures
│   ├── trainer.py         # Training loop
│   ├── transforms.py      # Data augmentation
│   └── utils.py           # Utility functions
├── train.py               # Training script
├── inference.py           # Inference and submission
├── attributes.csv         # Class-attribute matrix
└── class_names.csv        # Bird species names
```

## Usage

### Training

```bash
# Default training with BirdClassifierV2
python train.py

# With custom parameters
python train.py --model v2 --epochs 100 --batch_size 32 --lr 1e-3

# With SOTA augmentation
python train.py --sota_aug --cutmix --mixup

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_50.pth
```

### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | v2 | Model type: standard, light, v2, v2_gated |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--scheduler` | cosine_warmup | LR scheduler type |
| `--sota_aug` | False | Use SOTA augmentation |
| `--cutmix` | False | Enable CutMix |
| `--mixup` | False | Enable MixUp |

### Inference

```bash
# Generate predictions
python inference.py --checkpoint checkpoints/best_model.pth

# With Test-Time Augmentation
python inference.py --checkpoint checkpoints/best_model.pth --tta
```

## Model Architecture

### BirdClassifierV2

The main model architecture features:

1. **Stem**: Initial convolution block for feature extraction
2. **Backbone**: 4-stage residual network with SE attention blocks
3. **Shared Latent Head**: Projects features to a shared latent space
4. **Dual Heads**:
   - Classification head: Predicts 200 bird species
   - Attribute head: Predicts 312 visual attributes

```
Input (224x224x3)
    ↓
Stem (Conv + BN + ReLU + MaxPool)
    ↓
Stage 1-4 (ResBlocks with SE attention)
    ↓
Global Average Pooling
    ↓
Shared Latent Space (128-dim)
    ↓
┌─────────────────┴─────────────────┐
↓                                   ↓
Classification Head            Attribute Head
(200 classes)                  (312 attributes)
```

### Loss Function (CombinedLossV2)

The loss combines three components:

```
L_total = L_class + λ_attr(t) × L_attr + λ_cons × L_consistency
```

- **L_class**: Label-smoothed cross-entropy for classification
- **L_attr**: Binary cross-entropy for attribute prediction
- **L_consistency**: KL divergence between class and attribute-based predictions
- **λ_attr(t)**: Scheduled weight (decays from 0.5 to 0.1 over training)

## Results

| Model | Val Accuracy | Parameters | Notes |
|-------|-------------|------------|-------|
| BirdClassifierV2 | ~26% | 4.7M | With consistency loss |

## Data

The dataset contains:
- **Training**: 3,926 images across 200 bird species
- **Test**: 4,000 images for Kaggle submission
- **Attributes**: 312 visual attributes per class

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed data documentation.

## Configuration

Key configuration parameters in `src/config.py`:

```python
# Model
num_classes: 200
num_attributes: 312
image_size: 224

# Training
batch_size: 32
epochs: 100
lr: 1e-3
weight_decay: 1e-4

# Loss
label_smoothing: 0.1
attr_weight_initial: 0.5
attr_weight_final: 0.1
consistency_weight: 0.1
```

## Kaggle Submission

Generate a submission file:

```bash
python inference.py --checkpoint checkpoints/best_model.pth --output submissions/submission.csv
```

The submission format:
```csv
id,label
1,42
2,156
...
```

## License

This project is developed for educational purposes as part of the UvA Applied Machine Learning course.

## Acknowledgments

- University of Amsterdam - Applied Machine Learning Course
- Kaggle - Feathers in Focus Competition
