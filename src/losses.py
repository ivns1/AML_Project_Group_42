"""
Loss functions for Bird Classification with multi-task learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""

    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        """
        Args:
            smoothing: Label smoothing factor (0 = no smoothing)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C)
            target: Target labels (B,)

        Returns:
            Loss value
        """
        n_classes = pred.size(1)

        # Convert to one-hot
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)

        # Apply smoothing
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes

        # Compute loss
        log_probs = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_probs).sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C)
            target: Target labels (B,)

        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class AttributeLoss(nn.Module):
    """Binary cross entropy loss for attribute prediction."""

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted attribute logits (B, num_attributes)
            target: Target attributes (B, num_attributes)
            mask: Optional mask for valid attributes

        Returns:
            Loss value
        """
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-6)
            elif self.reduction == 'sum':
                return loss.sum()
            return loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task learning.

    L_total = L_classification + lambda * L_attributes
    """

    def __init__(
        self,
        num_classes: int = 200,
        label_smoothing: float = 0.1,
        attribute_weight: float = 0.3,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            num_classes: Number of classes
            label_smoothing: Label smoothing factor
            attribute_weight: Weight for attribute loss
            use_focal: Use focal loss for classification
            focal_gamma: Focal loss gamma parameter
            class_weights: Optional class weights for imbalance
        """
        super().__init__()

        self.attribute_weight = attribute_weight
        self.class_weights = class_weights

        # Classification loss
        if use_focal:
            self.class_loss = FocalLoss(gamma=focal_gamma)
        elif label_smoothing > 0:
            self.class_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            self.class_loss = nn.CrossEntropyLoss(weight=class_weights)

        # Attribute loss
        self.attr_loss = AttributeLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            outputs: Model outputs with 'class_logits' and 'attr_logits'
            targets: Target dict with 'label' and 'attributes'

        Returns:
            Dictionary with 'total', 'class_loss', 'attr_loss'
        """
        # Classification loss
        class_logits = outputs['class_logits']
        labels = targets['label']
        class_loss = self.class_loss(class_logits, labels)

        # Attribute loss (if available)
        attr_loss = torch.tensor(0.0, device=class_logits.device)
        if 'attributes' in targets and 'attr_logits' in outputs:
            attr_logits = outputs['attr_logits']
            attr_targets = targets['attributes']
            attr_loss = self.attr_loss(attr_logits, attr_targets)

        # Combined loss
        total_loss = class_loss + self.attribute_weight * attr_loss

        return {
            'total': total_loss,
            'class_loss': class_loss,
            'attr_loss': attr_loss
        }


class ClassificationOnlyLoss(nn.Module):
    """Simple classification loss without attributes."""

    def __init__(
        self,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()

        if label_smoothing > 0:
            self.loss_fn = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        class_loss = self.loss_fn(outputs['class_logits'], targets['label'])

        return {
            'total': class_loss,
            'class_loss': class_loss,
            'attr_loss': torch.tensor(0.0, device=class_loss.device)
        }


def create_loss(
    loss_type: str = "combined",
    num_classes: int = 200,
    label_smoothing: float = 0.1,
    attribute_weight: float = 0.3,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss modules.

    Args:
        loss_type: 'combined' or 'classification'
        num_classes: Number of classes
        label_smoothing: Label smoothing factor
        attribute_weight: Weight for attribute loss

    Returns:
        Loss module
    """
    if loss_type == "combined":
        return CombinedLoss(
            num_classes=num_classes,
            label_smoothing=label_smoothing,
            attribute_weight=attribute_weight,
            **kwargs
        )
    elif loss_type == "classification":
        return ClassificationOnlyLoss(
            label_smoothing=label_smoothing,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test losses
    print("Testing loss functions...")

    # Create dummy data
    batch_size = 4
    num_classes = 200
    num_attributes = 312

    outputs = {
        'class_logits': torch.randn(batch_size, num_classes),
        'attr_logits': torch.randn(batch_size, num_attributes)
    }

    targets = {
        'label': torch.randint(0, num_classes, (batch_size,)),
        'attributes': torch.rand(batch_size, num_attributes)
    }

    # Test combined loss
    loss_fn = CombinedLoss()
    losses = loss_fn(outputs, targets)

    print(f"Total loss: {losses['total'].item():.4f}")
    print(f"Class loss: {losses['class_loss'].item():.4f}")
    print(f"Attr loss: {losses['attr_loss'].item():.4f}")

    # Test label smoothing
    ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    ls_result = ls_loss(outputs['class_logits'], targets['label'])
    print(f"\nLabel smoothing loss: {ls_result.item():.4f}")

    # Test focal loss
    focal = FocalLoss()
    focal_result = focal(outputs['class_logits'], targets['label'])
    print(f"Focal loss: {focal_result.item():.4f}")

    print("\nLoss test complete!")
