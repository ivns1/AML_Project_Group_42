"""
Evaluation metrics for Bird Classification.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Calculate top-1 accuracy.

    Args:
        predictions: Predicted logits or probabilities (B, C)
        targets: Ground truth labels (B,)

    Returns:
        Accuracy value (0-1)
    """
    if predictions.dim() > 1:
        pred_labels = predictions.argmax(dim=1)
    else:
        pred_labels = predictions

    correct = (pred_labels == targets).sum().item()
    total = targets.size(0)

    return correct / total if total > 0 else 0.0


def top_k_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5
) -> float:
    """
    Calculate top-k accuracy.

    Args:
        predictions: Predicted logits or probabilities (B, C)
        targets: Ground truth labels (B,)
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy value (0-1)
    """
    batch_size = targets.size(0)

    # Get top-k predictions
    _, top_k_preds = predictions.topk(k, dim=1)

    # Check if target is in top-k
    targets_expanded = targets.view(-1, 1).expand_as(top_k_preds)
    correct = top_k_preds.eq(targets_expanded).any(dim=1).sum().item()

    return correct / batch_size if batch_size > 0 else 0.0


def per_class_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 200
) -> Dict[int, float]:
    """
    Calculate per-class accuracy.

    Args:
        predictions: Predicted logits (B, C)
        targets: Ground truth labels (B,)
        num_classes: Number of classes

    Returns:
        Dictionary mapping class index to accuracy
    """
    pred_labels = predictions.argmax(dim=1)

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for pred, target in zip(pred_labels, targets):
        target_idx = target.item()
        class_total[target_idx] += 1
        if pred.item() == target_idx:
            class_correct[target_idx] += 1

    accuracies = {}
    for cls in range(num_classes):
        if class_total[cls] > 0:
            accuracies[cls] = class_correct[cls] / class_total[cls]
        else:
            accuracies[cls] = 0.0

    return accuracies


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 200
) -> np.ndarray:
    """
    Calculate confusion matrix.

    Args:
        predictions: Predicted logits (B, C)
        targets: Ground truth labels (B,)
        num_classes: Number of classes

    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    pred_labels = predictions.argmax(dim=1).cpu().numpy()
    targets = targets.cpu().numpy()

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pred, target in zip(pred_labels, targets):
        matrix[target, pred] += 1

    return matrix


def precision_recall_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 200,
    average: str = 'macro'
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.

    Args:
        predictions: Predicted logits (B, C)
        targets: Ground truth labels (B,)
        num_classes: Number of classes
        average: 'macro' or 'weighted'

    Returns:
        Tuple of (precision, recall, f1)
    """
    pred_labels = predictions.argmax(dim=1)

    # Per-class counts
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)

    for pred, target in zip(pred_labels, targets):
        pred_idx = pred.item()
        target_idx = target.item()

        if pred_idx == target_idx:
            tp[target_idx] += 1
        else:
            fp[pred_idx] += 1
            fn[target_idx] += 1

    # Per-class precision and recall
    precision_per_class = tp / (tp + fp + 1e-6)
    recall_per_class = tp / (tp + fn + 1e-6)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + 1e-6)

    # Average
    if average == 'macro':
        precision = precision_per_class.mean().item()
        recall = recall_per_class.mean().item()
        f1 = f1_per_class.mean().item()
    elif average == 'weighted':
        weights = tp + fn
        weights = weights / (weights.sum() + 1e-6)
        precision = (precision_per_class * weights).sum().item()
        recall = (recall_per_class * weights).sum().item()
        f1 = (f1_per_class * weights).sum().item()
    else:
        raise ValueError(f"Unknown average type: {average}")

    return precision, recall, f1


def attribute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Calculate attribute prediction accuracy.

    Args:
        predictions: Predicted attribute logits (B, num_attrs)
        targets: Ground truth attributes (B, num_attrs)
        threshold: Threshold for binary prediction

    Returns:
        Accuracy value
    """
    pred_attrs = (torch.sigmoid(predictions) > threshold).float()
    target_attrs = (targets > threshold).float()

    correct = (pred_attrs == target_attrs).float().mean().item()
    return correct


class MetricTracker:
    """Track and aggregate metrics during training/validation."""

    def __init__(self, num_classes: int = 200):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = defaultdict(list)

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_dict: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Update metrics with batch results.

        Args:
            predictions: Predicted logits
            targets: Ground truth labels
            loss_dict: Dictionary of losses
        """
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())

        if loss_dict:
            for name, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    self.losses[name].append(value.item())
                else:
                    self.losses[name].append(value)

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary of metric values
        """
        # Concatenate all batches
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy(all_preds, all_targets),
            'top5_accuracy': top_k_accuracy(all_preds, all_targets, k=5),
        }

        # Add precision, recall, f1
        prec, rec, f1 = precision_recall_f1(all_preds, all_targets, self.num_classes)
        metrics['precision'] = prec
        metrics['recall'] = rec
        metrics['f1'] = f1

        # Average losses
        for name, values in self.losses.items():
            metrics[f'loss_{name}'] = np.mean(values)

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        return confusion_matrix(all_preds, all_targets, self.num_classes)

    def get_per_class_accuracy(self) -> Dict[int, float]:
        """Get per-class accuracy."""
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        return per_class_accuracy(all_preds, all_targets, self.num_classes)


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 200
) -> Dict[str, float]:
    """
    Calculate all metrics at once.

    Args:
        predictions: Predicted logits
        targets: Ground truth labels
        num_classes: Number of classes

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy(predictions, targets),
        'top5_accuracy': top_k_accuracy(predictions, targets, k=5),
    }

    prec, rec, f1 = precision_recall_f1(predictions, targets, num_classes)
    metrics['precision'] = prec
    metrics['recall'] = rec
    metrics['f1'] = f1

    return metrics


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")

    batch_size = 32
    num_classes = 200

    # Create dummy data
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Test accuracy
    acc = accuracy(predictions, targets)
    print(f"Top-1 Accuracy: {acc:.4f}")

    # Test top-5 accuracy
    top5_acc = top_k_accuracy(predictions, targets, k=5)
    print(f"Top-5 Accuracy: {top5_acc:.4f}")

    # Test precision, recall, f1
    prec, rec, f1 = precision_recall_f1(predictions, targets, num_classes)
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Test metric tracker
    tracker = MetricTracker(num_classes)
    tracker.update(predictions, targets)
    metrics = tracker.compute()
    print(f"\nMetric Tracker results: {metrics}")

    print("\nMetrics test complete!")
