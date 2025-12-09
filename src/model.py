"""
Custom CNN Model for Bird Classification with Attribute Prediction.

Architecture:
- Residual blocks with squeeze-and-excitation
- Multi-task head (classification + attributes)
- ~5M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for bottleneck
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """
    Residual block with optional squeeze-and-excitation.

    Structure:
    x -> Conv -> BN -> ReLU -> Conv -> BN -> (+x) -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = True,
        se_reduction: int = 16
    ):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            stride: Stride for first conv (for downsampling)
            use_se: Use squeeze-and-excitation
            se_reduction: SE reduction ratio
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SE block
        self.se = SqueezeExcitation(out_channels, se_reduction) if use_se else nn.Identity()

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class BirdClassifier(nn.Module):
    """
    Custom CNN for bird species classification with attribute prediction.

    Architecture:
    - Stem: 7x7 Conv + MaxPool
    - 4 Stages of residual blocks
    - Global Average Pooling
    - Multi-task heads: Classification + Attributes
    """

    def __init__(
        self,
        num_classes: int = 200,
        num_attributes: int = 312,
        in_channels: int = 3,
        base_channels: int = 64,
        use_se: bool = True,
        dropout_rate: float = 0.3
    ):
        """
        Args:
            num_classes: Number of bird species
            num_attributes: Number of attributes
            in_channels: Input image channels
            base_channels: Base channel count
            use_se: Use squeeze-and-excitation blocks
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_attributes = num_attributes

        # ================
        # Stem
        # ================
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ================
        # Stages
        # ================
        # Stage 1: 64 -> 64
        self.stage1 = self._make_stage(base_channels, base_channels, 2, 1, use_se)

        # Stage 2: 64 -> 128
        self.stage2 = self._make_stage(base_channels, base_channels * 2, 2, 2, use_se)

        # Stage 3: 128 -> 256
        self.stage3 = self._make_stage(base_channels * 2, base_channels * 4, 2, 2, use_se)

        # Stage 4: 256 -> 512
        self.stage4 = self._make_stage(base_channels * 4, base_channels * 8, 2, 2, use_se)

        # ================
        # Global Pooling
        # ================
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Feature dimension
        self.feature_dim = base_channels * 8  # 512

        # ================
        # Classification Head
        # ================
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(self.feature_dim // 2, num_classes)
        )

        # ================
        # Attribute Head
        # ================
        self.attribute_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(self.feature_dim // 2, num_attributes)
        )

        # Initialize weights
        self._initialize_weights()

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        use_se: bool
    ) -> nn.Sequential:
        """Create a stage of residual blocks."""
        layers = []

        # First block may downsample
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_se))

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, use_se))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification heads."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input images (B, C, H, W)
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing:
            - 'class_logits': Classification logits (B, num_classes)
            - 'attr_logits': Attribute logits (B, num_attributes)
            - 'features': Feature vector (optional)
        """
        # Extract features
        features = self.extract_features(x)

        # Classification head
        class_logits = self.classifier(features)

        # Attribute head
        attr_logits = self.attribute_head(features)

        output = {
            'class_logits': class_logits,
            'attr_logits': attr_logits
        }

        if return_features:
            output['features'] = features

        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions.

        Args:
            x: Input images

        Returns:
            Predicted class indices (0-indexed)
        """
        with torch.no_grad():
            output = self.forward(x)
            return output['class_logits'].argmax(dim=1)

    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with confidence scores.

        Args:
            x: Input images

        Returns:
            Tuple of (predictions, confidence scores)
        """
        with torch.no_grad():
            output = self.forward(x)
            probs = F.softmax(output['class_logits'], dim=1)
            confidence, predictions = probs.max(dim=1)
            return predictions, confidence


class SharedLatentHead(nn.Module):
    """
    Multi-task head with shared latent decomposition (Abdulnabi et al. style).

    Architecture:
        features (512) -> Latent Layer L -> z (K) -> S_class -> class_logits (200)
                                                  -> S_attr  -> attr_logits (312)

    This forces both tasks to share a common latent representation,
    enabling better knowledge transfer and reducing parameters.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        latent_dim: int = 128,
        num_classes: int = 200,
        num_attributes: int = 312,
        dropout_rate: float = 0.3
    ):
        """
        Args:
            feature_dim: Input feature dimension from backbone
            latent_dim: Shared latent space dimension (K)
            num_classes: Number of output classes
            num_attributes: Number of attributes
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # Shared latent projection: h (512) -> z (K)
        self.latent_layer = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2)
        )

        # Task-specific output layers: z (K) -> outputs
        self.class_head = nn.Linear(latent_dim, num_classes)
        self.attr_head = nn.Linear(latent_dim, num_attributes)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, feature_dim) from backbone

        Returns:
            dict with 'class_logits', 'attr_logits', 'latent'
        """
        # Shared latent representation
        z = self.latent_layer(features)  # (B, K)

        # Task-specific outputs
        class_logits = self.class_head(z)  # (B, 200)
        attr_logits = self.attr_head(z)    # (B, 312)

        return {
            'class_logits': class_logits,
            'attr_logits': attr_logits,
            'latent': z
        }


class AttributeGatedHead(nn.Module):
    """
    Attribute head with learned per-attribute feature gates.

    Each attribute learns which feature dimensions are important for its prediction.
    This is inspired by Qian et al.'s feature selection approach.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_attributes: int = 312,
        hidden_dim: int = 128,
        dropout_rate: float = 0.3
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout_rate)

        # Gate network: generates per-attribute attention weights
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_attributes),
            nn.Sigmoid()  # Produces (B, 312) attention weights
        )

        # Main attribute predictor
        self.attr_fc = nn.Linear(feature_dim, num_attributes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, feature_dim) backbone features

        Returns:
            attr_logits: (B, num_attributes) gated attribute predictions
        """
        features = self.dropout(features)

        # Generate per-attribute attention
        gates = self.gate_network(features)  # (B, 312)

        # Compute base predictions
        attr_logits = self.attr_fc(features)  # (B, 312)

        # Apply gating (element-wise multiplication)
        gated_logits = attr_logits * gates

        return gated_logits


class BirdClassifierV2(nn.Module):
    """
    Enhanced bird classifier with shared latent decomposition.

    Key improvements over BirdClassifier:
    1. Shared latent layer between classification and attribute heads
    2. Optional attribute-gated head for improved attribute prediction
    3. Reduced parameter count in heads (~40% reduction)

    Architecture:
    - Same ResNet-SE backbone as original
    - SharedLatentHead with configurable latent dimension
    - Optional AttributeGatedHead for attributes
    """

    def __init__(
        self,
        num_classes: int = 200,
        num_attributes: int = 312,
        in_channels: int = 3,
        base_channels: int = 64,
        latent_dim: int = 128,
        use_se: bool = True,
        use_gated_attrs: bool = False,
        dropout_rate: float = 0.3
    ):
        """
        Args:
            num_classes: Number of bird species
            num_attributes: Number of attributes
            in_channels: Input image channels
            base_channels: Base channel count for backbone
            latent_dim: Shared latent space dimension
            use_se: Use squeeze-and-excitation blocks
            use_gated_attrs: Use attribute-gated head
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.use_gated_attrs = use_gated_attrs

        # ================
        # Backbone (same as original BirdClassifier)
        # ================
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Stages
        self.stage1 = self._make_stage(base_channels, base_channels, 2, 1, use_se)
        self.stage2 = self._make_stage(base_channels, base_channels * 2, 2, 2, use_se)
        self.stage3 = self._make_stage(base_channels * 2, base_channels * 4, 2, 2, use_se)
        self.stage4 = self._make_stage(base_channels * 4, base_channels * 8, 2, 2, use_se)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = base_channels * 8  # 512

        # ================
        # Shared Latent Head
        # ================
        if use_gated_attrs:
            # Use separate heads with gated attributes
            self.latent_layer = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.feature_dim, latent_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate / 2)
            )
            self.class_head = nn.Linear(latent_dim, num_classes)
            self.attr_head = AttributeGatedHead(
                feature_dim=self.feature_dim,
                num_attributes=num_attributes,
                hidden_dim=latent_dim,
                dropout_rate=dropout_rate
            )
        else:
            # Use unified SharedLatentHead
            self.head = SharedLatentHead(
                feature_dim=self.feature_dim,
                latent_dim=latent_dim,
                num_classes=num_classes,
                num_attributes=num_attributes,
                dropout_rate=dropout_rate
            )

        # Initialize weights
        self._initialize_weights()

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        use_se: bool
    ) -> nn.Sequential:
        """Create a stage of residual blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_se))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, use_se))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification heads."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input images (B, C, H, W)
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing:
            - 'class_logits': Classification logits (B, num_classes)
            - 'attr_logits': Attribute logits (B, num_attributes)
            - 'features': Feature vector (optional)
            - 'latent': Latent representation (optional)
        """
        features = self.extract_features(x)

        if self.use_gated_attrs:
            # Separate paths for class (through latent) and attrs (gated)
            z = self.latent_layer(features)
            class_logits = self.class_head(z)
            attr_logits = self.attr_head(features)
            output = {
                'class_logits': class_logits,
                'attr_logits': attr_logits,
                'latent': z
            }
        else:
            # Unified SharedLatentHead
            output = self.head(features)

        if return_features:
            output['features'] = features

        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        with torch.no_grad():
            output = self.forward(x)
            return output['class_logits'].argmax(dim=1)

    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictions with confidence scores."""
        with torch.no_grad():
            output = self.forward(x)
            probs = F.softmax(output['class_logits'], dim=1)
            confidence, predictions = probs.max(dim=1)
            return predictions, confidence


class LightBirdClassifier(nn.Module):
    """
    Lighter version of BirdClassifier for faster training.
    ~2.5M parameters.
    """

    def __init__(
        self,
        num_classes: int = 200,
        num_attributes: int = 312,
        in_channels: int = 3,
        base_channels: int = 32,  # Reduced from 64
        dropout_rate: float = 0.3
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_attributes = num_attributes

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )

        # Stages (fewer blocks)
        self.stage1 = self._make_stage(base_channels * 2, base_channels * 2, 1, 1)
        self.stage2 = self._make_stage(base_channels * 2, base_channels * 4, 1, 2)
        self.stage3 = self._make_stage(base_channels * 4, base_channels * 8, 1, 2)
        self.stage4 = self._make_stage(base_channels * 8, base_channels * 16, 1, 2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = base_channels * 16  # 512

        # Heads
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, num_classes)
        )

        self.attribute_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, num_attributes)
        )

        self._initialize_weights()

    def _make_stage(self, in_ch, out_ch, num_blocks, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride, use_se=False)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch, 1, use_se=False))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        features = x.view(x.size(0), -1)

        output = {
            'class_logits': self.classifier(features),
            'attr_logits': self.attribute_head(features)
        }

        if return_features:
            output['features'] = features

        return output


def create_model(
    model_type: str = "standard",
    num_classes: int = 200,
    num_attributes: int = 312,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: 'standard', 'light', 'v2', or 'v2_gated'
        num_classes: Number of classes
        num_attributes: Number of attributes
        **kwargs: Additional model arguments

    Returns:
        Model instance
    """
    if model_type == "standard":
        return BirdClassifier(num_classes, num_attributes, **kwargs)
    elif model_type == "light":
        return LightBirdClassifier(num_classes, num_attributes, **kwargs)
    elif model_type == "v2":
        return BirdClassifierV2(num_classes, num_attributes, use_gated_attrs=False, **kwargs)
    elif model_type == "v2_gated":
        return BirdClassifierV2(num_classes, num_attributes, use_gated_attrs=True, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Get model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_str': f"{total_params / 1e6:.2f}M",
        'trainable_params_str': f"{trainable_params / 1e6:.2f}M"
    }


if __name__ == "__main__":
    # Test models
    print("=" * 60)
    print("Testing Bird Classification Models")
    print("=" * 60)

    x = torch.randn(2, 3, 224, 224)

    # Standard model
    print("\n1. BirdClassifier (Standard)")
    model = BirdClassifier()
    info = get_model_info(model)
    print(f"   Parameters: {info['total_params_str']}")
    output = model(x)
    print(f"   Class logits shape: {output['class_logits'].shape}")
    print(f"   Attr logits shape: {output['attr_logits'].shape}")

    # Light model
    print("\n2. LightBirdClassifier")
    light_model = LightBirdClassifier()
    light_info = get_model_info(light_model)
    print(f"   Parameters: {light_info['total_params_str']}")
    output = light_model(x)
    print(f"   Class logits shape: {output['class_logits'].shape}")

    # V2 model (shared latent)
    print("\n3. BirdClassifierV2 (Shared Latent)")
    v2_model = BirdClassifierV2(latent_dim=128, use_gated_attrs=False)
    v2_info = get_model_info(v2_model)
    print(f"   Parameters: {v2_info['total_params_str']}")
    output = v2_model(x)
    print(f"   Class logits shape: {output['class_logits'].shape}")
    print(f"   Attr logits shape: {output['attr_logits'].shape}")
    print(f"   Latent shape: {output['latent'].shape}")

    # V2 model with gated attributes
    print("\n4. BirdClassifierV2 (Gated Attributes)")
    v2_gated = BirdClassifierV2(latent_dim=128, use_gated_attrs=True)
    v2_gated_info = get_model_info(v2_gated)
    print(f"   Parameters: {v2_gated_info['total_params_str']}")
    output = v2_gated(x)
    print(f"   Class logits shape: {output['class_logits'].shape}")
    print(f"   Attr logits shape: {output['attr_logits'].shape}")

    # Summary
    print("\n" + "=" * 60)
    print("Parameter Summary:")
    print(f"  Standard:   {info['total_params_str']}")
    print(f"  Light:      {light_info['total_params_str']}")
    print(f"  V2:         {v2_info['total_params_str']}")
    print(f"  V2 (gated): {v2_gated_info['total_params_str']}")
    print("=" * 60)
    print("\nModel test complete!")
