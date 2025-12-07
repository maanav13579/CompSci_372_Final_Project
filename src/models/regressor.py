"""
Calorie Regression Model using pretrained backbone features.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Literal


class CalorieRegressor(nn.Module):
    """
    Calorie regression model with pretrained backbone as feature extractor.
    
    Architecture:
    - Frozen/trainable backbone (ResNet50/EfficientNet) for feature extraction
    - MLP head for regression
    """
    
    def __init__(
        self,
        backbone: Literal["resnet50", "efficientnet_b0"] = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        """
        Args:
            backbone: Feature extractor architecture
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone weights
            hidden_dim: Hidden dimension in MLP head
            dropout: Dropout rate in MLP head
        """
        super().__init__()
        
        self.backbone_name = backbone
        
        # Initialize backbone
        if backbone == "resnet50":
            self._init_resnet50(pretrained)
        elif backbone == "efficientnet_b0":
            self._init_efficientnet(pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        if freeze_backbone:
            self._freeze_backbone()
        
        # MLP regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        
        # Initialize regression head
        self._init_regressor()
    
    def _init_resnet50(self, pretrained: bool):
        """Initialize ResNet50 as feature extractor."""
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Remove classifier, keep everything up to avgpool
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
        )
        
        self.feature_dim = 2048
    
    def _init_efficientnet(self, pretrained: bool):
        """Initialize EfficientNet-B0 as feature extractor."""
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        effnet = models.efficientnet_b0(weights=weights)
        
        # Use features and avgpool
        self.backbone = nn.Sequential(
            effnet.features,
            effnet.avgpool,
            nn.Flatten(),
        )
        
        self.feature_dim = 1280
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def _init_regressor(self):
        """Initialize regression head weights."""
        for module in self.regressor.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            Predicted calories (normalized) of shape (B, 1)
        """
        features = self.backbone(x)
        return self.regressor(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract backbone features.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            Features of shape (B, feature_dim)
        """
        return self.backbone(x)
    
    def predict(self, x: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        """
        Predict calories in original scale.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            scale: Scale factor used during training
            
        Returns:
            Predicted calories in kcal of shape (B,)
        """
        normalized = self.forward(x)
        return normalized.squeeze(-1) * scale


class HuberLoss(nn.Module):
    """Huber loss for robust regression."""
    
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred.squeeze() - target)
        quadratic = torch.clamp(diff, max=self.delta)
        linear = diff - quadratic
        return torch.mean(0.5 * quadratic ** 2 + self.delta * linear)


def create_regressor(
    backbone: str = "resnet50",
    pretrained: bool = True,
    freeze_backbone: bool = True,
    hidden_dim: int = 256,
    dropout: float = 0.2,
) -> CalorieRegressor:
    """
    Factory function to create a calorie regressor.
    """
    return CalorieRegressor(
        backbone=backbone,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Quick test
    model = create_regressor()
    print(f"Model: {model.backbone_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test prediction with scaling
    calories = model.predict(x, scale=1000.0)
    print(f"Predicted calories shape: {calories.shape}")
