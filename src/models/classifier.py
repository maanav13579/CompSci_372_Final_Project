"""
Food Classification Model using pretrained ResNet50 with custom classifier head.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Literal


class FoodClassifier(nn.Module):
    """
    Food classification model with pretrained backbone.
    
    """
    
    def __init__(
        self,
        num_classes: int = 101,
        backbone: Literal["resnet50"] = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            num_classes: Number of food classes (101 for Food-101)
            backbone: Model architecture to use
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate before final classifier
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        if backbone == "resnet50":
            self._init_resnet50(pretrained, dropout, num_classes)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _init_resnet50(self, pretrained: bool, dropout: float, num_classes: int):
        """Initialize ResNet50 backbone."""
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # Get feature dimension
        in_features = self.backbone.fc.in_features  # 2048
        
        # Replace classifier head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        
        self.feature_dim = in_features
        
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters except the classifier head."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name and "heads" not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classifier head.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            Features of shape (B, feature_dim)
        """
        # Forward through all layers except fc
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class indices.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            Predicted class indices of shape (B,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            Probabilities of shape (B, num_classes)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


def create_classifier(
    num_classes: int = 101,
    backbone: str = "resnet50",
    pretrained: bool = True,
    dropout: float = 0.3,
) -> FoodClassifier:
    """
    Factory function to create a food classifier.
    """
    return FoodClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Quick test
    model = create_classifier()
    print(f"Model: {model.backbone_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    features = model.get_features(x)
    print(f"Feature shape: {features.shape}")
