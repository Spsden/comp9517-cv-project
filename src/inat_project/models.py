from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def create_resnet18(num_classes: int = 500, pretrained: bool = False) -> nn.Module:
    """Create the common ResNet-18 architecture used by both deep models."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

