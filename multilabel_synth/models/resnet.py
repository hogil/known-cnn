import torch.nn as nn


def build_resnet18(num_classes=20, pretrained=True):
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_resnet18_small(num_classes=8, in_ch=1):
    """ResNet-18 adapted for small gray inputs (e.g. 52x52 wafer maps):
    CIFAR-style 3x3 stride-1 stem, no maxpool, scratch-trained."""
    from torchvision.models import resnet18
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
