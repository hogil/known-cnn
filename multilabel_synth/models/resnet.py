import torch.nn as nn


def build_resnet18(num_classes=20, pretrained=True):
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_resnet18_gray(num_classes, in_ch=1, pretrained=True):
    """Standard ImageNet ResNet-18 (stride-2 7x7 stem + maxpool) adapted to
    in_ch-channel inputs by averaging the pretrained conv1 weights across the
    RGB axis. Right choice for mid-resolution gray domains (e.g. 128x128 CXR)
    where the stem downsampling is wanted and ImageNet features transfer."""
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = resnet18(weights=weights)
    if in_ch != 3:
        w = m.conv1.weight.data.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1)
        conv = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv.weight.data = w
        m.conv1 = conv
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
