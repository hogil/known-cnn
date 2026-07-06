import torch.nn as nn


class SmallCNN(nn.Module):
    """Small CPU-friendly CNN for MultiMNIST multi-label.

    NOTE: keep a spatial map (AdaptiveAvgPool2d(pool>1)) before the head.
    A previous version used AdaptiveAvgPool2d(1), which averaged away all
    spatial layout and bottlenecked digit identity (single-digit mAP stuck
    ~0.28 @ 15 epochs). Preserving a 4x4 map lifts it to ~0.95 @ 15 epochs.
    """

    def __init__(self, num_classes=10, in_ch=1, pool=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(pool),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * pool * pool, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
