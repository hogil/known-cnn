import torch
from multilabel_synth.models.small_cnn import SmallCNN


def test_forward_shape():
    model = SmallCNN(num_classes=10, in_ch=1)
    x = torch.zeros(4, 1, 40, 40)
    out = model(x)
    assert out.shape == (4, 10)


def test_param_count_small():
    model = SmallCNN(num_classes=10, in_ch=1)
    n = sum(p.numel() for p in model.parameters())
    assert n < 200_000   # stays tiny for CPU
