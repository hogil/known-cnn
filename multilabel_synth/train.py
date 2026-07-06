import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .models.small_cnn import SmallCNN
from .metrics import compute_map, exact_match, pos_neg_prob


def _loader(X, Y, bs, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


def train_one(train_X, train_Y, test_X, test_Y,
              epochs=3, bs=64, lr=1e-3, device="cpu", seed=0):
    torch.manual_seed(seed)
    model = SmallCNN(num_classes=train_Y.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in _loader(train_X, train_Y, bs, True):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in _loader(test_X, test_Y, bs, False):
            probs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    P = np.concatenate(probs)

    pos, neg = pos_neg_prob(P, test_Y)
    return {
        "mAP": compute_map(P, test_Y),
        "exact_match": exact_match(P, test_Y),
        "pos_prob": pos,
        "neg_prob": neg,
    }
