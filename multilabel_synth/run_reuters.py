"""4th domain family: text (Reuters-21578). Natural single-topic train pool ->
REAL multi-topic evaluation. Synthesis arms: concat (text concatenation = the
join/overlay analog: both topics genuinely present, label-honest) vs vector
averaging (the mixup/ghosting analog) vs single_only floor vs oracle (real
multi-topic docs).
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .metrics import bit_f1, far, exact_match, pos_neg_prob


def load_reuters(k=20):
    from nltk.corpus import reuters
    from collections import Counter
    fids = reuters.fileids()
    cnt = Counter()
    for f in fids:
        for c in reuters.categories(f):
            cnt[c] += 1
    top = [c for c, _ in cnt.most_common(k)]
    idx = {c: i for i, c in enumerate(top)}

    def label(f):
        y = np.zeros(k, np.float32)
        sel = [c for c in reuters.categories(f) if c in idx]
        for c in sel:
            y[idx[c]] = 1.0
        return y, len(sel), len(reuters.categories(f))

    tr_single, or_multi, te_multi = [], [], []
    for f in fids:
        y, nsel, ntot = label(f)
        txt = reuters.raw(f)
        if f.startswith("training"):
            if ntot == 1 and nsel == 1:
                tr_single.append((txt, y))
            elif nsel >= 2:
                or_multi.append((txt, y))
        else:
            if nsel >= 2:
                te_multi.append((txt, y))
    return top, tr_single, or_multi, te_multi


def train_eval(trX, trY, teX, teY, epochs=15, bs=64, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(trX.shape[1], 512), nn.ReLU(),
                          nn.Linear(512, trY.shape[1]))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lf = nn.BCEWithLogitsLoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(trX), torch.from_numpy(trY)),
                    batch_size=bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            lf(model(xb), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        P = torch.sigmoid(model(torch.from_numpy(teX))).numpy()
    pos, neg = pos_neg_prob(P, teY)
    return {"bitF1": bit_f1(P, teY), "FAR": far(P, teY),
            "exact": exact_match(P, teY), "pos": pos, "neg": neg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    top, tr_single, or_multi, te_multi = load_reuters(args.k)
    print(f"top-{args.k} cats | singles {len(tr_single)} oracle-multi {len(or_multi)} "
          f"test-multi {len(te_multi)}", flush=True)

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=5000, sublinear_tf=True)
    vec.fit([t for t, _ in tr_single])

    sY = np.stack([y for _, y in tr_single])
    sTxt = [t for t, _ in tr_single]
    teX = vec.transform([t for t, _ in te_multi]).toarray().astype(np.float32)
    teY = np.stack([y for _, y in te_multi])
    lab = sY.argmax(1)

    for arm in ["oracle", "concat", "vec_avg", "single_only"]:
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            if arm == "oracle":
                pick = rng.choice(len(or_multi), size=min(args.n_train, len(or_multi)),
                                  replace=False)
                trX = vec.transform([or_multi[i][0] for i in pick]).toarray().astype(np.float32)
                trY = np.stack([or_multi[i][1] for i in pick])
            elif arm == "single_only":
                pick = rng.choice(len(sTxt), size=min(args.n_train, len(sTxt)), replace=False)
                trX = vec.transform([sTxt[i] for i in pick]).toarray().astype(np.float32)
                trY = sY[pick]
            else:
                texts, ys = [], []
                tries = 0
                while len(texts) < args.n_train and tries < args.n_train * 5:
                    tries += 1
                    a, b = rng.integers(0, len(sTxt), size=2)
                    if lab[a] == lab[b]:
                        continue
                    if arm == "concat":
                        texts.append(sTxt[a] + "\n" + sTxt[b])
                    else:
                        texts.append((a, b))
                    ys.append(np.maximum(sY[a], sY[b]))
                if arm == "concat":
                    trX = vec.transform(texts).toarray().astype(np.float32)
                else:  # vec_avg: average of tfidf vectors (mixup/ghosting analog)
                    base = vec.transform(sTxt).toarray().astype(np.float32)
                    trX = np.stack([(base[a] + base[b]) / 2.0 for a, b in texts])
                trY = np.stack(ys)
            # every arm also sees the single pool (consistent with other datasets)
            pick2 = rng.choice(len(sTxt), size=min(2000, len(sTxt)), replace=False)
            spX = vec.transform([sTxt[i] for i in pick2]).toarray().astype(np.float32)
            trX = np.concatenate([trX, spX])
            trY = np.concatenate([trY, sY[pick2]])
            r = train_eval(trX, trY, teX, teY, epochs=args.epochs, seed=seed)
            print(f"{arm:12s} s{seed} | EVAL bitF1={r['bitF1']:.4f} FAR={r['FAR']:.4f} "
                  f"exact={r['exact']:.4f} pos={r['pos']:.4f} neg={r['neg']:.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
