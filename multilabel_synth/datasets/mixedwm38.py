import numpy as np

# MixedWM38 (Wafer_Map_Datasets.npz): X [38015,52,52] int 0..3, Y [38015,8]
# multi-hot. 1000 normal (0 bits), 7015 single (1 bit), 30000 mixed (2-4 bits,
# 29 unique combos). Pixel values: 0=off-wafer bg, 1=normal die, 2=defect die
# (3 rare artifact). We scale X to [0,1] by /2 and clip.

NPZ = "E:/data/mixedwm38/Wafer_Map_Datasets.npz"
N_BITS = 8


def load_wm38(npz_path=NPZ):
    d = np.load(npz_path)
    X = d["arr_0"].astype(np.float32)
    X = np.clip(X, 0, 2) / 2.0
    X = X[:, None, :, :]                       # [N,1,52,52]
    Y = d["arr_1"].astype(np.float32)          # [N,8]
    return X, Y


def split_groups(Y):
    nbits = Y.sum(1).astype(int)
    return {
        "normal": np.where(nbits == 0)[0],
        "single": np.where(nbits == 1)[0],
        "mixed": np.where(nbits >= 2)[0],
    }


def combo_key(y):
    return tuple(int(v) for v in y)


def mixed_combos(Y, idx):
    """Unique combo keys among mixed indices."""
    return sorted({combo_key(Y[i]) for i in idx})


def split_holdout_combos(combos, n_holdout, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(combos))
    hold = {combos[i] for i in perm[:n_holdout]}
    train = [c for c in combos if c not in hold]
    return train, sorted(hold)


def sample_idx(rng, idx, n):
    if n >= len(idx):
        return idx.copy()
    return rng.choice(idx, size=n, replace=False)
