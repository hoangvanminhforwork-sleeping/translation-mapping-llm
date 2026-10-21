"""Mapping EN representation -> VI representation.

Two hypotheses, identical inputs and identical evaluation:
  * ANN   : one MLP trained end-to-end on the pair loss (paperOverview §3.4)
  * ridge : 768 *independent* per-dimension linear regressions (the "hundred of
            linear regressions" baseline).  Solving them jointly in closed form is
            algebraically identical to fitting each dimension separately.
"""
import numpy as np
import torch
from torch import nn

from . import config as C


# ------------------------------------------------------------------ ridge
def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float):
    """Return (x_mean, y_mean, W) for y ~ (x - xm) W + ym."""
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    gram = Xc.T @ Xc
    gram.flat[:: gram.shape[0] + 1] += alpha
    W = np.linalg.solve(gram, Xc.T @ Yc)
    return xm, ym, W


def ridge_apply(X, params):
    xm, ym, W = params
    return (X.astype(np.float64) - xm) @ W + ym


# ------------------------------------------------------------------ MLP
class MLP(nn.Module):
    # NB: defaults are resolved inside __init__ - binding C.HIDDEN in the signature
    # would freeze it at import time and silently ignore later config changes.
    def __init__(self, dim: int | None = None, hidden: int | None = None):
        super().__init__()
        dim = C.DIM if dim is None else dim
        hidden = C.HIDDEN if hidden is None else hidden
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


def _standardiser(X):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    return mu, sd


def train_mlp(Xtr, Ytr, Xval, Yval, seed: int = C.SEED, verbose: bool = False):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    xmu, xsd = _standardiser(Xtr)
    ymu, ysd = _standardiser(Ytr)
    xt = torch.tensor((Xtr - xmu) / xsd, dtype=torch.float32, device=dev)
    yt = torch.tensor((Ytr - ymu) / ysd, dtype=torch.float32, device=dev)
    xv = torch.tensor((Xval - xmu) / xsd, dtype=torch.float32, device=dev)
    yv = torch.tensor((Yval - ymu) / ysd, dtype=torch.float32, device=dev)

    model = MLP().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    best, best_state, bad, curve = float("inf"), None, 0, []

    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xt), device=dev)
        total = 0.0
        for i in range(0, len(xt), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(xv), yv).item()
        curve.append({"epoch": epoch, "train_mse_std": total / len(xt), "val_mse_std": val})
        if verbose:
            print(f"  epoch {epoch:02d} train {total / len(xt):.5f} val {val:.5f}")
        if val < best - 1e-6:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= C.PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, (xmu, xsd, ymu, ysd), curve


def apply_mlp(model, scalers, X):
    xmu, xsd, ymu, ysd = scalers
    dev = next(model.parameters()).device
    with torch.no_grad():
        out = model(torch.tensor((X - xmu) / xsd, dtype=torch.float32, device=dev))
    return (out.cpu().numpy() * ysd + ymu).astype(np.float32)


# ------------------------------------------------------------------ metrics
def _l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def metrics(pred: np.ndarray, target: np.ndarray, ks=C.TOP_K) -> dict:
    """Representation metrics + closed-set retrieval (gallery = the test targets)."""
    diff = pred - target
    cos = (_l2(pred) * _l2(target)).sum(1)
    sims = _l2(pred) @ _l2(target).T                       # [n, n]
    true_idx = np.arange(len(target))
    rank = (sims > sims[true_idx, true_idx][:, None]).sum(1)   # 0 = perfect
    out = {
        "mse": float((diff ** 2).mean()),
        "mae": float(np.abs(diff).mean()),
        "cosine": float(cos.mean()),
        "rank_median": float(np.median(rank)),
        "n": int(len(target)),
    }
    for k in ks:
        out[f"r@{k}"] = float((rank < k).mean())
        out[f"chance@{k}"] = k / len(target)
    return out
