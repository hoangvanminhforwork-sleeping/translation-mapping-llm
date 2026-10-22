"""Mapping networks, refit here because experiment 01 never persisted them.

  ridge : 768 independent linear regressions (joint solve, algebraically identical)
  ann   : MLP 768 -> 2048 -> 768, frozen-encoder inputs
Same split protocol as experiment 01: fit on 90% of train, early stop on 10%.
"""
import numpy as np
import torch
from torch import nn

from . import config as C


# ------------------------------------------------------------------ ridge
def ridge_fit(X, Y, alpha: float):
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    gram = Xc.T @ Xc
    gram.flat[:: gram.shape[0] + 1] += alpha
    W = np.linalg.solve(gram, Xc.T @ Yc)
    return xm, ym, W


def ridge_apply(X, params) -> np.ndarray:
    xm, ym, W = params
    return (X.astype(np.float64) - xm) @ W + ym


def ridge_save(path, params):
    xm, ym, W = params
    np.savez(path, xm=xm, ym=ym, W=W)


def ridge_load(path):
    z = np.load(path)
    return z["xm"], z["ym"], z["W"]


# ------------------------------------------------------------------ MLP
class MLP(nn.Module):
    def __init__(self, dim: int | None = None, hidden: int | None = None):
        super().__init__()
        dim = C.DIM if dim is None else dim
        hidden = C.MLP_HIDDEN if hidden is None else hidden
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


def _std(X):
    return X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6


def train_mlp(Xtr, Ytr, Xval, Yval, seed: int = C.SEED, verbose: bool = False):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    xmu, xsd = _std(Xtr)
    ymu, ysd = _std(Ytr)
    xt = torch.tensor((Xtr - xmu) / xsd, dtype=torch.float32, device=dev)
    yt = torch.tensor((Ytr - ymu) / ysd, dtype=torch.float32, device=dev)
    xv = torch.tensor((Xval - xmu) / xsd, dtype=torch.float32, device=dev)
    yv = torch.tensor((Yval - ymu) / ysd, dtype=torch.float32, device=dev)

    model = MLP().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=C.MLP_LR, weight_decay=C.MLP_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    best, best_state, bad, curve = float("inf"), None, 0, []

    for epoch in range(1, C.MLP_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xt), device=dev)
        total = 0.0
        for i in range(0, len(xt), C.MLP_BATCH):
            idx = perm[i:i + C.MLP_BATCH]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(xv), yv).item()
        curve.append({"epoch": epoch, "train": total / len(xt), "val": val})
        if verbose:
            print(f"  epoch {epoch:02d} train {total / len(xt):.5f} val {val:.5f}")
        if val < best - 1e-6:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= C.MLP_PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, (xmu, xsd, ymu, ysd), curve


def apply_mlp(model, scalers, X) -> np.ndarray:
    xmu, xsd, ymu, ysd = scalers
    dev = next(model.parameters()).device
    with torch.no_grad():
        out = model(torch.tensor((X - xmu) / xsd, dtype=torch.float32, device=dev))
    return (out.cpu().numpy() * ysd + ymu).astype(np.float32)


def mlp_save(path, model, scalers):
    xmu, xsd, ymu, ysd = [torch.as_tensor(np.asarray(s), dtype=torch.float32) for s in scalers]
    torch.save({"state": model.state_dict(), "xmu": xmu, "xsd": xsd,
                "ymu": ymu, "ysd": ysd}, path)


def mlp_load(path):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = MLP().to("cpu").eval()
    model.load_state_dict(blob["state"])
    return model, tuple(blob[k].numpy() for k in ("xmu", "xsd", "ymu", "ysd"))
