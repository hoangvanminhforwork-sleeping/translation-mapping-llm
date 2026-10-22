"""Vector2 -> Token2 by embedding inversion.

Find tokens whose encoder output matches the target vector.  Forward pass always
uses *real* token embeddings (straight-through), so every iteration is a valid
discrete hypothesis; the gradient only steers which tokens get chosen.

This is the inverse of the arrow the pipeline already contains, so it measures
how much sentence the vector still carries.  It is a probe of the channel, NOT
the "target model generates" pathway of paperOverview.md section 3.1.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C


def pool(hidden, mask, pooling):
    if pooling == "mean":
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1)
    if pooling == "cls":
        return hidden[:, 0]
    lengths = mask.sum(1).clamp(min=1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]


def reference_ids(tok, texts):
    """Token ids of the *reference* translation.  Only the count L is used."""
    enc = tok(list(texts), truncation=True, max_length=C.MAX_TOKENS,
              add_special_tokens=True)["input_ids"]
    return [list(map(int, x)) for x in enc]


def _nearest(E, embn, allowed, chunk=8192):
    """Chunked nearest-vocabulary search; avoids a [B,L,V] tensor."""
    B, L, D = E.shape
    best = torch.full((B, L), -1e4, device=E.device)
    ids = torch.zeros((B, L), dtype=torch.long, device=E.device)
    for s in range(0, embn.shape[0], chunk):
        sims = E @ embn[s:s + chunk].T
        bad = ~allowed[s:s + chunk]
        if bad.any():
            sims[:, :, bad] = -1e4
        m, idx = sims.max(-1)
        upd = m > best
        best = torch.where(upd, m, best)
        ids = torch.where(upd, idx + s, ids)
    return ids


def _run(ids0, targets, mu, tok, model, emb, embn, allowed, pooling,
         steps, lr, project_every, use_amp):
    dev = ids0.device
    B, L = ids0.shape
    attn = torch.ones((B, L), dtype=torch.long, device=dev)
    tgt = F.normalize(targets - mu, dim=-1)
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id

    hard = emb[ids0].detach()
    soft = hard.clone().requires_grad_(True)
    best_ids, best = ids0.clone(), torch.full((B,), -2.0, device=dev)
    opt = torch.optim.Adam([soft], lr=lr)

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        x = hard + (soft - soft.detach())          # forward = hard, backward -> soft
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            h = model(inputs_embeds=x, attention_mask=attn).last_hidden_state
        v = pool(h.float(), attn, pooling)
        cos = (F.normalize(v - mu, dim=-1) * tgt).sum(-1)
        (-cos.mean()).backward()
        opt.step()

        if (step + 1) % project_every == 0 or step == steps - 1:
            with torch.no_grad():
                new = _nearest(F.normalize(soft.detach(), dim=-1), embn, allowed)
                new[:, 0], new[:, -1] = cls_id, sep_id
                hard = emb[new].detach()
                hh = model(input_ids=new, attention_mask=attn).last_hidden_state
                cc = (F.normalize(pool(hh, attn, pooling) - mu, dim=-1) * tgt).sum(-1)
                better = cc > best
                best_ids[better] = new[better]
                best[better] = cc[better]
    return best_ids, best


class Readout:
    """One frozen encoder used as the decoder of its own vector space."""

    def __init__(self, tok, model, pooling: str):
        self.tok = tok
        self.model = model
        self.pooling = pooling
        self.emb = model.get_input_embeddings().weight.detach()
        self.embn = F.normalize(self.emb, dim=-1)
        forbidden = [tok.pad_token_id, tok.unk_token_id, tok.cls_token_id,
                     tok.sep_token_id, tok.mask_token_id]
        allowed = torch.ones(self.emb.shape[0], dtype=torch.bool, device=self.emb.device)
        allowed[torch.tensor([i for i in forbidden if i is not None],
                            device=allowed.device)] = False
        self.allowed = allowed

    @torch.no_grad()
    def encode_texts(self, texts, batch=32):
        out = []
        for s in range(0, len(texts), batch):
            enc = self.tok(list(texts[s:s + batch]), padding=True, truncation=True,
                           max_length=C.MAX_TOKENS, return_tensors="pt")
            enc = {k: v.to(self.emb.device) for k, v in enc.items()}
            h = self.model(**enc).last_hidden_state
            out.append(pool(h, enc["attention_mask"], self.pooling).float().cpu().numpy())
        return np.concatenate(out)

    def invert(self, targets: dict, ref_texts, mu, token_budget: int = C.INV_TOKEN_BUDGET,
               steps: int = C.INV_STEPS, lr: float = C.INV_LR, seed: int = C.SEED,
               verbose: bool = True):
        """targets: arm -> [n,768].  mu: mean of the *target* vector space.

        Returns arm -> list[str], arm -> centered cosine of the best discrete sequence.

        Init is information-free: body tokens are drawn uniformly from the target
        vocabulary (seeded).  No reference token is used, only its count L.
        """
        tok = self.tok
        dev = self.emb.device
        gen = torch.Generator(device=dev).manual_seed(seed)
        mu_t = torch.tensor(np.asarray(mu), dtype=torch.float32, device=dev)
        ref = reference_ids(tok, ref_texts)
        n = len(ref)
        arms = list(targets)
        for a in arms:
            assert targets[a].shape == (n, C.DIM), (a, targets[a].shape)

        order = sorted(range(n), key=lambda i: len(ref[i]))
        texts = {a: [""] * n for a in arms}
        disc = {a: np.zeros(n, dtype=np.float32) for a in arms}
        use_amp = C.INV_AMP and dev.type == "cuda"

        pos, done, t0 = 0, 0, time.time()
        while pos < n:
            L = max(len(ref[order[pos]]), 3)
            group = []
            while pos < n and max(len(ref[order[pos]]), 3) == L:
                group.append(order[pos])
                pos += 1
            b = max(1, min(len(group), token_budget // (len(arms) * L)))
            for start in range(0, len(group), b):
                idx = group[start:start + b]
                ids0 = torch.full((len(idx), L), tok.mask_token_id, dtype=torch.long, device=dev)
                body = torch.randint(0, self.emb.shape[0], (len(idx), L - 2),
                                     generator=gen, device=dev)
                ids0[:, 1:-1] = torch.where(self.allowed[body], body,
                                            torch.zeros_like(body))
                ids0[:, 0], ids0[:, -1] = tok.cls_token_id, tok.sep_token_id
                tgt = torch.tensor(
                    np.concatenate([targets[a][idx] for a in arms]),
                    dtype=torch.float32, device=dev)
                best_ids, best = _run(ids0.repeat(len(arms), 1), tgt, mu_t, tok, self.model,
                                      self.emb, self.embn, self.allowed, self.pooling,
                                      steps, lr, C.INV_PROJECT_EVERY, use_amp)
                best_ids = best_ids.view(len(arms), len(idx), L).cpu()
                best = best.view(len(arms), len(idx)).cpu().numpy()
                for r, a in enumerate(arms):
                    for q, i in enumerate(idx):
                        texts[a][i] = tok.decode(best_ids[r, q, 1:-1].tolist(),
                                                 skip_special_tokens=True)
                        disc[a][i] = float(best[r, q])
                done += len(idx)
                if verbose and done % max(b, 1) == 0 and (done // b) % 5 == 0:
                    el = time.time() - t0
                    print(f"    L={L:3d} {done}/{n}  {el:6.1f}s  "
                          f"eta {el / max(done, 1) * (n - done):6.1f}s", flush=True)
        return texts, disc
