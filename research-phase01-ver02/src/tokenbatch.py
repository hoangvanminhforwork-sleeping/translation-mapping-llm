"""Fast aligned encoding: one batch of pairs -> (X, Y) token pairs on the GPU.

Direction-aware: `src` and `tgt` name the two models, and each side is tokenised with *its own*
tokenizer. (The first version hardcoded "vi source, en target", which silently computed the en→vi
geometry with the wrong text in each encoder.)

  * pad to the *batch* maximum rounded to 8, not to MAX_LEN;
  * cross-tokeniser pooling on the GPU as one weighted matmul;
  * tokenise the next batch on a worker thread while the GPU is busy.
"""
from __future__ import annotations

import queue
import threading

import numpy as np
import torch

from . import config as C


def crop_batch(b: dict, mult: int = 8) -> dict:
    n_src = int(b["src_mask"].sum(1).max())
    n_tgt = int(b["tgt_mask"].sum(1).max())
    l_src = min(b["src_ids"].shape[1], max(8, ((n_src + mult - 1) // mult) * mult))
    l_tgt = min(b["tgt_ids"].shape[1], max(8, ((n_tgt + mult - 1) // mult) * mult))
    out = {}
    for k in ("src_ids", "src_mask", "src_off"):
        out[k] = b[k][:, :l_src]
    for k in ("tgt_ids", "tgt_mask", "tgt_off"):
        out[k] = b[k][:, :l_tgt]
    return out


def align_torch(src_off, src_mask, tgt_off, tgt_mask, tgt_h):
    """Pool target states onto source positions on the GPU. Returns Y, keep, lens."""
    s, e = src_off[..., 0][:, :, None], src_off[..., 1][:, :, None]
    ts, te = tgt_off[..., 0][:, None, :], tgt_off[..., 1][:, None, :]
    overlap = (torch.minimum(e, te) - torch.maximum(s, ts)) > 0
    overlap &= (src_mask[:, :, None] > 0) & (tgt_mask[:, None, :] > 0)

    w = overlap.to(tgt_h.dtype)
    cnt = w.sum(-1)
    Y = (w / cnt.clamp(min=1)[..., None]) @ tgt_h

    empty = (cnt == 0) & (src_mask > 0)
    if bool(empty.any()):
        B, T = src_mask.shape
        pr = torch.arange(T, device=src_mask.device).float() / max(T - 1, 1)
        tgt_len = tgt_mask.sum(-1).clamp(min=1).float()
        j = (pr[None, :] * (tgt_len[:, None] - 1)).round().long().clamp(min=0)
        Y = torch.where(empty[..., None], tgt_h.gather(1, j[..., None].expand(-1, -1, tgt_h.shape[-1])), Y)

    keep = (cnt > 0) & (src_mask > 0)
    return Y, keep, keep.sum(-1)


def _to_cuda(b: dict):
    return {k: torch.from_numpy(v).cuda() for k, v in b.items()}


class AlignedBatcher:
    """Yields (start, n, X, Y, lens) float32 cuda tensors of aligned token pairs."""

    def __init__(self, enc, batch: int = 256, prefetch: int = 3, src: str = "vi", tgt: str = "en"):
        self.enc, self.batch, self.src, self.tgt = enc, batch, src, tgt
        self.q: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = False

    def _encode(self, src_texts, tgt_texts):
        a = self.enc.tok.encode_side(self.src, src_texts)
        b = self.enc.tok.encode_side(self.tgt, tgt_texts)
        return {"src_ids": a["ids"], "src_mask": a["mask"], "src_off": a["off"],
                "tgt_ids": b["ids"], "tgt_mask": b["mask"], "tgt_off": b["off"]}

    def _producer(self, src_texts, tgt_texts, order):
        try:
            for s in range(0, len(order), self.batch):
                idx = order[s:s + self.batch]
                b = crop_batch(self._encode([src_texts[i] for i in idx], [tgt_texts[i] for i in idx]))
                if self._stop:
                    return
                self.q.put((s, len(idx), b))
        finally:
            self.q.put(None)

    def batches(self, src_texts, tgt_texts, order=None):
        order = order if order is not None else sorted(range(len(src_texts)), key=lambda i: len(src_texts[i]))
        t = threading.Thread(target=self._producer, args=(src_texts, tgt_texts, order), daemon=True)
        t.start()
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                s, n, b = item
                cb = _to_cuda(b)
                with torch.no_grad():
                    h_src = self.enc.model[self.src].encoder(
                        input_ids=cb["src_ids"], attention_mask=cb["src_mask"]).last_hidden_state
                    h_tgt = self.enc.model[self.tgt].encoder(
                        input_ids=cb["tgt_ids"], attention_mask=cb["tgt_mask"]).last_hidden_state
                    Y, keep, lens = align_torch(cb["src_off"], cb["src_mask"],
                                                cb["tgt_off"], cb["tgt_mask"], h_tgt)
                yield s, n, h_src[keep].float(), Y[keep].float(), lens
        finally:
            self._stop = True


class SideBatcher:
    """Yields (start, n, pooled) for one side only: used for sentence embeddings."""

    def __init__(self, enc, side: str, batch: int = 256, prefetch: int = 3):
        self.enc, self.side, self.batch = enc, side, batch
        self.q: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = False

    def _producer(self, texts):
        try:
            for s in range(0, len(texts), self.batch):
                chunk = texts[s:s + self.batch]
                ids, mask, _off = self.enc.tok._encode(self.side, chunk)
                n = int(mask.sum(1).max())
                L = max(8, ((n + 7) // 8) * 8)
                if self._stop:
                    return
                self.q.put((s, len(chunk), ids[:, :L], mask[:, :L]))
        finally:
            self.q.put(None)

    def batches(self, texts):
        t = threading.Thread(target=self._producer, args=(texts,), daemon=True)
        t.start()
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                s, n, ids, mask = item
                with torch.no_grad():
                    h = self.enc.model[self.side].encoder(
                        input_ids=torch.from_numpy(ids).cuda(),
                        attention_mask=torch.from_numpy(mask).cuda()).last_hidden_state
                    m = torch.from_numpy(mask).cuda().unsqueeze(-1)
                    pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
                yield s, n, pooled.float().cpu().numpy()
        finally:
            self._stop = True
