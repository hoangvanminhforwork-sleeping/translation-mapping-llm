"""Fast aligned encoding: one batch of EN-VI pairs -> (X, Y) token pairs on the GPU.

Design decisions, each measured rather than assumed (see LEARNING_NOTE.md §2):
  * pad to the *batch* maximum, rounded to 8, not to MAX_LEN. Sentences here average 29.7 (vi) and
    27.6 (en) subwords, so padding to 64 halves throughput for nothing.
  * do the cross-tokeniser pooling on the GPU (one bf16 matmul with a 0/1 weight matrix), because the
    only CPU work that matters is the tokeniser itself (22 ms/batch).
  * keep A = XᵀX, B = XᵀY, C = YᵀY on the GPU in float64 and copy 3 x 4.7 MB to the host once per shard,
    instead of 3 x (768x768) per batch.
  * prefetch the *next* batch's tokenisation on a worker thread: the GPU is busy 560 ms/batch and the
    tokeniser only needs 30 ms, so it hides completely.
"""
from __future__ import annotations

import queue
import threading

import numpy as np
import torch

from .tokutil import PairTokenizer


def crop_batch(b: dict, mult: int = 8) -> dict:
    """Trim the padded width to the longest real token in this batch (rounded up to `mult`)."""
    n_vi = int(b["vi_mask"].sum(1).max())
    n_en = int(b["en_mask"].sum(1).max())
    l_vi = min(b["vi_ids"].shape[1], max(8, ((n_vi + mult - 1) // mult) * mult))
    l_en = min(b["en_ids"].shape[1], max(8, ((n_en + mult - 1) // mult) * mult))
    out = {}
    for k in ("vi_ids", "vi_mask", "vi_off"):
        out[k] = b[k][:, :l_vi]
    for k in ("en_ids", "en_mask", "en_off"):
        out[k] = b[k][:, :l_en]
    return out


def align_torch(src_off, src_mask, tgt_off, tgt_mask, tgt_h):
    """Pool target states onto source positions on the GPU.

    offs: (B,T,2) int64 cuda, masks: (B,T) int64 cuda, tgt_h: (B,T,D) bf16 cuda.
    Returns Y (B,T,D) bf16, keep (B,T) bool. Source positions that intersect no target token
    (whitespace pieces, punctuation dropped by the other normaliser) are filled from the
    proportionally nearest target token, so no source position is silently dropped.
    """
    s = src_off[..., 0][:, :, None]
    e = src_off[..., 1][:, :, None]
    ts = tgt_off[..., 0][:, None, :]
    te = tgt_off[..., 1][:, None, :]
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
    lens = keep.sum(-1)
    return Y, keep, lens


def _to_cuda(b: dict):
    return {k: torch.from_numpy(v).cuda() for k, v in b.items()}


class AlignedBatcher:
    """Yields (X, Y) float32 cuda tensors of aligned token pairs for a list of pairs."""

    def __init__(self, enc, batch: int = 256, prefetch: int = 3):
        self.enc = enc
        self.batch = batch
        self.q: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = False

    def _producer(self, vi, en, order):
        try:
            for s in range(0, len(order), self.batch):
                idx = order[s:s + self.batch]
                b = self.enc.tok.encode([vi[i] for i in idx], [en[i] for i in idx])
                b = crop_batch(b)
                if self._stop:
                    return
                self.q.put((s, len(idx), b))
        finally:
            self.q.put(None)

    def __iter__(self):
        raise NotImplementedError

    def batches(self, vi, en, order=None):
        order = order if order is not None else sorted(range(len(vi)), key=lambda i: len(vi[i]))
        t = threading.Thread(target=self._producer, args=(vi, en, order), daemon=True)
        t.start()
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                s, n, b = item
                cb = _to_cuda(b)
                with torch.no_grad():
                    h_vi = self.enc.vit5.encoder(input_ids=cb["vi_ids"], attention_mask=cb["vi_mask"]).last_hidden_state
                    h_en = self.enc.flan.encoder(input_ids=cb["en_ids"], attention_mask=cb["en_mask"]).last_hidden_state
                    Y, keep, lens = align_torch(cb["vi_off"], cb["vi_mask"], cb["en_off"], cb["en_mask"], h_en)
                X = h_vi[keep].float()
                Y = Y[keep].float()
                yield s, n, X, Y, lens
        finally:
            self._stop = True
