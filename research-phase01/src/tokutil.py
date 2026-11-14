"""Tokenisation with character offsets for both models, plus cross-tokeniser alignment.

Why this file exists: `VietAI/vit5-base` ships a *slow* sentencepiece tokenizer, so
`return_offsets_mapping` raises KeyError. Converting its sp_model to a `tokenizers.Tokenizer`
(chunk-level BPE over sentencepiece pieces) restores offsets, verified to reconstruct the source
string exactly. The flan tokenizer is already fast.

Alignment rule: target token j is attached to source token i when their character spans intersect
(after removing special tokens, whose span is (0, 0)). A source token with no intersecting target
token — punctuation dropped by the other normaliser, whitespace pieces — falls back to the
nearest target token by character midpoint, so every source position gets a target vector and the
mapped sequence has no holes.
"""
from __future__ import annotations

import numpy as np
from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import convert_slow_tokenizer


def _backend(hf_id: str):
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tok.is_fast:
        return tok, tok.backend_tokenizer
    # Slow sentencepiece tokenizer: build an equivalent fast backend to get offsets.
    be = convert_slow_tokenizer(tok)
    return tok, be


class PairTokenizer:
    """Encodes (vi, en) batches into ids + masks + char offsets, padded to a fixed width."""

    def __init__(self, max_len: int = 64):
        self.max_len = max_len
        self.vit5_tok, self.vit5_be = _backend("VietAI/vit5-base")
        self.flan_tok, self.flan_be = _backend("google/flan-t5-base")

    @staticmethod
    def _encode(be, texts: list[str], max_len: int):
        encs = be.encode_batch(texts, add_special_tokens=True)
        ids = np.zeros((len(texts), max_len), dtype=np.int64)
        mask = np.zeros((len(texts), max_len), dtype=np.int64)
        off = np.zeros((len(texts), max_len, 2), dtype=np.int64)
        for r, e in enumerate(encs):
            n = min(len(e.ids), max_len - 1)  # keep room for EOS
            ids[r, :n] = e.ids[:n]
            ids[r, n] = be.token_to_id("</s>") if be.token_to_id("</s>") is not None else 1
            mask[r, : n + 1] = 1
            off[r, :n] = e.offsets[:n]
        return ids, mask, off

    def encode(self, vi: list[str], en: list[str]):
        vi_ids, vi_mask, vi_off = self._encode(self.vit5_be, vi, self.max_len)
        en_ids, en_mask, en_off = self._encode(self.flan_be, en, self.max_len)
        return {
            "vi_ids": vi_ids, "vi_mask": vi_mask, "vi_off": vi_off,
            "en_ids": en_ids, "en_mask": en_mask, "en_off": en_off,
        }

    def encode_vi(self, vi: list[str]):
        """Only the vit5 side (the mapped channel never needs the English tokenisation)."""
        vi_ids, vi_mask, vi_off = self._encode(self.vit5_be, vi, self.max_len)
        return {"vi_ids": vi_ids, "vi_mask": vi_mask, "vi_off": vi_off}


def align(src_off: np.ndarray, src_mask: np.ndarray, tgt_off: np.ndarray, tgt_mask: np.ndarray,
          tgt_h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool target hidden states onto source token positions.

    src_off/tgt_off: (B,T,2); masks: (B,T); tgt_h: (B,T,D) float32.
    Returns (Y (B,T,D) float32, keep (B,T) bool). keep=False where nothing can be aligned
    (padding and special tokens).
    """
    s = src_off[..., 0][:, :, None].astype(np.int64)
    e = src_off[..., 1][:, :, None].astype(np.int64)
    ts = tgt_off[..., 0][:, None, :].astype(np.int64)
    te = tgt_off[..., 1][:, None, :].astype(np.int64)
    overlap = (np.minimum(e, te) - np.maximum(s, ts)) > 0
    overlap &= (src_mask[:, :, None] > 0) & (tgt_mask[:, None, :] > 0)

    cnt = overlap.sum(-1)
    w = overlap.astype(np.float32)
    w /= np.maximum(cnt, 1)[..., None]
    Y = np.einsum("bij,bjd->bid", w, tgt_h, optimize=True)

    # fallback for source tokens with no intersecting target token
    empty = (cnt == 0) & (src_mask > 0)
    if empty.any():
        center = (s + e) / 2.0
        tcenter = (ts + te) / 2.0
        dist = np.abs(center - tcenter)
        dist = np.where((tgt_mask[:, None, :] > 0), dist, 1e9)
        nearest = dist.argmin(-1)
        Y = np.where(empty[..., None], np.take_along_axis(tgt_h, nearest[..., None], axis=1), Y)

    keep = (cnt > 0) & (src_mask > 0)
    return Y.astype(np.float32), keep
