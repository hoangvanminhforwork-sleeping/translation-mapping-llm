"""Tokenisation with character offsets, and cross-tokeniser alignment.

Both checkpoints ship a `tokenizer.json`, so we load `T5TokenizerFast(tokenizer_file=...)`
directly. That sidesteps the broken `sentencepiece`/`protobuf` pair in the shared venv
(sentencepiece 0.2.x wants protobuf>=5.27, the venv has 4.25.9) and is also the only way to
get offsets from the Vietnamese checkpoint, whose `tokenizer_config` points at the slow class.

Alignment rule (identical to phase 01): target token j belongs to source token i when their
character spans intersect; a source token with no intersecting target token falls back to the
proportionally nearest target token, so the mapped sequence has no holes.
"""
from __future__ import annotations

import numpy as np
from transformers import T5TokenizerFast

from . import config as C


def load_tok(side: str) -> T5TokenizerFast:
    p = C.MODELS[side]["path"] / "tokenizer.json"
    tok = T5TokenizerFast(tokenizer_file=str(p))
    tok.backend_tokenizer.enable_truncation(max_length=C.MAX_LEN)
    return tok


class PairTokenizer:
    """Encodes (vi, en) batches into ids + masks + char offsets, padded to a fixed width."""

    def __init__(self, max_len: int = C.MAX_LEN):
        self.max_len = max_len
        self.tok = {"vi": load_tok("vi"), "en": load_tok("en")}
        self.be = {k: t.backend_tokenizer for k, t in self.tok.items()}
        self.vocab = {k: len(t) for k, t in self.tok.items()}

    def _encode(self, side: str, texts: list[str]):
        encs = self.be[side].encode_batch(texts, add_special_tokens=True)
        n = len(texts)
        ids = np.zeros((n, self.max_len), dtype=np.int64)
        mask = np.zeros((n, self.max_len), dtype=np.int64)
        off = np.zeros((n, self.max_len, 2), dtype=np.int64)
        vmax = C.MODELS[side]["vocab"]
        for r, e in enumerate(encs):
            m = min(len(e.ids), self.max_len)
            a = np.asarray(e.ids[:m], dtype=np.int64)
            a[a >= vmax] = 0  # <extra_id_96..99> overflow the vi embedding matrix (36096 rows)
            ids[r, :m] = a
            mask[r, :m] = 1
            off[r, :m] = np.asarray(e.offsets[:m], dtype=np.int64)
        return ids, mask, off

    def encode(self, vi: list[str], en: list[str]):
        vi_ids, vi_mask, vi_off = self._encode("vi", vi)
        en_ids, en_mask, en_off = self._encode("en", en)
        return {"vi_ids": vi_ids, "vi_mask": vi_mask, "vi_off": vi_off,
                "en_ids": en_ids, "en_mask": en_mask, "en_off": en_off}

    def encode_side(self, side: str, texts: list[str]):
        ids, mask, off = self._encode(side, texts)
        return {"ids": ids, "mask": mask, "off": off}


def align_np(src_off, src_mask, tgt_off, tgt_mask, tgt_h):
    """Pool target hidden states onto source token positions (numpy reference; GPU version below)."""
    s, e = src_off[..., 0][:, :, None], src_off[..., 1][:, :, None]
    ts, te = tgt_off[..., 0][:, None, :], tgt_off[..., 1][:, None, :]
    overlap = (np.minimum(e, te) - np.maximum(s, ts)) > 0
    overlap &= (src_mask[:, :, None] > 0) & (tgt_mask[:, None, :] > 0)
    cnt = overlap.sum(-1)
    w = overlap.astype(np.float32) / np.maximum(cnt, 1)[..., None]
    Y = np.einsum("bij,bjd->bid", w, tgt_h, optimize=True)
    empty = (cnt == 0) & (src_mask > 0)
    if empty.any():
        center = (s + e) / 2.0
        tcenter = (ts + te) / 2.0
        dist = np.where(tgt_mask[:, None, :] > 0, np.abs(center - tcenter), 1e9)
        nearest = dist.argmin(-1)
        Y = np.where(empty[..., None], np.take_along_axis(tgt_h, nearest[..., None], axis=1), Y)
    return Y.astype(np.float32), (cnt > 0) & (src_mask > 0)
