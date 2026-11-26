"""Shared infrastructure for the coincidence experiment.

Self-contained on purpose: it does NOT import `research-phase01-ver02/src` (both packages are
called `src`; see that folder's HANDOFF §"Import trap"). Paths and constants are copied from
`research-phase01-ver02/src/config.py` so the numbers stay comparable with E7/E8/E9.

Vocabulary
    E_vi     hidden states of the *source* model (t5-small-vietnamese) on the Vietnamese sentence
    H_ridge  E_vi @ W + b, the E9 map (263 200 params) — the "representation-fitted" memory
    H_nat    hidden states of the *target* model's own encoder reading the Vietnamese source
             (the decoder's `native` channel; in-distribution memory, weak content)
    H_ref    hidden states of the target model's encoder reading the ENGLISH REFERENCE
             ("the answer" — a ceiling memory, may be out of the adapter's training distribution)
    H*       whichever oracle is in play
    CE       mean per-token NLL of the reference under the decoder, teacher-forced
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]            # coincidence-solution
PROJECT = ROOT.parent                                 # repo root
V2 = PROJECT / "research-phase01-ver02"
DATA = V2 / "data"
RESULTS = ROOT / "results"
WORK = ROOT / "agent-work"
CACHE = WORK / "cache"
for _d in (RESULTS, WORK, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(PROJECT / "research-prototype" / "models" / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LOCAL = PROJECT / "research-prototype" / "models" / "hf" / "local"
MODELS = {
    "en": {"path": LOCAL / "t5-small", "d_model": 512, "vocab": 32128},
    "vi": {"path": LOCAL / "t5-small-vietnamese", "d_model": 512, "vocab": 36096},
}
ADAPTER = V2 / "adapters" / "trans-t5-small-vi-en"
MAP = DATA / "map_vi_en.npz"
CHOSEN = DATA / "chosen.parquet"

MAX_LEN = 64
DT = torch.bfloat16          # generation / caching
DT_A = torch.float32         # analysis (gradients need real precision)

# Split sizes fixed by E9 so tables line up with `GEN_VARIANTS.md`.
N_TEST = 1_000
N_VAL = 1_000
N_TRAIN_FIT = 20_000
SEED = 1234


# ---------------------------------------------------------------- data
def load_pairs(split: str, n: int | None = None, offset: int = 0) -> dict:
    """Rows of `chosen.parquet` whose `split` equals `split`, in file order (= E8/E9 order)."""
    import pyarrow.parquet as pq
    d = pq.read_table(CHOSEN, columns=["vi", "en", "split"]).to_pydict()
    idx = [i for i, s in enumerate(d["split"]) if s == split]
    if offset:
        idx = idx[offset:]
    if n is not None:
        idx = idx[:n]
    return {"vi": [d["vi"][i] for i in idx], "en": [d["en"][i] for i in idx], "row": idx}


def load_fit_pairs(n: int = N_TRAIN_FIT) -> dict:
    """A fixed random subset of the train split (seed 1234), same rule as the adapter's training."""
    import pyarrow.parquet as pq
    d = pq.read_table(CHOSEN, columns=["vi", "en", "split"]).to_pydict()
    idx = [i for i, s in enumerate(d["split"]) if s == "train"]
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(len(idx), size=min(n, len(idx)), replace=False))
    rows = [idx[j] for j in pick]
    return {"vi": [d["vi"][i] for i in rows], "en": [d["en"][i] for i in rows], "row": rows}


# ---------------------------------------------------------------- models
def load_tok(side: str):
    from transformers import T5TokenizerFast
    tok = T5TokenizerFast(tokenizer_file=str(MODELS[side]["path"] / "tokenizer.json"))
    tok.backend_tokenizer.enable_truncation(max_length=MAX_LEN)
    return tok


def load_encoder(side: str, dtype=DT):
    from transformers import T5ForConditionalGeneration
    m = T5ForConditionalGeneration.from_pretrained(str(MODELS[side]["path"]), dtype=dtype,
                                                   low_cpu_mem_usage=True).to("cuda").eval()
    m.requires_grad_(False)
    return m


def load_decoder(dtype=DT):
    """t5-small + the E8/E9 translation LoRA (vi→en), decoder modules only."""
    from peft import PeftModel
    from transformers import T5ForConditionalGeneration
    base = T5ForConditionalGeneration.from_pretrained(str(MODELS["en"]["path"]), dtype=dtype,
                                                      low_cpu_mem_usage=True)
    if not (ADAPTER / "adapter_config.json").exists():
        raise SystemExit(f"missing adapter {ADAPTER}")
    return PeftModel.from_pretrained(base, str(ADAPTER)).to("cuda").eval()


def load_map() -> tuple[torch.Tensor, torch.Tensor]:
    z = np.load(MAP)
    return (torch.from_numpy(z["W"].astype(np.float32)).cuda(),
            torch.from_numpy(z["b"].astype(np.float32)).cuda())


# ---------------------------------------------------------------- encoding
@torch.no_grad()
def encode(side: str, enc, texts: list[str], batch: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Plain encoder states (N, L, 512) fp16 + attention mask (N, L) int8.

    The tokenizer of `side` is used, exactly like `Channels.native` (which reads Vietnamese with
    the *target* tokenizer). Ids above the model's embedding table are zeroed (vi has
    <extra_id_96..99> beyond row 36 096).
    """
    tok = load_tok(side)
    vmax = MODELS[side]["vocab"]
    hs, ms = [], []
    for s in range(0, len(texts), batch):
        b = tok(texts[s:s + batch], padding=True, truncation=True, max_length=MAX_LEN,
                return_tensors="pt")
        ids = b["input_ids"].cuda()
        ids = ids.masked_fill(ids >= vmax, 0)
        am = b["attention_mask"].cuda()
        h = enc.encoder(input_ids=ids, attention_mask=am).last_hidden_state
        hs.append(h.to(torch.float16).cpu().numpy())
        ms.append(am.to(torch.int8).cpu().numpy())
    L = max(h.shape[1] for h in hs)
    out = np.zeros((len(texts), L, MODELS[side]["d_model"]), dtype=np.float16)
    msk = np.zeros((len(texts), L), dtype=np.int8)
    i = 0
    for h, m in zip(hs, ms):
        b, lh = h.shape[0], h.shape[1]
        out[i:i + b, :lh] = h
        msk[i:i + b, :lh] = m
        i += b
    return out, msk


@torch.no_grad()
def encode_vi_mapped(vi_enc, texts: list[str], batch: int = 128):
    """E_vi at the token positions the map sees, plus the `keep` mask E9 uses.

    Mirrors `Channels.mapped`: specials are kept in the tensor but masked out, so lengths line up
    with `map_vi_en.npz` (fitted on exactly these positions).
    """
    tok = load_tok("vi")
    be = tok.backend_tokenizer
    vmax = MODELS["vi"]["vocab"]
    hs, ms = [], []
    for s in range(0, len(texts), batch):
        encs = be.encode_batch(texts[s:s + batch], add_special_tokens=True)
        n = len(encs)
        L = min(MAX_LEN, max(len(e.ids) for e in encs))
        ids = np.zeros((n, L), dtype=np.int64)
        am = np.zeros((n, L), dtype=np.int64)
        keep = np.zeros((n, L), dtype=np.int64)
        for r, e in enumerate(encs):
            m = min(len(e.ids), L)
            a = np.asarray(e.ids[:m], dtype=np.int64)
            a[a >= vmax] = 0
            ids[r, :m] = a
            am[r, :m] = 1
            o = np.asarray(e.offsets[:m], dtype=np.int64)
            keep[r, :m] = (o[:, 1] - o[:, 0]) > 0
        ids_t, am_t = torch.from_numpy(ids).cuda(), torch.from_numpy(am).cuda()
        h = vi_enc.encoder(input_ids=ids_t, attention_mask=am_t).last_hidden_state
        hs.append(h.to(torch.float16).cpu().numpy())
        ms.append(keep.astype(np.int8))
    L = max(h.shape[1] for h in hs)
    out = np.zeros((len(texts), L, MODELS["vi"]["d_model"]), dtype=np.float16)
    msk = np.zeros((len(texts), L), dtype=np.int8)
    i = 0
    for h, m in zip(hs, ms):
        b, lh = h.shape[0], h.shape[1]
        out[i:i + b, :lh] = h
        msk[i:i + b, :lh] = m
        i += b
    return out, msk


# ---------------------------------------------------------------- decoder side
def labels(texts: list[str]) -> torch.Tensor:
    tok = load_tok("en")
    b = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    lab = b["input_ids"].clone()
    lab[b["attention_mask"] == 0] = -100
    return lab.cuda()


@torch.no_grad()
def generate(model, h: torch.Tensor, am: torch.Tensor, max_new: int = 64,
             batch: int = 32) -> list[str]:
    from transformers.modeling_outputs import BaseModelOutput
    tok = load_tok("en")
    hyps = []
    for s in range(0, h.shape[0], batch):
        g = model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=h[s:s + batch]),
                           attention_mask=am[s:s + batch], max_new_tokens=max_new,
                           num_beams=1, do_sample=False)
        hyps += tok.batch_decode(g, skip_special_tokens=True)
    return hyps


def ce_per_sentence(model, h: torch.Tensor, am: torch.Tensor, lab: torch.Tensor | None = None,
                    refs: list[str] | None = None, batch: int = 16) -> np.ndarray:
    """Mean per-token NLL of the reference (same definition as `generate.nll`).

    Pass `lab` (cached labels) to avoid re-tokenising 16 times per sweep; `refs` is the fallback.
    """
    from transformers.modeling_outputs import BaseModelOutput
    if lab is None:
        lab = labels(refs)
    out = []
    for s in range(0, h.shape[0], batch):
        b = slice(s, s + batch)
        o = model(encoder_outputs=BaseModelOutput(last_hidden_state=h[b]),
                  attention_mask=am[b], labels=lab[b])
        lp = torch.log_softmax(o.logits.float(), -1)
        tgt = lab[b].clone()
        tgt[tgt == -100] = 0
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        m = (lab[b] != -100).float()
        out.append((-(tok_lp * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- metrics
def bleu(hyps: list[str], refs: list[str]) -> float:
    import sacrebleu
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def chrf(hyps: list[str], refs: list[str]) -> float:
    import sacrebleu
    return sacrebleu.corpus_chrf(hyps, [refs]).score


def sent_chrf(hyps: list[str], refs: list[str]) -> np.ndarray:
    """Per-sentence chrF (0-100), for graded agreement between two outputs."""
    from sacrebleu.metrics import CHRF
    m = CHRF()
    return np.array([m.sentence_score(h, [r]).score for h, r in zip(hyps, refs)])


def degen_rate(hyps: list[str]) -> float:
    """Fraction with unique/total token ratio <= 0.5 (E8/E9 definition)."""
    bad = 0
    for h in hyps:
        t = h.split()
        if not t:
            bad += 1
            continue
        if len(set(t)) / len(t) <= 0.5:
            bad += 1
    return bad / max(len(hyps), 1)


# ---------------------------------------------------------------- alignment (for the ridge control)
def token_offsets(side: str, texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ids / mask / char-offsets at MAX_LEN, the same layout `PairTokenizer` uses."""
    be = load_tok(side).backend_tokenizer
    vmax = MODELS[side]["vocab"]
    n = len(texts)
    ids = np.zeros((n, MAX_LEN), dtype=np.int64)
    mask = np.zeros((n, MAX_LEN), dtype=np.int64)
    off = np.zeros((n, MAX_LEN, 2), dtype=np.int64)
    for r, e in enumerate(be.encode_batch(texts, add_special_tokens=True)):
        m = min(len(e.ids), MAX_LEN)
        a = np.asarray(e.ids[:m], dtype=np.int64)
        a[a >= vmax] = 0
        ids[r, :m] = a
        mask[r, :m] = 1
        off[r, :m] = np.asarray(e.offsets[:m], dtype=np.int64)
    return ids, mask, off


def align_np(src_off, src_mask, tgt_off, tgt_mask, tgt_h):
    """Pool target hidden states onto source token positions by character-span overlap.

    Copied verbatim from `research-phase01-ver02/src/tokutil.py:align_np` (the function that built
    the map E9 used). A source token with no overlap falls back to the proportionally nearest
    target token, so the mapped sequence has no holes.
    """
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
