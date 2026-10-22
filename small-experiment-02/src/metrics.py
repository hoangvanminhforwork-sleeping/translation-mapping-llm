"""Token-level translation metrics.

One function, one row per (direction, pooling, arm).  No hidden weighting.
"""
import re

import numpy as np
from sacrebleu.metrics import BLEU, CHRF, TER
from rouge_score import rouge_scorer

from . import config as C

_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub(" ", str(s).strip().lower())


def _nist(hyps, refs):
    from nltk.translate.nist_score import sentence_nist
    vals = []
    for h, r in zip(hyps, refs):
        try:
            vals.append(sentence_nist([r.split()], h.split()))
        except Exception:
            continue
    return float(np.mean(vals)) if vals else None


def _meteor(hyps, refs):
    import nltk
    nltk.data.path.insert(0, str(C.AGENT_WORK / "nltk_data"))
    for pkg in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, download_dir=str(C.AGENT_WORK / "nltk_data"), quiet=True)
    from nltk.translate.meteor_score import meteor_score
    return float(np.mean([meteor_score([r.split()], h.split()) for h, r in zip(hyps, refs)]))


def _rouge_l(hyps, refs):
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return float(np.mean([sc.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)]))


def compute(hyps, refs, tgt_lang: str) -> dict:
    hyps = [str(x) for x in hyps]
    refs = [str(x) for x in refs]
    out = {
        "n": len(hyps),
        "bleu": BLEU(tokenize="13a").corpus_score(hyps, [refs]).score,
        "chrf": CHRF().corpus_score(hyps, [refs]).score,
        "chrf_pp": CHRF(word_order=2).corpus_score(hyps, [refs]).score,
        "ter": TER().corpus_score(hyps, [refs]).score,
        "nist": _nist(hyps, refs),
        "rouge_l": _rouge_l(hyps, refs),
        "exact_match": float(np.mean([norm(h) == norm(r) for h, r in zip(hyps, refs)])),
        "meteor": _meteor(hyps, refs) if tgt_lang == "en" else None,
        "hyp_len": float(np.mean([len(h.split()) for h in hyps])),
        "ref_len": float(np.mean([len(r.split()) for r in refs])),
        "empty_rate": float(np.mean([len(h.strip()) == 0 for h in hyps])),
    }
    return out
