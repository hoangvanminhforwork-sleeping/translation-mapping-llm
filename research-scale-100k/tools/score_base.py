"""Score whatever base-decoder generations exist. CPU only; safe to run after an interrupted pipeline."""

import json

import numpy as np
import pyarrow.parquet as pq

from src import config as C


def main() -> None:
    import sacrebleu

    t = pq.read_table(C.PAIRS, columns=["vi", "en", "split"])
    en_te = [b for b, s in zip(t["en"].to_pylist(), t["split"].to_pylist()) if s == "test"]
    out = {"n_ref": len(en_te), "decoder": "google/flan-t5-base (no adapter)", "scores": {}}
    for name in ("mapped_base", "flan_encoder_base"):
        path = C.RESULTS / f"gen_{name}.jsonl"
        if not path.exists():
            out["scores"][name] = "missing"
            continue
        hyps = [json.loads(l)["hyp"] for l in path.read_text(encoding="utf-8").splitlines()]
        if len(hyps) != len(en_te):
            out["scores"][name] = f"partial: {len(hyps)}/{len(en_te)} hypotheses"
            continue
        out["scores"][name] = {
            "bleu": round(sacrebleu.corpus_bleu(hyps, [en_te]).score, 3),
            "chrf": round(sacrebleu.corpus_chrf(hyps, [en_te]).score, 3),
            "empty": sum(1 for h in hyps if not h.strip()),
            "mean_hyp_tokens": round(float(np.mean([len(h.split()) for h in hyps])), 2),
            "degenerate": round(float(np.mean([
                (1.0 - len(set(h.lower().split())) / max(1, len(h.split()))) > 0.5 for h in hyps])), 3),
        }
    (C.RESULTS / "bleu_vi_en_base_decoder.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
