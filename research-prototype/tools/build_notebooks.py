"""Build the analysis notebooks.

Why a builder instead of hand-written .ipynb JSON: seven notebooks share the same
preamble and the same table-reading idioms; keeping them generated means the
"restart and run all" contract cannot drift between notebooks.

    python tools/build_notebooks.py

Notebooks are the analysis surface (doc-50). The heavy generation lives in
src/run_all.py; the notebooks read its CSVs and re-run only small, live samples.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

PREAMBLE = '''\
import csv, json, os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TQDM_DISABLE", "1")
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name != "research-prototype" else Path.cwd()
sys.path.insert(0, str(ROOT))
import torch
from src import config as C, models as M, data as D

def read_csv(name):
    p = ROOT / "results" / name
    if not p.exists():
        print(f"missing {p.name} - run: python -m src.run_all (stage not executed yet)")
        return []
    return list(csv.DictReader(p.open(encoding="utf-8")))

print("python", sys.version.split()[0], "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
'''

PROMPTS = {
    "vi": [
        "Nói: Phở bò là món ăn nổi tiếng của Việt Nam.",
        "translate English to Vietnamese: Vietnamese food is rich and varied.",
        "1 + 3 = ?",
    ],
    "en": [
        "Say: Pho is a famous Vietnamese dish.",
        "translate Vietnamese to English: Phở bò là món ăn nổi tiếng của Việt Nam.",
        "1 + 3 = ?",
    ],
}

SENT = {
    "vi": "Hôm nay tôi ăn phở bò với bánh mì ở quán quen.",
    "en": "Today I ate pho bo with banh mi at my usual place.",
}


def cells(*spec):
    out = []
    for kind, text in spec:
        out.append(nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text))
    return out


def write(path: Path, cs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook(cells=cs, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    nbf.write(nb, str(path))
    print(f"built {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------
def explorer(key: str, idx: str) -> None:
    spec = C.MODELS[key]
    lang = spec["lang"]
    dish_sentence = SENT[lang]
    prompts = PROMPTS[lang]
    pair = "base" if spec["size_class"] == "base" else "large"
    mate = C.PAIRS[pair][0] if spec["size_class"] and C.PAIRS[pair][1] == key else C.PAIRS[pair][1]
    d_model = 768 if pair == "base" else 1024

    cs = cells(
        ("md", f"""# {spec['hf_id']} — model exploratory

**Three questions**
1. What is this model made of, and where would an embedding map attach?
2. How does it cut a Vietnamese/English dish name into tokens?
3. With no fine-tuning, can it copy a sentence — and where does it hesitate?

Pairing rule from the mission: this model is compared **{pair} with {pair}** (`{mate}`), never across sizes."""),
        ("code", PREAMBLE),
        ("md", "## 1. Load — and print the device\n\n`res-33`: a run that 'used the GPU' is a claim until the device string is printed."),
        ("code", f'tok, model, info = M.load("{key}")\ninfo'),
        ("md", "## 2. Architecture — the surface a mapping would touch"),
        ("code", """cfg = model.config
rows = [("d_model", cfg.d_model), ("encoder layers", cfg.num_layers),
        ("decoder layers", cfg.num_decoder_layers), ("heads", cfg.num_heads),
        ("d_ff", cfg.d_ff), ("vocab", cfg.vocab_size),
        ("relative buckets", cfg.relative_attention_num_buckets), ("max positions", cfg.n_positions)]
enc = sum(p.numel() for p in model.encoder.parameters())
dec = sum(p.numel() for p in model.decoder.parameters())
emb = model.shared.weight.numel()
print(f"{'property':<20}{'value':>12}")
for k, v in rows: print(f"{k:<20}{v:>12}")
print(f"{'params enc':<20}{enc/1e6:>10.1f}M")
print(f"{'params dec':<20}{dec/1e6:>10.1f}M")
print(f"{'params emb (shared)':<20}{emb/1e6:>10.1f}M")"""),
        ("md", f"""**Read this as:** every encoder layer outputs a `[seq, {d_model}]` matrix. The paper's map
sits between two sequences of such matrices, so the number of *candidate attach points* is
`layers + 1` per side, and the two models must agree on `d_model` ({d_model} → {d_model})."""),
        ("md", "## 3. Tokenizer — a dish name is a span, not a token"),
        ("code", f'''samples = D.build(force=False)
lang = C.MODELS["{key}"]["lang"]
texts = [r[lang] for r in samples]
dishes = [r["dish_vi"] if lang == "vi" else r["dish_en"] for r in samples]

tpw = [len(tok(t, add_special_tokens=False)["input_ids"]) / max(1, len(t.split())) for t in texts]
dtok = [len(tok(d, add_special_tokens=False)["input_ids"]) for d in dishes]
print(f"tokens per word      : mean {{sum(tpw)/len(tpw):.2f}}  max {{max(tpw):.2f}}")
print(f"dish-name subwords   : mean {{sum(dtok)/len(dtok):.2f}}  max {{max(dtok)}}")

ids = tok({dish_sentence!r}, add_special_tokens=False)["input_ids"]
print("\\nsentence:", {dish_sentence!r})
print("pieces  :", " | ".join(p.replace("\\u2581", "\\u2423") for p in tok.convert_ids_to_tokens(ids)))'''),
        ("md", f"""Vietnamese words are split more finely than English ones (diacritics + a {spec['lang'].upper()} vocabulary),
so a dish name can be 1–6 subword pieces. **Consequence for measurement:** "perplexity of the dish name"
must be computed over a token *span*, not a single token — which is what `src/score.py` does."""),
        ("md", "## 4. One forward pass — what the layers actually hold"),
        ("code", f'''enc_in = tok([{dish_sentence!r}], return_tensors="pt").to(M.device())
with torch.no_grad():
    enc_out = model.encoder(**enc_in, output_hidden_states=True)
hs = enc_out.hidden_states
print("hidden states returned:", len(hs), "| shape:", tuple(hs[0].shape))
for i, h in enumerate(hs):
    print(f"layer {{i:2d}}  pooled-mean norm {{h[0].mean(0).norm():7.2f}}  last-token norm {{h[0][-1].norm():7.2f}}")'''),
        ("md", f"""The norm grows with depth, and the last token of the encoder is the summary a decoder would
attend to. If you later probe "which layer carries the dish meaning", these 13 matrices are the grid."""),
        ("md", "## 5. Zero-shot behaviour (no fine-tuning)"),
        ("code", f'''def gen(prompt, n=64):
    ids = tok(prompt, return_tensors="pt").to(M.device())
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=n, num_beams=1)
    return tok.decode(out[0], skip_special_tokens=True)

for p in {prompts!r}:
    print("PROMPT:", p)
    print("OUTPUT:", gen(p))
    print()'''),
        ("md", "## 6. Copy test — 2,000 sentences"),
        ("code", f'''rows = read_csv("copy_{key}.csv")
if rows:
    bad = [r for r in rows if r["exact"] == "0"][:5]
    print(f"exact copies: {{sum(r['exact'] == '1' for r in rows)}} / {{len(rows)}}")
    for r in bad:
        print("SRC:", r["src"][:110])
        print("HYP:", r["hyp"][:110])
        print()
    s = [r for r in read_csv("copy_summary.csv") if r["model"] == "{key}"]
    if s:
        print({{k: s[0][k] for k in ("n", "exact_match", "exact_match_norm", "bleu", "chrf",
                                   "tok_recall", "mean_hyp_words", "mean_ref_words")}})'''),
        ("md", "## 7. Perplexity — is the dish name harder than the rest of the sentence?"),
        ("code", f'''s = [r for r in read_csv("ppl_summary.csv") if r["model"] == "{key}"]
if s:
    r = s[0]
    print({{k: r[k] for k in ("ppl_all", "ppl_dish", "ppl_other", "rank_dish", "rank_other",
                            "delta_nll", "delta_ci_lo", "delta_ci_hi", "dish_harder_sentences", "sentences")}})
toks = read_csv("ppl_tokens_{key}.csv")
if toks:
    print("\\nhardest tokens (highest NLL):")
    for r in sorted(toks, key=lambda x: -float(x["nll"]))[:10]:
        print(f"  {{float(r['nll']):6.2f}}  rank {{int(r['rank']):>6}}  dish={{r['in_dish']}}  {{r['span']!r}}")'''),
        ("md", """`delta_nll > 0` means dish tokens are *less* predictable than the sentence around them.
The CI is a paired bootstrap over the 100 sentences, so it answers "is the dish harder **in the same sentence**",
not "is this model worse than that model"."""),
        ("md", f"""## 8. What this means for the paper

- The *copy gate* is answered in §6: if `exact_match` is low, fine-tuning is not optional — the model will not
  even reproduce its own input, so a mapped vector cannot be read out.
- The *attach surface* in §2/§4 is `[seq, {d_model}]` at 13 depths; the pairing rule keeps `base` with `base`.
- The *dish span* in §3/§7 is the unit the paper's perplexity story has to be told in."""),
    )
    write(ROOT / "model-exploratory" / f"{idx}_{key}.ipynb", cs)


def instruction_following() -> None:
    cs = cells(
        ("md", """# Instruction-following: can the model copy what it is given?

**The gate.** `thought01.md` proposes teaching the model to *repeat* (`Say: {s_e}` / `Nói: {s_v}`) before
any mapping is trained. If the model already copies zero-shot, fine-tuning is unnecessary. If it does not,
the copy task is the first thing to fine-tune.

Number of sentences: **2,000** (deterministic sample of the MTET test split, seed 42).
Each model is prompted **in its own language**."""),
        ("code", PREAMBLE),
        ("md", "## 1. The exact prompt strings"),
        ("code", '''from src import copy_eval

rows = copy_eval.sample_sentences(6)
for r in rows[:6]:
    print("vi:", r["vi"][:90])
    print("en:", r["en"][:90])
    print("   prompt(vi model):", C.MODELS["vit5-base"]["copy_prefix"] + r["vi"][:60])
    print("   prompt(en model):", C.MODELS["flan-t5-base"]["copy_prefix"] + r["en"][:60])
    print()'''),
        ("md", """## 2. Live sample — 8 prompts per base model

Only 8, so the cell stays fast. Section 3 shows the full 2,000-row result."""),
        ("code", '''def demo(key, n=8):
    spec = C.MODELS[key]
    lang = spec["lang"]
    tok, model, _ = M.load(key)
    rs = copy_eval.sample_sentences(n)
    prompts = [spec["copy_prefix"] + r[lang] for r in rs]
    hyps = copy_eval._generate(tok, model, prompts, 96)
    for src, hyp in zip([r[lang] for r in rs], hyps):
        ok = "OK " if hyp.strip() == src.strip() else "BAD"
        print(f"{ok} src: {src[:80]}")
        print(f"    hyp: {hyp[:80]}")
    M.free(model)

demo("vit5-base")
demo("flan-t5-base")'''),
        ("md", "## 3. Full result, 2,000 sentences x 4 models"),
        ("code", '''table = read_csv("copy_summary.csv")
cols = ["model", "size_class", "n", "exact_match", "exact_match_norm", "bleu", "chrf", "tok_recall", "mean_hyp_words", "mean_ref_words"]
if table:
    print(" | ".join(cols))
    for r in table:
        print(" | ".join(str(r[c]) for c in cols))'''),
        ("md", "## 4. Failure taxonomy\n\nFour failure shapes matter, because they need different fixes: **empty**, **truncated**, **looped**, **paraphrased**."),
        ("code", '''def classify(src, hyp):
    if not hyp.strip():
        return "empty"
    if hyp.strip() == src.strip():
        return "exact"
    if len(hyp.split()) < 0.8 * len(src.split()):
        return "truncated"
    if len(hyp.split()) > 1.8 * len(src.split()):
        return "looped"
    return "paraphrased"

for r in read_csv("copy_summary.csv"):
    key = r["model"]
    rows = read_csv(f"copy_{key}.csv")
    if not rows:
        continue
    from collections import Counter
    cnt = Counter(classify(x["src"], x["hyp"]) for x in rows)
    total = sum(cnt.values())
    print(key, {k: f"{v/total:.1%}" for k, v in cnt.most_common()})'''),
        ("md", "## 5. Verdict"),
        ("code", '''for r in read_csv("copy_summary.csv"):
    verdict = "copies" if float(r["exact_match_norm"]) > 0.5 else "does not copy"
    print(f"{r['model']:<14} {verdict:<12} exact {float(r['exact_match_norm']):.1%}  BLEU {r['bleu']}")'''),
        ("md", """**What to conclude.** `exact_match_norm` is the number that decides whether fine-tuning on the copy task
is needed. BLEU on the copy task is a softer version of the same signal — it rewards partial copying that
exact match punishes, which is why both are reported.

If a model copies its own language reliably, the next experiment is the *mapping* (`small-experiment`),
not more fine-tuning."""),
    )
    write(ROOT / "model-instruction-following" / "01_instruction_following.ipynb", cs)


def ppl_notebook() -> None:
    cs = cells(
        ("md", """# Token generation & perplexity on dish names

**Question.** When a model reads a culinary sentence, is the *dish name* the hard part?

**Method.** Pseudo-perplexity: the sentence is the target, the decoder is causal, so token *t* is scored
given tokens *< t*. Every token is tagged **dish** or **other**; the comparison is paired within the sentence."""),
        ("code", PREAMBLE),
        ("md", "## 1. The test set — 100 authored pairs"),
        ("code", '''pairs = D.build(force=False)
for r in pairs[:5]:
    print(f"[{r['id']:3d}] {r['dish_vi']:<22} | {r['vi'][:70]}")
    print(f"      {r['dish_en']:<22} | {r['en'][:70]}")'''),
        ("md", "## 2. Live — per-token NLL for one sentence"),
        ("code", '''from src import score

tok, model, _ = M.load("vit5-base")
rows = score.explain(tok, model, pairs[4]["vi"], pairs[4]["dish_vi"])
print(f"{'pos':>4} {'piece':<14}{'span':<16}{'dish':<6}{'nll':>7}{'rank':>7}")
for r in rows:
    print(f"{r['pos']:>4} {r['piece']:<14}{r['span']:<16}{str(r['dish']):<6}{r['nll']:>7.2f}{r['rank']:>7}")

joined = "".join(r["span"] for r in rows if r["dish"]).strip()
assert joined.lower() == pairs[4]["dish_vi"].lower(), (joined, pairs[4]["dish_vi"])
print(f"\\nOK: dish-token spans concatenate back to {joined!r}")
M.free(model)'''),
        ("md", "`rank` = position of the true token in the model's own distribution. Rank 1 means the model would have produced it. This is the most human-readable version of the same quantity as NLL."),
        ("md", "## 3. Aggregate — dish vs the rest"),
        ("code", '''table = read_csv("ppl_summary.csv")
cols = ["model", "lang", "ppl_all", "ppl_dish", "rank_dish", "ppl_other", "rank_other", "delta_nll", "delta_ci_lo", "delta_ci_hi", "dish_harder_sentences", "sentences"]
if table:
    print(" | ".join(cols))
    for r in table:
        print(" | ".join(str(r[c]) for c in cols))'''),
        ("md", "## 4. How many sentences show the effect?"),
        ("code", '''import matplotlib.pyplot as plt

for r in read_csv("ppl_summary.csv"):
    lo, hi, d = float(r["delta_ci_lo"]), float(r["delta_ci_hi"]), float(r["delta_nll"])
    sig = "significant" if (lo > 0 or hi < 0) else "not significant"
    print(f"{r['model']:<14} delta {d:+.3f} nats  CI [{lo:+.3f}, {hi:+.3f}]  {sig}  "
          f"({r['dish_harder_sentences']}/{r['sentences']} sentences)")'''),
        ("md", """## 5. Caveats that decide how strong the claim can be

- This is a **copy loss**, not a corpus language-model perplexity: the encoder sees the whole sentence, so
  the model is scored on reproducing tokens it has already read. Comparable **within** a model only.
- A dish name is often a rare proper-ish noun, so some elevation is expected even without any culinary
  specialness. The honest control is the within-sentence comparison, plus the `rank` column.
- Vietnamese and English numbers are not on the same scale (different vocabularies, different tokenisation
  density: a Vietnamese dish name can be 1 piece, an English transliteration 6).
- Position matters: the first token has no decoder context. It is kept in the CSV (`has_context`) and
  excluded from the summary.
"""),
    )
    write(ROOT / "test-token-generation" / "01_testset_and_perplexity.ipynb", cs)


def bleu_notebook() -> None:
    cs = cells(
        ("md", """# BLEU — the copy baseline and zero-shot translation

**Two different BLEUs, two different questions.**

| BLEU | Question it answers |
|---|---|
| copy-task BLEU (2,000 sentences) | how much of the input survives the prompt |
| zero-shot translation BLEU (100 pairs) | what the methodology has to beat, with no training |

Directions are assigned so that the target language decides the model: `en->vi` uses ViT5, `vi->en` uses
Flan-T5 — one base pair, one large pair, never mixed."""),
        ("code", PREAMBLE),
        ("md", "## 1. The reference set"),
        ("code", '''pairs = D.build(force=False)
print("pairs:", len(pairs))
for r in pairs[:3]:
    print(f"[{r['id']}] VI: {r['vi']}")
    print(f"      EN: {r['en']}")'''),
        ("md", "## 2. Zero-shot translation — examples"),
        ("code", '''raw = ROOT / "results" / "raw"
for f in sorted(raw.glob("tr_*.jsonl")):
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()][:2]
    print("==", f.stem)
    for r in rows:
        p = pairs[r["id"] - 1]
        print("  hyp:", r["hyp"][:110])
        print("  ref:", (p["vi"] if "en2vi" in f.stem else p["en"])[:110])
        print()'''),
        ("md", "## 3. Scores"),
        ("code", '''table = read_csv("bleu_summary.csv")
cols = ["model", "direction", "n", "bleu", "chrf", "dish_name_kept", "empty_rate", "len_ratio"]
if table:
    print(" | ".join(cols))
    for r in table:
        print(" | ".join(str(r[c]) for c in cols))'''),
        ("md", """`dish_name_kept` is the paper-relevant number that BLEU cannot see: the fraction of outputs that still
contain the dish name (transliterated for `en->vi`, romanised for `vi->en`). A model can score BLEU 0 and
still keep every dish name — which is exactly the cultural-preservation question."""),
        ("md", "## 4. What this baseline means"),
        ("code", '''for r in read_csv("bleu_summary.csv"):
    print(f"{r['model']:<14} {r['direction']}  BLEU {r['bleu']:>6}  ->  the mapping must beat this with NO decoder fine-tuning")'''),
        ("md", """**Reading.** A T5-family base checkpoint has no instruction-tuned translation ability, so a low zero-shot
BLEU is expected and is *not* a failure of the methodology — it is the floor. The copy-task BLEU is the
ceiling for "the model can reproduce text at all"; the gap between the two is the work the mapping has to do."""),
    )
    write(ROOT / "test-token-generation" / "02_bleu_zero_shot.ipynb", cs)


def compare_notebook() -> None:
    cs = cells(
        ("md", """# All four models side by side

One table, three questions: does size buy copy fidelity, does it change dish-name perplexity, does it move
zero-shot BLEU. **Base is compared with base, large with large.**"""),
        ("code", PREAMBLE),
        ("code", '''import csv
def table(name):
    p = ROOT / "results" / name
    return {r["model"]: r for r in csv.DictReader(p.open(encoding="utf-8"))} if p.exists() else {}

copy_t, ppl_t, bleu_t = table("copy_summary.csv"), table("ppl_summary.csv"), table("bleu_summary.csv")
fmt = "{:<14}{:>9}{:>9}{:>10}{:>10}{:>9}"
print(fmt.format("model", "params", "exact", "copyBLEU", "ppl_dish", "d_nll"))
for k in C.MODEL_ORDER:
    c, p = copy_t.get(k), ppl_t.get(k)
    if not c:
        print(fmt.format(k, "-", "-", "-", "-", "-"))
        continue
    print(fmt.format(k, C.MODELS[k]["size_class"], c["exact_match"], c["bleu"],
                     p["ppl_dish"] if p else "-", p["delta_nll"] if p else "-"))'''),
        ("md", "## Zero-shot translation (one model per direction, matched size)"),
        ("code", '''print(f"{'model':<14}{'dir':<8}{'BLEU':>8}{'chrF':>8}{'dish_kept':>11}")
for k, r in bleu_t.items():
    print(f"{r['model']:<14}{r['direction']:<8}{r['bleu']:>8}{r['chrf']:>8}{r['dish_name_kept']:>11}")'''),
        ("md", """## What the table does and does not license

- ✅ "On this machine, at this size, the copy prompt works/does not work."
- ✅ "Dish tokens are X nats harder than the rest of the sentence, with a paired CI."
- ❌ "ViT5 is better than Flan-T5" — different languages, different tokenisers, different test text.
- ❌ "Large is better" — the large models see the same data but with a different token budget; a difference
  in `exact_match` can come from decoding length alone (see `mean_hyp_words` in the copy table)."""),
    )
    write(ROOT / "model-exploratory" / "00_compare_models.ipynb", cs)


def main() -> None:
    explorer("vit5-base", "01")
    explorer("flan-t5-base", "02")
    explorer("vit5-large", "03")
    explorer("flan-t5-large", "04")
    compare_notebook()
    instruction_following()
    ppl_notebook()
    bleu_notebook()


if __name__ == "__main__":
    main()
