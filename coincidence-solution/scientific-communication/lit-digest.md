# Literature digest — cross-lingual representation bridging / frozen-model translation

Extracted with `pypdf` from the first 3 pages of each local PDF (title page, abstract, first intro
paragraphs). Every title, author string, venue and number below is copied from the extracted text.
Where the PDF text did not state a venue, that is said explicitly rather than guessed.
All 26 requested PDFs were located and extracted successfully (2 of them sit under
`draft/neural-mining/before-screening/` rather than `draft/translation/before-screening/`).
No entry failed to extract.

---

## A. Latent communication between models

### Fu2026
- **Full title** Cache-to-Cache: Direct Semantic Communication Between Large Language Models
- **Authors** Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, Yu Wang
- **Year / venue** ICLR 2026 ("Published as a conference paper at ICLR 2026")
- **Claim** Existing multi-LLM systems force models to communicate through generated text, which loses
  semantic information and adds token-by-token latency; C2C instead projects and fuses the source
  model's KV-cache with the target model's cache, using a learnable gate to pick which target layers
  receive the transferred cache.
- **Quantitative fact** 6.4–14.2% higher average accuracy than individual models; ~3.1–5.4% over the
  text-communication paradigm; average 2.5× latency speedup.
- **How we would cite it** {latent communication between models}

### Yang2025
- **Full title** Direct Semantic Communication Between Large Language Models via Vector Translation
- **Authors** Fu-Chun Yang, Jason Eshraghian (University of California, Santa Cruz)
- **Year / venue** arXiv:2511.03945v1, 6 Nov 2025 (no conference venue stated)
- **Claim** A dual-encoder translator learns bidirectional mappings between the representation spaces of
  two frozen 7B models, and injecting the translated vectors into the receiver steers its generation
  without serialising to tokens.
- **Quantitative fact** Average cosine alignment of 0.538 between Llama-2-7B and Mistral-7B-Instruct;
  injection at 30% blending strength; bidirectional transfer asymmetry of 2.01:1.
- **How we would cite it** {latent communication between models}

### Dery2026
- **Full title** Latent Space Communication via K-V Cache Alignment
- **Authors** Lucio M. Dery, Zohar Yahav, Henry Prior, Qixuan Feng, Jiajun Shen, Arthur Szlam (Google DeepMind)
- **Year / venue** arXiv:2601.06123v1, 4 Jan 2026; "© 2026 Google DeepMind" (no venue stated)
- **Claim** Aligning the k-v caches of several models into a shared space via per-model adapters creates a
  high-bandwidth inter-model channel without altering pre-trained parameters, and also lets learned
  skills such as soft prompts be transferred between models.
- **Quantitative fact** none in abstract (experiments use "Gemma-2 models", no figure given on pp. 1–3)
- **How we would cite it** {latent communication between models}

### Moschella2023
- **Full title** Relative Representations Enable Zero-Shot Latent Space Communication
- **Authors** Luca Moschella, Valentino Maiorca, Marco Fumero, Antonio Norelli, Francesco Locatello, Emanuele Rodolà
- **Year / venue** ICLR 2023 ("Published as a conference paper at ICLR 2023")
- **Claim** Under the same data and modelling choices, angles between encodings are preserved across
  independently trained latent spaces, so re-expressing a sample by its similarity to a fixed set of
  anchors yields a representation that is invariant to latent isometries and rescalings — enabling
  zero-shot model stitching without any additional training.
- **Quantitative fact** none in abstract
- **How we would cite it** {latent communication between models}

---

## B. Premise: independently trained models converge on alignable geometry

### Peng2024
- **Full title** Concept Space Alignment in Multilingual LLMs
- **Authors** Qiwei Peng and Anders Søgaard (University of Copenhagen)
- **Year / venue** EMNLP 2024, pages 5511–5526
- **Claim** Multilingual LLMs generalise across languages partly because of implicit vector-space
  alignment: larger models admit very high-quality *linear* alignments between corresponding concepts,
  but alignment is weaker for typologically distant languages and for abstract concepts, and
  prompt-based embeddings align better while being less linear than word embeddings.
- **Quantitative fact** none in abstract (evaluation is precision@k on WordNet concepts; Fig. 1 shows
  four parallel concepts aligned across 7 languages)
- **How we would cite it** {premise: models converge / alignment is linear}

### Zhou2026
- **Full title** Cross-Lingual Alignment Without Joint Training: Do Monolingual Language Models Converge on Universal Representations?
- **Authors** Ej Zhou, Suchir Salhan, Catherine Arnett, Anna Korhonen
- **Year / venue** EMNLP 2026, pages 21811–21837
- **Claim** Strictly monolingual models trained on non-parallel data with no shared parameters already
  develop alignable representational geometry, a single Procrustes rotation fitted on parallel
  sentences maps hidden states between them, and patching a rotated English residual into a German
  model flips its factual-cloze prediction to the English model's answer — so alignment need not come
  from joint training.
- **Quantitative fact** Monolinguality check: "at most 0.1% English content in the non-English corpora".
- **How we would cite it** {premise: models converge / alignment is linear}

### Anonymous2026
- **Full title** Do Monolingual Language Models Learn Cross-Lingual Universal Conceptual Representations?
- **Authors** Anonymous authors (paper under double-blind review)
- **Year / venue** ICLR 2026 submission ("Under review as a conference paper at ICLR 2026")
- **Claim** Using the Goldfish family of independently trained monolingual models, the authors show via CKA
  that alignment survives beyond architectural baselines and scales with data and linguistic proximity,
  then use cross-lingual activation patching — injection with *no learned projection* — to show the
  representations are functionally, not merely correlationally, compatible.
- **Quantitative fact** 350 languages; per-language training budgets of 5MB, 10MB, 100MB and 1GB.
- **How we would cite it** {premise: models converge / alignment is linear}

### Huh2024
- **Full title** The Platonic Representation Hypothesis
- **Authors** Minyoung Huh, Brian Cheung, Tongzhou Wang, Phillip Isola (MIT)
- **Year / venue** ICML 2024, PMLR 235
- **Claim** Representations in deep networks are converging, across architectures, objectives and even
  modalities, toward a shared statistical model of reality; scaling model size, data and task
  diversity is the proposed driver.
- **Quantitative fact** none in abstract
- **How we would cite it** {premise: models converge / alignment is linear}

### Jha2025
- **Full title** Harnessing the Universal Geometry of Embeddings
- **Authors** Rishi Jha, Collin Zhang, Vitaly Shmatikov, John X. Morris (Cornell University)
- **Year / venue** NeurIPS 2025 (arXiv:2505.12540v4)
- **Claim** vec2vec translates text embeddings between the spaces of different encoders with no paired
  data, no encoders and no predefined matches, by routing through a learned universal latent space —
  a constructive, stronger form of the Platonic Representation Hypothesis for text.
- **Quantitative fact** Translations reach cosine similarity "as high as 0.96" to ground-truth target
  vectors, with perfect matching on over 8000 shuffled embeddings.
- **How we would cite it** {premise: models converge / alignment is linear}

### Park2024
- **Full title** The Linear Representation Hypothesis and the Geometry of Large Language Models
- **Authors** Kiho Park, Yo Joong Choe, Victor Veitch (University of Chicago)
- **Year / venue** ICML 2024, PMLR 235
- **Claim** Gives two counterfactual-based formalisations of "linear representation" (output/word space
  and input/context space), proves they connect to linear probing and model steering respectively, and
  identifies a causal inner product under which all notions of linear representation unify, enabling
  probes and steering vectors to be built from counterfactual pairs.
- **Quantitative fact** none in abstract
- **How we would cite it** {background: linear representation / stitching}

### Hernandez2023
- **Full title** Model Stitching: Looking For Functional Similarity Between Representations
- **Authors** Adriano Hernandez, Rumen Dangovski, Peter Y. Lu, Marin Soljačić (MIT Physics)
- **Year / venue** arXiv:2303.11277v2, 31 Aug 2023 (no conference venue stated)
- **Claim** Extends model stitching to layers of different shapes and to different architectures, and
  finds that stitches can reach high accuracy even when the stitched layers are far apart and ordered
  "backwards" — evidence that a stitch may find *some* workable representation rather than recovering
  the one the receiver expects, so naive stitching is not a reliable similarity measure.
- **Quantitative fact** Setup: all 16 small ResNets (10–18 layers) trained on CIFAR-10 reach above 90% accuracy.
- **How we would cite it** {background: linear representation / stitching}

---

## C. Metric critique: alignment ≠ transfer

### Veitsman2026
- **Full title** Why Better Cross-Lingual Alignment Fails for Better Cross-Lingual Transfer: Case of Encoders
- **Authors** Yana Veitsman, Yihong Liu, Hinrich Schütze
- **Year / venue** MRL 2026 workshop (6th Workshop on Multilingual Representation Learning), pages 119–134
- **Claim** XLM-R models explicitly aligned with token-level, sentence-level and MLM objectives can improve
  representational alignment without improving — or while degrading — zero-shot downstream performance;
  embedding-based alignment metrics are unreliable predictors, and alignment/downstream gradients are
  often nearly orthogonal.
- **Quantitative fact** Four language pairs aligned (no downstream score is quoted in the abstract).
- **How we would cite it** {metric critique: alignment != transfer}

### Pallucchini2025
- **Full title** Lost in Alignment: A Survey on Cross-Lingual Alignment Methods for Contextualized Representation
- **Authors** Filippo Pallucchini, Lorenzo Malandri, Fabio Mercorio, Mario Mezzanzanica (University of Milan-Bicocca)
- **Year / venue** ACM Computing Surveys 58(5), Article 116, November 2025
- **Claim** Surveys and taxonomises cross-lingual contextual embedding methods by data requirements and
  objective functions, and critically reviews the evaluation methodologies used to claim alignment,
  flagging open challenges.
- **Quantitative fact** none in abstract
- **How we would cite it** {metric critique: alignment != transfer}

### Ravisankar2026
- **Full title** Can you map it to English? The Role of Cross-Lingual Alignment in the Multilingual Performance of LLMs
- **Authors** Kartik Ravisankar, HyoJung Han, Sarah Wiegreffe, Marine Carpuat (University of Maryland)
- **Year / venue** EACL 2026, pages 4854–4872
- **Claim** Introduces the Discriminative Alignment Index to measure *instance-level* alignment to English
  and shows incorrect NLU predictions are strongly associated with lower alignment in middle layers;
  activation patching of parallel English activations in those layers repairs errors, giving causal
  evidence that alignment (not just correlation) drives cross-lingual correctness.
- **Quantitative fact** 24 languages other than English, 3 NLU tasks.
- **How we would cite it** {metric critique: alignment != transfer} (counterpoint: instance-level
  alignment does predict correctness)

### Rajaee2022
- **Full title** An Isotropy Analysis in the Multilingual BERT Embedding Space
- **Authors** Sara Rajaee, Mohammad Taher Pilehvar
- **Year / venue** arXiv:2110.04504v2, 16 Mar 2022 (venue not stated in the extracted text; file dated 2021)
- **Claim** mBERT, unlike monolingual BERT, shows no single outlier dimension but a highly anisotropic
  space; increasing isotropy measurably improves representation quality, and different languages'
  subspaces are only *partially* structurally similar despite sharing degenerate directions.
- **Quantitative fact** Six languages analysed: English, Spanish, Arabic, Turkish, Sundanese, Swahili.
- **How we would cite it** {metric critique: alignment != transfer}

### Zhao2020
- **Full title** Non-Linearity in mapping based Cross-Lingual Word Embeddings
- **Authors** Jiawei Zhao, Andrew Gilman (Massey University)
- **Year / venue** LREC 2020, pages 3583–3589
- **Claim** The linear-mapping assumption behind most cross-lingual embedding work (similar meanings keep
  similar geometric arrangements) is violated for some language pairs and concepts; a kernel CCA
  non-linear mapping beats linear state-of-the-art in both supervised and self-learning settings.
- **Quantitative fact** Five language pairs evaluated.
- **How we would cite it** {premise: models converge / alignment is linear} (as the counter-evidence
  that the bridge need not be linear)

---

## D. Intervention: train/adapt the model

### Liu2025
- **Full title** Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs
- **Authors** Danni Liu, Jan Niehues (Karlsruhe Institute of Technology)
- **Year / venue** ACL 2025, pages 15979–15996
- **Claim** Analysing internal representations across 1,000+ language pairs shows middle layers have the
  strongest cross-lingual alignment potential; adding a middle-layer alignment objective to
  task-specific training improves transfer on slot filling, MT and structured generation, and separately
  trained alignment modules can be merged into existing task modules without full retraining.
- **Quantitative fact** Over 1,000+ language pairs analysed.
- **How we would cite it** {intervention: train the model}

### Nakai2025
- **Full title** TRepLiNa: Layer-wise CKA+REPINA Alignment Improves Low-Resource Machine Translation in Aya-23 8B
- **Authors** Toshiki Nakai, Ravi Kiran Chikkala, Lena Sophie Oberkircher, Nicholas Jennings, Natalia Skachkova, Tatiana Anikina, Jesujoba O. Alabi
- **Year / venue** MMLoSo 2025 workshop, pages 25–34
- **Claim** Enforcing cross-lingual similarity at a chosen layer of a decoder-only multilingual LLM, by
  combining CKA with REPINA regularisation against representation drift, is a low-cost way to improve
  LRL→HRL translation in data-scarce settings; gains concentrate in mid layers.
- **Quantitative fact** Mid-layer alignment "roughly layers 10–15" is most effective, with layer 15
  preferred in limited-data settings; score is 0.6×BLEU + 0.4×ChrF.
- **How we would cite it** {intervention: train the model}

### Borchert2025
- **Full title** Language Fusion for Parameter-Efficient Cross-lingual Transfer
- **Authors** Philipp Borchert, Ivan Vulić, Marie-Francine Moens, Jochen De Weerdt
- **Year / venue** ACL 2025, pages 25848–25868
- **Claim** FLARE integrates source- and target-language representations inside LoRA adapters with
  lightweight linear transformations, improving non-English representation quality and downstream
  cross-lingual NLU while staying parameter-efficient.
- **Quantitative fact** +4.9% (Llama 3.1) and +2.2% (Gemma 2) exact match on question answering versus
  standard LoRA fine-tuning.
- **How we would cite it** {intervention: train the model}

### Chen2026
- **Full title** LatentMT: Machine Translation with Latent Reasoning
- **Authors** Wei-Rui Chen, Samar M. Magdy, Chiyu Zhang, Wenhui Zhu, Zhipeng Wang, Muhammad Abdul-Mageed
- **Year / venue** arXiv:2607.18618v1, 21 Jul 2026 (no venue stated)
- **Claim** Adapts a small looped LM with per-language-pair LoRA adapters so that extra recurrent
  computation happens in hidden states rather than in emitted reasoning tokens; quality gains saturate
  after a few reasoning steps, and hidden-representation differences shrink along the reasoning-step axis.
- **Quantitative fact** 2.6B-parameter backbone; 32 translation directions; comparable to models 3–5× larger.
- **How we would cite it** {intervention: train the model}

### Vardasbi2023
- **Full title** State Spaces Aren't Enough: Machine Translation Needs Attention
- **Authors** Ali Vardasbi, Telmo Pessoa Pires, Robin M. Schmidt, Stephan Peitz
- **Year / venue** arXiv:2304.12776v1, 25 Apr 2023 (venue not stated in the extracted text)
- **Claim** S4-based MT encoders lag the Transformer by roughly 4 BLEU and, counter-intuitively, degrade on
  long sentences, because compressing the source into a single hidden state loses token-level access;
  adding attention closes the gap.
- **Quantitative fact** Approximately 4 BLEU points behind the Transformer on WMT'14/WMT'16.
- **How we would cite it** {background: linear representation / stitching} (architecture caveat: one
  compressed state is not enough for MT)

---

## E. Intervention: frozen pair + bridge

### Bu2026
- **Full title** Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality
- **Authors** Mengyu Bu, Yang Feng
- **Year / venue** ACL 2026, pages 20863–20885
- **Claim** XBridge is a compositional encoder–LLM–decoder architecture that offloads multilingual
  understanding and generation to external pretrained translation models while keeping an English-centric
  LLM core, using lightweight cross-model mapping layers plus an optimal-transport alignment objective to
  bridge the heterogeneous representation spaces; it improves low-resource and unseen-language
  performance without retraining the LLM.
- **Quantitative fact** Evaluated on four LLMs.
- **How we would cite it** {intervention: frozen pair + bridge}

---

## F. Data or domain

### Ngo2019
- **Full title** MTet: Multi-domain Translation for English and Vietnamese
- **Authors** Chinh Ngo, Trieu H. Trinh, Long Phan, Hieu Tran, Tai Dang, Hieu Nguyen, Minh Nguyen, Minh-Thang Luong (VietAI Research)
- **Year / venue** arXiv:2210.05610v2, 19 Oct 2022 (no conference venue stated in the extracted text)
- **Claim** Releases the largest public English–Vietnamese parallel corpus plus the first EnViT5
  encoder-decoder pretrained model for the pair, with a multi-domain test set curated by the Vietnamese
  research community, and shows multi-domain training data helps at fixed data budget.
- **Quantitative fact** 4.2M high-quality training pairs; 6.2M total when combined with prior work;
  up to +2 BLEU while being 1.6× smaller.
- **How we would cite it** {data or domain}

### Conia2024
- **Full title** Towards Cross-Cultural Machine Translation with Retrieval-Augmented Generation from Multilingual Knowledge Graphs
- **Authors** Simone Conia, Daniel Lee, Min Li, Umar Farooq Minhas, Saloni Potdar, Yunyao Li
- **Year / venue** EMNLP 2024, pages 16343–16360
- **Claim** Introduces XC-Translate (a manually created benchmark of culturally nuanced entity names) and
  KG-MT, which injects multilingual-knowledge-graph evidence into an NMT model through dense retrieval;
  current MT systems and LLMs still mishandle such entities.
- **Quantitative fact** 129% relative improvement over NLLB-200 and 62% over GPT-4.
- **How we would cite it** {data or domain}

### Xu2026
- **Full title** Cultural and Knowledge Biases in LLMs Through the Lens of Entity-Aware Machine Translation
- **Authors** Lu Xu, Luca Moroni, Roberto Navigli
- **Year / venue** LREC 2026, pages 8794–8812
- **Claim** Proposes a three-level cultural-specificity taxonomy (Culturally Agnostic, Culturally Sensitive,
  Culturally Local Knowledge) and shows that external knowledge helps entity translation far more for
  culturally local entities than for agnostic ones, with different failure modes across model tiers.
- **Quantitative fact** 11 LLMs, 10 languages; up to 70% improvement in m-ETA for culturally local entities.
- **How we would cite it** {data or domain}

---

## G. Background outside NLP

### Gallistel2011
- **Full title** Prelinguistic Thought
- **Authors** C. R. Gallistel (Rutgers University)
- **Year / venue** Language Learning and Development, 7(4), 253–262, 2011
- **Claim** Evidence from animal cognition shows that abstract, recursively structured symbol processing
  and predicate–argument representations are evolutionarily ancient and independent of language; what is
  uniquely human is translating these private representations into a communicable symbol system of
  comparable power.
- **Quantitative fact** none in abstract
- **How we would cite it** {premise: models converge / alignment is linear} (cognitive premise: thought
  has a language-independent format that translation externalises)

---

## Gap statement

1. The convergence literature (Huh2024, Peng2024, Zhou2026, Anonymous2026, Jha2025, Gallistel2011)
   establishes that *representations* of different models/languages are alignable, and in the best case
   a single Procrustes rotation or a learned latent space suffices (Zhou2026, Jha2025, Moschella2023).
2. But the intervention results that actually change what a model does are all "train the model":
   middle-layer alignment losses (Liu2025), CKA+REPINA at layer 15 (Nakai2025), FLARE inside LoRA
   (Borchert2025), looped latent reasoning (Chen2026).
3. Where models are kept frozen, the object being bridged is never the decoder's *next-token
   distribution*: C2C and KVCacheAlign fuse or align KV-caches for assistant-style collaboration
   (Fu2026, Dery2026), Yang2025 injects translated vectors at 30% blending, and Jha2025 translates
   embeddings only.
4. Only XBridge (Bu2026) composes a frozen encoder–frozen LLM–frozen decoder, and it exists to give an
   English-centric LLM multilingual *understanding*, not to produce a translation with a frozen decoder.
5. No paper here fits the bridge by asking the frozen decoder to generate and scoring the generated
   translation; every bridge objective found is a representation-error objective — Procrustes/MSE
   (Zhou2026, Peng2024), CKA+REPINA (Nakai2025), cosine/contrastive vector translation (Yang2025),
   optimal transport on hidden states (Bu2026), or adversarial latent matching (Jha2025).
6. That choice is exactly what Veitsman2026 warns against: better embedding alignment can fail to improve,
   or degrade, downstream behaviour, because alignment and task gradients are nearly orthogonal.
7. Ravisankar2026 sharpens the tension from the other side: instance-level middle-layer alignment *does*
   track correctness, so alignment is neither irrelevant nor sufficient — it is the wrong *training
   signal* even when it is a good *diagnostic*.
8. The architecture evidence (Vardasbi2023) says a single compressed source state loses token-level
   information that MT needs — so a bridge that only maps one pooled vector cannot be expected to drive
   a decoder.
9. So the unoccupied cell is: two frozen, independently trained monolingual encoder–decoder models, one
   bridge module between encoder output and decoder input, decoder weights untouched, and the bridge
   fitted on the *behaviour* of the frozen decoder (likelihood/BLEU of generated translations) rather
   than on hidden-state reconstruction error.
10. That requires abandoning the linear/isometric assumption that dominates the alignment literature
    (Zhao2020 already shows it fails for some pairs), while keeping the premise that the two spaces are
    related enough to be bridged (Zhou2026, Anonymous2026).
11. It also requires the domain/culture resources (Ngo2019, Conia2024, Xu2026) to be handled by the
    frozen decoder, which is where the empirical value of a behaviour-fitted bridge would show up.
12. In short: convergence is established, frozen-pair latent communication is established, and
    behaviour-fitted bridging of a *frozen decoder for translation* is established by none of these papers.
