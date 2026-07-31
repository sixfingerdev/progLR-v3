# ProgLR / V3-All — Training LLM-scale models on 6GB consumer GPUs

A research project on a **unified weight format** (`W = R + α/r · U@Vᵀ`) that compresses Transformer weights 4× while preserving out-of-distribution (OOD) reasoning. Includes the full pipeline that trains a **250K-vocab WikiText-103 language model (12.1M params) on an RTX 3060 Laptop (6GB VRAM)** — from dataset build to interactive chat.

**Hardware:** RTX 3060 Laptop GPU (6GB VRAM) | PyTorch 2.5.1+cu121 | CUDA 13.2 | Windows 11

---

## The Core Idea

Every weight matrix uses the **V3 format**:

```
W_eff = R + (α/r) · U @ Vᵀ
```

| Component | Shape | Trained? | Purpose |
|-----------|-------|----------|---------|
| R (base) | out × in | No (fixed) | Free full-rank structure |
| U, V | low-rank factors | Yes | Learned corrections |
| α, r | scalars | Hyperparams | LoRA scaling / rank |

Three modes cover every part of the network:

- **`random`** — R = frozen random: used for **embedding & head** (input is one-hot, so token identity survives; **500× compression**).
- **`standard`** — R = learned dense: bottom layers, full-rank quality.
- **`zero`** — R = 0, pure `U@Vᵀ`: top layers, **10× hidden compression**.

Key insight: random R works for embeddings (one-hot input) but **not** hidden layers (continuous input gets destroyed by random projection).

## Architecture: ProgLR (V3-Embed + Progressive Low-Rank)

```
Embedding:   V3-Embed  W = R + (α/r)·U@Vᵀ   (R frozen random, 500× compression)
Layer 0:     Standard Transformer            (full-rank, representation quality)
Layer 1:     Low-Rank Transformer            (W = (α/r)·U@Vᵀ, r=16)
Head:        V3-Embed, weight-tied with embedding (same R/U/V)
Norm:        LayerNorm
```

| Component | Params (V=50K) | Params (V=250K) |
|---|---|---|
| V3-Embed | 812,288 | 4,012,288 |
| Std layer | 7,077,888 | 7,077,888 |
| LR layer r=16 | 221,184 | 221,184 |
| **Total (trainable)** | **8,905,472** | **12,105,472** |

## Key Results

### Arithmetic OOD (a+b mod 100, train a<50 → test a≥50)

| Model | Params | OOD Acc |
|---|---|---|
| Full Transformer | 867K | 99.6% |
| **V3-All 2Std+2LR r=8** | **500K** | **92.2%** |
| V3-All 1Std+3LR r=16 | 377K | 58.6% |

### WikiText-103 (this project's main run)

| Stage | Vocab | Params | Val PPL |
|---|---|---|---|
| ProgLR 50K (16K steps) | 50K | 8.9M | **95.18** |
| ProgLR + 250K finetune | 250K | 12.1M | **108.48** |
| (reference) TransfoXL | 267K | — | 75.53 |

### VRAM scaling (V=500K, 6GB budget)

| Config | Params | VRAM | Fits? |
|---|---|---|---|
| D=4096, L=24, 2Std+22LR r=8 | 422M | 4.2GB | ✅ |
| Standard D=4096, L=24 | 1.6B | 16.3GB | ❌ |

### v2 training acceleration

Analytic chunked head (no 1.6GB logits tensor ever materialized): **12.17× faster** (4,589 → 377 ms/step) and fits 6GB VRAM (4.68GB peak vs 10.35GB paging). Math verified bitwise-identical via gradcheck (`tests/bench_head.py`).

## Repository Structure

```
law/
├── FINAL_REPORT.md / .pdf   # Full report: WT103 pipeline, v2 acceleration, 250K finetune, chat
├── benchmark.py             # Condenser vs BaselineTransformer benchmark
├── train_compare.py         # Training comparison scripts
├── make_report_pdf.py       # Generates FINAL_REPORT.pdf (chart-rich)
├── config.py                # Python interpreter path
├── condenser/               # Research line: V3 / V3-All implementations
│   ├── AGENTS.md            # Research log: V3-Embed → Progressive Low-Rank → V3-All
│   ├── TECHNICAL_SUMMARY.md # Technical report (paper-ready)
│   ├── v3_all.py            # V3-All: V3Embedding + StdLayer + LRLayer
│   ├── v3_embed.py          # V3-Embed module (R + α/r·U@Vᵀ)
│   ├── baseline_transformer.py
│   ├── scale_projection.py  # VRAM calculator
│   ├── build_wt103_dataset.py / build_vocab_250k.py  # data/vocab builders
│   └── ...                  # experiments (genomatrix, prism, scm, hybrid, etc.)
├── tests/
│   ├── final_report_train.py      # 50K main training (dense head)
│   ├── final_report_train_v2.py   # v2 analytic-chunked-head training (12× faster)
│   ├── finetune_vocab250k.py      # 250K vocab expansion + finetune
│   ├── bench_head.py              # gradcheck + orig-vs-v2 benchmark
│   ├── eval_ckpt.py               # val_ppl evaluator
│   ├── chat_250k.py               # interactive streamed chat
│   ├── bench_v3_all.py            # V3-All arithmetic benchmark + VRAM scale
│   └── report_figs/               # report figures
├── experiments/               # PRISM-vs-Transformer analyses
└── report_out/ report_out_250k/ report_out_v2/  # checkpoint dirs + reports
```

## Reproduction

```bash
# 1. 50K main training (original):   python tests/final_report_train.py
# 2. v2 training (12× faster):       $env:V2_STEPS=20000; python tests/final_report_train_v2.py
# 3. gradcheck + benchmark:          python tests/bench_head.py 15
# 4. eval a 50K ckpt:                python tests/eval_ckpt.py report_out/checkpoints/ckpt_16000.pt
# 5. 250K finetune:                  python tests/finetune_vocab250k.py
# 6. chat:                           python tests/chat_250k.py
```

**Note:** checkpoints (`*.pt`), datasets (`*.pkl`) and raw WT103 parquet files are gitignored (exceed GitHub's 100MB file limit). `tests/chat_250k.py` requires the trained checkpoint from the 250K finetune stage.

## Documentation

- [FINAL_REPORT.md](FINAL_REPORT.md) — end-to-end report of the WT103 pipeline (data → 50K train → v2 → 250K finetune → chat)
- [condenser/TECHNICAL_SUMMARY.md](condenser/TECHNICAL_SUMMARY.md) — V3-All architecture report (paper-ready)
- [condenser/AGENTS.md](condenser/AGENTS.md) — research log: discovery, failure catalog, results

## Findings & Limitations

- **V3-Embed works only where input is one-hot** (embedding/head); random R destroys continuous hidden states.
- **2Std+2LR is the sweet spot** — more low-rank layers degrade OOD accuracy (noise accumulation).
- Freeze design for finetune: only new-vocab U rows train; old-word CE is preserved bitwise.
- Raw WT103 language model — completes text, not instruction-tuned; word-level tokenizer (no BPE).
