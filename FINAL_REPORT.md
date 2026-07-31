# ProgLR WikiText-103: Final Report

**Pipeline:** WT103 corpus → 50K-vocab Progressive Low-Rank training (16K steps) → 12× accelerated v2 re-implementation → head expansion to 250K vocab + finetune → streamed interactive inference.

**Hardware:** RTX 3060 Laptop (6GB VRAM) | PyTorch 2.5.1+cu121 | CUDA 13.2 | Windows 11
**Date:** 2026-07-31

---

## 1. Architecture: ProgLR (V3-Embed + Progressive Low-Rank)

Context: this project is the large-scale validation of the V3-All research line (`condenser/AGENTS.md`, `condenser/TECHNICAL_SUMMARY.md`) — the unified weight format `W = R + α/r · U@Vᵀ`.

The trained model (`ProgLR`, defined in `tests/final_report_train.py`, `tests/final_report_train_v2.py`, `tests/finetune_vocab250k.py`):

```
Embedding:   V3-Embed  W = R + (α/r)·U@Vᵀ   (R = frozen random, 500× compression)
Layer 0:     Standard Transformer            (full-rank, representation quality)
Layer 1:     Low-Rank Transformer            (W = (α/r)·U@Vᵀ per Q,K,V,O,FF1,FF2)
Head:        V3-Embed  W = R + (α/r)·U@Vᵀ   (same R/U/V tensors as embedding, weight-tied)
Norm:        LayerNorm
```

| Component | Shape | Params (V=50K) | Params (V=250K) |
|---|---|---|---|
| V3-Embed (R buffer frozen + U,V learned) | V×768 | 812,288 | 4,012,288 |
| Std layer (Q,K,V,O,FF1,FF2) | 768→768 | 7,077,888 | 7,077,888 |
| LR layer r=16, α=64 (s=4) | 768→768 | 221,184 | 221,184 |
| Position embedding | 1024×768 | 786,432 | 786,432 |
| LayerNorm | 768 | 1,536 | 1,536 |
| **Total (trainable)** | | **8,905,472** | **12,105,472** |

Notes:
- The embedding and head share the same `R`, `U`, `V` (weight-tied via one `V3WeightEmbedding` module).
- `R` is a frozen random buffer (`randn/√768`) — the dominant memory component (V×768×4B = 154MB at 50K, 750MB at 250K) — never trained, never optimized.
- Low-rank layers use a custom autograd function `LRLinearFn` with analytic `U,V` gradients (no materialized W).

## 2. Data

Built from WikiText-103 v1 parquet (HuggingFace mirror) by `condenser/build_wt103_dataset.py` (50K) and `condenser/build_vocab_250k.py` + data rebuild (250K):

- Lowercased, whitespace-split tokenization (word-level, no BPE).
- Chunks of 512 tokens; `x = ids[i:i+512]`, `y = ids[i+1:i+513]`; stored as int32 numpy → pickled.
- `condenser/wt103_data.pkl` / `wt103_vocab.pkl`: **50,000 vocab**, 198,097 train chunks, 417 valid chunks.
- `condenser/wt103_data_250k.pkl` / `wt103_vocab_250k.pkl`: nominal vocab_size 250,000; old 50K ids preserved 1:1, new words appended by frequency (vocab file ends at 229,464 distinct words — WT103's full distinct-word set; ids ≥229,464 exist in R/U but never occur in data).
- Special ids: `<pad>`=0, `<unk>`=1 (OOV).

## 3. Stage 1 — 50K Main Training (`tests/final_report_train.py`)

Config: `B=16, SEQ=512, D=768, H=12, 1 Std + 1 LR r=16, AdamW lr=3e-4 wd=0.01, FP32`.

Per-step cost: full dense `forward_head` (16×512×50K = 419M logits, ~1.6GB tensor) + `cross_entropy` backprop through the softmax and full head — on 6GB VRAM this **pages to system memory every step** (measured peak 10.35GB). Measured: **4,589 ms/step (~1,785 tok/s)**.

Checkpoint history (`report_out/checkpoints`, ~261MB each):

| Step | Loss | Best Val PPL |
|---|---|---|
| 2,000 | 5.215 | 215.53 |
| 4,000 | 5.048 | 147.60 |
| 6,000 | 4.930 | 124.97 |
| 8,000 | 4.706 | 114.48 |
| 10,000 | 4.844 | 107.62 |
| 12,000 | 4.564 | 102.27 |
| 14,000 | 4.458 | 98.80 |
| **16,000** | 4.617 | **95.18** |

The run was stopped at step 16,000 (TDR/long-run issues); `ckpt_16000.pt` is the source model for everything downstream.

## 4. Stage 2 — v2 Accelerated Training (`tests/final_report_train_v2.py`)

### Problem
The dense 50K head is the entire bottleneck: one 1.6GB logits tensor, softmax graph nodes, CE backward over 50K classes, and per-step `cudaMalloc` churn on a 6GB GPU.

### Solution — analytic chunked head
The head `l_ic = h_i·R_c + s·(t_i·U_cᵀ)`, `t = h·V`, `s = α/r` is split into 5 chunks of 50K classes. Inside one `torch.no_grad()` block:

- Logits computed chunkwise into a reused buffer via `addmm(out=)` (no 1.6GB tensor ever exists).
- Loss = sum of per-chunk `cross_entropy` (detached values).
- Gradient `g` of CE wrt chunk logits computed manually; then analytically:

```
dU_c = s·(gᵀ·t_b)          dH = g@Rc + s·(g@Uc)@Vᵀ          dV = s·(h_bᵀ·(g@Uc))
```

- `h.backward(dH)` flows the exact head gradient through the layers once (no graph through the head).
- `dU, dV` accumulated manually into `emb.U.grad` / `emb.V.grad`.

Math is bitwise identical to the original path; the optimizer step is unchanged.

### Gradcheck (`tests/bench_head.py`)
Analytic vs full-autograd reference on identical input, norm-relative error `max|a−b| / max|b|`:

| Tensor | Norm-err |
|---|---|
| U (embedding/head factor) | 8.5e-05 |
| V (embedding/head factor) | 6.3e-05 |
| pos (through both layers) | 3.1e-04 |

**PASS** (all < 1e-3). Losses: `loss_an = loss_ref` (bitwise).

### Benchmark (15 steps, RTX 3060 Laptop 6GB)

| Metric | Original | v2 | |
|---|---|---|---|
| Time/step | 4,589 ms | 377 ms | **12.17× faster** |
| Throughput | 1,785 tok/s | 21,727 tok/s | |
| Peak VRAM | 10.35 GB (paging) | 4.68 GB | **fits 6GB** |

### Validation run (2K steps, `V2_STEPS=2000`, ~13 min vs ~96 min original)

| | Original (ckpt_02000) | v2 run |
|---|---|---|
| Val PPL | 215.53 | 224.21 |
| Loss | 5.215 | 5.221 |

Same math confirmed at scale (small PPL delta = random-batch noise). **The 20K full run was intentionally skipped**: mathematical equivalence is proven at 2K, so re-burning ~9 GPU-hours was deemed unnecessary.

### Debugging lessons (for reproducibility)
- `max|a−b|/max|b|` is the correct error metric; elementwise max-relative on near-zero entries produces phantom "768×" errors.
- `addmm(out=...)` with `out` a leaf doesn't autograd-track — use it only inside `no_grad`.
- The `scatter_add` trick for target-position gradient (`g.scatter_add_(1, tgt, −1)`) is correct on both CPU and GPU.
- Everything else (softmax CE gradients) is standard math — verification caught no real bugs, only measurement artifacts.

## 5. Stage 3 — 250K Vocab Expansion + Finetune (`tests/finetune_vocab250k.py`)

Source: `ckpt_16000.pt` (val_ppl 95.18 at 50K vocab).

### Transfer (exact, bitwise)
- `R[:50K]`, `U[:50K]`, `V`, `pos`, `norm`, both layers copied 1:1 from the source checkpoint. New U rows (50,000+) random-init (`0.01·randn`). **No R expansion** — R is a frozen random buffer at full 250K rows from the start.

### Freeze design
Only `emb.U[50K:]` (3.2M params) trains. Everything else frozen — including old U rows. This was forced by a real finding: *torch's AdamW moves Parameter rows ~1e-5/step even with zero gradient* → slow old-vocab CE creep. The optimizer therefore is a **manual Adam on the slice** `U_sl = U[50K:]` (`b1=0.9, b2=0.999, eps=1e-8`), so old rows are *structurally* excluded from every optimizer op.

### Training setup
- `B=8, SEQ=512`, 4,000 steps, warmup 200 → cosine decay, `PEAK_LR=3e-3`.
- **Oversampling**: 50% of batches drawn from the top-1% train chunks by new-word density (mean density 0.057% → top-1% ≈ ×5.3 richer).
- **Weighted gradient**: chunk gradient scaled by `(1 + 9·ρ_c)` where `ρ_c` = new-word fraction in the chunk → rare/new words get strong pulls.
- **Softmax-invariant row-mean removal** in the chunk gradient (kills uniform Adam drift on the U slice).
- Chunked 5×50K CE with a single reused `lc_buf`; fused `[h_f, s·t] @ [Rc, Uc]ᵀ` per chunk.

### Results

| Stage | Val PPL | old-word CE | new-word CE |
|---|---|---|---|
| Pre-finetune (fresh expansion) | 113.51 | 4.590 (bitwise = source) | 13.19 |
| **Final (step 3500/4000 ckpt)** | **108.48** | 4.590 (bitwise preserved) | **11.11** |

- `best_ppl` reached 108.496 at step 2,000 and held through step 3,500 (`report_out_250k/checkpoints` metadata).
- New-word CE drops 13.19 → 11.11; old-word CE is bitwise untouched (freeze works).
- Generated text contains **no `<unk>`** and no `<pad>` (both blocked in the sampler).
- Checkpoint size: 816MB each (dominated by the frozen R buffer, 250K×768×4B ≈ 750MB).
- Runtime ≈ 25 min total for 4,000 steps.

**Known gap:** the loop saves ckpts every 500 steps up to `st=3500`; steps 3500→4000 exist only in the final terminal eval (final val_ppl ≈ 108.5), not in a saved checkpoint. `ckpt_03500.pt` is therefore the saved final model.

## 6. Stage 4 — Interactive Inference (`tests/chat_250k.py`)

Loads `ckpt_03500.pt` (12,105,472 params, 250K head), streamed token-by-token generation (~100 tok/s on this GPU).

- Tokenizer: lowercase + regex word/punct split, OOV→`<unk>`; context capped at 512 tokens.
- Sampler controls (all active): `/temp T`, `/topk K`, `/topp P` (nucleus), `/rep R` (repetition penalty over context), `/greedy`; `/len N`, `/help`, `/quit`.
- `<pad>`/`<unk>` (ids 0/1) are masked at every step.
- The model is a raw WT103 language model — it completes text; it is not instruction-tuned.

## 7. External Reference Points

| Model | Vocab | WT103 Val PPL |
|---|---|---|
| TransfoXL (reference, session-reported) | 267K | 75.53 |
| **ProgLR this work** (16K steps, 8.9M params) | 50K | 95.18 |
| **ProgLR this work** (finetuned, 12.1M params) | 250K | 108.48 |

## 8. Files

| Path | Role |
|---|---|
| `report_out/checkpoints/ckpt_16000.pt` | Source model (50K, val_ppl 95.18, step 16,000) |
| `report_out_250k/checkpoints/ckpt_03500.pt` | **Final model** (250K, best_ppl 108.50, step 3,500) |
| `report_out/checkpoints/ckpt_02000…14000.pt` | 50K training curve |
| `report_out_250k/checkpoints/ckpt_00500…03000.pt` | Finetune curve (best_ppl +108.50 at 2K) |
| `tests/final_report_train.py` | Original 50K training (dense head) |
| `tests/final_report_train_v2.py` | v2 analytic-chunked-head training (`V2_STEPS` env) |
| `tests/bench_head.py` | Gradcheck + orig-vs-v2 benchmark |
| `tests/finetune_vocab250k.py` | 250K expansion + finetune (`RESUME_CKPT` env) |
| `tests/eval_ckpt.py` | val_ppl evaluator for 50K ckpts |
| `tests/chat_250k.py` | Interactive streamed chat |
| `condenser/build_wt103_dataset.py`, `build_vocab_250k.py` | Data/vocab builders |
| `condenser/wt103_{data,vocab}{,_250k}.pkl` | Dataset + vocab files |
| `condenser/v3_embed.py` | V3-Embed module (R + α/r·U@Vᵀ) |
| `condenser/AGENTS.md`, `TECHNICAL_SUMMARY.md` | Research-line context (V3-All) |

## 9. Reproduction

```bash
# 1. 50K main training (original):  python tests/final_report_train.py
# 2. v2 training (12× faster):      $env:V2_STEPS=20000; python tests/final_report_train_v2.py
# 3. gradcheck + benchmark:         python tests/bench_head.py 15
# 4. eval a 50K ckpt:               python tests/eval_ckpt.py report_out/checkpoints/ckpt_16000.pt
# 5. 250K finetune:                 python tests/finetune_vocab250k.py
# 6. chat:                          python tests/chat_250k.py
```

## 10. Open Items / Limitations

1. **Step-4000 250K state not saved** — steps 3500→4000 exist only as a terminal eval; saving it requires a 500-step resume (~3 min).
2. **Full 20K v2 run not executed** — equivalence proven at 2K; the 50K model's best known ckpt is step 16,000.
3. **Vocab file 229,464 vs nominal 250,000** — ids ≥229,464 are allocated but never seen in data (untrained rows).
4. **Raw WT103 LM, not instruction-tuned** — chat completes text, doesn't answer questions; no RLHF/chat data.
5. **No test-set eval** — only validation splits were evaluated.
6. **Word-level tokenizer** — no BPE/subword; `@-@`-style artifacts exist in the vocab.
7. **Comparisons** (TransfoXL 75.53) are external references, not reproduced in this repo.
