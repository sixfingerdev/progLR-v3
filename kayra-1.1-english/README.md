---
language:
- en
license: mit
library_name: transformers
tags:
- kayra
- proglr
- v3-embed
- low-rank
- custom_code
- causal-lm
pipeline_tag: text-generation
base_model: sixfingerdev/kayra-1
datasets:
- wikimedia/wikipedia
- legacy-datasets/mc4
---

# Kayra-1.1 English

**ProgLR** — V3-Embed + progressive low-rank architecture — trained on WikiText-103 with a
250K-vocabulary word-level tokenizer. A proof that very large vocabularies (250K) can be
trained on consumer hardware (6GB VRAM) using an analytic (never-materialized) head.

## Model Details

| Field | Value |
|---|---|
| Architecture | ProgLR: V3-Embed + 1 Std + 1 LR layer |
| Hidden size | 768 |
| Heads | 12 |
| Max position | 1024 |
| Low-rank r / alpha | 16 / 64 (s = 4) |
| Vocabulary | 250,000 (229,464 real words) |
| Params | 12,105,472 |
| Head | analytic: `x @ R.T + (a/r) (x @ V) @ U.T` |
| Checkpoint | step 3500, val PPL 108.5 |
| License | MIT |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("./kayra-1.1-english", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./kayra-1.1-english",
                                             trust_remote_code=True)

ids = tok("the history of")["input_ids"]
out = model.generate(torch.tensor([ids]), max_new_tokens=32, do_sample=False)
print(tok.decode(out[0]))
```

Notes:
- Word-level tokenizer: input is lowercased and split on whitespace; unknown words map to `<unk>`.
- `torch_dtype` is float32; R is a frozen random buffer (does not count as params).
- Not instruction-tuned; raw language-model checkpoint.

## Technical Background

V3-Embed generates the embedding/head as `W = R + α/r · U@Vᵀ` with a frozen random `R`,
giving ~500× embedding compression and enabling vocabularies that are impossible for
standard models on the same GPU (a 250K×768 logits matrix is ≈8GB — it is never built).
Progressive low-rank keeps the bottom layers dense for representation quality and uses
factorized `U@Vᵀ` layers on top. See the Kayra series documentation for full results.

## Citation

```bibtex
@misc{kayra11-english,
  title  = {Kayra-1.1 English: ProgLR language model with a 250K vocabulary trained on 6GB VRAM},
  author = {Six Finger Dev},
  year   = {2026},
  license = {MIT}
}
```
