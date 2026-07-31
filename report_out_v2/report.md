# V3-Embed + Progressive Low-Rank: Technical Report (v2 optimized)

## Model
- **D=768, H=12, V=50,000**: 1 Std TF + 1 LR (r=16)
- **Params**: 8,905,472
- **Batch**: 16 × 512

## Dataset
- **WikiText-103**: 198,097 train seq, 417 valid seq

## Training
- **AdamW** (lr=3e-4, wd=0.01), FP32, 2000 steps
- **Duration**: 794s, **Throughput**: 20627 tok/s

## Results
- **Best Validation PPL**: 224.21
- **Final Validation PPL**: 224.21

## Sample
the <unk> <unk>, and <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>

## Architecture
- **V3-Embed**: 812,288 params
- **Standard TF**: 7,077,888 params
- **Low-Rank (r=16)**: 221,184 params
