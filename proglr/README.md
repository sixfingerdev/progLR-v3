# proglr

**ProgLR** — V3-Embed + Progressive Low-Rank transformer eğitim kütüphanesi.
WikiText-103 üzerinde 6 GB'lık tüketici GPU'sunda (RTX 3060 Laptop) 50K–250K kelime
hazineli modelleri eğitmek için tasarlandı. Çıktı katmanı (head) **exact CE** veya
**NCE** (sampled softmax) olarak seçilebilir; tüm eğitim ayarları tek bir config
objesiyle yönetilir.

```
from proglr import TrainConfig, train

train(TrainConfig(head='nce', nce_k=512, fp16=True, d=1536, h=24, steps=20000))
```

---

## 1. Kurulum

- Python 3.10+, PyTorch 2.x (CUDA), ReportLab yalnızca PDF üreticileri için gerekir (kütüphane değil).
- Paket `law/` kökünün altında yaşar; `condenser.v3_embed` modülünü kullanır.
  İkisi de `sys.path` üzerinden import edilir:

```powershell
cd C:\Users\enesa\law
$env:PYTHONPATH = 'C:\Users\enesa\law'      # geçerli oturum için
& .\venv\Scripts\python.exe -c "import proglr; print(proglr.__version__)"
```

İsteğe bağlı (kalıcı kurulum):

```powershell
& .\venv\Scripts\python.exe -m pip install -e .   # law/ içinde bir pyproject.toml ile
```

## 2. Hızlı başlangıç

```python
from proglr import TrainConfig, train, train_many

# --- 1) Exact CE, 50K vocab, WT103 (referans koşu) ---
res = train(TrainConfig(
    data=r'C:\Users\enesa\law\condenser\wt103_data.pkl',
    d=768, h=12, n_std=1, n_lr=1,
    head='exact',
    steps=2000, eval_every=2000,
))
print(res['best'], res['tok_per_sec'], res['vram'])

# --- 2) Aynı model, NCE head (sampled softmax, K=512) ---
res = train(TrainConfig(
    data=r'C:\Users\enesa\law\condenser\wt103_data.pkl',
    d=768, h=12, n_std=1, n_lr=1,
    head='nce', nce_k=512,
    steps=2000, eval_every=2000,
))

# --- 3) 250K vocab ölçek koşusu (fp16 gövde + NCE) ---
res = train(TrainConfig(
    data=r'C:\Users\enesa\law\condenser\wt103_data_250k.pkl',
    d=1536, h=24, n_std=2, n_lr=6,
    head='nce', nce_k=128,
    fp16=True, batch=16, sub_batch=1024,
    steps=500, eval_every=0,
))

# --- 4) Adil karşılaştırma (aynı harness, seed 7) ---
results = train_many([
    TrainConfig(d=768, h=12, n_std=1, n_lr=1, head='exact', steps=500, eval_every=0),
    TrainConfig(d=768, h=12, n_std=1, n_lr=1, head='nce',  nce_k=512, steps=500, eval_every=0),
])

# --- 5) Hibrit: NCE ısınma (2000 adım) -> exact CE rötuş (2000 adım) ---
res = train(TrainConfig(
    d=768, h=12, n_std=1, n_lr=1,
    head_schedule=[(0, 'nce', 512), (2000, 'exact', 0)],
    steps=4000, eval_every=500,
))
```

## 3. TrainConfig referansı

Tüm ayarlar tek dataclass'ta. `as_dict()` her koşunun yapılandırmasını
checkpoint'lere kaydeder.

| Alan | Default | Açıklama |
|---|---|---|
| `data` | `law/condenser/wt103_data.pkl` | Veri pkl'i (`tr_x/tr_y/te_x/te_y/vocab_size` anahtarları) |
| `out_dir` | `law/report_out_scale` | Checkpoint + çıktı klasörü |
| `chunk` | 50000 | Exact head'de vocab chunk genişliği (bellek kontrolü) |
| `d` | 768 | Model genişliği |
| `h` | 12 | Başlık (attention head) sayısı |
| `n_std` | 1 | Standart (dense) transformer katmanı sayısı |
| `n_lr` | 1 | Düşük ranklı (LR) katman sayısı |
| `r` | 16 | LR rank |
| `a` | 64 | LR alpha (a/r çarpanı) |
| `seq` | 512 | Bağlam uzunluğu |
| `max_pos` | 1024 | Pozisyon embedding boyutu |
| `batch` | 16 | Mini-batch (16×512 = 8192 token/adım) |
| `sub_batch` | 1024 | Exact head'in adım başına işlediği token alt kümesi (bellek sınırı) |
| `steps` | 20000 | Toplam adım |
| `lr` | 3e-4 | AdamW öğrenme oranı |
| `weight_decay` | 0.01 | AdamW WD |
| `grad_clip` | 1.0 | Gradyan norm kırpma |
| `seed` | 7 | Global seed (karşılaştırılabilir koşular için sabit tutun) |
| `device` | `cuda` | Eğitim cihazı |
| `fp16` | False | Gövde matmul'larını fp16 tensor core'da çalıştırır (R buffer kalıcı half) |
| `checkpoint_all` | False | Her katmanda gradient-checkpoint (bellek tasarrufu, hız kaybı) |
| `head` | `exact` | `'exact'` (full softmax) veya `'nce'` (sampled) |
| `nce_k` | 512 | NCE adım başına paylaşımlı negatif sayısı |
| `head_schedule` | None | Adım bazlı head planı: `[(step, 'exact'\|'nce', k), ...]`; örn. `[(0,'nce',512),(2000,'exact',0)]` = ilk 2000 adım NCE, sonra exact CE. `None` → tüm koşu tek head |
| `log_every` | 100 | Loss/tok-s log aralığı |
| `eval_every` | 2000 | Periyodik val_ppl (0 = kapalı) |
| `save_best` | True | Her eval'da `best_<head>.pt` checkpoint'i kaydeder |
| `eval_at_end` | True | Koşu sonunda val_ppl ölçer |

## 4. Head'ler

### `head='exact'` — Analytic chunked head
`l = h·Rᶜ + (α/r)·(h·V)·Uᶜ` her chunk'ta yeniden kullanılan bir tampon üzerinde
hesaplanır, softmax yerinde yapılır, `dH/dU/dV` analitik (autograd yok). Matematiksel
olarak full softmax ile bit-bit aynı; 250K vocab'ta `(N×V)` logit tensörünü hiç
oluşturmaz (vanilla'da ~8GB paging → burada 5.7GB peak).

### `head='nce'` — NCE sampled head
Her token hedef sınıfı + **adım boyunca paylaşılan K rastgele negatif** puanlar.
Head maliyeti tek küçük GEMM'e iner ve **kelime hazinesi boyutundan bağımsız** olur.
Logit'ler ve softmax fp32'de (TF32 matmul) — fp16'da büyüyen hedef logit'leri
65,504'ü aşıp `inf→NaN` ürettiği için (gövde yine fp16 kalabilir).

### Ölçülen sonuçlar (RTX 3060 Laptop 6GB, aynı model/veri/batch, seed 7)

| Senaryo | Exact CE | NCE | Fark |
|---|---|---|---|
| V=250K, D=1536 (64.9M), fp16 | 5.0K tok/s, 5.70GB | **14.9K tok/s**, 5.58GB | 3.0× hız |
| V=50K, D=768 (8.9M), fp32 | 20.6K tok/s, 4.68GB | **90.9K tok/s**, 1.16GB | 4.4× hız |
| 8.9M val_ppl @ 20K adım | ~94 | 265.55 | ~2.8× PPL kaybı (aynı adım) |

Yorum: NCE bir **hız kolu**, kalite kolu değil — aynı adımda PPL ~2.8× geride;
ancak aynı duvar saati süresinde ~4.4× fazla token işler ve vocab'tan bağımsız
ölçeklenir. Kalite metriği istendiğinde `head='exact'`, hız/bütçe istendiğinde
`head='nce'` kullanın; doğal hibrit: NCE ile warmup → exact CE ile refine.

### Hibrit (`head_schedule`)
NCE birkaç yüz ucuz adımda kaba yapıyı öğrenir (loss 5.4→2.0), exact CE ince
dağılımı bitirir. `head_schedule` ikisini tek koşuda birleştirir. Ölçülen
(8.9M, V=50K, WT103, FP32, seed 7):

| Koşu | val_ppl | Duvar saati |
|---|---|---|
| Exact CE 2000 (ref) | 224.21 | ~13.5 dk |
| Exact CE 4000 (ref) | 147.60 | ~27 dk |
| NCE 2000 (ref) | 797.27 | ~3 dk |
| NCE 2000 → exact 500 | 388.33 | ~6 dk |
| **NCE 2000 → exact 2000** | **233.10** | **~14 dk** |
| NCE 20000 (ref) | 265.55 | ~80 dk |

Rötuş eğrisi: 797 → 390 (500 exact adım) → 233 (2000 exact adım). Aynı ~14
dakikalık bütçede exact-CE-2000 kalitesinin %96'sına ulaşır ve saf NCE'nin
20K adımlık koşusunu (~80 dk) **3.5× az sürede %12 geçer**. Hibrit, bu
donanımdaki en iyi hız/kalite dengesidir.

## 5. Çıktılar

- `train()` → dict: `{name, ppl, best, params, sec, vram, tok_per_sec}`
- `save_best=True` → `out_dir/best_<head>.pt`:
  `{step, model_state, best_ppl, config}` (config tam kayıtlı — koşu tekrarlanabilir)
- `eval_ppl` her zaman **exact full softmax** ile ölçer (eğitim head'inden bağımsız).

## 6. CLI (eski env uyumluluğu)

`SCALE_*` değişkenleriyle `python -m proglr` aynı şeyi yapar:

```powershell
$env:SCALE_DATA = 'C:\Users\enesa\law\condenser\wt103_data_250k.pkl'
$env:SCALE_D = '1536'; $env:SCALE_H = '24'
$env:SCALE_NSTD = '2'; $env:SCALE_NLR = '6'
$env:SCALE_B = '16'; $env:SCALE_SUBBATCH = '1024'
$env:SCALE_STEPS = '500'; $env:SCALE_LOG_EVERY = '50'
$env:SCALE_FP16 = '1'
$env:SCALE_NCE = '128'          # 0 = exact CE
python -m proglr
```

`SCALE_NCE=0` (veya boş) exact CE demektir. `SCALE_EVAL_EVERY` eval aralığını
kontrol eder.

## 7. Bellek notları

- 6GB kartta **D=1536 + B=16 + sub_batch=1024** maksimum güvenli noktadır;
  `sub_batch=2048` veya `B=32` CUDA paging'e düşer (tok/s çakılır).
- 250K uzun fp32 koşuları GPU TDR (driver watchdog) ile kilitlenebiliyor;
  fp16 + kısa periyodik eval önerilir.
- `eval_at_end=False` + `eval_every=0` ile eval maliyeti (250K'da ~25s/koşu) atlanır.

## 8. Dosyalar

| Dosya | Rol |
|---|---|
| `proglr/__init__.py` | Public API: `TrainConfig, ProgLR, head_loss_fn, eval_ppl, train, train_many, load_data, make_model` |
| `proglr/config.py` | `TrainConfig` (tüm ayarlar) + `from_env()` |
| `proglr/model.py` | `LRLayer`, `StdLayer`, `ProgLR` (V3-Embed + katmanlar) |
| `proglr/head.py` | `head_loss_fn` (exact + NCE), `eval_ppl` |
| `proglr/train.py` | `train(cfg)`, `train_many(cfgs)`, `load_data`, `make_model` |
| `proglr/__main__.py` | CLI: `python -m proglr` (env odaklı) |

Tarihsel karşılaştırma koşuları ve rapor üretimi için:
`tests/scale_compare_train.py`, `tests/final_report_train.py`,
`make_report_pdf.py`, `make_exec_pdf(_tr).py`.
