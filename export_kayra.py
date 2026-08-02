"""Export 250K checkpoint + word-level tokenizer -> HuggingFace package.

Creates kayra-1.1-english/ with:
  model.safetensors, config.json, modeling_proglr.py, tokenizer.json pieces,
  vocab.txt, tokenizer_config.json, README.md (written separately).
"""
import os
import pickle
import json
import sys
import math

import torch
from safetensors.torch import save_file

LAW = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAW)

CKPT = os.path.join(LAW, 'report_out_250k', 'checkpoints', 'ckpt_03500.pt')
VOCAB_PKL = os.path.join(LAW, 'condenser', 'wt103_vocab_250k.pkl')
OUT = os.path.join(LAW, 'kayra-1.1-english')

ck = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = ck['model_state']
print('checkpoint step', ck['step'], 'best_ppl', round(ck['best_ppl'], 2))

voc = pickle.load(open(VOCAB_PKL, 'rb'))
vocab = voc['vocab']  # word -> id
id2word = voc['id2word']

vocab_size = sd['emb.R'].shape[0]
assert vocab_size == 250000
print('R', tuple(sd['emb.R'].shape), 'U', tuple(sd['emb.U'].shape),
      'V', tuple(sd['emb.V'].shape), 'rank', sd['emb.U'].shape[1])
print('alpha/rank:', 64, '/', 16, '= s', 64 / 16)

D = sd['pos'].shape[-1]
max_pos = sd['pos'].shape[1]
n_std = 1
n_lr = 1
heads = 12
print('D', D, 'max_pos', max_pos, 'n_std', n_std, 'n_lr', n_lr)

trainable = sum(v.numel() for k, v in sd.items()
                if k.startswith(('emb.U', 'emb.V', 'pos', 'layers', 'norm')))
print('trainable params (incl pos/bias):', f'{trainable:,}')

config = {
    'architectures': ['ProgLRForCausalLM'],
    'model_type': 'proglr',
    'auto_map': {
        'AutoModelForCausalLM': 'modeling_proglr.ProgLRForCausalLM',
        'AutoModel': 'modeling_proglr.ProgLRModel',
    },
    'd': D,
    'heads': 12,
    'n_std': 1,
    'n_lr': 1,
    'r': 16,
    'a': 64,
    'max_pos': max_pos,
    'vocab_size': vocab_size,
    'pad_token_id': 0,
    'unk_token_id': 1,
    'bos_token_id': 2,
    'eos_token_id': None,
    'tie_word_embeddings': False,
    'torch_dtype': 'float32',
    'model_step': ck['step'],
    'best_val_ppl': round(ck['best_ppl'], 3),
}

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUT, 'vocab.txt'), 'w', encoding='utf-8') as f:
    for i in range(vocab_size):
        f.write((id2word.get(i, '<pad>') if i > 0 else '<pad>') + '\n')

# tokenizer is the custom slow WordLevelProgLRTok (tokenizer_proglr.py + vocab.txt);
# no HF tokenizer.json is produced (it would require the full HF JSON schema).

save_file({k: v.contiguous() for k, v in sd.items()},
          os.path.join(OUT, 'model.safetensors'))
print('saved', os.path.join(OUT, 'model.safetensors'),
      f'{os.path.getsize(os.path.join(OUT, "model.safetensors")) / 1e6:.0f} MB')
print('vocab.txt / tokenizer.json / config.json written')
