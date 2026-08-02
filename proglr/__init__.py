"""proglr — ProgLR (V3-Embed + Progressive Low-Rank) training library.

One import, everything controllable:

    from proglr import TrainConfig, train

    train(TrainConfig(
        data=r'C:\\Users\\enesa\\law\\condenser\\wt103_data_250k.pkl',
        d=1536, h=24, n_std=2, n_lr=6,
        head='nce', nce_k=512,          # 'exact' | 'nce'
        fp16=True, batch=16, sub_batch=1024,
        steps=20000, eval_every=2000,
    ))
"""
import os
import sys

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)  # allow `from condenser.v3_embed import ...`

from .config import TrainConfig
from .model import LRLayer, StdLayer, ProgLR
from .head import head_loss_fn, eval_ppl
from .train import train, train_many, load_data, make_model

__version__ = '0.1.0'
__all__ = ['TrainConfig', 'ProgLR', 'LRLayer', 'StdLayer', 'head_loss_fn', 'eval_ppl',
           'train', 'train_many', 'load_data', 'make_model', '__version__']
