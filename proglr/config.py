"""TrainConfig: single source of truth for all training settings."""
from dataclasses import dataclass, field, fields
import os


@dataclass
class TrainConfig:
    # --- data ---
    data: str = ''                      # path to wt103 pkl (tr_x/tr_y/te_x/te_y/vocab_size)
    out_dir: str = ''                   # checkpoint / result output dir
    chunk: int = 50000                  # vocab chunk width for the analytic head

    # --- architecture ---
    d: int = 768
    h: int = 12
    n_std: int = 1
    n_lr: int = 1
    r: int = 16
    a: int = 64
    seq: int = 512
    max_pos: int = 1024

    # --- training ---
    batch: int = 16
    sub_batch: int = 1024               # head sub-batch (memory control)
    steps: int = 20000
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 7
    device: str = 'cuda'

    # --- speed / precision ---
    fp16: bool = False                  # fp16 tensor-core body (R buffer cast to half)
    checkpoint_all: bool = False        # gradient-checkpoint every layer (memory saver)
    head: str = 'exact'                 # 'exact' (full softmax) | 'nce' (sampled)
    nce_k: int = 512                    # negatives per step (shared set, NCE only)
    head_schedule: list = None          # [(step, 'exact'|'nce', k), ...]; e.g. [(0,'nce',512),(8000,'exact',0)]
                                        # None -> single head from `head`/`nce_k` for all steps

    # --- logging / eval ---
    log_every: int = 100
    eval_every: int = 2000              # periodic val_ppl eval (0 = disabled)
    save_best: bool = True              # save checkpoint at each eval
    eval_at_end: bool = True

    def resolve(self, base_dir: str):
        """Fill defaults that depend on the repo layout."""
        if not self.data:
            self.data = os.path.join(base_dir, 'condenser', 'wt103_data.pkl')
        if not self.out_dir:
            self.out_dir = os.path.join(base_dir, 'report_out_scale')
        os.makedirs(self.out_dir, exist_ok=True)
        return self

    @property
    def n_layers(self) -> int:
        return self.n_std + self.n_lr

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @staticmethod
    def from_env() -> 'TrainConfig':
        """Build from SCALE_* / V2_* env vars (legacy CLI compatibility)."""
        i = int
        cfg = TrainConfig(
            data=os.environ.get('SCALE_DATA', ''),
            d=i(os.environ.get('SCALE_D', '768')),
            h=i(os.environ.get('SCALE_H', '12')),
            n_std=i(os.environ.get('SCALE_NSTD', '1')),
            n_lr=i(os.environ.get('SCALE_NLR', '1')),
            r=i(os.environ.get('SCALE_R', '16')),
            a=i(os.environ.get('SCALE_A', '64')),
            batch=i(os.environ.get('SCALE_B', '16')),
            sub_batch=i(os.environ.get('SCALE_SUBBATCH', '1024')),
            steps=i(os.environ.get('SCALE_STEPS', '20000')),
            fp16=os.environ.get('SCALE_FP16', '0') == '1',
            checkpoint_all=os.environ.get('SCALE_NOCKPT', '0') != '1',
            head='nce' if int(os.environ.get('SCALE_NCE', '0')) > 0 else 'exact',
            nce_k=i(os.environ.get('SCALE_NCE', '0')),
            log_every=i(os.environ.get('SCALE_LOG_EVERY', '100')),
            eval_every=i(os.environ.get('SCALE_EVAL_EVERY', '2000')),
        )
        if os.environ.get('V2_STEPS'):
            cfg.steps = i(os.environ['V2_STEPS'])
        return cfg
