"""Load a final_report checkpoint and evaluate val_ppl with the standard eval."""
import os, sys, math, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

torch.set_float32_matmul_precision('high')

from final_report_train_v2 import ProgLR, eval_ppl, CHUNK, D, VS, SEQ, DATA, VOCAB

dev = torch.device('cuda')
with open(DATA, 'rb') as f:
    d = pickle.load(f)
te_x = torch.tensor(d['te_x'], device=dev, dtype=torch.int32)
te_y = torch.tensor(d['te_y'], device=dev, dtype=torch.int32)
del d

parser = argparse.ArgumentParser()
parser.add_argument('ckpt')
args = parser.parse_args()

ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
model = ProgLR(VS, D, 2, 12, 1, 16, 64).to(dev)
model.load_state_dict(ckpt['model_state'])
model.eval()

ppl = eval_ppl(model, te_x, te_y)
print(f'{args.ckpt}: step={ckpt.get("step")} | val_ppl={ppl:.2f} | best={ckpt.get("best_ppl"):.2f} | loss={ckpt.get("loss"):.3f}')
