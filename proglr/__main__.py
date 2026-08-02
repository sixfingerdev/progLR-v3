"""CLI: python -m proglr — trains with TrainConfig.from_env() (SCALE_* env vars)."""
from .config import TrainConfig
from .train import train


def main():
    cfg = TrainConfig.from_env()
    res = train(cfg)
    print(f'\nRESULT: {res["name"]} | best_ppl={res["best"]:.2f} | {res["sec"]:.0f}s | '
          f'{res["vram"]:.2f}GB | {res["tok_per_sec"]:.0f} tok/s')


if __name__ == '__main__':
    main()
