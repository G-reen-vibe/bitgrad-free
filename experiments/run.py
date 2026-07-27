"""Run experiments: python3 -m experiments.run --exp mnist_mlp_bnn [--seeds 0 1 2] [--smoke]

Writes one JSON per (experiment, seed) to results/<exp>/seed<k>.json; skips seeds
that already have results (idempotent — safe to resume across sessions since the
filesystem is ephemeral but the git repo is not). --smoke caps steps/eval for a
fast end-to-end correctness check and writes to results/smoke/ instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bitgrad.train import train_one  # noqa: E402
from experiments.configs import CONFIGS, SEED_DEFAULTS  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override epoch budget")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = dict(CONFIGS[args.exp])
    seeds = args.seeds if args.seeds is not None else SEED_DEFAULTS[cfg["dataset"]]
    if args.epochs:
        cfg["epochs"] = args.epochs
    outdir = os.path.join(RESULTS, "smoke" if args.smoke else "", args.exp)
    if args.smoke:
        cfg.update(epochs=1, max_steps=120, eval_limit=2000)
        seeds = seeds[:1]
    os.makedirs(outdir, exist_ok=True)

    for seed in seeds:
        path = os.path.join(outdir, f"seed{seed}.json")
        if os.path.exists(path):
            print(f"[skip] {args.exp} seed {seed} (exists)")
            continue
        print(f"[run] {args.exp} seed {seed} cfg={cfg}", flush=True)
        res = train_one(cfg, seed)
        res["config"] = cfg
        res["experiment"] = args.exp
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
        print(f"[done] {args.exp} seed {seed}: final={res['final_acc']:.4f} "
              f"best={res['best_acc']:.4f} ({res['wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
