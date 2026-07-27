#!/usr/bin/env bash
# Sequential baseline queue (resumable; skips completed seeds).
set -u
cd "$(dirname "$0")/.."
for exp in mnist_mlp_fp32 mnist_mlp_bc mnist_mlp_bnn mnist_mlp_bop \
           fashion_mnist_mlp_fp32 fashion_mnist_mlp_bc fashion_mnist_mlp_bnn fashion_mnist_mlp_bop; do
  python3 -m experiments.run --exp "$exp" --seeds 0 1 2 3 4
done
python3 -m experiments.aggregate
