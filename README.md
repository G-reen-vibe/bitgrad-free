# bitgrad-free

**Research project: a natively 1-bit machine learning framework.**

Current binary/ternary neural networks (BinaryConnect, XNOR-Net, BNN, BitNet, etc.)
are trained with straight-through estimators (STE) or other continuous proxies. This
means training still requires latent full-precision weights and full-precision
gradients, the discrete forward pass and continuous backward pass are mismatched,
and the architectures themselves are inherited from the continuous regime where they
may not make sense.

**Goal:** design a learning system from scratch where

1. the *model* is natively 1-bit (or ternary) — weights, and ideally activations;
2. the *training procedure* is natively efficient — no shadow FP32 weights, no
   dense FP gradients if avoidable;
3. accuracy is competitive with SOTA on basic vision classification
   (MNIST → Fashion-MNIST → CIFAR-10).

All options are on the table: unconventional architectures, non-gradient
optimizers, or abandoning neural networks entirely.

## Repo layout (planned)

```
docs/        research notes, direction surveys, decision logs
src/         framework code
experiments/ runnable experiment scripts + results
```

## Status

- [x] Repo initialized
- [x] Direction survey & critical review (`docs/01–03`)
- [x] Environment: JAX (CPU) + Flax + Optax on a 1-core/3 GB box (`requirements.txt`)
- [x] Data pipeline: MNIST / Fashion-MNIST / CIFAR-10 via proxy-reachable mirrors,
      checksummed npz caches (`scripts/fetch_data.sh`, `scripts/data.md5`)
- [x] Baselines implemented: FP32, BinaryConnect, BNN (STE), Bop (latent-free)
- [x] Evaluation harness: multi-seed runner + 95% t-CI aggregation (`docs/04`)
- [ ] Baseline numbers on all benchmarks (`results/summary.md`, filling in)
- [ ] Prototype v0 (Phase 1 bake-off)

## Reproduce

```
pip install -r requirements.txt
./scripts/fetch_data.sh                                  # rebuild data caches
python3 -m experiments.run --exp mnist_mlp_bnn --smoke   # fast correctness check
./scripts/queue_phase1.sh                                # full baseline queue (resumable)
python3 -m experiments.aggregate                         # results/summary.md
```
