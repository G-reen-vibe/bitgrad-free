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
- [ ] Direction survey & critical review (`docs/`)
- [ ] Baselines
- [ ] Prototype v0
