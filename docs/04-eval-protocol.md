# 04 — Evaluation Protocol

## Benchmarks
| Dataset | Size | Role |
|---|---|---|
| MNIST | 60k/10k, 28×28×1 | sanity gate (kill-criterion: method < 97% dies) |
| Fashion-MNIST | 60k/10k, 28×28×1 | harder texture task, same cost |
| CIFAR-10 | 50k/10k, 32×32×3 | main benchmark (+ standard flip/pad-crop augmentation) |

**CIFAR-10 data caveat:** canonical hosts are blocked by this machine's egress
proxy; we use the `YoongiKim/CIFAR-10-images` GitHub mirror, which is a *lossy
JPEG re-encode*. Every method and baseline in this repo uses the identical cache
(md5s in `scripts/data.md5`), so internal comparisons are unaffected; comparisons
against literature numbers carry a small (≲0.5 pp) dataset-shift asterisk.

## Baselines (all implemented in `src/bitgrad`, JAX/Flax)
| Key | Method | Weights | Acts | Optimizer | Train state bits/w |
|---|---|---|---|---|---|
| `fp32` | full-precision reference | FP32 | FP32 | Adam | 96 |
| `bc` | BinaryConnect (Courbariaux '15) | sign(STE latent) | FP | Adam | 96 |
| `bnn` | BNN (Courbariaux/Hubara '16) | sign(STE latent) | sign(STE) | Adam | 96 |
| `bop` | Bop (Helwegen '19, latent-free) | native ±1 | sign(STE) | Bop(γ,τ) + Adam(BN) | 33 |

`train state bits per weight` = weight storage during training + optimizer state,
for binary-layer weights. This is the metric STE pipelines fail (32 latent + 64
Adam); Bop needs 33 (1 + 32 FP EMA); our future methods target ≤ 9.

Architectures: `mlp` (784-256-256-10, BN, ~270k params) for MNIST/Fashion;
`cnn_s` (VGG-style 2 blocks, 32/64 ch, GAP head, ~66k params) and `cnn_m`
(3 blocks, 64/128/256) for CIFAR-10. BatchNorm follows every weight layer;
activations binarize after BN (standard BNN practice — at inference BN reduces
to an integer threshold).

## Statistics
- MNIST / Fashion-MNIST: **5 seeds** (0–4); CIFAR-10: **3 seeds** (0–2).
- Report mean ± **95% Student-t interval** over seeds (`src/bitgrad/stats.py`),
  for both final-epoch and best-epoch test accuracy.
- Runner is idempotent per (experiment, seed) → resumable across sessions;
  results JSONs are committed to git (the filesystem is ephemeral, the repo isn't).
- Also logged per run: wall time, steps/s, param counts, state bits, model bits.

## Compute-matched budgets (1 CPU core, 3 GB RAM)
Paper-scale baselines (VGG-Small/ResNet + hundreds of epochs) are impossible
here; we instead fix a *shared* budget per benchmark, identical for every method
— fair internal comparison, honest external framing:

| Benchmark | Arch | Epochs | ~time/seed | measured steps/s |
|---|---|---|---|---|
| MNIST | mlp | 15 | ~4.5 min | ~35 |
| Fashion-MNIST | mlp / cnn_s | 15 | ~4.5 / ~25 min | ~35 / ~3 |
| CIFAR-10 | cnn_s | 12 | ~2 h | ~0.8 |
| CIFAR-10 | cnn_m | 30 | (needs bigger box) | — |

## Literature context (paper-scale numbers, quoted not reproduced)
For calibration only — different architectures/budgets than ours:
FP32 VGG-Small CIFAR-10 ≈ 93–94%; BinaryConnect ≈ 90–91.7%; BNN (bin W+A)
≈ 88–89.9%; XNOR-Net ≈ 89–90%; Bop ≈ 91.3% (BinaryNet arch); ReActNet-style
SOTA BNNs ≈ 92–95% with FP teachers and multi-stage training. MNIST MLP-class:
FP ≈ 98.5–99.2%, BNN ≈ 98.6–99%. Our compute-matched table lives in
`results/summary.md`; both tables must always be read together.

## What "winning" means (from docs/01)
A new method must Pareto-dominate on (accuracy, train-state bits, training
arithmetic class, inference arithmetic class) — not accuracy alone.
