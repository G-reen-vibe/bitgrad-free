# 03 — Roadmap & Decision Log

## Phase 0 — Groundwork (next)
- [ ] Repo scaffolding: `src/`, `experiments/`, seed utilities (bit-packing,
      popcount kernels via numpy uint64, dataset loaders with binary encodings).
- [ ] Binary input encoders: thermometer coding, local binary patterns, sign of
      random projections — shared by all candidates.
- [ ] Reference baselines: FP MLP/LeNet (accuracy anchor), simple STE-BNN
      (cost anchor). Log accuracy + training-state-bits + bit-ops.

## Phase 1 — Prototype bake-off (MNIST, 1 day each)
- [ ] **C-A Counting Logic Nets** (D5+D8+D6+D7): LUT layers filled by counting.
- [ ] **C-B Integer-Bop** (D1+D6): hysteresis counters, 1-bit backward error.
- [ ] **C-C Boosted binary blocks** (D9+D5+D3).
- Gate: ≥97% MNIST, integer-only inner loop. Losers get a post-mortem note here.

## Phase 2 — Scale survivors
- Fashion-MNIST gate (≥91%), then CIFAR-10 with conv/patch-local connectivity.
- Ablations: error bit-width (C-B), target-propagation scheme (C-A),
  block depth (C-C), front-end encoder choice (all).

## Phase 3 — Consolidate
- Unify the winner into a small framework API; benchmark suite; writeup.

## Decision log
| Date | Decision | Rationale |
|---|---|---|
| 2026-07-26 | Repo created; problem framed as discrete search + model-class redesign, not STE repair | See docs/01, docs/02 |
| 2026-07-26 | Three composite candidates selected for bake-off | Coverage of the three axes (local rule / credit assignment / architecture) with different risk profiles |
| 2026-07-26 | Stack = JAX(CPU)+Flax+Optax+NumPy | 1-core/3GB box; PyTorch pulls multi-GB CUDA deps & its CPU index is proxy-blocked; jaxlib PyPI wheel is CPU-native and small; XLA best single-core FP throughput; integer-native methods will be NumPy bit-ops |
| 2026-07-26 | CIFAR-10 from JPEG GitHub mirror, checksummed | Canonical hosts + LFS media blocked by egress proxy; identical cache for all methods keeps internal comparisons valid (caveat in docs/04) |
| 2026-07-26 | Baselines = fp32 / BinaryConnect / BNN / Bop; compute-matched budgets; 5 seeds (3 for CIFAR) with 95% t-CI | Covers FP anchor + both STE regimes + the strongest latent-free prior method; paper-scale infeasible on 1 core, so shared budgets + quoted literature context (docs/04) |
