# 01 — Problem Analysis: Why 1-bit Learning Is Broken Today

## 1. What "1-bit" actually buys you

Before designing anything, be precise about the prize:

- **Inference compute:** binary weights × binary activations turn multiply-accumulate
  into XNOR + popcount. On commodity hardware a 64-lane XNOR-popcount replaces 64
  FP multiplies; on custom hardware the area/energy win is ~2–3 orders of magnitude.
- **Memory:** 32× smaller weights than FP32; activations too if binarized.
- **Training compute (the neglected one):** essentially *nobody* gets this today.
  STE-based training is *more* expensive than FP training (extra quantize/dequantize
  ops on top of full FP backward pass + FP32 latent weights + FP32 optimizer state,
  so Adam on a "1-bit" model stores 96 bits/weight during training).

So the honest scoreboard for any proposed method has three columns:
**inference cost, training cost, accuracy.** Existing work optimizes column 1 only.

## 2. The three sins of the STE paradigm

**Sin 1 — Gradient mismatch.** Forward uses `sign(w)`, backward pretends the
derivative is 1 (or a clipped/soft variant). The optimization landscape being
descended is not the landscape being evaluated. Empirically this shows up as:
instability late in training, sensitivity to LR schedules, the need for tricks
(ReActNet's RPReLU, knowledge distillation from an FP teacher, multi-stage
training). The tricks work, but they are evidence the core estimator is wrong.

**Sin 2 — Latent continuous weights.** The FP32 shadow weight is the *real*
parameter; the binary weight is a projection of it. That means (a) training memory
and compute are FP-class, (b) the "model" being trained is secretly continuous —
we never actually learned in the discrete space, we learned nearby and rounded.

**Sin 3 — Architecture inheritance.** Conv/BN/ReLU stacks were co-evolved with
SGD on continuous weights. Under 1-bit constraints, several of their design
assumptions fail:

- *Dot products lose magnitude information.* A binary dot product is a popcount:
  an integer in [-n, n]. Information per neuron is bounded by log2(n+1) bits, not
  the ~real-valued channel a FP neuron has. Narrow layers are catastrophically
  lossy; width has to substitute for precision.
- *BatchNorm is doing secret full-precision work.* In most BNN pipelines the BN
  scale/shift (FP32) carries a large share of representational capacity, and at
  inference it collapses into an integer threshold — i.e. the "neural network" is
  really a learned threshold circuit. We should embrace that instead of hiding it.
- *Residual connections in FP addition* are the accuracy lifeline of modern BNNs
  (Bi-Real Net). This again smuggles precision back in.
- *Softmax + cross-entropy* assumes real-valued logits. With integer popcount
  logits, margin-style or counting-style losses may be more natural.

## 3. Reframing: what problem class is this really?

A network with binary weights and threshold activations **is a Boolean circuit** /
threshold circuit (TC0-ish). Learning it is a *combinatorial* problem:

- Learning a single binary perceptron is NP-hard in the worst case, but
  average-case it is solvable, and statistical-physics analysis (Baldassi,
  Zecchina et al.) shows the solution space has rare dense clusters of solutions
  that are *accessible* to certain heuristics (reinforced belief propagation,
  entropy-driven local search) — and generalize better than isolated solutions.
- This tells us the right mental model is **search in a discrete space with a
  smoothed/entropic objective**, not "gradient descent with a broken derivative."

Two distinct strategies follow, and everything in the literature is one of them:

- **(S1) Smooth the *parameters*:** maintain a distribution/relaxation over
  discrete weights, optimize it with continuous tools, then collapse.
  (STE is a degenerate version of this; stochastic weights / Gumbel / Bayesian
  BNNs are principled versions. All pay FP training cost.)
- **(S2) Search the discrete space directly:** bit-flip local search, evolutionary
  methods, message passing, one-shot constructive rules. Training can be cheap,
  but naive versions don't scale in depth.

The interesting unexplored territory is **(S3): change the model class** so that
the discrete learning problem becomes easy or even closed-form — lookup tables,
weightless neural networks, hyperdimensional computing, logic networks, boosted
threshold committees. Here "1-bit" is not a constraint fought against; it is the
native substrate.

## 4. Success criteria for this project

| Metric | Target |
|---|---|
| MNIST accuracy | ≥ 99.0% |
| Fashion-MNIST | ≥ 91% |
| CIFAR-10 | ≥ 90% (stretch: 93%+, ReActNet-class) |
| Training state per weight | ≤ 8 bits total (vs ~96 for STE+Adam) |
| Training arithmetic | integer/bitwise dominant; no dense FP32 backward |
| Inference | pure XNOR/popcount/threshold (+ small integer bias), no FP |

A method that hits CIFAR-10 ≥ 90% with integer-only training would be a genuine
contribution regardless of whether it looks like a neural network.
