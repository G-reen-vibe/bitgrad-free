# 02 — Direction Survey & Critical Review

Nine families of approaches, deliberately spanning "fix the optimizer," "fix the
architecture," and "abandon neural networks." Each gets: idea, why it could work,
why it could fail, verdict.

---

## D1. Integer-state flip optimizers (latent-free "gradient" training)

**Idea.** Keep binary weights as the *only* weights. Per weight, keep a small
signed integer counter c (4–8 bits). Each step, compute a cheap *direction signal*
(sign of a gradient surrogate, or a perturbation-based estimate), accumulate it
into c with saturation and decay; flip the weight when c crosses ±threshold, then
reset. This generalizes Bop (Helwegen 2019) and Madaline-style "minimal
disturbance," but with all optimizer state quantized to integers.

- **Strengths.** Drop-in for deep nets; training memory ~1+8 bits/weight;
  hysteresis naturally prevents oscillation (the classic BNN failure mode);
  compatible with any surrogate signal — including *forward-only* estimates,
  removing the FP backward pass entirely.
- **Weaknesses.** If the direction signal is still an FP backprop gradient, we've
  only fixed memory, not compute or Sin 1. Forward-only estimators (SPSA, evolution
  strategies, forward-mode with random tangents) have variance that grows with
  parameter count — the classic scaling killer. Threshold/decay are new
  hyperparameters with per-layer sensitivity.
- **Critical question.** Can gradient *sign* information survive extreme
  quantization of the backward pass (e.g., 1-bit errors backpropagated, à la
  sign-sign variants of feedback alignment)? If yes, the whole training loop
  becomes bitwise/integer.
- **Verdict: strong candidate.** Most likely path to CIFAR-10-class accuracy with
  small training state. Novelty lever: fully integer backward pass + hysteresis
  counters, co-designed with D6 architecture.

## D2. Stochastic weights / distributional training (Bernoulli parameters)

**Idea.** Parameterize each weight as Bernoulli(p). Train p by REINFORCE,
Gumbel-softmax, expectation backpropagation (Soudry 2014), or natural evolution
strategies over the product distribution. Sample or threshold at deployment.

- **Strengths.** Principled — the expected loss is a smooth function of p; no
  gradient mismatch in the limit. Local-reparameterization gives low-variance
  gradients through the *pre-activation* distribution (central limit theorem:
  popcounts are approximately Gaussian, so you can propagate means/variances
  analytically — this is elegant and underused for BNNs).
- **Weaknesses.** p is a continuous parameter → Sin 2 returns through the back
  door; training cost ≥ FP training. Sampling noise at deployment vs. MAP
  thresholding gap. Entropy collapses slowly → long training.
- **Verdict: useful as theory and as teacher.** Probably not the final framework,
  but the analytic mean/variance propagation trick is stealable: it gives a
  *deterministic, integer-friendly* approximation of ensemble behavior that could
  supply the direction signal for D1 counters. Keep as a component, not a product.

## D3. Direct discrete search (bit-flip local search, SA, evolution)

**Idea.** Treat training as combinatorial optimization: greedy/stochastic bit
flips scored by loss deltas, simulated annealing, population methods.

- **Strengths.** Zero gradient machinery; loss deltas for a single flipped weight
  can be computed *incrementally* (a flip changes one popcount lane → integer
  delta propagation), which is dramatically cheaper than a full forward pass.
  Physics results say entropic/replicated local search finds flat, generalizing
  solutions on shallow nets.
- **Weaknesses.** Credit assignment through depth is the wall: a flip in layer 1
  changes downstream activations discontinuously; incremental evaluation gets
  complicated and the search landscape becomes deceptive. Sequential flips = slow
  wall-clock unless heavily batched/parallelized.
- **Verdict: viable for shallow/last-mile.** Use as (a) fine-tuning polish after
  another method, (b) the trainer for *small local blocks* (see D5/D6 where blocks
  are tiny and search is tractable). Not the backbone for deep end-to-end training.

## D4. Message passing / statistical-physics solvers

**Idea.** Model the binary network as a factor graph over weight variables; run
(reinforced) belief propagation / survey propagation to find dense solution
clusters, as in Baldassi–Zecchina binary perceptron learning.

- **Strengths.** The only family with *theory* saying binary perceptrons are
  learnable in practice and that the found solutions generalize. Messages can be
  heavily quantized. fBP already works for single layers and committee machines.
- **Weaknesses.** Deep, loopy, convolutional graphs break BP assumptions;
  message-passing on conv weight-sharing is painful; engineering complexity is
  high; nobody has scaled it past toy depth.
- **Verdict: high-risk, high-novelty.** Don't bet the project on it, but its core
  insight — *seek dense solution regions, i.e., flat minima in Hamming space* —
  should be encoded into whatever we build (e.g., D1 counters with entropy-style
  regularization: prefer flips that many samples agree on).

## D5. Learned logic: lookup-table & logic-gate networks (leave "neurons" behind)

**Idea.** The model is a feed-forward graph of k-input Boolean functions (LUTs,
k=2–6), i.e., literally an FPGA netlist. Differentiable Logic Gate Networks
(Petersen 2022) train a softmax over the 16 two-input gates per node; LUTNet /
LogicNets / ULEEN learn LUT contents directly. Inference is *pure logic* — faster
than XNOR-nets, no popcount even.

- **Strengths.** Radically honest about the substrate: under 1-bit, a "neuron" is
  a Boolean function anyway, so learn Boolean functions. SOTA-ish results exist
  (CIFAR-10 in the 80s–90+% range in follow-up work with convolutional logic
  networks). Inference cost is absurdly low (millions of gates ≈ nanojoules).
- **Weaknesses.** Training the softmax-over-gates is *continuous* (Sin 2 again)
  and currently expensive/finicky; random fixed connectivity wastes capacity;
  learning *connectivity* is the real problem and it's discrete and hard.
  Depth-2^k blowup limits k.
- **Novel angles worth owning:**
  1. **Train LUTs by counting, not gradients:** a k-LUT with frozen inputs has
     2^k entries; each entry can be set by *majority vote of the desired outputs*
     over the training set (a closed-form, one-pass rule). Layer-wise, with
     targets propagated by (D8) local rules → gradient-free logic training.
  2. **Connectivity by feature selection:** choose the k inputs of each LUT by
     mutual-information / χ² screening — integer statistics, one pass.
- **Verdict: the most exciting "outside the box" direction.** Pairs naturally
  with D3 (per-LUT search is over 2^(2^k) options but entries are independent
  given targets) and D8 (needs local targets).

## D6. Architecture co-design for the 1-bit regime (if we keep NN-shaped models)

Design choices that respect the popcount information bottleneck:

- **Width-for-precision:** wide, *sparse* binary layers; fan-in capped so
  popcount ranges stay small and thresholds stay meaningful (also keeps D3
  incremental deltas cheap).
- **Majority/threshold units with integer thresholds** learned by counting
  statistics (no BN; the threshold *is* the normalizer). Per-channel thresholds =
  medians of pre-activation popcounts — computable online with integer sketches.
- **Ternary {−1,0,+1} weights with sparsity prior:** the 0 state is prunable and
  makes counters (D1) semantically clean (0 = "undecided").
- **Binary residual = XOR-skip or concat-skip** instead of FP addition.
- **Ensembles / ECOC output layer:** many small binary heads with
  error-correcting output codes; integer Hamming decoding; robustness to
  individual head errors — matches the physics intuition that binary solutions
  should be redundant.
- **Fixed binary feature front-end for images:** learned first layers are where
  BNNs bleed accuracy; consider a *fixed* overcomplete binary transform (sign of
  random/DCT/Gabor projections, local binary patterns, thermometer-coded pixels)
  so learnable layers operate on rich binary codes from step one.
- **Verdict: not a standalone method but multiplies every other direction.**
  Thermometer/LBP front-ends are known to add multiple points on CIFAR for
  logic/LUT models.

## D7. Hyperdimensional computing (HDC) & weightless NNs (WiSARD family)

**Idea.** Encode inputs as ~10k-bit hypervectors; "training" = bundling
(majority-summing) vectors per class, or writing RAM cells (WiSARD). One-pass
training, Hamming-distance inference.

- **Strengths.** Training is almost free (single pass, integer counters);
  absurdly robust to bit errors; hardware-trivial.
- **Weaknesses.** Ceiling problem: pure HDC/WiSARD tops out ~95–98% MNIST,
  ~55–70% CIFAR-10 raw. They are effectively fixed-feature linear models in
  Hamming space; no feature *hierarchy*.
- **Verdict: not competitive alone, but the training rule is the treasure.**
  "Bundle with saturating integer counters, then threshold" is exactly a
  closed-form binary-weight learner — reuse it as the layer-local learning rule
  inside deeper stacks (→ D5 LUT-filling, D8 targets). Also ideal as the *output
  classifier* on top of learned binary features.

## D8. Layer-local / backprop-free credit assignment

**Idea.** Replace end-to-end backprop with local objectives so each layer/block
trains with integer statistics: greedy layer-wise training, Forward-Forward
(goodness = popcount magnitude vs. threshold — natively integer!), feedback
alignment with 1-bit random feedback, target propagation with binary targets,
information-bottleneck-style local losses (maximize class MI of binary codes).

- **Strengths.** Kills the FP backward pass — the biggest training-cost item.
  Forward-Forward's "goodness" is literally *sum of squared activations*; for
  binary activations that's just a popcount → the algorithm becomes integer-native
  almost by accident. Local rules compose with D1 counters and D5 LUT filling.
- **Weaknesses.** Local training historically lags end-to-end by a few points on
  CIFAR; greedy layers can learn redundant features; FF needs good negative
  samples; theory thin.
- **Verdict: strong candidate as the credit-assignment backbone**, because it's
  the enabling piece that lets D1/D5/D7 stack into deep models without FP
  gradients.

## D9. Boosting / constructive committees of threshold units

**Idea.** Don't train a fixed architecture; *grow* one. AdaBoost over weak binary
threshold learners (each trainable in closed form or by tiny D3 search); the final
model is a weighted-majority — with integer weights via quantized boosting — i.e.,
still popcount-implementable. Deep variant: boosted stacks where each round's
features feed the next (layer = ensemble round).

- **Strengths.** Boosting is a *native discrete-learner amplifier* with actual
  generalization theory (margins). No gradients anywhere. Anytime property:
  accuracy scales with rounds/compute. Sample weights can be quantized.
- **Weaknesses.** Sequential (rounds don't parallelize trivially); flat ensembles
  don't build compositional features → likely capped on CIFAR unless stacked;
  stacking boosted layers is under-explored (that's an opportunity).
- **Verdict: dark horse.** Especially attractive fused with D5: boost *LUT
  blocks* rather than stumps.

---

# Cross-cutting critique & synthesis

**The pattern across all directions:** every method needs (a) a *local learning
rule* that is closed-form or integer-cheap, and (b) a *credit-assignment scheme*
to stack rules into depth, and (c) an *architecture* whose units are natural for
bits. The families just mix these differently. STE fails because it forces all
three to impersonate continuous deep learning simultaneously.

**Three concrete composite candidates to prototype (ranked):**

1. **C-A "Counting Logic Nets"** = D5 + D8 + D6 front-end + D7 output.
   Thermometer/LBP binary front-end → sparse-connectivity LUT layers whose
   entries are filled by counting/majority against locally-propagated binary
   targets → HDC/ECOC integer classifier head. *Entirely gradient-free,
   one-to-few passes, integer-only.* Risk: target propagation quality.

2. **C-B "Integer-Bop"** = D1 + D6 + quantized backward.
   Ternary weights, hysteresis counters (8-bit), 1-bit backpropagated error
   signs (sign-feedback-alignment), median-threshold units, ECOC head. *Closest
   to known-good BNN accuracy; every tensor in training is ≤8-bit integer.*
   Risk: 1-bit backward signal too noisy at depth → ablate error bit-width.

3. **C-C "Boosted binary blocks"** = D9 + D5 + D3.
   Grow the network block-by-block with quantized boosting; each block trained
   by closed-form counting + local bit-flip polish. Risk: depth/compositionality.

**Kill-criteria discipline:** each candidate gets a 1-day MNIST prototype; drop
anything <97% MNIST or with FP ops in the inner loop. Fashion-MNIST gates entry
to CIFAR-10. Baselines to beat/report: FP LeNet/ResNet-20, STE-BNN, Bop,
diff-logic-net numbers from literature.

**What we measure (per §01):** accuracy, training bit-ops & bytes of optimizer
state per weight, inference bit-ops — reported together, always.
