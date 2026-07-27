"""Optimizers and training-cost accounting.

Bop (Helwegen et al., NeurIPS 2019, "Latent Weights Do Not Exist"): weights live in
{-1,+1}; keep a gradient EMA m; flip w when |m| > tau AND sign(m) == sign(w)
(gradient consistently says the current sign increases loss). No latent weights.
Implemented as an optax GradientTransformation producing additive updates
u = -2w at flip positions (so w + u = -w).

Cost accounting: we report *training state bits per weight* — the honest metric
that STE pipelines fail (latent FP32 + Adam moments = 96 bits/weight).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from .binary import is_binary_kernel_path


class BopState(NamedTuple):
    m: optax.Updates


def bop(gamma: float = 1e-3, tau: float = 1e-6) -> optax.GradientTransformation:
    def init_fn(params):
        return BopState(m=jax.tree_util.tree_map(jnp.zeros_like, params))

    def update_fn(updates, state, params):
        m = jax.tree_util.tree_map(
            lambda mo, g: (1.0 - gamma) * mo + gamma * g, state.m, updates)

        def flip_update(mi, w):
            flip = (jnp.abs(mi) > tau) & (jnp.sign(mi) == jnp.sign(w))
            return jnp.where(flip, -2.0 * w, 0.0)

        new_updates = jax.tree_util.tree_map(flip_update, m, params)
        return new_updates, BopState(m=m)

    return optax.GradientTransformation(init_fn, update_fn)


def make_optimizer(method: str, lr: float, bop_gamma: float = 1e-3,
                   bop_tau: float = 1e-6) -> optax.GradientTransformation:
    """fp32/bc/bnn -> Adam on everything.
    bop -> Bop on binary kernels, Adam on the rest (BN scales/shifts)."""
    if method != "bop":
        return optax.adam(lr)

    def label_tree(params):
        return jax.tree_util.tree_map_with_path(
            lambda path, _: "bin" if is_binary_kernel_path(path) else "other", params)

    return optax.multi_transform(
        {"bin": bop(bop_gamma, bop_tau), "other": optax.adam(lr)}, label_tree)


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------

def cost_report(params, method: str) -> dict:
    leaves = jax.tree_util.tree_leaves_with_path(params)
    n_bin = sum(l.size for p, l in leaves if is_binary_kernel_path(p))
    n_other = sum(l.size for p, l in leaves if not is_binary_kernel_path(p))
    if method == "fp32":
        n_bin, n_other = 0, n_bin + n_other

    weight_bits = {  # storage of the weight itself during training
        "fp32": 32, "bc": 32, "bnn": 32,  # latent FP32 shadow weights
        "bop": 1,
    }[method]
    opt_bits = {  # optimizer state per *binary-layer* weight
        "fp32": 64, "bc": 64, "bnn": 64,  # Adam: two FP32 moments
        "bop": 32,                        # one FP32 EMA (future work: 8-bit int)
    }[method]
    infer_bits = 32 if method == "fp32" else 1

    return {
        "n_params_binary_layers": int(n_bin),
        "n_params_other": int(n_other),
        "train_state_bits_per_weight": weight_bits + opt_bits,
        "inference_weight_bits": infer_bits,
        "model_size_bits": int(n_bin * infer_bits + n_other * 32),
    }
