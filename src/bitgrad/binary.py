"""Binary layers (the baselines we are arguing against, implemented faithfully).

Two weight modes:
  * 'ste' : latent FP32 weights, forward = sign(w), backward = clipped straight-through
            (BinaryConnect / BNN style, Courbariaux et al. 2015/2016).
  * 'raw' : weights are stored directly as +-1 and used as-is; gradients flow straight
            through (identity). This is what the Bop optimizer (Helwegen et al. 2019)
            expects: no latent weights, the optimizer decides flips.

Activation binarization = sign with hardtanh STE (gradient passes iff |x| <= 1).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn


def sign_pm1(x):
    """sign() mapping 0 -> +1 so outputs are exactly {-1, +1}."""
    return jnp.where(x >= 0, 1.0, -1.0)


def binarize_ste(x):
    """Forward: sign(x). Backward: identity clipped to |x| <= 1 (hardtanh STE)."""
    clipped = jnp.clip(x, -1.0, 1.0)
    return clipped + jax.lax.stop_gradient(sign_pm1(x) - clipped)


class BinaryDense(nn.Module):
    features: int
    mode: str = "ste"  # 'ste' | 'raw'
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", nn.initializers.glorot_normal(),
                            (x.shape[-1], self.features))
        w = binarize_ste(kernel) if self.mode == "ste" else kernel
        y = x @ w
        if self.use_bias:
            y = y + self.param("bias", nn.initializers.zeros, (self.features,))
        return y


class BinaryConv(nn.Module):
    features: int
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)
    mode: str = "ste"

    @nn.compact
    def __call__(self, x):
        kh, kw = self.kernel_size
        kernel = self.param("kernel", nn.initializers.glorot_normal(),
                            (kh, kw, x.shape[-1], self.features))
        w = binarize_ste(kernel) if self.mode == "ste" else kernel
        return jax.lax.conv_general_dilated(
            x, w, window_strides=self.strides, padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"))


def binarize_params_pm1(params):
    """Project all binary-layer kernels to exactly +-1 (init for 'raw'/Bop mode)."""
    def f(path, leaf):
        names = [getattr(p, "key", getattr(p, "name", "")) for p in path]
        if any(str(n).startswith(("BinaryDense", "BinaryConv")) for n in names) and \
           str(names[-1]) == "kernel":
            return sign_pm1(leaf)
        return leaf
    return jax.tree_util.tree_map_with_path(f, params)


def is_binary_kernel_path(path) -> bool:
    names = [str(getattr(p, "key", getattr(p, "name", ""))) for p in path]
    return any(n.startswith(("BinaryDense", "BinaryConv")) for n in names) and \
        names[-1] == "kernel"


def clip_latent(params):
    """Clip latent weights of STE binary layers to [-1, 1] (BinaryConnect rule)."""
    def f(path, leaf):
        return jnp.clip(leaf, -1.0, 1.0) if is_binary_kernel_path(path) else leaf
    return jax.tree_util.tree_map_with_path(f, params)
