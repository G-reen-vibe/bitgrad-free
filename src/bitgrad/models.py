"""Model zoo, sized for a 1-core / 3 GB budget.

Methods (per model family):
  fp32 : full-precision reference (accuracy anchor)
  bc   : BinaryConnect — binary weights via STE, FP activations
  bnn  : binary weights AND binary activations via STE (Courbariaux/Hubara BNN)
  bop  : same architecture as bnn, but 'raw' +-1 weights trained by the Bop optimizer

Architectural notes for the binary regime (standard practice from the BNN
literature): BatchNorm after every binary conv/dense (it absorbs scale and turns
into an integer threshold at inference), binarize activations *after* BN, first
and last layers kept FP in 'bc'/'bnn' only in the sense that input images are
continuous and logits are BN-scaled (we binarize all weight layers, which is the
harder, honest setting).

CNN sizes are compute-matched to our server, not to paper-scale VGG-Small; the
results table therefore also quotes literature numbers for paper-scale context.
"""
from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp

from .binary import BinaryConv, BinaryDense, binarize_ste


def _act(x, binary_act: bool):
    return binarize_ste(x) if binary_act else nn.relu(x)


class MLP(nn.Module):
    """784-256-256-10 MLP (mnist/fashion_mnist)."""
    method: str = "fp32"
    hidden: int = 256
    n_classes: int = 10

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.reshape((x.shape[0], -1))
        binary_w = self.method in ("bc", "bnn", "bop")
        binary_a = self.method in ("bnn", "bop")
        mode = "raw" if self.method == "bop" else "ste"
        for _ in range(2):
            x = (BinaryDense(self.hidden, mode=mode)(x) if binary_w
                 else nn.Dense(self.hidden, use_bias=False)(x))
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = _act(x, binary_a)
        x = (BinaryDense(self.n_classes, mode=mode)(x) if binary_w
             else nn.Dense(self.n_classes, use_bias=False)(x))
        x = nn.BatchNorm(use_running_average=not train)(x)
        return x


class ConvNet(nn.Module):
    """Small VGG-style net: [C(w) C(w) pool] x len(widths) blocks -> GAP -> dense.

    widths=(32, 64) is the mnist tier (~90k params);
    widths=(64, 128, 256) is the cifar tier (~1.2M params, heavier).
    """
    method: str = "fp32"
    widths: tuple = (32, 64)
    n_classes: int = 10

    @nn.compact
    def __call__(self, x, train: bool = True):
        binary_w = self.method in ("bc", "bnn", "bop")
        binary_a = self.method in ("bnn", "bop")
        mode = "raw" if self.method == "bop" else "ste"

        def conv(x, w):
            return (BinaryConv(w, (3, 3), mode=mode)(x) if binary_w
                    else nn.Conv(w, (3, 3), use_bias=False)(x))

        for w in self.widths:
            x = conv(x, w)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = _act(x, binary_a)
            x = conv(x, w)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = _act(x, binary_a)
            x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = jnp.mean(x, axis=(1, 2))  # global average pool
        x = (BinaryDense(self.n_classes, mode=mode)(x) if binary_w
             else nn.Dense(self.n_classes, use_bias=False)(x))
        x = nn.BatchNorm(use_running_average=not train)(x)
        return x


def make_model(arch: str, method: str):
    if arch == "mlp":
        return MLP(method=method)
    if arch == "cnn_s":
        return ConvNet(method=method, widths=(32, 64))
    if arch == "cnn_m":
        return ConvNet(method=method, widths=(64, 128, 256))
    raise ValueError(arch)
