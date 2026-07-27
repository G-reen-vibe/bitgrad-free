"""Dataset loading for bitgrad-free.

Constraints of our environment:
  * egress proxy only allows github.com / raw.githubusercontent.com / codeload.github.com
    (canonical dataset hosts are blocked), so we pull from well-known GitHub mirrors;
  * 3 GB RAM -> datasets are kept as uint8 in memory and normalized per-batch;
  * ephemeral filesystem -> everything is cached to data/*.npz and re-fetchable.

Datasets: mnist, fashion_mnist, cifar10. All return uint8 images in NHWC and int labels.
"""
from __future__ import annotations

import gzip
import io
import os
import struct
import tarfile
import urllib.request

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

_IDX_MIRRORS = {
    "mnist": "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "fashion_mnist": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
}
_IDX_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
_CIFAR_TARBALL = "https://codeload.github.com/YoongiKim/CIFAR-10-images/tar.gz/refs/heads/master"


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def _parse_idx_images(raw: bytes) -> np.ndarray:
    magic, n, rows, cols = struct.unpack(">IIII", raw[:16])
    assert magic == 2051, f"bad image magic {magic}"
    return np.frombuffer(raw, np.uint8, offset=16).reshape(n, rows, cols, 1)


def _parse_idx_labels(raw: bytes) -> np.ndarray:
    magic, n = struct.unpack(">II", raw[:8])
    assert magic == 2049, f"bad label magic {magic}"
    return np.frombuffer(raw, np.uint8, offset=8).astype(np.int32)


def _build_idx_dataset(name: str, path: str) -> None:
    base = _IDX_MIRRORS[name]
    arrays = {}
    for key, fname in _IDX_FILES.items():
        raw = gzip.decompress(_fetch(base + fname))
        arrays[key] = _parse_idx_images(raw) if key.endswith("x") else _parse_idx_labels(raw)
    np.savez_compressed(path, **arrays)


def _build_cifar10(path: str) -> None:
    """Build CIFAR-10 from the YoongiKim/CIFAR-10-images GitHub mirror.

    CAVEAT (documented in docs/04): this mirror stores images as JPEGs, i.e. a lossy
    re-encode of the canonical dataset (canonical hosts are blocked by our egress
    proxy; the only bit-exact GitHub mirrors use LFS/annex, also blocked). All
    methods AND baselines in this repo train/test on the same cache, so internal
    comparisons are unaffected; comparisons to literature numbers carry a small
    (<~0.5pp) dataset-shift asterisk. Labels = sorted class dirs, which matches the
    canonical CIFAR-10 label order since it is alphabetical."""
    from PIL import Image

    buf = io.BytesIO(_fetch(_CIFAR_TARBALL))
    splits = {"train": ([], []), "test": ([], [])}
    classes = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
    cls_idx = {c: i for i, c in enumerate(classes)}
    with tarfile.open(fileobj=buf, mode="r:gz") as tf:
        for member in tf:
            if not member.name.endswith(".jpg"):
                continue
            parts = member.name.split("/")  # repo/train/<class>/<i>.png
            split, cls = parts[1], parts[2]
            img = Image.open(tf.extractfile(member))
            splits[split][0].append(np.asarray(img, dtype=np.uint8))
            splits[split][1].append(cls_idx[cls])
    out = {}
    for split, (xs, ys) in splits.items():
        key = "train" if split == "train" else "test"
        out[f"{key}_x"] = np.stack(xs)
        out[f"{key}_y"] = np.asarray(ys, np.int32)
    np.savez_compressed(path, **out)


def load(name: str):
    """Return (train_x, train_y, test_x, test_y); images uint8 NHWC."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.npz")
    if not os.path.exists(path):
        print(f"[data] building {name} cache ...", flush=True)
        if name in _IDX_MIRRORS:
            _build_idx_dataset(name, path)
        elif name == "cifar10":
            _build_cifar10(path)
        else:
            raise ValueError(name)
    z = np.load(path)
    return z["train_x"], z["train_y"], z["test_x"], z["test_y"]


def normalize(x_u8: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float32 [-1, 1] (done per-batch to respect the RAM budget)."""
    return x_u8.astype(np.float32) / 127.5 - 1.0


def augment_cifar(x_u8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Standard light augmentation: random horizontal flip + pad-4 random crop (uint8 in/out)."""
    n, h, w, c = x_u8.shape
    flip = rng.random(n) < 0.5
    x = x_u8.copy()
    x[flip] = x[flip, :, ::-1]
    padded = np.pad(x, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
    out = np.empty_like(x_u8)
    offs = rng.integers(0, 9, size=(n, 2))
    for i in range(n):  # per-sample crop; cheap relative to the train step on 1 core
        r, cc = offs[i]
        out[i] = padded[i, r:r + h, cc:cc + w]
    return out


def batches(x, y, batch_size, rng: np.random.Generator, aug=False, shuffle=True):
    idx = rng.permutation(len(x)) if shuffle else np.arange(len(x))
    for s in range(0, len(idx) - batch_size + 1 if shuffle else len(idx), batch_size):
        b = idx[s:s + batch_size]
        xb = x[b]
        if aug:
            xb = augment_cifar(xb, rng)
        yield normalize(xb), y[b]


if __name__ == "__main__":
    import sys
    names = ["mnist", "fashion_mnist", "cifar10"] if "--all" in sys.argv else sys.argv[1:]
    for n in names:
        tx, ty, sx, sy = load(n)
        print(n, tx.shape, tx.dtype, ty.shape, "| test", sx.shape, "| classes",
              int(ty.max()) + 1, flush=True)
