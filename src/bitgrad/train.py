"""Generic training loop (jit-compiled, BN batch_stats handled, timed)."""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import data as D
from .binary import binarize_params_pm1, clip_latent
from .models import make_model
from .optim import cost_report, make_optimizer


def cross_entropy(logits, labels):
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


def train_one(cfg: dict, seed: int) -> dict:
    """Train a single (config, seed) run; returns a metrics dict."""
    rng_np = np.random.default_rng(seed)
    tx_u8, ty, sx_u8, sy = D.load(cfg["dataset"])
    model = make_model(cfg["arch"], cfg["method"])

    key = jax.random.PRNGKey(seed)
    dummy = D.normalize(tx_u8[:2])
    variables = model.init(key, jnp.asarray(dummy), train=True)
    params, batch_stats = variables["params"], variables.get("batch_stats", {})
    if cfg["method"] == "bop":
        params = binarize_params_pm1(params)

    opt = make_optimizer(cfg["method"], cfg["lr"],
                         cfg.get("bop_gamma", 1e-3), cfg.get("bop_tau", 1e-6))
    opt_state = opt.init(params)

    @jax.jit
    def train_step(params, batch_stats, opt_state, xb, yb):
        def loss_fn(p):
            out, upd = model.apply({"params": p, "batch_stats": batch_stats}, xb,
                                   train=True, mutable=["batch_stats"])
            return cross_entropy(out, yb), upd["batch_stats"]
        (loss, new_bs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        if cfg["method"] in ("bc", "bnn"):
            new_params = clip_latent(new_params)
        return new_params, new_bs, new_opt, loss

    @jax.jit
    def eval_step(params, batch_stats, xb):
        logits = model.apply({"params": params, "batch_stats": batch_stats}, xb,
                             train=False)
        return jnp.argmax(logits, -1)

    def evaluate(params, batch_stats, x_u8, y, limit=None):
        if limit:
            x_u8, y = x_u8[:limit], y[:limit]
        correct = 0
        bs = 500
        for s in range(0, len(x_u8), bs):
            xb = jnp.asarray(D.normalize(x_u8[s:s + bs]))
            correct += int((np.asarray(eval_step(params, batch_stats, xb))
                            == y[s:s + bs]).sum())
        return correct / len(y)

    aug = cfg.get("augment", False)
    max_steps = cfg.get("max_steps")           # smoke-mode cap
    eval_limit = cfg.get("eval_limit")         # smoke-mode cap
    history, step, t0 = [], 0, time.time()
    for epoch in range(cfg["epochs"]):
        for xb, yb in D.batches(tx_u8, ty, cfg["batch_size"], rng_np, aug=aug):
            params, batch_stats, opt_state, loss = train_step(
                params, batch_stats, opt_state, jnp.asarray(xb), jnp.asarray(yb))
            step += 1
            if max_steps and step >= max_steps:
                break
        acc = evaluate(params, batch_stats, sx_u8, sy, eval_limit)
        history.append(acc)
        print(f"  [seed {seed}] epoch {epoch + 1}/{cfg['epochs']} "
              f"test_acc={acc:.4f} loss={float(loss):.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if max_steps and step >= max_steps:
            break

    wall = time.time() - t0
    return {
        "seed": seed,
        "final_acc": history[-1],
        "best_acc": max(history),
        "history": history,
        "wall_seconds": wall,
        "steps": step,
        "steps_per_sec": step / wall,
        **cost_report(params, cfg["method"]),
    }
