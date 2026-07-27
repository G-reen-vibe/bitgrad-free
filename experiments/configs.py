"""Experiment registry: every (benchmark x baseline) cell of the evaluation grid.

Naming: <dataset>_<arch>_<method>.
Epoch budgets are sized for a single CPU core; see docs/04-eval-protocol.md for the
protocol (seeds, CI reporting, literature context numbers).
"""

_BASE = {
    "batch_size": 100,
    "lr": 1e-3,
    "bop_gamma": 1e-3,
    "bop_tau": 1e-6,
}

CONFIGS = {}


def _add(dataset, arch, method, epochs, **kw):
    name = f"{dataset}_{arch}_{method}"
    CONFIGS[name] = {**_BASE, "dataset": dataset, "arch": arch, "method": method,
                     "epochs": epochs, **kw}


for _m in ("fp32", "bc", "bnn", "bop"):
    _add("mnist", "mlp", _m, epochs=15)
    _add("fashion_mnist", "mlp", _m, epochs=15)
    _add("fashion_mnist", "cnn_s", _m, epochs=15)
    _add("cifar10", "cnn_s", _m, epochs=12, augment=True)   # ~2h/seed on 1 core
    _add("cifar10", "cnn_m", _m, epochs=30, augment=True)   # aspirational; needs bigger box

SEED_DEFAULTS = {  # more seeds where runs are cheap
    "mnist": [0, 1, 2, 3, 4],
    "fashion_mnist": [0, 1, 2, 3, 4],
    "cifar10": [0, 1, 2],
}
