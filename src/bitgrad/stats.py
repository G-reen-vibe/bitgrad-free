"""Multi-seed aggregation statistics (no scipy on this box; small t-table inlined)."""
from __future__ import annotations

import math

# Two-sided 95% Student-t critical values for df = 1..30.
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    return _T95[df - 1] if df <= 30 else 1.96


def mean_ci(values):
    """Return (mean, std, ci95_halfwidth) with sample std (ddof=1)."""
    n = len(values)
    mu = sum(values) / n
    if n == 1:
        return mu, 0.0, float("nan")
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    return mu, sd, t95(n - 1) * sd / math.sqrt(n)


def fmt_pct(mu, ci):
    if math.isnan(ci):
        return f"{100 * mu:.2f} (1 seed)"
    return f"{100 * mu:.2f} ± {100 * ci:.2f}"
