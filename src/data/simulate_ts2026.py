"""Simulate 2 x 50 time series (T=100) and write them to ts/ts2026 as CSV.

Batch 1: pure AR(p) or MA(q), order 1-3, random coefficients.
Batch 2: ARIMA(p, 0, q) with p in 1-3, q in 1-2, random coefficients.

Usage: uv run python src/data/simulate_ts2026.py [outdir]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess

T = 100
N = 50
SEED = 2026
BURN = 200  # burn-in so the series starts in the stationary regime
OUT = Path(__file__).resolve().parents[2] / "ts" / "ts2026"


def is_valid(phi, theta, margin=0.05):
    """True if the ARMA process is stationary and invertible.

    Uses statsmodels' checks and additionally requires all roots to lie at
    least `margin` outside the unit circle (avoids near-unit-root series).
    """
    proc = ArmaProcess(np.r_[1, -phi], np.r_[1, theta])
    if not (proc.isstationary and proc.isinvertible):
        return False
    roots = np.r_[proc.arroots, proc.maroots]
    return bool(np.all(np.abs(roots) > 1 + margin))


def draw_coefs(rng, p, q):
    """Rejection-sample AR/MA coefficients until the process is valid."""
    while True:
        phi = rng.uniform(-0.9, 0.9, size=p)
        theta = rng.uniform(-0.9, 0.9, size=q)
        if is_valid(phi, theta):
            return phi, theta


def simulate(rng, phi, theta, sigma):
    proc = ArmaProcess(np.r_[1, -phi], np.r_[1, theta])
    return proc.generate_sample(
        nsample=T, scale=sigma, distrvs=rng.standard_normal, burnin=BURN
    )


def main(out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    meta = []

    for batch in (1, 2):
        for i in range(1, N + 1):
            sigma = float(rng.uniform(0.5, 2.0))
            if batch == 1:
                kind = "AR" if rng.random() < 0.5 else "MA"
                order = int(rng.integers(1, 4))
                p, q = (order, 0) if kind == "AR" else (0, order)
                phi, theta = draw_coefs(rng, p, q)
            else:
                kind = "ARIMA"
                p, q = int(rng.integers(1, 4)), int(rng.integers(1, 3))
                phi, theta = draw_coefs(rng, p, q)

            y = simulate(rng, phi, theta, sigma)
            name = f"batch{batch}_{i:02d}"
            pd.DataFrame({"t": np.arange(1, T + 1), "y": y}).to_csv(
                out / f"{name}.csv", index=False
            )
            meta.append(
                {
                    "file": f"{name}.csv",
                    "batch": batch,
                    "model": kind,
                    "p": p,
                    "d": 0,
                    "q": q,
                    "ar_coefs": " ".join(f"{c:.4f}" for c in phi),
                    "ma_coefs": " ".join(f"{c:.4f}" for c in theta),
                    "sigma": round(sigma, 4),
                }
            )

    pd.DataFrame(meta).to_csv(out / "parameters.csv", index=False)
    print(f"Wrote {len(meta)} series + parameters.csv to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUT)
