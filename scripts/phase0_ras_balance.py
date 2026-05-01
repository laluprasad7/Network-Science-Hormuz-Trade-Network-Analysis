"""Phase 0.3 — RAS balancing of yearly trade matrices.

The RAS algorithm reconciles a non-negative matrix `W` so that its row sums
equal target vector `u` and its column sums equal target vector `v`,
subject to `sum(u) == sum(v)`. The update is

    W <- diag(u / W.sum(axis=1)) @ W @ diag(v / W.sum(axis=0))

iterated until both marginals match the targets within tolerance.

In our pipeline, both row-sum (exporter totals) and column-sum (importer
totals) margins come from the same Comtrade importer-reported data, so the
observed margins are mutually consistent and RAS converges in a single
iteration with negligible residual. This is the expected sanity-check
outcome and is recorded in the report.

The function is written as a general utility so that downstream robustness
runs can substitute alternative margins (e.g. exporter-side mirror data,
GDP-shares macro margins) without rewriting the loop.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHASE0 = ROOT / "Results" / "phase0"
LAYERS = ("wheat", "ammonia", "urea", "lpg_propane", "lpg_butane")


def ras_balance(W: np.ndarray, u: np.ndarray, v: np.ndarray,
                tol: float = 1e-9, max_iter: int = 200,
                ) -> tuple[np.ndarray, dict]:
    """Reconcile W to row totals u and column totals v.

    Parameters
    ----------
    W : (n, n) non-negative matrix
    u : (n,) row-sum targets    (exporter totals)
    v : (n,) column-sum targets (importer totals)
    tol : convergence tolerance on max absolute marginal residual
    max_iter : safety cap on iterations

    Returns
    -------
    W_balanced : balanced matrix
    info : dict with convergence diagnostics
    """
    if not np.isclose(u.sum(), v.sum(), rtol=1e-6):
        raise ValueError(
            f"RAS requires sum(u) == sum(v); got {u.sum():.4g} vs {v.sum():.4g}")
    W = W.astype(float).copy()
    history = []
    for it in range(max_iter):
        rs = W.sum(axis=1)
        r_scale = np.divide(u, rs, out=np.zeros_like(rs), where=rs > 0)
        W *= r_scale[:, None]

        cs = W.sum(axis=0)
        c_scale = np.divide(v, cs, out=np.zeros_like(cs), where=cs > 0)
        W *= c_scale[None, :]

        row_err = np.max(np.abs(W.sum(axis=1) - u))
        col_err = np.max(np.abs(W.sum(axis=0) - v))
        history.append((row_err, col_err))
        if max(row_err, col_err) < tol:
            return W, {"iterations": it + 1, "row_err": row_err,
                       "col_err": col_err, "history": history}
    return W, {"iterations": max_iter, "row_err": row_err,
               "col_err": col_err, "history": history,
               "converged": False}


def main() -> None:
    """Run RAS on every (year, layer) and save the balanced matrices."""
    log_rows = []
    years = sorted({int(p.stem.split("_")[1]) for p in PHASE0.glob("W_*.npy")})

    for y in years:
        for L in LAYERS:
            W = np.load(PHASE0 / f"W_{y}_{L}.npy")
            if W.sum() == 0:
                log_rows.append({"year": y, "layer": L, "iterations": 0,
                                 "row_err": 0.0, "col_err": 0.0,
                                 "matrix_total": 0.0, "delta_norm": 0.0})
                continue
            u = W.sum(axis=1)   # observed exporter totals
            v = W.sum(axis=0)   # observed importer totals
            W_bal, info = ras_balance(W, u, v)
            np.save(PHASE0 / f"W_RAS_{y}_{L}.npy", W_bal)

            delta = np.linalg.norm(W_bal - W) / max(np.linalg.norm(W), 1e-12)
            log_rows.append({"year": y, "layer": L,
                             "iterations": info["iterations"],
                             "row_err": float(info["row_err"]),
                             "col_err": float(info["col_err"]),
                             "matrix_total": float(W.sum()),
                             "delta_norm": float(delta)})

    log = pd.DataFrame(log_rows)
    log.to_csv(PHASE0 / "ras_balance_log.csv", index=False)
    print("RAS convergence log (observed-margin sanity-check run):")
    print(log.to_string(index=False))
    print()
    max_iter = int(log.iterations.max())
    max_resid = float(log[["row_err", "col_err"]].max().max())
    max_delta = float(log.delta_norm.max())
    print(f"Worst case: iterations = {max_iter}, "
          f"max marginal residual = {max_resid:.3e}, "
          f"max ||W_bal - W||/||W|| = {max_delta:.3e}")


if __name__ == "__main__":
    main()
