"""Phase 1 — exposure metrics.

For every (year, layer) pair we compute eight country-level scores measuring
exposure to the Hormuz set H = {ARE, BHR, IRN, IRQ, KWT, QAT, SAU}:

    DE                Direct Exposure: column-sum of A over Hormuz rows.
    SR                Systemic Risk:   DE * total imports of layer (USD).
    betweenness       directed-weighted betweenness centrality on the
                       trade graph (cost = 1 / value, weighting frequent
                       large flows as cheap edges).
    katz              Katz centrality with alpha = 0.85 / rho(A^T).
    ppr               personalized PageRank seeded uniformly on Hormuz.
    leontief          ((I - alpha A)^{-1} - I) propagated from Hormuz,
                       totals indirect + direct dependency on H per
                       importer (alpha = 0.95 / rho(A)).
    debtrank          Battiston-style discrete DebtRank: distress
                       propagation in U/D/I states starting from h0=1
                       on Hormuz; output is equilibrium per-country
                       distress.
    pivi              Sum_{m not in H, m != j} A[m,j] * sum_{h in H} A[h,m].
                       Indirect vulnerability through any non-Hormuz
                       intermediary.

Outputs (under Results/phase1/):
    exposure_long.csv     tidy: (year, layer, country, metric, value)
    exposure_wide.csv     pivot:  one row per (year, layer, country),
                                  one column per metric.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
PHASE0 = ROOT / "Results" / "phase0"
OUT = ROOT / "Results" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)

HORMUZ = ["ARE", "BHR", "IRN", "IRQ", "KWT", "QAT", "SAU"]
LAYERS = ("wheat", "ammonia", "urea", "lpg_propane", "lpg_butane")


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def direct_exposure(A: np.ndarray, h_idx: list[int]) -> np.ndarray:
    """DE_j = sum_{h in H} A[h, j]: importer j's share-weighted Hormuz reliance."""
    return A[h_idx, :].sum(axis=0)


def systemic_risk(de: np.ndarray, W: np.ndarray) -> np.ndarray:
    """SR_j = DE_j * total imports of j (USD).  Scale-aware vulnerability."""
    return de * W.sum(axis=0)


def betweenness_directed(W: np.ndarray, names: list[str]) -> np.ndarray:
    """Directed weighted betweenness; edge cost = 1/value (frequent flow = cheap)."""
    G = nx.DiGraph()
    G.add_nodes_from(names)
    rows, cols = np.nonzero(W)
    for i, j in zip(rows, cols):
        G.add_edge(names[i], names[j], weight=1.0 / W[i, j])
    bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
    return np.array([bc[n] for n in names])


def katz_centrality(A: np.ndarray, alpha_frac: float = 0.85) -> np.ndarray:
    """Katz centrality on A^T with alpha = alpha_frac / spectral_radius(A^T).

    Interpretation: long-path attenuated downstream vulnerability via the
    dependency matrix.  We use A^T because Katz aggregates inbound paths,
    and inbound on A^T = upstream-supplier paths to importers.
    """
    M = A.T
    eigs = np.abs(np.linalg.eigvals(M))
    rho = eigs.max()
    if rho < 1e-12:
        return np.zeros(M.shape[0])
    alpha = alpha_frac / rho
    n = M.shape[0]
    katz = np.linalg.solve(np.eye(n) - alpha * M, np.ones(n)) - 1.0
    return katz


def personalized_pagerank(W: np.ndarray, names: list[str],
                          h_idx: list[int]) -> np.ndarray:
    """Personalized PageRank seeded uniformly on Hormuz exporters.

    Edges of W go exporter -> importer; PPR scores reachability from H
    along directed paths (downstream diffusion of a Hormuz signal).
    """
    G = nx.DiGraph()
    G.add_nodes_from(names)
    rows, cols = np.nonzero(W)
    for i, j in zip(rows, cols):
        G.add_edge(names[i], names[j], weight=float(W[i, j]))
    p = {names[i]: (1.0 / len(h_idx) if i in h_idx else 0.0)
         for i in range(len(names))}
    pr = nx.pagerank(G, alpha=0.85, personalization=p, weight="weight",
                     max_iter=200, tol=1e-9)
    return np.array([pr[n] for n in names])


def leontief_indirect(A: np.ndarray, h_idx: list[int],
                      alpha_frac: float = 0.95) -> np.ndarray:
    """Indirect-plus-direct Hormuz dependency via damped Leontief inverse.

    With trade-share matrix A (column-stochastic on importers with imports),
    the geometric series sum_{k>=1} (alpha*A)^k = (I - alpha*A)^{-1} - I.
    Path interpretation: (A^k)_{h,j} = j's k-step upstream dependence on h.
    We sum over h in Hormuz to get total dependency of every importer j.
    The damping keeps the inverse well-defined when rho(A) = 1.
    """
    n = A.shape[0]
    eigs = np.abs(np.linalg.eigvals(A))
    rho = max(eigs.max(), 1e-12)
    alpha = alpha_frac / rho
    L = np.linalg.solve(np.eye(n) - alpha * A, np.eye(n))
    seed = np.zeros(n)
    seed[h_idx] = 1.0
    return seed @ L - seed


def debtrank(A: np.ndarray, h_idx: list[int]) -> np.ndarray:
    """Battiston-Puliga-Kaushik-Tasca-Caldarelli DebtRank.

    Each node holds distress h_i in [0,1] and a state in {U, D, I}.
    Initially Hormuz nodes are D with h=1, all others U with h=0.
    Each step:
       new_h[U] = min(1, old_h[U] + sum_{i in D} A[i, U] * old_h[i])
       D -> I,  any U with new_h > 0 -> D.
    Stops when D is empty.  Returns the per-country equilibrium distress.
    """
    n = A.shape[0]
    h = np.zeros(n)
    h[h_idx] = 1.0
    state = np.full(n, "U", dtype="<U1")
    state[h_idx] = "D"
    while np.any(state == "D"):
        d_mask = (state == "D")
        u_mask = (state == "U")
        shock = (h * d_mask) @ A          # (n,) inbound from currently-D
        new_h = h.copy()
        new_h[u_mask] = np.minimum(1.0, h[u_mask] + shock[u_mask])
        state[d_mask] = "I"
        newly_d = u_mask & (new_h > h)
        state[newly_d] = "D"
        h = new_h
    return h


def pivi(A: np.ndarray, h_idx: list[int]) -> np.ndarray:
    """Potential Indirect Vulnerability Index.

    PIVI_j = sum over non-Hormuz intermediaries m (m != j) of
              [importer-j's dependency on m] * [m's total dependency on Hormuz].
    Captures countries that look insulated directly but flow indirectly
    through a Hormuz-dependent supplier.
    """
    n = A.shape[0]
    is_hormuz = np.zeros(n, dtype=bool)
    is_hormuz[h_idx] = True
    m_dep_on_H = A[h_idx, :].sum(axis=0)        # length n: m's direct exposure
    out = np.zeros(n)
    for j in range(n):
        weights = A[:, j].copy()
        weights[is_hormuz] = 0.0
        weights[j] = 0.0
        out[j] = float(np.dot(weights, m_dep_on_H))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    countries = pd.read_csv(PHASE0 / "country_index.csv")["iso3"].tolist()
    h_idx = [countries.index(c) for c in HORMUZ if c in countries]
    print(f"Country index size : {len(countries)}")
    print(f"Hormuz indices     : {dict(zip([countries[i] for i in h_idx], h_idx))}")

    rows = []
    years = sorted({int(p.stem.split("_")[1])
                    for p in PHASE0.glob("W_*.npy")
                    if not p.stem.startswith("W_RAS")})
    for y in years:
        for L in LAYERS:
            W = np.load(PHASE0 / f"W_{y}_{L}.npy")
            A = np.load(PHASE0 / f"A_{y}_{L}.npy")

            de  = direct_exposure(A, h_idx)
            sr  = systemic_risk(de, W)
            bc  = betweenness_directed(W, countries)
            kz  = katz_centrality(A)
            ppr = personalized_pagerank(W, countries, h_idx)
            lt  = leontief_indirect(A, h_idx)
            dr  = debtrank(A, h_idx)
            pv  = pivi(A, h_idx)

            for k, c in enumerate(countries):
                rows.append({"year": y, "layer": L, "country": c,
                             "DE": de[k], "SR": sr[k],
                             "betweenness": bc[k], "katz": kz[k],
                             "ppr": ppr[k], "leontief": lt[k],
                             "debtrank": dr[k], "pivi": pv[k]})
            print(f"  done {y} {L}")

    wide = pd.DataFrame(rows)
    wide.to_csv(OUT / "exposure_wide.csv", index=False)

    long = wide.melt(id_vars=["year", "layer", "country"],
                     value_vars=["DE", "SR", "betweenness", "katz",
                                 "ppr", "leontief", "debtrank", "pivi"],
                     var_name="metric", value_name="value")
    long.to_csv(OUT / "exposure_long.csv", index=False)

    print()
    print(f"Wrote exposure_wide.csv ({wide.shape}) and exposure_long.csv ({long.shape}).")
    print()
    print("2024 LPG-butane top-10 by direct exposure:")
    sub = wide[(wide.year == 2024) & (wide.layer == "lpg_butane")]
    sub = sub[~sub.country.isin(HORMUZ)].nlargest(10, "DE")
    print(sub[["country", "DE", "SR", "katz", "ppr", "leontief",
               "debtrank", "pivi"]].to_string(index=False))


if __name__ == "__main__":
    main()
