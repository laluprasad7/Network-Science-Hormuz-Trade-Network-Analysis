"""Phase 0 — build yearly directed weighted trade matrices and dependency shares.

For each (year, layer) we produce two matrices indexed by a unified country set:

    W[i, j] = USD flow from exporter i to importer j during year `year`
    A[i, j] = W[i, j] / sum_i W[i, j]  (importer j's dependency share on i)

A is column-stochastic on importers that have any imports of the layer; columns
for importers with zero observed imports are left as zero so we never invent
flows. Both matrices are saved as long-format CSV edge lists keyed on
(year, layer, exporter, importer) so downstream code can rebuild a sparse
matrix without loading the dense object.

We also save the per-importer total-import margins (used as RAS column targets)
and the per-exporter total-export margins (RAS row targets).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "Processed_Data"
OUT = ROOT / "Results" / "phase0"
OUT.mkdir(parents=True, exist_ok=True)


HORMUZ = {"ARE", "BHR", "IRN", "IRQ", "KWT", "QAT", "SAU"}
LAYERS = ("wheat", "ammonia", "urea", "lpg_propane", "lpg_butane")


def load_bilateral_yearly() -> pd.DataFrame:
    """Aggregate the monthly bilateral panel to yearly USD flows."""
    df = pd.read_csv(PROC / "comtrade_bilateral_monthly.csv")
    g = (df.groupby(["refYear", "layer", "partnerISO", "reporterISO"])
           ["primaryValue"].sum().reset_index()
           .rename(columns={"partnerISO": "exporter",
                            "reporterISO": "importer",
                            "primaryValue": "value_usd"}))
    return g


def load_world_yearly() -> pd.DataFrame:
    """Reporter-side world totals — used as importer-side RAS column targets."""
    df = pd.read_csv(PROC / "comtrade_world_imports_monthly.csv")
    g = (df.groupby(["refYear", "layer", "reporterISO"])
           ["primaryValue"].sum().reset_index()
           .rename(columns={"reporterISO": "importer",
                            "primaryValue": "world_total_imports_usd"}))
    return g


def country_index(bilat: pd.DataFrame) -> list[str]:
    """Unified country index: union of all exporters and importers."""
    countries = sorted(set(bilat["exporter"]).union(bilat["importer"]))
    return countries


def yearly_matrices(bilat: pd.DataFrame, year: int, layer: str,
                    countries: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (W, A) for one (year, layer) on the given country index."""
    sub = bilat[(bilat.refYear == year) & (bilat.layer == layer)]
    n = len(countries)
    idx = {c: k for k, c in enumerate(countries)}
    W = np.zeros((n, n), dtype=float)
    for _, r in sub.iterrows():
        i, j = idx[r.exporter], idx[r.importer]
        W[i, j] += r.value_usd

    col_sum = W.sum(axis=0)
    A = np.zeros_like(W)
    nz = col_sum > 0
    A[:, nz] = W[:, nz] / col_sum[nz]
    return W, A


def hormuz_share_by_year(bilat: pd.DataFrame, layers: tuple[str, ...]) -> pd.DataFrame:
    """Hormuz exporters' share of global flow value, per year per layer."""
    df = bilat.assign(is_hormuz=lambda d: d.exporter.isin(HORMUZ))
    out = (df.groupby(["refYear", "layer", "is_hormuz"])
             ["value_usd"].sum().unstack("is_hormuz", fill_value=0.0))
    out.columns = ["non_hormuz_usd", "hormuz_usd"]
    out["hormuz_share"] = out["hormuz_usd"] / (out["hormuz_usd"] + out["non_hormuz_usd"])
    return out.reset_index()


def main() -> None:
    bilat = load_bilateral_yearly()
    world = load_world_yearly()
    countries = country_index(bilat)
    print(f"Years in panel  : {sorted(bilat.refYear.unique())}")
    print(f"Layers          : {sorted(bilat.layer.unique())}")
    print(f"Country index   : {len(countries)} (union of importers and exporters)")

    bilat.to_csv(OUT / "trade_yearly_edges.csv", index=False)
    world.to_csv(OUT / "trade_yearly_world_margins.csv", index=False)
    pd.Series(countries, name="iso3").to_csv(OUT / "country_index.csv", index=False)

    rows = []
    for y in sorted(bilat.refYear.unique()):
        for L in LAYERS:
            W, A = yearly_matrices(bilat, y, L, countries)
            np.save(OUT / f"W_{y}_{L}.npy", W)
            np.save(OUT / f"A_{y}_{L}.npy", A)
            rows.append({"year": y, "layer": L,
                         "n_edges": int((W > 0).sum()),
                         "total_value_usd": float(W.sum()),
                         "n_importers_active": int((W.sum(axis=0) > 0).sum()),
                         "n_exporters_active": int((W.sum(axis=1) > 0).sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "trade_matrix_summary.csv", index=False)
    print()
    print("Per (year, layer) matrix summary:")
    print(summary.to_string(index=False))

    hs = hormuz_share_by_year(bilat, LAYERS)
    hs.to_csv(OUT / "hormuz_share_by_year_layer.csv", index=False)
    print()
    print("Hormuz exporter share of global trade value by year x layer:")
    print(hs.pivot(index="refYear", columns="layer", values="hormuz_share").round(4).to_string())


if __name__ == "__main__":
    main()
