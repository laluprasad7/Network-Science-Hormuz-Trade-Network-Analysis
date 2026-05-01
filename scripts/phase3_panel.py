"""Phase 3.0 — Build the monthly econometric panel.

Sources joined:
    Comtrade bilateral monthly   → country-level monthly import value per layer
    CMO Pink Sheet               → global commodity benchmark prices
                                     wheat     → wheat_us_hrw  ($/mt)
                                     urea      → urea           ($/mt)
                                     ammonia   → dap            ($/mt, fertilizer proxy)
                                     lpg_propane → FRED propane ($/gallon)
                                     lpg_butane  → LNG Japan    ($/mmbtu)
    FRED propane monthly         → propane price (backfilled with LNG Japan index
                                     for 2020-01 to 2021-02, where FRED is absent)
    PortWatch Hormuz monthly     → n_tanker, capacity_tanker  (monthly tanker DWT)
    Phase 1 yearly exposure      → DE, composite_01 per (country, year, layer)

Output:
    Results/phase3/panel_monthly.csv
        country, year, month, period_dt, layer,
        import_value_usd,          log_import,
        price_usd,                 log_price,
        hormuz_tanker_n,           hormuz_tanker_DWT,
        hormuz_DWT_norm,           hormuz_tanker_shock,  (month-on-month DWT % change)
        DE, composite_01

    Results/phase3/price_panel.csv
        period_dt, layer, price_usd, log_price   (global monthly prices)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "Processed_Data"
PHASE1 = ROOT / "Results" / "phase1"
OUT = ROOT / "Results" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

HORMUZ = ["ARE", "BHR", "IRN", "IRQ", "KWT", "QAT", "SAU"]
LAYERS = ("wheat", "ammonia", "urea", "lpg_propane", "lpg_butane")


# ---------------------------------------------------------------------------
# 1. Monthly country-level import aggregates from Comtrade
# ---------------------------------------------------------------------------

def load_import_panel() -> pd.DataFrame:
    df = pd.read_csv(PROC / "comtrade_bilateral_monthly.csv",
                     usecols=["refYear", "refMonth", "reporterISO",
                               "partnerISO", "layer", "primaryValue"])
    df = df[~df.reporterISO.isin(HORMUZ)]
    agg = (df.groupby(["refYear", "refMonth", "reporterISO", "layer"],
                      as_index=False)["primaryValue"].sum()
             .rename(columns={"refYear": "year", "refMonth": "month",
                               "reporterISO": "country",
                               "primaryValue": "import_value_usd"}))
    agg["period_dt"] = pd.to_datetime(
        agg["year"].astype(str) + "-" + agg["month"].astype(str).str.zfill(2) + "-01")
    agg = agg[agg["import_value_usd"] > 0].copy()
    return agg


# ---------------------------------------------------------------------------
# 2. Global commodity price series (one per layer, monthly)
# ---------------------------------------------------------------------------

def load_prices() -> pd.DataFrame:
    cmo = pd.read_csv(PROC / "cmo_monthly_prices.csv",
                      usecols=["period_dt", "wheat_us_hrw", "urea",
                                "dap", "liquefied_natural_gas_japan"])
    cmo["period_dt"] = pd.to_datetime(cmo["period_dt"])

    fred = pd.read_csv(PROC / "fred_propane_monthly.csv",
                       usecols=["period_dt", "propane_mean"])
    fred["period_dt"] = pd.to_datetime(fred["period_dt"])

    m = cmo.merge(fred, on="period_dt", how="left")
    # Fill early propane (2020-01 to 2021-02) with LNG Japan index scaled to
    # a 2021 overlap mean so the series is continuous in log-changes.
    mask = m["propane_mean"].isna() & (m["period_dt"] >= "2020-01-01")
    if mask.any():
        overlap = m[(m.period_dt >= "2021-03-01") & (m.period_dt <= "2021-12-31")]
        scale = (overlap["propane_mean"].mean() /
                 overlap["liquefied_natural_gas_japan"].mean())
        m.loc[mask, "propane_mean"] = (
            m.loc[mask, "liquefied_natural_gas_japan"] * scale)

    layer_col = {
        "wheat":       "wheat_us_hrw",
        "ammonia":     "dap",
        "urea":        "urea",
        "lpg_propane": "propane_mean",
        "lpg_butane":  "liquefied_natural_gas_japan",
    }
    rows = []
    for layer, col in layer_col.items():
        tmp = m[["period_dt", col]].copy()
        tmp.columns = ["period_dt", "price_usd"]
        tmp["layer"] = layer
        rows.append(tmp)
    prices = pd.concat(rows, ignore_index=True)
    prices = prices.dropna(subset=["price_usd"])
    prices["log_price"] = np.log(prices["price_usd"])
    return prices


# ---------------------------------------------------------------------------
# 3. PortWatch Hormuz monthly tanker series
# ---------------------------------------------------------------------------

def load_hormuz() -> pd.DataFrame:
    h = pd.read_csv(PROC / "portwatch_hormuz_monthly.csv",
                    usecols=["period_dt", "n_tanker", "capacity_tanker"])
    h["period_dt"] = pd.to_datetime(h["period_dt"])
    h = h.rename(columns={"n_tanker": "hormuz_tanker_n",
                           "capacity_tanker": "hormuz_tanker_DWT"})
    # Normalise DWT to its own 2020-2024 mean
    mean_DWT = h.loc[h.period_dt.between("2020-01-01", "2024-12-31"),
                     "hormuz_tanker_DWT"].mean()
    h["hormuz_DWT_norm"] = h["hormuz_tanker_DWT"] / mean_DWT
    h["hormuz_tanker_shock"] = h["hormuz_tanker_DWT"].pct_change()
    return h


# ---------------------------------------------------------------------------
# 4. Phase 1 yearly exposure (DE + composite_01)
# ---------------------------------------------------------------------------

def load_exposure() -> pd.DataFrame:
    exp = pd.read_csv(PHASE1 / "composite_exposure.csv",
                      usecols=["year", "layer", "country", "DE", "composite_01"])
    return exp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading Comtrade import aggregates …")
    imp = load_import_panel()
    print(f"  import panel: {imp.shape}")

    print("Loading commodity prices …")
    prices = load_prices()
    prices.to_csv(OUT / "price_panel.csv", index=False)
    print(f"  price panel: {prices.shape}")

    print("Loading PortWatch Hormuz series …")
    hormuz = load_hormuz()
    print(f"  hormuz panel: {hormuz.shape}")

    print("Loading Phase 1 exposure …")
    exp = load_exposure()
    print(f"  exposure: {exp.shape}")

    # Merge: import × price (on period_dt + layer)
    panel = imp.merge(prices[["period_dt", "layer", "price_usd", "log_price"]],
                      on=["period_dt", "layer"], how="left")
    # Merge: × Hormuz (on period_dt only — same shock for all countries/layers)
    panel = panel.merge(
        hormuz[["period_dt", "hormuz_tanker_n", "hormuz_tanker_DWT",
                "hormuz_DWT_norm", "hormuz_tanker_shock"]],
        on="period_dt", how="left")
    # Merge: × exposure (on year + layer + country)
    panel = panel.merge(exp[["year", "layer", "country", "DE", "composite_01"]],
                        on=["year", "layer", "country"], how="left")

    panel["log_import"] = np.log(panel["import_value_usd"].clip(lower=1))
    panel = panel.sort_values(["country", "layer", "period_dt"]).reset_index(drop=True)

    # Restrict to 2020-2024 (full-year reporting)
    panel = panel[panel.period_dt.between("2020-01-01", "2024-12-31")]

    print(f"Final panel: {panel.shape}")
    print("Null counts:\n", panel[["price_usd", "hormuz_tanker_DWT",
                                    "DE", "composite_01"]].isna().sum())

    panel.to_csv(OUT / "panel_monthly.csv", index=False)

    # Quick summary
    print("\nCountries per layer:")
    print(panel.groupby("layer")["country"].nunique())
    print("\nDate range:", panel.period_dt.min(), "–", panel.period_dt.max())
    print("\nHormuz DWT descriptives:")
    print(hormuz[hormuz.period_dt.between("2020-01-01", "2024-12-31")][
        ["hormuz_tanker_DWT", "hormuz_DWT_norm"]].describe().round(3))


if __name__ == "__main__":
    main()
