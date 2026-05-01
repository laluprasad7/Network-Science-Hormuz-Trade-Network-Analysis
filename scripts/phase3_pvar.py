"""Phase 3.1 — Time-series VAR per commodity layer.

For each layer we estimate a bivariate VAR(p) on the global monthly series:

    [Δ log_price_t,  hormuz_DWT_norm_t]

with lag order p selected by BIC (max 4).  The Hormuz DWT series enters in
*levels* (it is already normalised and mean-stationary around 1.0).  Prices
enter as log first-differences (growth rates) to remove the dominant trend.

Identification: recursive (Cholesky) ordering — Hormuz DWT is ordered first,
log-price second.  Interpretation: a negative shock to Hormuz tanker capacity
raises commodity prices.  We do *not* order in the opposite direction because
the Hormuz transit is a physical capacity variable that commodity prices
cannot simultaneously cause (the reverse causality runs at multi-month lags
if at all).

We also run a simple linear Granger-causality F-test: "does lagged DWT help
predict log_price_growth?"

Additionally, as an interactive panel check, we run per-layer OLS of

    Δ log_import_{i,t} = α_i + α_t + γ_1 Δlog_price_t + γ_2 ΔDWT_t
                         + γ_3 DE_i × ΔDWT_t + ε_{i,t}

where DE_i is the Phase-1 direct-exposure score for country i (year-averaged).
The coefficient γ_3 tests the network claim: more-exposed countries suffer
disproportionately larger import contractions when Hormuz capacity falls.

Outputs (Results/phase3/):
    var_results.csv          lag order, Granger p-value, IRF peak, IRF horizon
    panel_regression.csv     OLS panel-FE coefficients per layer
Figures (figures/phase3/):
    fig3_price_vs_DWT.png    time series: prices + Hormuz DWT
    fig3_irf_{layer}.png     12-month IRF of price to a −1σ DWT shock (5 panels)
    fig3_panel_gamma3.png    γ_3 bar chart (exposure × shock interaction)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PHASE3 = ROOT / "Results" / "phase3"
FIG = ROOT / "figures" / "phase3"
FIG.mkdir(parents=True, exist_ok=True)

LAYERS = ("wheat", "ammonia", "urea", "lpg_propane", "lpg_butane")
LAYER_LABEL = {"wheat": "Wheat", "ammonia": "Ammonia (DAP proxy)",
               "urea": "Urea", "lpg_propane": "LPG propane",
               "lpg_butane": "LPG butane"}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
NBOOT = 500


def load_data():
    panel = pd.read_csv(PHASE3 / "panel_monthly.csv", parse_dates=["period_dt"])
    prices = pd.read_csv(PHASE3 / "price_panel.csv", parse_dates=["period_dt"])
    return panel, prices


def build_ts(prices: pd.DataFrame, layer: str) -> pd.DataFrame:
    """Global monthly time series for one layer: DWT + Δlog_price."""
    p = prices[prices.layer == layer].set_index("period_dt")[["log_price"]].sort_index()
    # Merge in DWT from panel (same for all countries/layers)
    panel = pd.read_csv(PHASE3 / "panel_monthly.csv", parse_dates=["period_dt"])
    dwtcol = (panel[["period_dt", "hormuz_DWT_norm"]].drop_duplicates()
                   .set_index("period_dt").sort_index())
    ts = p.join(dwtcol, how="inner")
    ts = ts.loc["2020-01-01":"2024-12-31"]
    ts["dlog_price"] = ts["log_price"].diff()
    ts = ts.dropna()
    return ts


def run_var(ts: pd.DataFrame, layer: str, max_lags: int = 4) -> dict:
    endog = ts[["dlog_price", "hormuz_DWT_norm"]].copy()
    model = VAR(endog)
    bic_sel = model.select_order(maxlags=max_lags)
    p = int(bic_sel.bic) if int(bic_sel.bic) > 0 else 1
    p = min(p, max_lags)
    res = model.fit(p)

    # Granger causality: DWT → dlog_price
    gc_tests = grangercausalitytests(ts[["dlog_price", "hormuz_DWT_norm"]], maxlag=p,
                                      verbose=False)
    gc_pval = min(gc_tests[l][0]["ssr_ftest"][1] for l in gc_tests)

    # IRF: shock to DWT (second variable), response in dlog_price (first)
    irf = res.irf(12)
    # Cholesky: shock to column 1 (DWT) observed in row 0 (dlog_price)
    irf_vals = irf.orth_irfs[:, 0, 1]   # periods × response × shock
    stderr = irf.stderr(orth=True)[:, 0, 1]
    ci_lo = irf_vals - 1.96 * stderr
    ci_hi = irf_vals + 1.96 * stderr

    peak_idx = int(np.argmax(np.abs(irf_vals)))
    return {
        "layer": layer,
        "lag_order": p,
        "gc_pval": gc_pval,
        "irf_vals": irf_vals,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "irf_peak": float(irf_vals[peak_idx]),
        "irf_peak_horizon": peak_idx,
    }


def plot_irf(result: dict) -> None:
    layer = result["layer"]
    irf_vals = result["irf_vals"]
    ci_lo = result["ci_lo"]
    ci_hi = result["ci_hi"]
    horizons = np.arange(len(irf_vals))
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.fill_between(horizons, ci_lo, ci_hi, alpha=0.25, color="steelblue",
                    label="95 % CI")
    ax.plot(horizons, irf_vals, color="steelblue", marker="o", ms=4,
            label="IRF (Cholesky)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Months after shock")
    ax.set_ylabel("Δlog price response")
    ax.set_title(f"{LAYER_LABEL[layer]}: price response to −1σ Hormuz DWT shock")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"fig3_irf_{layer}.png", dpi=150)
    plt.close(fig)


def run_panel_regression(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-layer OLS in log-levels with country + time FE.

    log_import_{it} = alpha_i + alpha_t + b1*log_price_t
                      + b2*DWT_t + g3*DE_i*DWT_t + eps

    DE_i is the Phase-1 direct exposure (year-averaged across 2020-2024,
    time-invariant).  DWT_t is the monthly normalised tanker DWT (mean=1).
    Identification of g3: within-country variation in DWT interacted with
    the cross-sectional heterogeneity in DE.
    Positive g3 would mean more-exposed countries are *more* sensitive to
    DWT increases (supply improvement).  We expect g3 < 0: more-exposed
    countries suffer larger import contractions when DWT falls.
    """
    from numpy.linalg import lstsq
    from sklearn.preprocessing import LabelEncoder

    rows = []
    sub_all = panel.copy()
    # Time-invariant DE: average over years per (country, layer)
    de_avg = (sub_all.groupby(["country", "layer"])["DE"].mean()
                     .reset_index().rename(columns={"DE": "DE_avg"}))
    sub_all = sub_all.merge(de_avg, on=["country", "layer"], how="left")
    sub_all["DE_x_DWT"] = sub_all["DE_avg"] * sub_all["hormuz_DWT_norm"]

    for layer in LAYERS:
        sub = sub_all[sub_all.layer == layer].dropna(
            subset=["log_import", "log_price", "hormuz_DWT_norm",
                    "DE_avg", "DE_x_DWT"])
        if len(sub) < 100:
            continue
        # Country FE dummies
        le_c = LabelEncoder()
        c_enc = le_c.fit_transform(sub["country"].values)
        n_c = c_enc.max() + 1
        le_t = LabelEncoder()
        t_enc = le_t.fit_transform(sub["period_dt"].astype(str).values)
        n_t = t_enc.max() + 1
        C_FE = np.eye(n_c)[c_enc][:, :-1]
        T_FE = np.eye(n_t)[t_enc][:, :-1]
        X_regs = np.column_stack([
            sub["log_price"].values,
            sub["hormuz_DWT_norm"].values,
            sub["DE_x_DWT"].values,
        ])
        X = np.column_stack([np.ones(len(sub)), X_regs, C_FE, T_FE])
        y = sub["log_import"].values
        beta, _, _, _ = lstsq(X, y, rcond=None)
        # Residual std
        resid = y - X @ beta
        s2 = resid @ resid / max(len(y) - X.shape[1], 1)
        try:
            cov = s2 * np.linalg.inv(X[:, :4].T @ X[:, :4])
            se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            se = np.full(4, np.nan)
        rows.append({"layer": layer, "n_obs": len(sub),
                     "b_log_price": beta[1], "se_log_price": se[1],
                     "b_DWT": beta[2], "se_DWT": se[2],
                     "gamma3_DE_x_DWT": beta[3], "se_gamma3": se[3]})
        print(f"  {layer}: n={len(sub)} b_price={beta[1]:.3f} "
              f"b_DWT={beta[2]:.3f} g3={beta[3]:.3f}")
    return pd.DataFrame(rows)


def main() -> None:
    panel, prices = load_data()
    prices = prices[prices.period_dt.between("2020-01-01", "2024-12-31")]

    # Time-series overview plot
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 8.0))
    axes = axes.ravel()
    dwtcol = (panel[["period_dt", "hormuz_DWT_norm"]].drop_duplicates()
                   .sort_values("period_dt"))
    color_cycle = sns.color_palette("tab10", n_colors=5)
    for k, layer in enumerate(LAYERS):
        ax = axes[k]
        p = prices[prices.layer == layer].sort_values("period_dt")
        ax2 = ax.twinx()
        ax.plot(p["period_dt"], p["log_price"], color=color_cycle[k], lw=1.5,
                label="log price")
        ax2.plot(dwtcol["period_dt"], dwtcol["hormuz_DWT_norm"], color="gray",
                 lw=1.0, ls="--", alpha=0.7, label="Hormuz DWT (norm)")
        ax.set_title(LAYER_LABEL[layer])
        ax.set_xlabel("")
        ax.set_ylabel("log price", color=color_cycle[k])
        ax2.set_ylabel("DWT norm", color="gray")
    axes[-1].set_visible(False)
    fig.suptitle("Global commodity prices vs Hormuz tanker capacity (2020–2024)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_price_vs_DWT.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # VAR + IRF per layer
    var_rows = []
    for layer in LAYERS:
        ts = build_ts(prices, layer)
        print(f"\n{layer}: T={len(ts)}")
        result = run_var(ts, layer)
        print(f"  p={result['lag_order']} Granger p={result['gc_pval']:.4f} "
              f"IRF_peak={result['irf_peak']:.4f} @ h={result['irf_peak_horizon']}")
        plot_irf(result)
        var_rows.append({k: v for k, v in result.items()
                         if k not in ("irf_vals", "ci_lo", "ci_hi")})
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(PHASE3 / "var_results.csv", index=False)
    print("\nVAR summary:")
    print(var_df.to_string(index=False))

    # Panel regression
    print("\nPanel regression (country + time FE):")
    pr = run_panel_regression(panel)
    pr.to_csv(PHASE3 / "panel_regression.csv", index=False)

    # Bar chart of γ3
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    colors = ["#c0392b" if v < 0 else "#2980b9" for v in pr["gamma3_DE_x_DWT"]]
    ax.bar(pr["layer"], pr["gamma3_DE_x_DWT"], color=colors)
    for i, (_, r) in enumerate(pr.iterrows()):
        v = r["gamma3_DE_x_DWT"]
        ax.text(i, v + (0.002 if v >= 0 else -0.015),
                f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("g3 (DE x dDWT coefficient)")
    ax.set_title("Exposure-amplified import response to Hormuz capacity drop")
    ax.set_xlabel("Layer")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_panel_gamma3.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote figures under {FIG}")


if __name__ == "__main__":
    main()
