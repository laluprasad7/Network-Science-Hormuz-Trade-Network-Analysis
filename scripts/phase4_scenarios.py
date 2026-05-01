"""Phase 4 — Scenario Simulation Engine.

Three Hormuz disruption classes:
    S1  14-day transient shock        (half-month at full closure)
    S2  30-45-day rupture             (near-closure for ~1.5 months)
    S3  6-month persistent realignment (70 pct capacity loss for 6 months)

Country-level import impacts via two channels:
    (1) Exposure channel : g3 x DE_i x dDWT[t]   (panel-reg interaction term)
    (2) Price channel    : b_price x dlog_price[t] (SDM b_log_price x VAR IRF path)
    (3) Network multiplier 1/(1-rho)               (SDM spatial lag coefficient)

Structural benchmark: log(1 - DE_i x disruption_factor) x 12 months, where
disruption_factor = mean(-dDWT) over disruption months. This is a pure
trade-network lower bound independent of the regression estimates.

Outputs (Results/phase4/):
    scenario_country_impacts.csv
    scenario_summary.csv
    top_losers_S3.csv
Figures (figures/phase4/):
    fig4_price_paths.png
    fig4_scenario_comparison.png
    fig4_heatmap_S1_14day.png
    fig4_heatmap_S2_45day_rupture.png
    fig4_heatmap_S3_6month_realignment.png
    fig4_top_losers.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "Results" / "phase2"
PHASE3 = ROOT / "Results" / "phase3"
PHASE4 = ROOT / "Results" / "phase4"
FIG = ROOT / "figures" / "phase4"
PHASE4.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

LAYERS = ("wheat", "ammonia", "urea", "lpg_propane", "lpg_butane")
LAYER_LABEL = {
    "wheat": "Wheat",
    "ammonia": "Ammonia",
    "urea": "Urea",
    "lpg_propane": "LPG propane",
    "lpg_butane": "LPG butane",
}
HORMUZ = ["ARE", "BHR", "IRN", "IRQ", "KWT", "QAT", "SAU"]

# Monthly dDWT_norm profiles (subtract from baseline DWT_norm = 1.0)
# Negative = capacity reduction; 12 months total
SCENARIOS: dict[str, dict] = {
    "S1_14day": {
        "label": "S1: 14-day transient shock",
        "dDWT": [-0.50] + [0.0] * 11,
    },
    "S2_45day_rupture": {
        "label": "S2: 30-45 day rupture",
        "dDWT": [-0.90, -0.90, -0.40] + [0.0] * 9,
    },
    "S3_6month_realignment": {
        "label": "S3: 6-month persistent realignment",
        "dDWT": [-0.70, -0.70, -0.70, -0.70, -0.70, -0.70,
                 -0.50, -0.30, -0.10] + [0.0] * 3,
    },
}

SC_COLORS = {
    "S1_14day": "#2980b9",
    "S2_45day_rupture": "#e67e22",
    "S3_6month_realignment": "#c0392b",
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)


# ---------------------------------------------------------------------------
# VAR / IRF helpers
# ---------------------------------------------------------------------------

def get_irf(layer: str) -> np.ndarray:
    """Fit VAR(p) on [dlog_price, DWT_norm]; return 13-step Cholesky IRF.

    Response variable : dlog_price (index 0)
    Shock variable    : DWT_norm   (index 1)
    Returns array of shape (13,): IRF[h] for h = 0..12.
    """
    panel = pd.read_csv(PHASE3 / "panel_monthly.csv", parse_dates=["period_dt"])
    prices = pd.read_csv(PHASE3 / "price_panel.csv", parse_dates=["period_dt"])
    p = (prices[prices.layer == layer]
             .set_index("period_dt")[["log_price"]].sort_index())
    dwt = (panel[["period_dt", "hormuz_DWT_norm"]]
               .drop_duplicates().set_index("period_dt").sort_index())
    ts = p.join(dwt, how="inner").loc["2020-01-01":"2024-12-31"].copy()
    ts["dlog_price"] = ts["log_price"].diff()
    ts = ts.dropna()
    m = VAR(ts[["dlog_price", "hormuz_DWT_norm"]])
    sel = m.select_order(maxlags=4)
    p_ord = max(int(sel.bic), 1)
    p_ord = min(p_ord, 4)
    res = m.fit(p_ord)
    return res.irf(12).orth_irfs[:, 0, 1]  # shape (13,)


def steady_state_dprice(irf_vals: np.ndarray, dDWT_t: float,
                         sigma_DWT: float) -> float:
    """Approximate steady-state log_price deviation when DWT is held at dDWT_t.

    Uses the long-run (sum of) IRF as the multiplier: how much does log_price
    shift in the long run for a permanent 1-sigma DWT change?  Applying this
    per period avoids explosive accumulation from sustained shocks.
    """
    lrun = float(irf_vals.sum())   # long-run log_price response per 1-sigma DWT
    return lrun * (dDWT_t / sigma_DWT)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def main() -> None:
    panel = pd.read_csv(PHASE3 / "panel_monthly.csv", parse_dates=["period_dt"])
    panel_reg = pd.read_csv(PHASE3 / "panel_regression.csv")
    sdm_res = pd.read_csv(PHASE3 / "sdm_results.csv")
    attack = pd.read_csv(PHASE2 / "hormuz_attack.csv")

    sigma_DWT = float(panel["hormuz_DWT_norm"].std())
    print(f"sigma_DWT = {sigma_DWT:.4f}")

    # 2024 baselines
    mask_2024 = panel["period_dt"].between("2024-01-01", "2024-12-31")
    baseline = (panel[mask_2024]
                .groupby(["country", "layer"])["import_value_usd"].sum()
                .reset_index()
                .rename(columns={"import_value_usd": "import_2024_usd"}))
    baseline = baseline[(baseline["import_2024_usd"] > 0) &
                        (~baseline["country"].isin(HORMUZ))]

    de_avg = (panel[mask_2024]
              .groupby(["country", "layer"])["DE"].mean().reset_index())

    attack_2024 = (attack[attack.year == 2024]
                   .set_index("layer")[["flow_loss", "eff_drop"]])

    # Fit IRFs
    print("Fitting VARs and extracting IRFs ...")
    irfs: dict[str, np.ndarray] = {}
    for layer in LAYERS:
        irfs[layer] = get_irf(layer)
        ph = int(np.argmax(np.abs(irfs[layer])))
        print(f"  {layer}: peak IRF = {irfs[layer][ph]:+.5f} @ h={ph}")

    # Simulation loop
    print("\nSimulating scenarios ...")
    rows = []
    for sc_name, sc in SCENARIOS.items():
        dDWT = np.array(sc["dDWT"], dtype=float)
        disruption_factor = float(np.mean(np.maximum(-dDWT, 0.0)))

        for layer in LAYERS:
            sdm_r = sdm_res[sdm_res.layer == layer].iloc[0]
            b_price = float(sdm_r["b_log_price"])
            rho = float(sdm_r["rho"]) if not pd.isna(sdm_r["rho"]) else 0.0
            rho = max(0.0, min(rho, 0.99))
            multiplier = 1.0 / (1.0 - rho)

            pr = panel_reg[panel_reg.layer == layer].iloc[0]
            g3 = float(pr["gamma3_DE_x_DWT"])

            flow_loss_full = float(attack_2024.loc[layer, "flow_loss"])

            layer_bl = baseline[baseline.layer == layer]
            de_layer = de_avg[de_avg.layer == layer]

            for _, bl_row in layer_bl.iterrows():
                cty = bl_row["country"]
                imp = float(bl_row["import_2024_usd"])

                de_r = de_layer[de_layer.country == cty]
                DE = float(de_r["DE"].iloc[0]) if len(de_r) > 0 else 0.0

                # Monthly impact: structural channel + price channel (per period)
                # structural: country loses DE x dDWT fraction of imports from
                #   Hormuz route each month
                # price:      per-period steady-state log_price response drives
                #   imports via SDM b_log_price
                monthly = np.zeros(12)
                cum_price = 0.0
                for t in range(12):
                    disruption_t = max(-dDWT[t], 0.0)
                    structural_t = np.log(max(1.0 - DE * disruption_t, 0.01))
                    dprice_t = steady_state_dprice(irfs[layer], dDWT[t], sigma_DWT)
                    price_t = b_price * dprice_t
                    monthly[t] = (structural_t + price_t) * multiplier
                    cum_price += dprice_t

                cum12 = float(monthly.sum())
                # Normalise to average monthly log-change so % makes sense
                avg_monthly = cum12 / 12.0
                peak_loss = float(monthly.min())
                pct = float(np.expm1(avg_monthly))      # avg monthly % change
                dollar = imp * pct                       # annual equivalent

                # Structural benchmark: DE-only, no price, no multiplier
                struct_t_vals = [
                    np.log(max(1.0 - DE * max(-dDWT[t], 0.0), 0.01))
                    for t in range(12)
                ]
                structural_benchmark = float(np.mean(struct_t_vals))

                rows.append({
                    "scenario": sc_name,
                    "scenario_label": sc["label"],
                    "country": cty,
                    "layer": layer,
                    "DE": DE,
                    "rho": rho,
                    "multiplier": multiplier,
                    "cum_dDWT": float(dDWT.sum()),
                    "avg_dprice": cum_price / 12.0,
                    "exposure_channel": float(
                        np.mean([g3 * DE * dDWT[t] * multiplier for t in range(12)])
                    ),
                    "price_channel": float(b_price * cum_price / 12.0 * multiplier),
                    "structural_benchmark": structural_benchmark,
                    "avg_monthly_dlog": avg_monthly,
                    "peak_month_dlog": peak_loss,
                    "pct_change_import": pct,
                    "import_2024_usd": imp,
                    "dollar_impact_usd": dollar,
                })

    impacts = pd.DataFrame(rows)
    impacts.to_csv(PHASE4 / "scenario_country_impacts.csv", index=False)
    print(f"  Saved scenario_country_impacts.csv  ({len(impacts)} rows)")

    # Scenario summary
    summ = (impacts.groupby(["scenario", "scenario_label", "layer"]).agg(
        n_countries=("country", "count"),
        mean_pct=("pct_change_import", "mean"),
        median_pct=("pct_change_import", "median"),
        p10_pct=("pct_change_import", lambda x: float(x.quantile(0.1))),
        total_dollar_impact=("dollar_impact_usd", "sum"),
        structural_bm=("structural_benchmark", "mean"),
    ).reset_index())
    summ.to_csv(PHASE4 / "scenario_summary.csv", index=False)

    print("\nMean % import change by scenario x layer:")
    piv = summ.pivot(index="scenario", columns="layer", values="mean_pct")
    piv = piv[[L for L in LAYERS if L in piv.columns]]
    print((piv * 100).round(2).to_string())

    # Top losers S3
    s3 = impacts[impacts.scenario == "S3_6month_realignment"].copy()
    s3_cty = (s3.groupby("country").agg(
        total_dollar=("dollar_impact_usd", "sum"),
        total_base=("import_2024_usd", "sum"),
    ).reset_index())
    s3_cty["pct_total"] = s3_cty["total_dollar"] / s3_cty["total_base"].clip(lower=1)
    s3_cty.sort_values("total_dollar").head(30).to_csv(
        PHASE4 / "top_losers_S3.csv", index=False)

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------

    months = np.arange(1, 13)

    # Fig 1 — Price level paths
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()
    for k, layer in enumerate(LAYERS):
        ax = axes[k]
        for sc_name, sc in SCENARIOS.items():
            pp = np.array([
                steady_state_dprice(irfs[layer], d, sigma_DWT)
                for d in sc["dDWT"]
            ])
            ax.plot(months, pp, color=SC_COLORS[sc_name], lw=1.8,
                    label=sc["label"] if k == 0 else "")
        ax.axhline(0, color="black", lw=0.7, ls="--")
        ax.set_title(LAYER_LABEL[layer])
        ax.set_xlabel("Month")
        ax.set_ylabel("dlog price")
    axes[-1].set_visible(False)
    handles = [plt.Line2D([0], [0], color=SC_COLORS[s], lw=2,
               label=SCENARIOS[s]["label"]) for s in SCENARIOS]
    fig.legend(handles=handles, loc="lower right", frameon=False, fontsize=9,
               bbox_to_anchor=(0.98, 0.05))
    fig.suptitle("Commodity price deviation under Hormuz disruption scenarios",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_price_paths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_price_paths.png")

    # Fig 2 — DWT profiles + aggregate impact comparison
    # Bar chart shows trade-weighted aggregate impact (in USD bn) for the
    # transient (S1) and rupture (S2) scenarios; the 6-month realignment
    # is omitted from this comparison because the IRF is extrapolated far
    # outside its estimation horizon for sustained shocks of that length.
    SHORT_SC = ["S1_14day", "S2_45day_rupture"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    for sc_name, sc in SCENARIOS.items():
        ax.plot(months, 1.0 + np.array(sc["dDWT"]),
                color=SC_COLORS[sc_name], lw=2, marker="o", ms=4,
                label=sc["label"])
    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("Month after onset")
    ax.set_ylabel("Hormuz DWT norm (baseline = 1.0)")
    ax.set_title("DWT capacity profiles")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    x = np.arange(len(LAYERS))
    w = 0.35
    for i, sc_name in enumerate(SHORT_SC):
        vals = []
        for L in LAYERS:
            sub = summ[(summ.scenario == sc_name) & (summ.layer == L)]
            # USD billions, signed
            v = float(sub["total_dollar_impact"].iloc[0]) / 1e9 if len(sub) > 0 else 0.0
            vals.append(v)
        ax.bar(x + (i - 0.5) * w, vals, w,
               label=SCENARIOS[sc_name]["label"],
               color=SC_COLORS[sc_name], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([LAYER_LABEL[L] for L in LAYERS], rotation=15, ha="right")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Annualised import-value impact (USD bn)")
    ax.set_title("Aggregate import-value impact by commodity (S1, S2)")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG / "fig4_scenario_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved fig4_scenario_comparison.png")

    # Fig 3 — Heatmaps (one per scenario)
    top_countries = (baseline.groupby("country")["import_2024_usd"]
                     .sum().nlargest(40).index.tolist())

    for sc_name in SCENARIOS:
        sub = impacts[impacts.scenario == sc_name]
        piv = sub.pivot(index="country", columns="layer",
                        values="avg_monthly_dlog")
        piv = piv.reindex(index=[c for c in top_countries if c in piv.index])
        piv = piv[[L for L in LAYERS if L in piv.columns]] * 100
        if piv.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 10))
        vmax = float(piv.abs().quantile(0.95).max())
        vmax = max(vmax, 1.0)
        sns.heatmap(piv, ax=ax, cmap="RdYlGn_r", center=0,
                    vmin=-vmax, vmax=vmax,
                    linewidths=0.3, linecolor="white",
                    cbar_kws={"label": "% import change (12-month cumulative)"})
        ax.set_title(f"Import impact — {SCENARIOS[sc_name]['label']}")
        ax.set_xlabel("Commodity")
        ax.set_ylabel("Country (ISO3)")
        fig.tight_layout()
        fig.savefig(FIG / f"fig4_heatmap_{sc_name}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved fig4_heatmap_{sc_name}.png")

    # Fig 4 — Top losers by dollar impact (S3)
    top30 = (s3_cty.sort_values("total_dollar").head(30).copy())
    top30 = top30.sort_values("total_dollar", ascending=False)
    bar_colors = ["#c0392b" if v < 0 else "#27ae60" for v in top30["total_dollar"]]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top30["country"], top30["total_dollar"] / 1e6, color=bar_colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("12-month import change (USD million)")
    ax.set_title("S3 (6-month realignment): most-affected countries by dollar impact")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_top_losers.png", dpi=150)
    plt.close(fig)
    print("  Saved fig4_top_losers.png")

    print(f"\nPhase 4 complete. Results -> {PHASE4} | Figures -> {FIG}")


if __name__ == "__main__":
    main()
