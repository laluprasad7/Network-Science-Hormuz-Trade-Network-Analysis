"""Phase 1 composite exposure index, ranking tables, and figures.

The composite index is built per (year, layer) by:

    1. z-scoring each of the eight raw metrics across countries (within
       the year x layer slice),
    2. averaging the z-scores into a single composite,
    3. re-rescaling the composite to [0, 1] by min-max so it can be
       compared across slices (the *ranking* is what matters; the level
       of the composite z-score is not directly meaningful otherwise).

We exclude the seven Hormuz countries from the composite distribution
because they are the SOURCE of the disruption: their own "exposure" to
themselves is not interpretable.

Outputs (under Results/phase1/):
    composite_exposure.csv      one row per (year, layer, country)
    top_decile_2024.csv         top-decile vulnerability list, 2024
                                  (year fully reported)
    metric_correlation_2020_2024.csv     pooled cross-metric correlations
Figures (under figures/phase1/):
    fig1_top_countries_lpg_butane_2024.png
    fig1_composite_heatmap_2024.png
    fig1_metric_correlation.png
    fig1_top5_timeseries.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "Results" / "phase1"
FIG = ROOT / "figures" / "phase1"
FIG.mkdir(parents=True, exist_ok=True)

HORMUZ = ["ARE", "BHR", "IRN", "IRQ", "KWT", "QAT", "SAU"]
METRICS = ["DE", "SR", "betweenness", "katz", "ppr", "leontief", "debtrank", "pivi"]
LAYERS = ["wheat", "ammonia", "urea", "lpg_propane", "lpg_butane"]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)


def composite_index(wide: pd.DataFrame) -> pd.DataFrame:
    """z-score within (year, layer); average; min-max scale to [0, 1]."""
    df = wide.copy()
    df_nonh = df[~df.country.isin(HORMUZ)].copy()

    z_cols = []
    for m in METRICS:
        zname = f"z_{m}"
        df_nonh[zname] = (df_nonh
                           .groupby(["year", "layer"])[m]
                           .transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12)))
        z_cols.append(zname)
    df_nonh["composite_z"] = df_nonh[z_cols].mean(axis=1)
    df_nonh["composite_01"] = (df_nonh
                                .groupby(["year", "layer"])["composite_z"]
                                .transform(lambda s: (s - s.min()) /
                                                     (s.max() - s.min() + 1e-12)))
    df_nonh["rank"] = (df_nonh
                        .groupby(["year", "layer"])["composite_z"]
                        .rank(ascending=False, method="dense").astype(int))
    return df_nonh


def top_decile(comp: pd.DataFrame, year: int) -> pd.DataFrame:
    """Top-decile vulnerability list for one year, across all layers."""
    sub = comp[comp.year == year]
    n = sub.country.nunique()
    cutoff = max(1, n // 10)
    return (sub.sort_values(["layer", "composite_z"], ascending=[True, False])
              .groupby("layer").head(cutoff))


# --- figures -----------------------------------------------------------------

def fig_top_countries(wide: pd.DataFrame, year: int = 2024,
                      layer: str = "lpg_butane", topn: int = 15) -> None:
    sub = wide[(wide.year == year) & (wide.layer == layer) &
               (~wide.country.isin(HORMUZ))].copy()
    sub = sub.sort_values("DE", ascending=False).head(topn)

    cols = ["DE", "leontief", "debtrank", "pivi"]
    melted = sub.melt(id_vars="country", value_vars=cols,
                      var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    sns.barplot(data=melted, x="country", y="value", hue="metric", ax=ax,
                order=sub.country.tolist())
    ax.set_xlabel("")
    ax.set_ylabel("Exposure score")
    ax.set_title(f"Top-{topn} non-Hormuz importers, {layer}, {year}")
    ax.legend(frameon=False, fontsize=8, ncols=4)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIG / f"fig1_top_countries_{layer}_{year}.png", dpi=150)
    plt.close(fig)


def fig_composite_heatmap(comp: pd.DataFrame, year: int = 2024) -> None:
    """For one year, heatmap of composite exposure across (country, layer)
    showing the 25 most exposed countries (rank-sum across layers)."""
    sub = comp[comp.year == year]
    pv = sub.pivot(index="country", columns="layer", values="composite_z")[LAYERS]
    score = pv.sum(axis=1).sort_values(ascending=False)
    top = score.head(25).index
    pv = pv.loc[top]

    fig, ax = plt.subplots(figsize=(6.0, 6.5))
    sns.heatmap(pv, annot=True, fmt=".1f", cmap="rocket_r",
                cbar_kws={"label": "Composite exposure (z)"}, ax=ax)
    ax.set_title(f"Composite exposure z-score, top-25 importers, {year}")
    ax.set_xlabel("Commodity layer")
    ax.set_ylabel("Country (ISO-3)")
    fig.tight_layout()
    fig.savefig(FIG / f"fig1_composite_heatmap_{year}.png", dpi=150)
    plt.close(fig)


def fig_metric_correlation(wide: pd.DataFrame) -> pd.DataFrame:
    """Pooled across 2020-2024 and all layers; non-Hormuz only.
    Spearman because metrics are highly skewed and we care about ranks."""
    sub = wide[(wide.year.between(2020, 2024)) & (~wide.country.isin(HORMUZ))]
    corr = sub[METRICS].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag",
                vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"label": "Spearman rho"})
    ax.set_title("Cross-metric Spearman correlation (2020-2024 pooled)")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_metric_correlation.png", dpi=150)
    plt.close(fig)
    return corr


def fig_top5_timeseries(comp: pd.DataFrame, layer: str = "lpg_butane") -> None:
    """Country-level composite exposure across 2020-2024 for the top-5
    countries by 2024 composite score, restricted to one layer."""
    sub = comp[(comp.layer == layer) & (comp.year.between(2020, 2024))]
    top = (sub[sub.year == 2024].nlargest(5, "composite_z").country.tolist())
    sub = sub[sub.country.isin(top)].sort_values(["country", "year"])

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    palette = sns.color_palette("tab10", n_colors=len(top))
    for k, c in enumerate(top):
        d = sub[sub.country == c]
        ax.plot(d.year, d.composite_z, marker="o", label=c, color=palette[k])
    ax.set_xlabel("Year")
    ax.set_ylabel("Composite exposure z-score")
    ax.set_title(f"Top-5 {layer} importers by composite exposure (2024 ranking)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"fig1_top5_timeseries_{layer}.png", dpi=150)
    plt.close(fig)


# --- main --------------------------------------------------------------------

def main() -> None:
    wide = pd.read_csv(PHASE1 / "exposure_wide.csv")
    comp = composite_index(wide)
    comp.to_csv(PHASE1 / "composite_exposure.csv", index=False)

    td = top_decile(comp, 2024)
    td.to_csv(PHASE1 / "top_decile_2024.csv", index=False)

    print("Phase 1 composite — top-decile per layer (2024):")
    for L in LAYERS:
        sub = td[td.layer == L].nlargest(8, "composite_z")
        print(f"\n  layer = {L}")
        print(sub[["country", "composite_z", "DE", "leontief", "debtrank", "pivi"]]
                  .to_string(index=False))

    fig_top_countries(wide)
    fig_composite_heatmap(comp)
    corr = fig_metric_correlation(wide)
    corr.to_csv(PHASE1 / "metric_correlation_2020_2024.csv")
    fig_top5_timeseries(comp)
    print()
    print("Cross-metric Spearman rho (2020-2024 pooled):")
    print(corr.round(2).to_string())


if __name__ == "__main__":
    main()
