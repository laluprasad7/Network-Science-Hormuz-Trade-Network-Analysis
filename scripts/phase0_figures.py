"""Phase 0 figures — what does the raw trade panel look like?

Outputs (under figures/phase0/):
    fig0_layer_volume_timeseries.png   total USD value per layer per year
    fig0_hormuz_share.png              Hormuz exporters' share by layer over time
    fig0_dependency_share_cdf.png      CDF of dependency shares per layer (2024)
    fig0_top_importers_lpg.png         top-10 LPG importers' Hormuz share, 2024
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PHASE0 = ROOT / "Results" / "phase0"
FIG = ROOT / "figures" / "phase0"
FIG.mkdir(parents=True, exist_ok=True)

HORMUZ = ["ARE", "BHR", "IRN", "IRQ", "KWT", "QAT", "SAU"]
LAYERS = ["wheat", "ammonia", "urea", "lpg_propane", "lpg_butane"]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)


def fig_volume_timeseries() -> None:
    """One panel per layer; bar height = total USD; light bar marks partial year."""
    summary = pd.read_csv(PHASE0 / "trade_matrix_summary.csv")
    summary["partial_year"] = summary.year >= 2025  # 2025 lagged, 2026 = Q1 only

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    palette = sns.color_palette("tab10", n_colors=len(LAYERS))
    for k, L in enumerate(LAYERS):
        sub = summary[summary.layer == L].sort_values("year")
        ax.plot(sub.year, sub.total_value_usd / 1e9, marker="o",
                label=L, color=palette[k])
    ax.set_xlabel("Year")
    ax.set_ylabel("Total bilateral imports (USD billion)")
    ax.set_title("Annual bilateral trade value by commodity layer")
    ax.legend(frameon=False, ncols=2, fontsize=8)
    ax.axvspan(2024.5, 2026.5, alpha=0.08, color="gray")
    ax.text(2025.5, ax.get_ylim()[1] * 0.95, "partial / lagged",
            ha="center", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(FIG / "fig0_layer_volume_timeseries.png", dpi=150)
    plt.close(fig)


def fig_hormuz_share() -> None:
    hs = pd.read_csv(PHASE0 / "hormuz_share_by_year_layer.csv")
    pv = hs.pivot(index="refYear", columns="layer", values="hormuz_share")[LAYERS]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    pv.loc[:2024].plot(ax=ax, marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of global flow value")
    ax.set_title("Hormuz exporters' share of global imports, by layer")
    ax.set_ylim(0, max(0.55, pv.loc[:2024].values.max() * 1.1))
    ax.legend(title="layer", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig0_hormuz_share.png", dpi=150)
    plt.close(fig)


def fig_dependency_share_cdf(year: int = 2024) -> None:
    """For each layer, CDF of importer dependency shares pooled across all
    (importer, exporter) pairs with positive flow that year."""
    countries = pd.read_csv(PHASE0 / "country_index.csv")["iso3"].tolist()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    palette = sns.color_palette("tab10", n_colors=len(LAYERS))
    for k, L in enumerate(LAYERS):
        A = np.load(PHASE0 / f"A_{year}_{L}.npy")
        shares = A[A > 0]
        x = np.sort(shares)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, label=L, color=palette[k], lw=1.4)
    ax.set_xscale("log")
    ax.set_xlabel("Dependency share (importer j on exporter i)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"Distribution of bilateral dependency shares, {year}")
    ax.axvline(0.25, ls="--", color="gray", lw=0.7)
    ax.text(0.26, 0.05, "25%", color="gray", fontsize=8)
    ax.legend(title="layer", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig0_dependency_share_cdf.png", dpi=150)
    plt.close(fig)


def fig_top_importers_lpg(year: int = 2024) -> None:
    """For LPG butane and propane (the most Hormuz-exposed layers), pick the
    top-10 importers by total imports and show their Hormuz dependency share."""
    countries = pd.read_csv(PHASE0 / "country_index.csv")["iso3"].tolist()
    h_idx = [countries.index(c) for c in HORMUZ if c in countries]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=False)
    for ax, L in zip(axes, ["lpg_butane", "lpg_propane"]):
        A = np.load(PHASE0 / f"A_{year}_{L}.npy")
        W = np.load(PHASE0 / f"W_{year}_{L}.npy")
        total_imports = W.sum(axis=0)
        h_share = A[h_idx, :].sum(axis=0)
        order = np.argsort(-total_imports)[:10]
        names = [countries[i] for i in order]
        h = h_share[order]
        non_h = 1 - h
        ax.bar(names, h, color="#d62728", label="Hormuz")
        ax.bar(names, non_h, bottom=h, color="#cccccc", label="other")
        ax.set_title(f"{L}, top-10 importers ({year})")
        ax.set_ylabel("Share of imports")
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig0_top_importers_lpg.png", dpi=150)
    plt.close(fig)


def main() -> None:
    fig_volume_timeseries()
    fig_hormuz_share()
    fig_dependency_share_cdf()
    fig_top_importers_lpg()
    print(f"Wrote 4 figures to {FIG}")


if __name__ == "__main__":
    main()
