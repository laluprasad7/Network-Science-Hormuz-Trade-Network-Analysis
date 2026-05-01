"""Phase 2.1 — directed community detection on yearly trade networks.

We use InfoMap (igraph's community_infomap) on each (year, layer) graph.
InfoMap is a flow-based method designed for directed weighted networks and
preserves the importer/exporter hierarchy without symmetrising.

For each (year, layer):
    - Build a directed igraph from W (exporter -> importer).
    - Run InfoMap with weights = W.
    - Identify the *primary Hormuz community*: the InfoMap module containing
      the largest count of Hormuz members.  This is a strict, single-community
      definition of the Hormuz supply-chain bloc, which we found to be much
      more stable across years than the original "union of every module that
      touches Hormuz".  (The latter conflated 'genuine Hormuz cluster' with
      'one Hormuz country happens to sit in a generic European wheat module',
      producing year-on-year swings that were artefacts of InfoMap relabelling
      rather than economic change.)
    - Report internal-flow share of Hormuz exports inside that community,
      so the reader can see how strongly anchored the bloc is.

Outputs (Results/phase2/):
    communities.csv             year, layer, country, community membership,
                                  primary-community flag, codelength, modularity
    hormuz_bloc_membership.csv  per (year, layer) summary: number of
                                  communities, modularity Q, primary-community
                                  size, non-Hormuz countries in primary comm,
                                  share of Hormuz imports captured by the
                                  primary community
Figures (figures/phase2/):
    fig2_community_count.png       n_communities and modularity per (year, layer)
    fig2_hormuz_bloc_size.png      number of non-Hormuz countries in the
                                    primary Hormuz community over time
    fig2_hormuz_bloc_share.png     fraction of all Hormuz exports flowing to
                                    members of the primary Hormuz community
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PHASE0 = ROOT / "Results" / "phase0"
OUT = ROOT / "Results" / "phase2"
FIG = ROOT / "figures" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

HORMUZ = ["ARE", "BHR", "IRN", "IRQ", "KWT", "QAT", "SAU"]
LAYERS = ("wheat", "ammonia", "urea", "lpg_propane", "lpg_butane")
YEARS = range(2020, 2025)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)


def build_graph(W: np.ndarray, names: list[str]) -> ig.Graph:
    rows, cols = np.nonzero(W)
    edges = list(zip(rows.tolist(), cols.tolist()))
    weights = W[rows, cols].tolist()
    g = ig.Graph(n=len(names), edges=edges, directed=True)
    g.vs["name"] = names
    g.es["weight"] = weights
    return g


def infomap_partition(g: ig.Graph) -> tuple[list[int], float, float]:
    """Run InfoMap; return per-vertex community label, codelength, modularity."""
    part = g.community_infomap(edge_weights="weight")
    membership = part.membership
    cl = float(getattr(part, "codelength", float("nan")))
    q = g.modularity(membership, weights="weight")
    return membership, cl, q


def primary_hormuz_community(membership: list[int],
                             countries: list[str]) -> int | None:
    """Return the community label that contains the most Hormuz countries.

    Ties are broken by community size (smaller wins, since a tighter cluster
    is the more interpretable bloc).  Returns None if no Hormuz country is
    present in any community."""
    h_in = [(membership[countries.index(h)], h)
            for h in HORMUZ if h in countries]
    if not h_in:
        return None
    counts: dict[int, int] = {}
    for c, _ in h_in:
        counts[c] = counts.get(c, 0) + 1
    sizes: dict[int, int] = {}
    for m in membership:
        sizes[m] = sizes.get(m, 0) + 1
    best = sorted(counts.items(), key=lambda kv: (-kv[1], sizes[kv[0]]))[0]
    return best[0]


def main() -> None:
    countries = pd.read_csv(PHASE0 / "country_index.csv")["iso3"].tolist()
    rows, hb_rows = [], []

    for y in YEARS:
        for L in LAYERS:
            W = np.load(PHASE0 / f"W_{y}_{L}.npy")
            active = (W.sum(axis=0) > 0) | (W.sum(axis=1) > 0)
            if active.sum() < 5:
                continue
            g = build_graph(W, countries)
            membership, codelength, modularity = infomap_partition(g)
            primary = primary_hormuz_community(membership, countries)

            for k, c in enumerate(countries):
                if not active[k]:
                    continue
                rows.append({"year": y, "layer": L, "country": c,
                             "community": membership[k],
                             "is_primary_hormuz": int(membership[k] == primary),
                             "codelength": codelength,
                             "modularity": modularity})

            n_active = int(active.sum())
            n_comm = len({m for m, a in zip(membership, active) if a})
            primary_members = [c for k, c in enumerate(countries)
                               if active[k] and membership[k] == primary]
            primary_size = len(primary_members)
            primary_nonh = sum(1 for c in primary_members if c not in HORMUZ)
            primary_h = sum(1 for c in primary_members if c in HORMUZ)
            h_idx = [countries.index(h) for h in HORMUZ if h in countries]
            primary_idx = [k for k in range(len(countries))
                           if membership[k] == primary]
            h_total_export = float(W[h_idx, :].sum())
            h_to_primary = float(W[np.ix_(h_idx, primary_idx)].sum())
            primary_share = (h_to_primary / h_total_export) if h_total_export > 0 else 0.0

            hb_rows.append({"year": y, "layer": L,
                            "n_active": n_active,
                            "n_communities": n_comm,
                            "modularity": modularity,
                            "codelength": codelength,
                            "primary_size": primary_size,
                            "primary_n_hormuz": primary_h,
                            "primary_n_nonH": primary_nonh,
                            "primary_share_of_H_exports": primary_share})

    comm = pd.DataFrame(rows)
    comm.to_csv(OUT / "communities.csv", index=False)
    summary = pd.DataFrame(hb_rows)
    summary.to_csv(OUT / "hormuz_bloc_membership.csv", index=False)

    print("Phase 2.1 community-detection summary:")
    print(summary.to_string(index=False))

    # Figure: community count + modularity per (year, layer)
    pv_n = summary.pivot(index="year", columns="layer", values="n_communities")[list(LAYERS)]
    pv_q = summary.pivot(index="year", columns="layer", values="modularity")[list(LAYERS)]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    pv_n.plot(ax=axes[0], marker="o")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Number of communities (InfoMap)")
    axes[0].set_title("Community count by layer")
    axes[0].legend(title="layer", frameon=False, fontsize=8)
    pv_q.plot(ax=axes[1], marker="s")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Modularity Q (weighted, directed)")
    axes[1].set_title("Modularity by layer")
    axes[1].legend(title="layer", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_community_count.png", dpi=150)
    plt.close(fig)

    # Figure: primary-bloc size over time
    pv2 = summary.pivot(index="year", columns="layer",
                        values="primary_n_nonH")[list(LAYERS)]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    pv2.plot(ax=ax, marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Non-Hormuz countries in primary Hormuz community")
    ax.set_title("Size of the primary Hormuz-anchored community")
    ax.legend(title="layer", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_hormuz_bloc_size.png", dpi=150)
    plt.close(fig)

    # Figure: share of Hormuz exports captured by the primary community
    pv3 = summary.pivot(index="year", columns="layer",
                        values="primary_share_of_H_exports")[list(LAYERS)]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    pv3.plot(ax=ax, marker="o")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of Hormuz exports inside primary community")
    ax.set_title("How much Hormuz flow the bloc actually captures")
    ax.legend(title="layer", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_hormuz_bloc_share.png", dpi=150)
    plt.close(fig)

    print(f"Wrote 3 figures under {FIG}")


if __name__ == "__main__":
    main()
