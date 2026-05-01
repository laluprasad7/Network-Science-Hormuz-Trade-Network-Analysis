"""Phase 2.2 — topology over time + targeted-attack resilience under Hormuz removal.

For each (year, layer) we compute:

    Static topology
    ---------------
        n_nodes_active, n_edges, density,
        reciprocity (Garlaschelli-Loffredo on the binary support),
        avg_clustering (undirected projection, unweighted),
        lwcc_frac   (size of largest weakly-connected component / n_active),
        lscc_frac   (size of largest strongly-connected component / n_active),
        global_efficiency  ( mean over (i,j) of 1/d(i,j), cost = 1/W ).

    Targeted-attack resilience (zero out Hormuz rows = no Hormuz exports)
    ---------------------------------------------------------------------
        flow_loss        fraction of total trade volume removed,
        importers_cut    number of countries whose import in-degree drops to 0,
        lwcc_after       LWCC size as a fraction of pre-attack LWCC,
        eff_after        global efficiency as a fraction of pre-attack,
        eff_drop         1 - eff_after.

These complement Phase 2.1 (community structure) by quantifying how much of
each network's connectivity *depends* on the Hormuz set.  A drop close to
zero means the network reroutes; a drop close to 1 means Hormuz nodes are
articulation points the network cannot work around with its current edges.

Outputs:
    Results/phase2/topology.csv             one row per (year, layer)
    Results/phase2/hormuz_attack.csv        one row per (year, layer)
Figures:
    figures/phase2/fig2_topology_grid.png       2x2 grid: density, reciprocity,
                                                 avg-clustering, LWCC fraction
    figures/phase2/fig2_attack_efficiency.png   efficiency drop after Hormuz
                                                 removal, by year x layer
    figures/phase2/fig2_attack_flow_loss.png    flow-loss bars, by year x layer
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
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


def make_digraph(W: np.ndarray, names: list[str]) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(names)
    rows, cols = np.nonzero(W)
    for i, j in zip(rows, cols):
        G.add_edge(names[i], names[j], weight=float(W[i, j]),
                   cost=1.0 / float(W[i, j]))
    return G


def global_efficiency_weighted(G: nx.DiGraph) -> float:
    """Mean inverse shortest-path distance on cost = 1/weight.

    Uses Dijkstra over the strongly-connected reachable set; pairs with no
    path contribute zero.  Normalised by N*(N-1)."""
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    nodes = list(G.nodes)
    total = 0.0
    for src in nodes:
        lengths = nx.single_source_dijkstra_path_length(G, src, weight="cost")
        for tgt, d in lengths.items():
            if tgt == src or d <= 0:
                continue
            total += 1.0 / d
    return total / (n * (n - 1))


def static_topology(W: np.ndarray, names: list[str]) -> dict:
    active = (W.sum(axis=0) > 0) | (W.sum(axis=1) > 0)
    sub_idx = np.where(active)[0]
    sub_names = [names[i] for i in sub_idx]
    Wsub = W[np.ix_(sub_idx, sub_idx)]
    G = make_digraph(Wsub, sub_names)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = m / (n * (n - 1)) if n > 1 else 0.0
    reciprocity = nx.overall_reciprocity(G) if m > 0 else 0.0
    Gu = G.to_undirected(reciprocal=False)
    avg_clust = nx.average_clustering(Gu) if n > 1 else 0.0
    lwcc = max((len(c) for c in nx.weakly_connected_components(G)), default=0) / n if n else 0.0
    lscc = max((len(c) for c in nx.strongly_connected_components(G)), default=0) / n if n else 0.0
    eff = global_efficiency_weighted(G)
    return {"n_nodes_active": n, "n_edges": m, "density": density,
            "reciprocity": reciprocity, "avg_clustering": avg_clust,
            "lwcc_frac": lwcc, "lscc_frac": lscc,
            "global_efficiency": eff}


def hormuz_attack(W: np.ndarray, names: list[str]) -> dict:
    """Zero out Hormuz outflows; report damage relative to baseline."""
    base_total = float(W.sum())
    h_idx = [names.index(h) for h in HORMUZ if h in names]
    in_deg_before = (W.sum(axis=0) > 0)

    Wp = W.copy()
    Wp[h_idx, :] = 0.0
    in_deg_after = (Wp.sum(axis=0) > 0)
    importers_cut = int(((in_deg_before) & (~in_deg_after)).sum())
    flow_loss = 1.0 - float(Wp.sum()) / base_total if base_total > 0 else 0.0

    base = static_topology(W, names)
    after = static_topology(Wp, names)
    lwcc_after_rel = (after["lwcc_frac"] * after["n_nodes_active"]) / max(
        base["lwcc_frac"] * base["n_nodes_active"], 1)
    eff_after_rel = after["global_efficiency"] / max(base["global_efficiency"], 1e-12)

    return {"flow_loss": flow_loss,
            "importers_cut": importers_cut,
            "lwcc_after_rel": lwcc_after_rel,
            "eff_after_rel": eff_after_rel,
            "eff_drop": 1.0 - eff_after_rel}


def main() -> None:
    countries = pd.read_csv(PHASE0 / "country_index.csv")["iso3"].tolist()
    topo_rows, atk_rows = [], []

    for y in YEARS:
        for L in LAYERS:
            W = np.load(PHASE0 / f"W_{y}_{L}.npy")
            if W.sum() == 0:
                continue
            topo = static_topology(W, countries)
            topo.update({"year": y, "layer": L})
            topo_rows.append(topo)

            atk = hormuz_attack(W, countries)
            atk.update({"year": y, "layer": L})
            atk_rows.append(atk)
            print(f"  done {y} {L}: density={topo['density']:.3f} "
                  f"eff_drop={atk['eff_drop']:.3f} flow_loss={atk['flow_loss']:.3f}")

    topo = pd.DataFrame(topo_rows)
    topo.to_csv(OUT / "topology.csv", index=False)
    atk = pd.DataFrame(atk_rows)
    atk.to_csv(OUT / "hormuz_attack.csv", index=False)

    print("\nPhase 2.2 — static topology (mean across years per layer):")
    print(topo.groupby("layer")[["density", "reciprocity", "avg_clustering",
                                  "lwcc_frac", "lscc_frac",
                                  "global_efficiency"]].mean().round(3)
              .reindex(list(LAYERS)).to_string())

    print("\nPhase 2.2 — Hormuz-removal damage (mean across years per layer):")
    print(atk.groupby("layer")[["flow_loss", "importers_cut",
                                 "lwcc_after_rel", "eff_drop"]].mean().round(3)
              .reindex(list(LAYERS)).to_string())

    # Topology grid
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4), sharex=True)
    for ax, col, title in zip(
        axes.ravel(),
        ["density", "reciprocity", "avg_clustering", "lwcc_frac"],
        ["Density", "Reciprocity", "Avg clustering (undirected)",
         "Largest weakly-connected component (frac)"]):
        pv = topo.pivot(index="year", columns="layer", values=col)[list(LAYERS)]
        pv.plot(ax=ax, marker="o", legend=False)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel(col)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="layer", loc="upper center",
               ncols=5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "fig2_topology_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Efficiency drop figure
    pv = atk.pivot(index="year", columns="layer", values="eff_drop")[list(LAYERS)]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    pv.plot(ax=ax, marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Drop in global efficiency after Hormuz removal")
    ax.set_title("Network efficiency loss when Hormuz exports are removed")
    ax.set_ylim(0, max(0.05, pv.values.max() * 1.1))
    ax.legend(title="layer", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_attack_efficiency.png", dpi=150)
    plt.close(fig)

    # Flow-loss bar chart (2024 only, comparable across layers)
    sub = atk[atk.year == 2024].copy()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    order = list(LAYERS)
    sub = sub.set_index("layer").reindex(order).reset_index()
    sns.barplot(data=sub, x="layer", y="flow_loss", ax=ax,
                hue="layer", palette="rocket_r", legend=False)
    for k, v in enumerate(sub["flow_loss"]):
        ax.text(k, v + 0.005, f"{v*100:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, max(0.05, sub["flow_loss"].max() * 1.25))
    ax.set_ylabel("Share of trade volume lost")
    ax.set_xlabel("")
    ax.set_title("2024 — flow loss from removing Hormuz exports")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_attack_flow_loss.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote 3 figures under {FIG}")


if __name__ == "__main__":
    main()
