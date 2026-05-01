"""Phase 5 — generate trade-network graph figures for the final report.

Two flavours:
  * single-layer focused network (LPG butane 2024 — the most exposed layer)
  * 5-layer comparison panel (one network per layer, 2024)

In all plots:
    nodes    = countries (ISO3 codes)
    edges    = directed bilateral trade flows (exporter -> importer)
    node size = country's total trade involvement = weighted (in + out) degree
    node colour = red for Hormuz exporter states, blue otherwise
    edge width / alpha = log of trade weight
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PHASE0 = ROOT / "Results" / "phase0"
FIG = ROOT / "figures" / "phase5"
FIG.mkdir(parents=True, exist_ok=True)

HORMUZ = {"ARE", "BHR", "IRN", "IRQ", "KWT", "QAT", "SAU"}

LAYER_LABEL = {
    "wheat": "Wheat",
    "ammonia": "Ammonia",
    "urea": "Urea",
    "lpg_propane": "LPG propane",
    "lpg_butane": "LPG butane",
}


def build_top_edge_graph(layer: str, year: int, top_n: int):
    countries = pd.read_csv(PHASE0 / "country_index.csv")["iso3"].tolist()
    W = np.load(PHASE0 / f"W_{year}_{layer}.npy")
    flat = [(i, j, W[i, j]) for i in range(len(countries))
            for j in range(len(countries)) if W[i, j] > 0]
    flat.sort(key=lambda x: -x[2])
    keep = flat[:top_n]

    G = nx.DiGraph()
    nodes = sorted({countries[i] for i, j, _ in keep} |
                   {countries[j] for i, j, _ in keep})
    G.add_nodes_from(nodes)
    for i, j, w in keep:
        G.add_edge(countries[i], countries[j], weight=float(w))
    return G, W


def draw_network(G: nx.DiGraph, W: np.ndarray, title: str,
                  ax: plt.Axes, label_top: int = 18) -> None:
    in_deg = dict(G.in_degree(weight="weight"))
    out_deg = dict(G.out_degree(weight="weight"))
    total = {n: in_deg.get(n, 0) + out_deg.get(n, 0) for n in G.nodes}
    max_t = max(total.values()) if total else 1.0

    pos = nx.spring_layout(G, seed=42, k=1.7, iterations=120)

    sizes = [60 + 1500 * (total[n] / max_t) for n in G.nodes]
    colors = ["#c0392b" if n in HORMUZ else "#3498db" for n in G.nodes]

    wmax = max(W.max(), 1e-9)
    edge_widths = [0.3 + 2.4 * (G[u][v]["weight"] / wmax) for u, v in G.edges]
    edge_alphas = [min(0.85, 0.2 + 0.7 * (G[u][v]["weight"] / wmax))
                   for u, v in G.edges]

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                            alpha=edge_alphas, edge_color="#7f8c8d",
                            arrows=True, arrowsize=7,
                            connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                            node_color=colors, edgecolors="black",
                            linewidths=0.5)
    if total:
        thr_idx = min(label_top, len(total) - 1)
        thr = sorted(total.values(), reverse=True)[thr_idx]
        labels = {n: n for n in G.nodes if total[n] >= thr}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                                 font_size=7, font_weight="bold")
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def plot_single_layer(layer: str, year: int = 2024, top_n: int = 90) -> None:
    G, W = build_top_edge_graph(layer, year, top_n)
    fig, ax = plt.subplots(figsize=(11, 8))
    title = (f"{LAYER_LABEL[layer]} trade network ({year}) — top {top_n} edges. "
             f"Red = Hormuz exporter; node size {chr(0x221d)} weighted degree.")
    draw_network(G, W, title, ax, label_top=22)
    fig.tight_layout()
    out = FIG / f"fig5_network_{layer}_{year}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_layer_panel(year: int = 2024, top_n: int = 60) -> None:
    layers = ["wheat", "ammonia", "urea", "lpg_propane", "lpg_butane"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    axes = axes.ravel()
    for k, layer in enumerate(layers):
        G, W = build_top_edge_graph(layer, year, top_n)
        draw_network(G, W, f"{LAYER_LABEL[layer]} ({year})", axes[k],
                     label_top=10)
    axes[-1].set_axis_off()
    # Legend in last panel
    axes[-1].scatter([0.2], [0.7], s=200, c="#c0392b",
                     edgecolors="black", label="Hormuz exporter")
    axes[-1].scatter([0.2], [0.5], s=200, c="#3498db",
                     edgecolors="black", label="Non-Hormuz country")
    axes[-1].text(0.05, 0.85,
                   "Nodes  : countries\n"
                   "Edges  : directed trade flow\n"
                   "Size   $\\propto$ weighted degree\n"
                   "Width  $\\propto$ trade volume",
                   fontsize=10, va="top")
    axes[-1].legend(loc="lower left", frameon=False, fontsize=10)
    axes[-1].set_xlim(0, 1)
    axes[-1].set_ylim(0, 1)
    fig.suptitle(f"Bilateral trade networks across five commodity layers ({year}, top {top_n} edges per layer)",
                 fontsize=12)
    fig.tight_layout()
    out = FIG / f"fig5_network_panel_{year}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_layer_panel(2024)
    plot_single_layer("lpg_butane", 2024)
    plot_single_layer("urea", 2024)
    plot_single_layer("wheat", 2024)
