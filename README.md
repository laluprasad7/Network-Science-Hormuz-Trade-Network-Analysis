# Network Science: Hormuz Trade-Network Analysis

A multi-layer trade-network resilience study of five commodity layers
(wheat, ammonia, urea, LPG propane, LPG butane) under Strait of Hormuz
disruption. Combines weighted-network topology, community detection,
targeted-attack simulation, panel and spatial econometrics, and a
scenario engine to estimate country-level import impacts.

> **Course:** DAL-311 Network Science · IIT Roorkee · May 2026
>
> **Group members:** Preetham Gunta, Vineel Krishna Manchala,
> Laluprasad Ramavath, Prabhath Kolli

The final write-up is in [`Final_Report/report.pdf`](Final_Report/report.pdf).

---

## What this repository contains

```
.
├── Project_Data/        Raw upstream CSV / XLSX files (Comtrade, JODI,
│                        World Bank Pink Sheet, FRED, IMF PortWatch)
├── Processed_Data/      Cleaned tidy CSVs produced by the preprocess_*
│                        scripts (input to the analysis pipeline)
├── scripts/             All pipeline code (preprocessing + Phase 0–5)
├── Results/             Generated CSV outputs by phase
│   ├── phase0/   Per-(year, layer) trade matrices A, W (.npy) + index
│   ├── phase1/   Composite exposure scores per (country, year, layer)
│   ├── phase2/   Community detection, topology, Hormuz-attack damage
│   ├── phase3/   Monthly econometric panel, VAR, panel-FE, SDM, SLX
│   └── phase4/   Scenario simulation country-level impacts
├── figures/             All plots (PNG, 150–170 dpi)
│   ├── IITR_logo.png    Institute logo used on the report cover
│   └── phase0/ … phase5/
├── Final_Report/        LaTeX source + compiled PDF + .bib
│   ├── report.tex
│   ├── references.bib
│   └── report.pdf
├── Reference Papers/    Literature and related papers
├── README.md            (this file)
├── requirements.txt     Pinned Python dependencies
└── .gitignore
```

---

## Headline findings

| Finding | Evidence |
|---|---|
| LPG butane is the most fragile layer to a Hormuz closure | 48.5 % of 2024 trade volume + 32.4 % of weighted global efficiency lost on Hormuz removal |
| Hormuz tanker capacity Granger-causes ammonia (DAP) prices | VAR Granger *p* = 0.010, IRF peak = −0.015 at *h* = 1 |
| Modest network contagion across countries | SDM ρ ≈ 0.11 → 13 % Leontief amplifier |
| India is the single largest absolute scenario loser | −\$7.6 B annual import value at risk under a 6-month realignment (LPG dependence ≈ 97 %) |
| Wheat is structurally resilient | only 0.13 % flow loss when Hormuz is removed |

See [`Final_Report/report.pdf`](Final_Report/report.pdf) §5 (Results) and
§6 (Discussion) for the full account.

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/PreethamGunta/Network-Science-Propane-Trade-Network-Analysis.git
cd Network-Science-Propane-Trade-Network-Analysis
```

### 2. Set up the Python environment

Tested with **Python 3.12** on Windows 11. From the repo root:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, the direct dependencies are:

```
numpy pandas matplotlib seaborn networkx python-igraph
statsmodels spreg libpysal scikit-learn openpyxl
```

### 3. Run the pipeline

Each script is self-contained and writes to `Results/` and `figures/`.
Run in this order (subsequent phases depend on earlier outputs):

```bash
# Preprocessing — cleans Project_Data → Processed_Data
python scripts/preprocess_comtrade.py
python scripts/preprocess_cmo_pinksheet.py
python scripts/preprocess_fred_propane.py
python scripts/preprocess_portwatch.py
python scripts/preprocess_jodi.py

# Phase 0 — trade-matrix construction
python scripts/phase0_trade_matrices.py
python scripts/phase0_ras_balance.py
python scripts/phase0_figures.py

# Phase 1 — composite exposure scoring
python scripts/phase1_exposure.py
python scripts/phase1_composite.py

# Phase 2 — networks, communities, attack
python scripts/phase2_communities.py
python scripts/phase2_resilience.py

# Phase 3 — panel econometrics
python scripts/phase3_panel.py
python scripts/phase3_pvar.py
python scripts/phase3_sdm.py

# Phase 4 — scenario simulation
python scripts/phase4_scenarios.py

# Phase 5 — network-graph figures for the report
python scripts/phase5_network_figure.py
```

### 4. Build the PDF report

Requires a TeX distribution (TeXLive, MiKTeX, MacTeX). From
`Final_Report/`:

```bash
cd Final_Report
pdflatex report.tex
bibtex   report
pdflatex report.tex
pdflatex report.tex       # second pass for cross-references
```

The output is `Final_Report/report.pdf`.

---

## Methodology in brief

**Phase 0** — bilateral monthly Comtrade flows are aggregated into
153 × 153 weighted directed adjacency matrices (`W_{year,layer}`) and
binary supports (`A_{year,layer}`), one per (year, layer) for
2020–2026. A single-pass RAS rebalance reconciles row / column
marginals.

**Phase 1** — eight country-level exposure metrics are computed
(*Direct Exposure*, *Source Redundancy*, *Betweenness*, *Katz*,
*Personalised PageRank*, *Leontief multiplier*, *DebtRank*, *PIVI*),
*z*-scored within each (year, layer) and combined into a
`composite_01` score in [0, 1].

**Phase 2** — InfoMap community detection on each weighted directed
graph; topology metrics (density, reciprocity, clustering, weighted
global efficiency, LWCC / LSCC); a targeted Hormuz-removal attack
that zeroes out export rows of {ARE, BHR, IRN, IRQ, KWT, QAT, SAU}
and reports flow loss + efficiency drop.

**Phase 3** — bivariate VAR(1) per layer on
[Δlog price, Hormuz DWT] with Cholesky IRF; panel-FE regression
with country × time effects and a DE × DWT interaction; spatial
Durbin (`spreg.Panel_FE_Lag`) with W = row-normalised 2020 trade
share; SLX side-model.

**Phase 4** — three Hormuz disruption profiles (14-day, 30–45 day,
6-month). Country impacts decomposed into a structural channel
(`log(1 − DE × disruption)`) and a price channel (long-run IRF ×
SDM β\_log_price), amplified by the SDM Leontief multiplier
1 / (1 − ρ).

Full details (with formulas, citations and caveats) are in
`Final_Report/report.pdf`.

---

## Data sources

All raw files in `Project_Data/` are publicly downloadable:

| Dataset | Source | URL |
|---|---|---|
| Bilateral monthly trade | UN Comtrade Plus | <https://comtradeplus.un.org> |
| Hormuz tanker DWT (daily) | IMF PortWatch | <https://portwatch.imf.org> |
| Commodity benchmark prices (monthly) | World Bank Pink Sheet | <https://www.worldbank.org/en/research/commodity-markets> |
| Mont-Belvieu propane (daily) | FRED (DPROPANEMBTX) | <https://fred.stlouisfed.org/series/DPROPANEMBTX> |
| NGL production / stocks (monthly) | JODI World Database | <https://www.jodidata.org> |

If you re-download upstream files, the preprocessing scripts will
regenerate `Processed_Data/` and the rest of the pipeline.

---

## Reproducibility notes

- All randomised steps (InfoMap, spring layout) use a fixed seed
  (42) so figures and community labels are reproducible.
- Trade matrices are committed as `.npy` to avoid recomputation
  cost on slow machines.
- The pipeline is single-threaded; full re-run is ~10–15 minutes
  on a recent laptop.

---

## License

Code and figures: MIT.
Data files in `Project_Data/` retain the licenses of their upstream
providers (cited above).

---

## Citation

If you use this work, please cite the project report:

> Preetham Gunta, Vineel Krishna Manchala, Laluprasad Ramavath,
> and Prabhath Kolli. *Trade-Network Resilience and the Strait of
> Hormuz: A Multi-Layer Exposure Analysis of Five Commodity Networks
> (2020–2024).* Course project, DAL-311 Network Science,
> IIT Roorkee, May 2026.
