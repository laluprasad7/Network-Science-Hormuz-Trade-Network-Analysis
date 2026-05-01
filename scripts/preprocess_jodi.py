"""Clean the JODI energy stocks/flows yearly extracts and assess viability.

Inputs:
    Project_Data/jodi 2020.csv ... jodi 2026.csv

Outputs:
    Processed_Data/jodi_long.csv          tidy long-format panel (KTONS)
    Processed_Data/jodi_ngl_wide.csv      NGL-only wide-format panel
    Processed_Data/jodi_coverage_report.csv     per-(product,flow) coverage

Why this script exists in this exact form:
    * The user flagged JODI as the highest-risk dataset, full of missing
      values and zeros. We do NOT drop it pre-emptively. We clean it,
      compute coverage, and document the result so a later modelling step
      can decide whether the *processed* JODI is good enough. If 2.3
      Phase 2 reserve calibration only needs the OECD + major-exporter
      subset, the data is usable as-is.

Cleaning rules:
    1. Concatenate all yearly files into a single panel.
    2. Filter to UNIT_MEASURE == 'KTONS'. The raw files store every
       observation in 5 unit-measure rows (CONVBBL, KBBL, KBD, KL, KTONS),
       which is purely redundant duplication once we pick a unit. KTONS
       is chosen because it covers every FLOW_BREAKDOWN (KBD has
       0% coverage on stock-level flows: CLOSTLV, STOCKCH, OSOURCES,
       TRANSBAK).
    3. Recode missing-value sentinels: OBS_VALUE strings '-' and 'x'
       (and any other non-numeric) become NaN.
    4. Add `period_dt` (first of month) parsed from TIME_PERIOD = 'YYYY-MM'.
    5. Drop `ASSESSMENT_CODE` (1/2/3 assessment quality flag — internal
       JODI use, not modelling-relevant in our framework).
    6. Re-name columns to lowercase snake_case, add a `flow_label` column
       expanding the JODI FLOW_BREAKDOWN codes for human readability.
    7. Compute and emit a per-(product, flow) coverage table so the
       sparseness picture is documented alongside the data.
    8. Pivot the NGL slice (Natural Gas Liquids — our LPG-relevant layer)
       to a wide panel keyed on (ref_area, period_dt) with one numeric
       column per flow. This is the table Phase 2 reserve-threshold
       calibration will read directly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "Project_Data"
OUT = ROOT / "Processed_Data"
OUT.mkdir(exist_ok=True)


FLOW_LABELS = {
    "CLOSTLV":  "closing_stock_level",
    "DIRECUSE": "direct_use",
    "INDPROD":  "indigenous_production",
    "OSOURCES": "from_other_sources",
    "REFINOBS": "refinery_observed_intake",
    "STATDIFF": "statistical_difference",
    "STOCKCH":  "stock_change",
    "TOTEXPSB": "total_exports",
    "TOTIMPSB": "total_imports",
    "TRANSBAK": "transfer_backflow",
}


def load_all() -> pd.DataFrame:
    parts = []
    for f in sorted(RAW_DIR.glob("jodi *.csv")):
        parts.append(pd.read_csv(f, low_memory=False))
    return pd.concat(parts, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["UNIT_MEASURE"] == "KTONS"].copy()
    df["obs_value"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df["period_dt"] = pd.to_datetime(df["TIME_PERIOD"] + "-01")

    df = df.rename(columns={
        "REF_AREA": "ref_area",
        "ENERGY_PRODUCT": "energy_product",
        "FLOW_BREAKDOWN": "flow_breakdown",
        "UNIT_MEASURE": "unit_measure",
    })
    df["flow_label"] = df["flow_breakdown"].map(FLOW_LABELS)
    df = df[["ref_area", "period_dt", "energy_product",
             "flow_breakdown", "flow_label", "unit_measure", "obs_value"]]
    return df


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["energy_product", "flow_breakdown", "flow_label"])
    out = g["obs_value"].agg(
        n_total="size",
        n_present=lambda s: s.notna().sum(),
        n_zero=lambda s: (s == 0).sum(),
    ).reset_index()
    out["coverage_pct"] = (100 * out["n_present"] / out["n_total"]).round(1)
    out["zero_pct"]     = (100 * out["n_zero"]    / out["n_total"]).round(1)
    return out.sort_values(["energy_product", "flow_breakdown"])


def ngl_wide(df: pd.DataFrame) -> pd.DataFrame:
    ngl = df[df.energy_product == "NGL"].copy()
    wide = (ngl.pivot_table(index=["ref_area", "period_dt"],
                            columns="flow_label",
                            values="obs_value",
                            aggfunc="first")
              .reset_index())
    wide.columns.name = None
    return wide


def main() -> None:
    raw = load_all()
    print(f"raw concatenated rows  : {len(raw):,}")

    df = clean(raw)
    print(f"after KTONS filter     : {len(df):,}")
    print(f"period range           : {df.period_dt.min().date()} -> {df.period_dt.max().date()}")
    print(f"areas / products       : {df.ref_area.nunique()} / {df.energy_product.nunique()}")
    print(f"numeric obs            : {df.obs_value.notna().sum():,}  ({100*df.obs_value.notna().mean():.1f}%)")
    print(f"zero obs (of numeric)  : {(df.obs_value==0).sum():,}  ({100*(df.obs_value==0).sum()/df.obs_value.notna().sum():.1f}% of present)")

    long_out = OUT / "jodi_long.csv"
    df.to_csv(long_out, index=False)

    cov = coverage(df)
    cov_out = OUT / "jodi_coverage_report.csv"
    cov.to_csv(cov_out, index=False)
    print()
    print("Coverage by (product, flow), KTONS:")
    print(cov.to_string(index=False))

    wide = ngl_wide(df)
    wide_out = OUT / "jodi_ngl_wide.csv"
    wide.to_csv(wide_out, index=False)
    print()
    print(f"NGL wide panel: {wide.shape}  -> {wide_out.name}")
    # Coverage of the most modelling-relevant NGL columns
    relevant = ["closing_stock_level", "indigenous_production",
                "total_imports", "total_exports"]
    have = [c for c in relevant if c in wide.columns]
    print()
    print("NGL wide modelling-window coverage (2020-01..2026-02):")
    win = wide[(wide.period_dt >= "2020-01-01") &
               (wide.period_dt <= "2026-02-01")]
    for c in have:
        n_areas_with_any = (win.groupby("ref_area")[c]
                              .apply(lambda s: s.notna().any()).sum())
        cell_cov = 100 * win[c].notna().mean()
        print(f"  {c:30s}  cells: {cell_cov:5.1f}%  areas with >=1 obs: {n_areas_with_any}")


if __name__ == "__main__":
    main()
