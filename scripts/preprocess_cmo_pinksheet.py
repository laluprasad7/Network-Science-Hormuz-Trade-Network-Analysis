"""Clean the World Bank Commodities (Pink Sheet) workbook.

Inputs:
    Project_Data/CMO-Historical-Data-Monthly.xlsx

Outputs:
    Processed_Data/cmo_monthly_prices.csv
    Processed_Data/cmo_monthly_indices.csv

Cleaning rules:
    1. Both useful sheets ("Monthly Prices", "Monthly Indices") have a multi-row
       header band followed by a column of period strings of the form
       "YYYYMNN". We rebuild a clean flat header from the band, parse the
       period column to a real date (`period_dt`, first of month), and
       discard the descriptive header rows.
    2. Missing values are encoded as "..." in source which renders as the
       Unicode replacement character (U+FFFD) when read with cp1252; we
       coerce any non-numeric cell in the data block to NaN.
    3. Monthly Indices has a hierarchical header band (top level / sub-level
       / sub-sub-level) where some labels span multiple columns visually but
       are stored in a single column position. We flatten the hierarchy to
       short snake_case names (see CMO_INDEX_COLS).
    4. Price column names are snake_cased (`crude_oil_brent`,
       `liquefied_natural_gas_japan`, `urea`, `wheat_us_hrw`, ...) with the
       unit appended in a separate metadata file
       (`cmo_monthly_prices.metadata.csv`) so the data column itself stays
       compact and joinable.
    5. We keep the entire 1960M01 onwards history. Filtering to the
       2020-01 .. 2026-02 modelling window happens in downstream scripts so
       the cleaned long-history file remains reusable.
"""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Project_Data" / "CMO-Historical-Data-Monthly.xlsx"
OUT = ROOT / "Processed_Data"
OUT.mkdir(exist_ok=True)


CMO_INDEX_COLS = {
    1:  "total_index",
    2:  "energy",
    3:  "non_energy",
    4:  "agriculture",
    5:  "beverages",
    6:  "food",
    7:  "oils_meals",
    8:  "grains",
    9:  "other_food",
    10: "raw_materials",
    11: "timber",
    12: "other_raw_mat",
    13: "fertilizers",
    14: "metals_minerals",
    15: "base_metals_ex_iron",
    16: "precious_metals",
}


def _slug(s: str) -> str:
    s = s.strip().rstrip("*").strip()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    return s.strip("_").lower()


def _parse_period(s: str) -> pd.Timestamp:
    # "1960M01" -> 1960-01-01
    return pd.Timestamp(year=int(s[:4]), month=int(s[5:7]), day=1)


def clean_monthly_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(RAW, sheet_name="Monthly Prices", header=None)
    name_row = raw.iloc[0]
    unit_row = raw.iloc[1]
    data = raw.iloc[2:].reset_index(drop=True)

    period = data.iloc[:, 0].astype(str)
    period_mask = period.str.match(r"^\d{4}M\d{2}$")
    data = data[period_mask].copy()

    # Build clean column names (col 0 -> period_dt; col i -> slug(name))
    cols = ["period_dt"]
    metadata = []
    for i in range(1, raw.shape[1]):
        name = name_row.iat[i]
        unit = unit_row.iat[i]
        slug = _slug(str(name)) if pd.notna(name) else f"col_{i}"
        cols.append(slug)
        metadata.append({"col": slug,
                         "raw_name": str(name).strip().rstrip("*").strip()
                                       if pd.notna(name) else "",
                         "unit": str(unit).strip() if pd.notna(unit) else ""})
    data.columns = cols

    data["period_dt"] = data["period_dt"].astype(str).map(_parse_period)
    for c in cols[1:]:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    return data.reset_index(drop=True), pd.DataFrame(metadata)


def clean_monthly_indices() -> pd.DataFrame:
    raw = pd.read_excel(RAW, sheet_name="Monthly Indices", header=None)
    # Data starts where col 0 matches "YYYYMNN"
    period = raw.iloc[:, 0].astype(str)
    data = raw[period.str.match(r"^\d{4}M\d{2}$")].copy().reset_index(drop=True)

    cols = ["period_dt"] + [CMO_INDEX_COLS[i] for i in range(1, raw.shape[1])]
    data.columns = cols
    data["period_dt"] = data["period_dt"].astype(str).map(_parse_period)
    for c in cols[1:]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    return data


def main() -> None:
    prices, prices_meta = clean_monthly_prices()
    indices = clean_monthly_indices()

    p = OUT / "cmo_monthly_prices.csv"
    pmeta = OUT / "cmo_monthly_prices.metadata.csv"
    i = OUT / "cmo_monthly_indices.csv"
    prices.to_csv(p, index=False)
    prices_meta.to_csv(pmeta, index=False)
    indices.to_csv(i, index=False)

    print(f"prices : {prices.shape} {prices.period_dt.min().date()} -> {prices.period_dt.max().date()}  -> {p.name}")
    print(f"indices: {indices.shape} {indices.period_dt.min().date()} -> {indices.period_dt.max().date()}  -> {i.name}")

    # Quick coverage check on the modelling window
    win = prices[(prices.period_dt >= "2020-01-01") & (prices.period_dt <= "2026-02-01")]
    pct_missing = win.iloc[:, 1:].isna().mean().sort_values(ascending=False)
    print()
    print("Top-10 columns with most missing values inside 2020-01..2026-02:")
    print(pct_missing.head(10).to_string())
    print()
    print("Pipeline-relevant series, missingness in window:")
    relevant = ["crude_oil_brent", "crude_oil_dubai", "crude_oil_average",
                "natural_gas_us", "natural_gas_europe",
                "liquefied_natural_gas_japan", "natural_gas_index",
                "wheat_us_hrw", "wheat_us_srw", "urea", "dap"]
    have = [c for c in relevant if c in pct_missing.index]
    print(pct_missing.reindex(have).to_string())


if __name__ == "__main__":
    main()
