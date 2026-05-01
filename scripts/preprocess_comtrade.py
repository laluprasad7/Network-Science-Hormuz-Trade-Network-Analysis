"""Clean the UN Comtrade monthly extract into bilateral and world-total panels.

Inputs:
    Project_Data/Comtrade_monthly_2020_to_2026.csv

Outputs (under Processed_Data/):
    comtrade_bilateral_monthly.csv   bilateral importer-reported flows
    comtrade_world_imports_monthly.csv   importer x commodity x month totals (partner=World)

What this script fixes:
    1. The raw header has 47 column names but every data row stores 46 real
       values + 1 trailing empty cell. The intended layout drops the leading
       `typeCode` column, so we shift the headers left by one and discard the
       trailing pad.
    2. After the shift `freqCode` is constant 'M', `flowCode` is constant 'M'
       (Import) and several flag columns add no information for our use case;
       they are dropped.
    3. Self-loops (reporterISO == partnerISO) are dropped: they appear as
       customs-warehouse / re-import noise in some reporters and they cannot
       form a meaningful edge in the trade network.
    4. World-total rows (partnerISO == 'W00') are split into a separate file
       so that dependency-share computations downstream can use them as
       importer-side denominators without contaminating the bilateral edge
       list.
    5. Adds:
         period_dt       first day of the month, ISO date
         layer           short commodity tag: lpg_propane, lpg_butane,
                         ammonia, urea, wheat
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Project_Data" / "Comtrade_monthly_2020_to_2026.csv"
OUT = ROOT / "Processed_Data"
OUT.mkdir(exist_ok=True)

LAYER_MAP = {
    1001:   "wheat",
    2814:   "ammonia",
    271112: "lpg_propane",
    271113: "lpg_butane",
    310210: "urea",
}

DROP_COLS = [
    "freqCode",                  # always 'M'
    "flowCode",                  # always 'M' (Import)
    "flowDesc",                  # always 'Import'
    "classificationCode",        # always 'H5'
    "classificationSearchCode",  # always 'HS'
    "isOriginalClassification",  # always True
    "customsCode", "customsDesc",
    "mosCode",
    "motCode", "motDesc",
    "partner2Code", "partner2ISO", "partner2Desc",  # always 'World' / 0
    "isQtyEstimated", "isAltQtyEstimated",
    "isNetWgtEstimated", "isGrossWgtEstimated",
    "legacyEstimationFlag",
    "isReported",
    "isAggregate",
    "isLeaf",
    "altQtyUnitCode", "altQtyUnitAbbr", "altQty",
    "grossWgt",
    "refPeriodId",
]


def load_raw() -> pd.DataFrame:
    with open(RAW, encoding="cp1252") as f:
        header = next(csv.reader(f))
    real_cols = header[1:]                          # drop unused 'typeCode'
    df = pd.read_csv(RAW, encoding="cp1252", skiprows=1, header=None, low_memory=False)
    df = df.iloc[:, : len(real_cols)]               # drop trailing empty pad
    df.columns = real_cols
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Period -> first-of-month date
    df["period_dt"] = pd.to_datetime(
        df["refYear"].astype(str) + "-" + df["refMonth"].astype(str).str.zfill(2) + "-01"
    )

    # Commodity short label
    df["layer"] = df["cmdCode"].map(LAYER_MAP)
    if df["layer"].isna().any():
        unknown = sorted(df.loc[df.layer.isna(), "cmdCode"].unique())
        raise RuntimeError(f"Unmapped cmdCode values: {unknown}")

    # Drop self-loops
    df = df[df["reporterISO"] != df["partnerISO"]].copy()

    return df


def main() -> None:
    df = load_raw()
    n0 = len(df)
    df = clean(df)
    n1 = len(df)

    bilateral = df[df["partnerISO"] != "W00"].copy()
    world = df[df["partnerISO"] == "W00"].copy()
    world = world.drop(columns=["partnerCode", "partnerISO", "partnerDesc"])

    bilateral_out = OUT / "comtrade_bilateral_monthly.csv"
    world_out = OUT / "comtrade_world_imports_monthly.csv"
    bilateral.to_csv(bilateral_out, index=False)
    world.to_csv(world_out, index=False)

    print(f"raw rows                : {n0:>9,}")
    print(f"after drop self-loops   : {n1:>9,}")
    print(f"bilateral rows          : {len(bilateral):>9,}  -> {bilateral_out.name}")
    print(f"world-total rows        : {len(world):>9,}  -> {world_out.name}")
    print(f"period range            : {df.period_dt.min().date()} -> {df.period_dt.max().date()}")
    print(f"reporters / partners    : {bilateral.reporterISO.nunique()} / {bilateral.partnerISO.nunique()}")
    print("layer counts (bilateral):")
    print(bilateral.layer.value_counts().to_string())


if __name__ == "__main__":
    main()
