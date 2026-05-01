"""Clean the FRED Mont Belvieu Propane Spot Price series (DPROPANEMBTX).

Inputs:
    Project_Data/DPROPANEMBTX.csv.xls (despite the extension, this is a CSV)

Outputs:
    Processed_Data/fred_propane_daily.csv
    Processed_Data/fred_propane_monthly.csv

Cleaning rules:
    1. The .xls extension is misleading; the file is a plain comma-delimited
       CSV with two columns: observation_date, DPROPANEMBTX.
    2. The DPROPANEMBTX column has 57 blank cells (weekends are already
       excluded from the index, so the gaps are holidays + occasional
       reporting gaps). They become NaN on read.
    3. We add `period_dt` (first day of month) and produce a monthly file
       with three aggregates: mean, last (end-of-month price), and count of
       valid trading days. The mean is used as the canonical monthly price;
       last is kept as a robustness alternative; count is the data-quality
       column for any monthly observation.
    4. The series starts 2021-03-23, so the modelling window (2020-01) is
       missing the first ~14 months. This is documented but not patched
       here; see report.md for fallback options (Brent, LNG-Japan).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Project_Data" / "DPROPANEMBTX.csv.xls"
OUT = ROOT / "Processed_Data"
OUT.mkdir(exist_ok=True)


def main() -> None:
    df = pd.read_csv(RAW)
    df.columns = ["date", "propane_usd_per_gal"]
    df["date"] = pd.to_datetime(df["date"])
    df["propane_usd_per_gal"] = pd.to_numeric(df["propane_usd_per_gal"],
                                              errors="coerce")

    daily_out = OUT / "fred_propane_daily.csv"
    df.to_csv(daily_out, index=False)

    df["period_dt"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df.groupby("period_dt")["propane_usd_per_gal"]
          .agg(propane_mean="mean",
               propane_last="last",
               n_valid_days="count")
          .reset_index()
    )
    monthly_out = OUT / "fred_propane_monthly.csv"
    monthly.to_csv(monthly_out, index=False)

    print(f"daily   : {df.shape}  range {df.date.min().date()} -> {df.date.max().date()}  -> {daily_out.name}")
    print(f"monthly : {monthly.shape}  range {monthly.period_dt.min().date()} -> {monthly.period_dt.max().date()}  -> {monthly_out.name}")
    print(f"daily NaNs: {df.propane_usd_per_gal.isna().sum()}")
    short = monthly[monthly.n_valid_days < 15]
    print(f"months with <15 valid trading days: {len(short)}  (heuristic for 'thin' months)")
    if not short.empty:
        print(short.to_string(index=False))


if __name__ == "__main__":
    main()
