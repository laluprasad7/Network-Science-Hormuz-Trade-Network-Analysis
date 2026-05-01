"""Clean the IMF PortWatch daily-chokepoint dataset.

Inputs:
    Project_Data/daily-chokepoint.csv

Outputs:
    Processed_Data/portwatch_chokepoint_daily.csv         all 28 chokepoints
    Processed_Data/portwatch_chokepoint_monthly.csv       monthly aggregates
    Processed_Data/portwatch_hormuz_daily.csv             Strait of Hormuz only
    Processed_Data/portwatch_hormuz_monthly.csv           Hormuz monthly

Cleaning rules:
    1. The `date` column is an ISO timestamp with UTC offset; we drop the
       time component and tz info so it joins cleanly with all the
       monthly tables.
    2. The `chokepointN` ids are not memorable; we add a `chokepoint_slug`
       column with snake_case names (`hormuz`, `bab_el_mandeb`, `suez`,
       `cape_of_good_hope`, ...) for safer joins.
    3. Capacity is in DWT (deadweight tonnage); n_* are vessel counts.
       Both are kept verbatim. 2 361 rows have `capacity == 0`; these are
       legitimate days at minor straits with no traffic, not missing data.
    4. Daily Hormuz shocks are the operational signal for Phase 4 scenario
       triggering, so we ship a Hormuz-only file alongside the all-chokepoint
       file. The all-chokepoint monthly file is what feeds the trade-flow
       monthly panel.
    5. Monthly aggregation: vessel counts and capacity are summed across
       days (totals matter, not averages); we also report `n_days` for each
       (chokepoint, month) tuple as a data-quality flag.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Project_Data" / "daily-chokepoint.csv"
OUT = ROOT / "Processed_Data"
OUT.mkdir(exist_ok=True)


SLUG = {
    "Suez Canal": "suez",
    "Panama Canal": "panama",
    "Bosporus Strait": "bosporus",
    "Bab el-Mandeb Strait": "bab_el_mandeb",
    "Malacca Strait": "malacca",
    "Strait of Hormuz": "hormuz",
    "Cape of Good Hope": "cape_of_good_hope",
    "Gibraltar Strait": "gibraltar",
    "Dover Strait": "dover",
    "Oresund Strait": "oresund",
    "Taiwan Strait": "taiwan",
    "Korea Strait": "korea",
    "Tsugaru Strait": "tsugaru",
    "Luzon Strait": "luzon",
    "Lombok Strait": "lombok",
    "Ombai Strait": "ombai",
    "Bohai Strait": "bohai",
    "Torres Strait": "torres",
    "Sunda Strait": "sunda",
    "Makassar Strait": "makassar",
    "Magellan Strait": "magellan",
    "Yucatan Channel": "yucatan",
    "Windward Passage": "windward",
    "Mona Passage": "mona",
    "Balabac Strait": "balabac",
    "Bering Strait": "bering",
    "Mindoro Strait": "mindoro",
    "Kerch Strait": "kerch",
}

VESSEL_COUNT_COLS = [
    "n_container", "n_dry_bulk", "n_general_cargo", "n_roro",
    "n_tanker", "n_cargo", "n_total",
]
CAPACITY_COLS = [
    "capacity_container", "capacity_dry_bulk", "capacity_general_cargo",
    "capacity_roro", "capacity_tanker", "capacity_cargo", "capacity",
]


def load() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    df["chokepoint_slug"] = df["portname"].map(SLUG)
    if df["chokepoint_slug"].isna().any():
        unmapped = df.loc[df.chokepoint_slug.isna(), "portname"].unique().tolist()
        raise RuntimeError(f"Unmapped portname values: {unmapped}")
    df["period_dt"] = df["date"].values.astype("datetime64[M]")  # first-of-month
    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    agg = {c: "sum" for c in VESSEL_COUNT_COLS + CAPACITY_COLS}
    agg["date"] = "count"   # number of daily observations contributing
    out = (df.groupby(["period_dt", "chokepoint_slug", "portid", "portname"])
             .agg(agg).rename(columns={"date": "n_days"}).reset_index())
    return out


def main() -> None:
    df = load()

    daily_out = OUT / "portwatch_chokepoint_daily.csv"
    df.to_csv(daily_out, index=False)

    monthly = aggregate_monthly(df)
    monthly_out = OUT / "portwatch_chokepoint_monthly.csv"
    monthly.to_csv(monthly_out, index=False)

    h_daily = df[df.chokepoint_slug == "hormuz"].copy()
    h_monthly = monthly[monthly.chokepoint_slug == "hormuz"].copy()
    (OUT / "portwatch_hormuz_daily.csv").write_text(h_daily.to_csv(index=False))
    (OUT / "portwatch_hormuz_monthly.csv").write_text(h_monthly.to_csv(index=False))

    print(f"daily   : {df.shape}  range {df.date.min().date()} -> {df.date.max().date()}  -> {daily_out.name}")
    print(f"monthly : {monthly.shape}  range {monthly.period_dt.min().date()} -> {monthly.period_dt.max().date()}  -> {monthly_out.name}")
    print(f"hormuz daily/monthly rows : {len(h_daily)} / {len(h_monthly)}")
    # Spot-check Hormuz monthly tanker capacity in our window
    h_w = h_monthly[(h_monthly.period_dt >= "2020-01-01") &
                    (h_monthly.period_dt <= "2026-02-01")]
    print()
    print("Hormuz monthly tanker capacity, head/tail in modelling window:")
    print(h_w[["period_dt", "n_tanker", "capacity_tanker", "n_days"]].head(3).to_string(index=False))
    print("...")
    print(h_w[["period_dt", "n_tanker", "capacity_tanker", "n_days"]].tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
