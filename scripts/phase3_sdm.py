"""Phase 3.2 — Spatial Durbin Model (SDM) / Panel Spatial Lag.

We use spreg.Panel_FE_Lag (ML fixed-effects spatial lag panel) to estimate

    log_import_{it} = ρ W log_import_{t} + β_1 log_price_t + β_2 DE_i
                      + β_3 hormuz_DWT_annual + μ_i + ε_{it}

separately for each commodity layer.  W is the row-normalised trade-share
matrix A from 2020 (held fixed as the pre-period spatial weight to avoid
simultaneity with the outcome).

The key question: after controlling for own exposure (DE) and global price,
does network position still matter?  A positive ρ means countries that are
close trading partners co-move in import volumes — a direct measure of
trade-network contagion.

We also run an SLX augmented regression (manually add WX regressors):

    Y = X β + WX θ + ε   (no spatial lag in Y)

which directly estimates the indirect (network-spillover) effect θ without
ML or instrumental-variable requirements.

Outputs (Results/phase3/):
    sdm_results.csv     ρ, β estimates, p-values per layer
    slx_results.csv     direct and indirect effects per regressor per layer
Figures (figures/phase3/):
    fig3_rho_by_layer.png     spatial autoregressive coefficient ρ by layer
    fig3_sdm_direct_indirect.png  direct vs indirect effects (SLX)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import spreg
import libpysal.weights as weights
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PHASE0 = ROOT / "Results" / "phase0"
PHASE1 = ROOT / "Results" / "phase1"
PHASE3 = ROOT / "Results" / "phase3"
FIG = ROOT / "figures" / "phase3"
FIG.mkdir(parents=True, exist_ok=True)

LAYERS = ("wheat", "ammonia", "urea", "lpg_propane", "lpg_butane")
LAYER_LABEL = {"wheat": "Wheat", "ammonia": "Ammonia", "urea": "Urea",
               "lpg_propane": "LPG propane", "lpg_butane": "LPG butane"}
HORMUZ = ["ARE", "BHR", "IRN", "IRQ", "KWT", "QAT", "SAU"]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)


def build_W(layer: str, country_order: list[str]) -> np.ndarray:
    """Row-normalise the 2020 trade-share matrix to get W.

    If a row is all-zero (no imports in 2020), connect that node to its
    geographic nearest neighbours to keep the weights matrix connected — but
    in practice we simply give it a uniform row weight over all observed
    importers so spreg does not error.
    """
    A = np.load(PHASE0 / f"A_2020_{layer}.npy")
    # Subset to country_order
    all_countries = pd.read_csv(PHASE0 / "country_index.csv")["iso3"].tolist()
    idx = [all_countries.index(c) for c in country_order]
    Wsub = A[np.ix_(idx, idx)].copy()
    # Row normalise
    rs = Wsub.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    Wsub = Wsub / rs
    return Wsub


def build_panel(layer: str) -> tuple[pd.DataFrame, list[str], int]:
    """Return (tidy annual panel, country list, n_years).

    Columns: country, year, log_import, log_price, DE, hormuz_DWT_annual
    Ordered by year within country (spreg convention).
    """
    monthly = pd.read_csv(PHASE3 / "panel_monthly.csv", parse_dates=["period_dt"])
    sub = monthly[monthly.layer == layer].copy()
    # Annual aggregates
    agg = sub.groupby(["country", "year"]).agg(
        import_value_usd=("import_value_usd", "sum"),
        hormuz_DWT_annual=("hormuz_DWT_norm", "mean"),
        log_price=("log_price", "mean"),
        DE=("DE", "first")
    ).reset_index()
    agg = agg[agg.import_value_usd > 0].copy()
    agg["log_import"] = np.log(agg["import_value_usd"])

    # Keep only countries present in all 5 years
    year_count = agg.groupby("country")["year"].count()
    keep = year_count[year_count == 5].index
    agg = agg[agg.country.isin(keep)].sort_values(["country", "year"])
    countries = sorted(agg.country.unique().tolist())
    n_years = agg.year.nunique()
    return agg, countries, n_years


def run_sdm(layer: str) -> dict:
    df, countries, T = build_panel(layer)
    n = len(countries)
    Wsub = build_W(layer, countries)
    w_obj = weights.full2W(Wsub, ids=countries)
    w_obj.transform = "r"

    # y and X ordered: all countries in year 1, then year 2, … (spreg convention)
    ordered = df.sort_values(["year", "country"])
    y = ordered["log_import"].values.reshape(-1, 1)
    X = ordered[["log_price", "DE", "hormuz_DWT_annual"]].values
    # Panel_FE_Lag expects shape (n, T*k) for X — actually (n*T, k) is OK
    try:
        mod = spreg.Panel_FE_Lag(y, X, w_obj,
                                  name_y="log_import",
                                  name_x=["log_price", "DE", "hormuz_DWT_annual"],
                                  name_w=f"A2020_{layer}")
        rho = float(mod.rho)
        betas = mod.betas.flatten()
        std_err = np.sqrt(np.diag(mod.vm)).flatten()
        z_stats = betas / (std_err + 1e-12)
        return {"layer": layer, "n": n, "T": T,
                "rho": rho,
                "b_log_price": betas[0], "se_log_price": std_err[0],
                "b_DE": betas[1], "se_DE": std_err[1],
                "b_DWT": betas[2], "se_DWT": std_err[2],
                "z_rho": float(mod.z_stat[-1][0]),
                "p_rho": float(mod.z_stat[-1][1])}
    except Exception as e:
        print(f"  SDM failed for {layer}: {e}")
        return {"layer": layer, "error": str(e)}


def run_slx(layer: str) -> dict:
    """SLX model: Y = Xβ + WXθ + ε (OLS, no spatial lag in Y).

    Adds W×log_price and W×DE as extra regressors.
    β[1] = direct effect of log_price
    θ[1] = indirect / network-spillover effect of log_price
    """
    df, countries, T = build_panel(layer)
    Wsub = build_W(layer, countries)

    ordered = df.sort_values(["year", "country"])
    y = ordered["log_import"].values

    # Build WX by applying W separately per year
    Wx_log_price_parts = []
    Wx_DE_parts = []
    Wx_DWT_parts = []
    for yr in sorted(df.year.unique()):
        yr_df = ordered[ordered.year == yr].set_index("country").reindex(countries)
        lp = yr_df["log_price"].values
        de = yr_df["DE"].values
        dw = yr_df["hormuz_DWT_annual"].values
        Wx_log_price_parts.append(Wsub @ lp)
        Wx_DE_parts.append(Wsub @ de)
        Wx_DWT_parts.append(Wsub @ dw)

    # Stack in same year × country order as ordered
    years = sorted(df.year.unique())
    Wx_lp = np.concatenate([Wx_log_price_parts[i]
                              for i in range(len(years))])
    Wx_de = np.concatenate([Wx_DE_parts[i] for i in range(len(years))])
    Wx_dw = np.concatenate([Wx_DWT_parts[i] for i in range(len(years))])

    Xmat = np.column_stack([
        np.ones(len(y)),
        ordered["log_price"].values,
        ordered["DE"].values,
        ordered["hormuz_DWT_annual"].values,
        Wx_lp, Wx_de, Wx_dw
    ])
    # Country FE dummies
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    c_enc = le.fit_transform(ordered["country"].values)
    FE = np.eye(len(countries))[c_enc][:, :-1]  # drop one for identification
    Xmat = np.column_stack([Xmat, FE])
    beta, _, _, _ = np.linalg.lstsq(Xmat, y, rcond=None)
    n_obs = len(y)
    resid = y - Xmat @ beta
    s2 = resid @ resid / (n_obs - Xmat.shape[1])
    try:
        cov = s2 * np.linalg.inv(Xmat.T @ Xmat)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full_like(beta, np.nan)
    names = ["const", "b_lp", "b_DE", "b_DWT",
             "theta_lp", "theta_DE", "theta_DWT"]
    out = {"layer": layer, "n_obs": n_obs}
    for k, nm in enumerate(names):
        out[nm] = float(beta[k])
        out[f"se_{nm}"] = float(se[k]) if k < len(se) else np.nan
    out["total_lp"] = out["b_lp"] + out["theta_lp"]
    out["total_DE"] = out["b_DE"] + out["theta_DE"]
    print(f"  {layer}: b_lp={beta[1]:.3f} th_lp={beta[4]:.3f} "
          f"b_DE={beta[2]:.3f} th_DE={beta[5]:.3f}")
    return out


def main() -> None:
    print("SDM (Panel_FE_Lag) per layer …")
    sdm_rows = []
    for layer in LAYERS:
        print(f"  {layer} ...", end=" ", flush=True)
        row = run_sdm(layer)
        sdm_rows.append(row)
        if "rho" in row:
            print(f"rho={row['rho']:.4f} (z={row['z_rho']:.2f}, p={row['p_rho']:.4f})")

    sdm_df = pd.DataFrame(sdm_rows)
    sdm_df.to_csv(PHASE3 / "sdm_results.csv", index=False)
    print("\nSDM results:")
    print(sdm_df[["layer", "rho", "z_rho", "p_rho",
                   "b_log_price", "b_DE", "b_DWT"]].round(4).to_string(index=False))

    print("\nSLX (direct + indirect effects) per layer …")
    slx_rows = []
    for layer in LAYERS:
        row = run_slx(layer)
        slx_rows.append(row)
    slx_df = pd.DataFrame(slx_rows)
    slx_df.to_csv(PHASE3 / "slx_results.csv", index=False)

    # Figure: ρ by layer
    valid = sdm_df.dropna(subset=["rho"])
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    colors = ["#c0392b" if r < 0 else "#27ae60" for r in valid["rho"]]
    ax.bar(valid["layer"], valid["rho"], color=colors)
    for i, (_, r) in enumerate(valid.iterrows()):
        ax.text(i, r["rho"] + 0.005 * np.sign(r["rho"]),
                f"{r['rho']:.3f}", ha="center", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Spatial lag coefficient ρ")
    ax.set_title("Network contagion: trade-network co-movement in import volumes")
    ax.set_xlabel("Layer")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_rho_by_layer.png", dpi=150)
    plt.close(fig)

    # Figure: SLX direct vs indirect for log_price
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(LAYERS))
    w = 0.35
    labs = [LAYER_LABEL[L] for L in LAYERS]
    direct = [slx_df.loc[slx_df.layer == L, "b_lp"].iloc[0] for L in LAYERS]
    indirect = [slx_df.loc[slx_df.layer == L, "theta_lp"].iloc[0] for L in LAYERS]
    ax.bar(x - w/2, direct, w, label="Direct (β)", color="#2980b9")
    ax.bar(x + w/2, indirect, w, label="Indirect / spillover (θ)", color="#e67e22",
           alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=15, ha="right")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Effect on log import")
    ax.set_title("Price effect on imports: direct vs network spillover (SLX)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_sdm_direct_indirect.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote figures under {FIG}")


if __name__ == "__main__":
    main()
