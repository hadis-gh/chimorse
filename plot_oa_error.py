"""Plot OA potential E(r) statistics and the Fourier-Morse fit error vs distance."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chimorse.analysis import (
    compute_near_equilibrium_energy_rmse,
    extract_energy_minimums,
    get_screw_dir,
)
from chimorse.config import PLOT_PARAMS, load_molecule_info
from chimorse.dataio import load_data
from chimorse.fitting import generate_fourier_morse_data

plt.rcParams.update(PLOT_PARAMS)

ENERGY_RANGE = (8.3, 10.3)

MOLECULE = "PA"
INTERACTION = "OA"
HARMONIC_CEILS = {"EP": (8, 1), "EA": (8, 1), "OP": (20, 1), "OA": (20, 1)}
ALPHA_FIT = True

molecule = load_molecule_info(MOLECULE)
df_ref = load_data(molecule, INTERACTION, zero_zeta=True)
df_model = generate_fourier_morse_data(
    df_ref, molecule, INTERACTION, HARMONIC_CEILS,
    alpha_fit=ALPHA_FIT, print_errors=False,
)

def make_plot(df_ref, df_model, title_suffix, out, stats_file=None):
    """Aggregate E(r) statistics and fit error per r and save a 2-panel figure.

    Adds an RMSE-vs-r curve. If stats_file is given, writes MAE/RMSE of the
    filtered data (overall and restricted to ENERGY_RANGE) to that file.
    """
    # ---- reference statistics per r over (phi1, phi2) ----
    agg = df_ref.groupby("r")["e"].agg(["max", "min", "mean"])

    # ---- fit error per r: model - reference ----
    join = pd.merge(
        df_model,
        df_ref,
        on=["phi1", "phi2", "r"],
        suffixes=("_model", "_ref"),
    )
    join["delta_e"] = join["e_model"] - join["e_ref"]
    err = join.groupby("r")["delta_e"].agg(
        [lambda s: np.mean(np.abs(s)),
         lambda s: np.sqrt(np.mean(s**2)),
         lambda s: np.max(np.abs(s))]
    )
    err.columns = ["mean_abs_err", "rmse", "max_abs_err"]

    r = agg.index.to_numpy()
    e_max, e_min, e_mean = agg["max"].to_numpy(), agg["min"].to_numpy(), agg["mean"].to_numpy()
    err_mean, err_rmse, err_max = (
        err["mean_abs_err"].to_numpy(),
        err["rmse"].to_numpy(),
        err["max_abs_err"].to_numpy(),
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2]},
    )

    # ---- panel 1: E(r) max/min/mean ----
    ax1.plot(r, e_max, color="tab:red", ls="-", lw=1.6, label=r"$\max_{\phi_1,\phi_2} E(r)$")
    ax1.plot(r, e_min, color="tab:blue", ls="-", lw=1.6, label=r"$\min_{\phi_1,\phi_2} E(r)$")
    ax1.plot(r, e_mean, color="tab:green", ls="-", lw=1.6, label=r"$\langle E(r)\rangle_{\phi_1,\phi_2}$")
    ax1.axhline(0, color="k", ls=":", lw=0.8)
    ax1.set_ylabel("E (eV)")
    ax1.set_title(f"OA interaction — {MOLECULE}: E(r) statistics vs. distance{title_suffix}")
    ax1.legend(loc="upper right", ncol=3)

    # ---- panel 2: fit error ----
    ax2.plot(r, err_mean, color="tab:purple", ls="-", lw=1.6, label="mean |error| (model-ref)")
    ax2.plot(r, err_rmse, color="tab:olive", ls="-", lw=1.6, label="RMSE")
    ax2.plot(r, err_max, color="tab:orange", ls="-", lw=1.6, label="max |error|")
    ax2.axhline(0, color="k", ls=":", lw=0.8)
    ax2.set_xlabel("r (Å)")
    ax2.set_ylabel("ΔE (eV)")
    ax2.set_title(f"Fourier-Morse fit error vs. distance{title_suffix}")
    ax2.legend(loc="upper right", ncol=3)

    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)

    if stats_file is not None:
        de = join["delta_e"].to_numpy() * 1000
        lines = [
            f"# OA fit-error statistics ({len(de)} points)",
            f"r range            : {join['r'].min():.1f} - {join['r'].max():.1f} Å",
            f"MAE (all r)        : {np.mean(np.abs(de)):.4f} meV",
            f"RMSE (all r)       : {np.sqrt(np.mean(de**2)):.4f} meV",
        ]
        sel = join["r"].between(*ENERGY_RANGE)
        de_sel = join["delta_e"][sel].to_numpy() * 1000
        lines += [
            f"r range restricted : {ENERGY_RANGE[0]} - {ENERGY_RANGE[1]} Å  ({len(de_sel)} points)",
            f"MAE ({ENERGY_RANGE[0]}-{ENERGY_RANGE[1]} Å)    : {np.mean(np.abs(de_sel)):.4f} meV",
            f"RMSE ({ENERGY_RANGE[0]}-{ENERGY_RANGE[1]} Å)   : {np.sqrt(np.mean(de_sel**2)):.4f} meV",
        ]

        # --- near-equilibrium RMSE, two definitions ---
        # 1. fixed spatial window: |r - re_ref| <= 0.5 Å around each orientation's min
        rmse_spatial = compute_near_equilibrium_energy_rmse(
            df_ref, df_model, delta_r=0.5, r_max=12
        )
        lines.append(
            f"E RMSE near eq (spatial ±0.5 Å) : {rmse_spatial*1000:.4f} meV"
        )

        # 2. energy window: points at most 100 meV above each orientation's minimum
        e_min = extract_energy_minimums(df_ref, r_max=12)
        e_join = join.merge(
            e_min[["phi1", "phi2", "e"]].rename(columns={"e": "e_min"}),
            on=["phi1", "phi2"],
            how="left",
        )
        energy_mask = (e_join["e_ref"] - e_join["e_min"]) <= 0.1  # 100 meV
        de_energy = (e_join["e_model"] - e_join["e_ref"])[energy_mask].to_numpy() * 1000
        lines.append(
            f"E RMSE near eq (energy ≤ +100 meV) : "
            f"{np.sqrt(np.mean(de_energy**2)):.4f} meV  ({len(de_energy)} points)"
        )

        text = "\n".join(lines) + "\n"
        with open(stats_file, "w") as f:
            f.write(text)
        print("Saved", stats_file)


# full dataset
make_plot(df_ref, df_model, "", "Figures/oa_E_statistics_and_error.pdf")

# bound region only: disregard all points with E > 0
mask = df_ref["e"] <= 0
df_ref_bound = df_ref[mask]
keys = pd.MultiIndex.from_frame(df_ref_bound[["phi1", "phi2", "r"]].drop_duplicates())
df_model_bound = df_model[
    pd.MultiIndex.from_frame(df_model[["phi1", "phi2", "r"]]).isin(keys)
].reset_index(drop=True)
make_plot(
    df_ref_bound, df_model_bound, " (E ≤ 0 only)",
    "Figures/oa_E_statistics_and_error_bound.pdf",
    stats_file="Figures/oa_fit_error_stats.txt",
)
