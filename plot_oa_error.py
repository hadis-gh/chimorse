"""Plot OA potential E(r) statistics and the Fourier-Morse fit error vs distance."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chimorse.analysis import get_screw_dir
from chimorse.config import PLOT_PARAMS, load_molecule_info
from chimorse.dataio import load_data
from chimorse.fitting import generate_fourier_morse_data

plt.rcParams.update(PLOT_PARAMS)

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

def make_plot(df_ref, df_model, title_suffix, out):
    """Aggregate E(r) statistics and fit error per r and save a 2-panel figure."""
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
        [lambda s: np.mean(np.abs(s)), lambda s: np.max(np.abs(s))]
    )
    err.columns = ["mean_abs_err", "max_abs_err"]

    r = agg.index.to_numpy()
    e_max, e_min, e_mean = agg["max"].to_numpy(), agg["min"].to_numpy(), agg["mean"].to_numpy()
    err_mean, err_max = err["mean_abs_err"].to_numpy(), err["max_abs_err"].to_numpy()

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
    ax2.plot(r, err_max, color="tab:orange", ls="-", lw=1.6, label="max |error|")
    ax2.axhline(0, color="k", ls=":", lw=0.8)
    ax2.set_xlabel("r (Å)")
    ax2.set_ylabel("ΔE (eV)")
    ax2.set_title(f"Fourier-Morse fit error vs. distance{title_suffix}")
    ax2.legend(loc="upper right", ncol=2)

    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)


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
)
