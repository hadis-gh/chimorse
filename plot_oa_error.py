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
COARSE_R = 0.1       # coarse r-sampling spacing (Å)
FINE_R = 0.0125      # fine r-sampling spacing near the minimum (Å)

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


def shifted_aligned(df_ref, df_model, e_lt_zero=True):
    """Align all orientations at their equilibrium distance on the fine grid.

    For each (phi1, phi2) the equilibrium distance r_e is rounded to the
    nearest coarse-grid point and subtracted from r, so the shifted distance
    s = r - r_e (rounded) falls on the fine grid. Only bound points (e < 0)
    are kept.
    """
    join = pd.merge(
        df_model, df_ref,
        on=["phi1", "phi2", "r"],
        suffixes=("_model", "_ref"),
    )
    join["delta_e"] = join["e_model"] - join["e_ref"]

    e_min = extract_energy_minimums(df_ref, r_max=12)[["phi1", "phi2", "r", "e"]]
    e_min["re_shift"] = np.round(e_min["r"] / COARSE_R) * COARSE_R
    e_min = e_min.rename(columns={"e": "e_min"})
    join = join.merge(e_min[["phi1", "phi2", "re_shift", "e_min"]], on=["phi1", "phi2"], how="left")
    join["s"] = join["r"] - join["re_shift"]
    join["e_rel"] = join["e_ref"] - join["e_min"]

    if e_lt_zero:
        join = join[join["e_ref"] < 0]
    return join


def plot_shifted_error(df_ref, df_model, out, csv_out=None):
    """MAE and RMSE vs. shifted distance from equilibrium (fine grid, e < 0)."""
    join = shifted_aligned(df_ref, df_model, e_lt_zero=True)

    # statistics per fine-grid shifted distance s
    g = join.groupby("s")["delta_e"].agg(
        [lambda x: np.mean(np.abs(x)),
         lambda x: np.sqrt(np.mean(x**2)),
         lambda x: np.max(np.abs(x))]
    )
    g.columns = ["mae", "rmse", "maxerr"]
    e_agg = join.groupby("s")["e_rel"].agg(["min", "max"])
    g = g.join(e_agg)

    s_min, s_max = np.floor(join["s"].min() / FINE_R) * FINE_R, np.ceil(join["s"].max() / FINE_R) * FINE_R
    s_grid = np.round(np.arange(s_min, s_max + FINE_R / 2, FINE_R), 4)

    def to_grid(values):
        arr = np.full(len(s_grid), np.nan)
        idx = np.round((values.index.to_numpy() - s_grid[0]) / FINE_R).astype(int)
        arr[idx] = values.to_numpy()
        return arr

    mae = to_grid(g["mae"])
    rmse = to_grid(g["rmse"])
    maxerr = to_grid(g["maxerr"])
    emin = to_grid(g["min"])
    emax = to_grid(g["max"])

    # drop empty (NaN) fine-grid bins once; the same rows feed both plot and CSV
    frame = pd.DataFrame(
        {
            "s": s_grid, "MAE_meV": mae * 1000, "RMSE_meV": rmse * 1000,
            "MaxErr_meV": maxerr * 1000, "Emin_eV": emin, "Emax_eV": emax,
        }
    ).dropna()

    if csv_out is not None:
        frame.to_csv(csv_out, index=False)
        print("Saved", csv_out, f"({len(frame)} rows)")

    s = frame["s"].to_numpy()
    mae_pl = frame["MAE_meV"].to_numpy()
    rmse_pl = frame["RMSE_meV"].to_numpy()
    maxerr_pl = frame["MaxErr_meV"].to_numpy()
    emin_pl = frame["Emin_eV"].to_numpy()
    emax_pl = frame["Emax_eV"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(s, mae_pl, color="tab:purple", ls="-", lw=1.6, label="MAE (model-ref)")
    ax.plot(s, rmse_pl, color="tab:olive", ls="-", lw=1.6, label="RMSE")
    ax.plot(s, maxerr_pl, color="tab:red", ls=":", lw=1.4, label="Max |error|")
    ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("shifted distance $r - r_e$ (Å)")
    ax.set_ylabel("ΔE (meV)")
    ax.set_title(f"OA {MOLECULE}: MAE/RMSE vs. distance from equilibrium (E < 0)")

    axr = ax.twinx()
    axr.plot(s, emin_pl, color="tab:blue", ls="--", lw=1.2,
             label=r"$\min_{\phi_1,\phi_2} [E - E(r_e)]$")
    axr.plot(s, emax_pl, color="tab:red", ls=":", lw=1.2,
             label=r"$\max_{\phi_1,\phi_2} [E - E(r_e)]$")
    axr.set_ylabel(r"$E - E(r_e)$ (eV)")

    ax.set_xlim(s.min(), s.max())
    ax.set_ylim(0, 300)
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = axr.get_legend_handles_labels()
    ax.legend(lines_l + lines_r, labels_l + labels_r, loc="upper right", ncol=2)

    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)


def plot_waterfall(df_ref, df_model, out, n_curves=None):
    """Waterfall of all shifted bound E(r) curves (E < 0).

    The figure is sized so it can be scrolled on screen (tall aspect ratio),
    giving each of the (up to ~7200) orientations visible vertical separation.
    """
    join = shifted_aligned(df_ref, df_model, e_lt_zero=True)
    curves = join.groupby(["phi1", "phi2"])
    orientations = list(curves.groups.keys())
    if n_curves is not None:
        orientations = orientations[:n_curves]
    n = len(orientations)

    offset = 0.05          # eV of vertical offset per curve
    curves_per_inch = 40   # vertical resolution for screen scrolling
    fig_width = 10
    fig_height = max(8.0, n / curves_per_inch)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    cmap = plt.get_cmap("viridis")
    vmax = max(n - 1, 1)
    for k, key in enumerate(orientations):
        sub = curves.get_group(key)
        s = sub["s"].to_numpy()
        e = sub["e_ref"].to_numpy()
        ax.plot(s, e + k * offset, color=cmap(k / vmax), lw=0.5)
    ax.set_xlabel("shifted distance $r - r_e$ (Å)")
    ax.set_ylabel("E + offset (eV)")
    ax.set_title(f"OA {MOLECULE}: waterfall of aligned E(r), E < 0 ({n} orientations)")
    fig.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)


plot_shifted_error(
    df_ref, df_model, "Figures/oa_shifted_error.pdf",
    csv_out="Figures/oa_shifted_error_data.csv",
)
plot_waterfall(df_ref, df_model, "Figures/oa_waterfall.pdf")


def plot_top_rmse(df_ref, df_model, out_shifted, out_abs,
                  e_rel_sel=0.1, e_rel_plot=1.0):
    """Select the worst (top-3) and best orientations by RMSE computed in the
    region e - e_min < e_rel_sel (100 meV), but plot the selected curves over
    the wider window e - e_min < e_rel_plot (1 eV).
    Produces one figure with shifted distance s and one with absolute distance r.
    """
    join = shifted_aligned(df_ref, df_model, e_lt_zero=False)
    sel = join[join["e_rel"] < e_rel_sel]           # selection window (100 meV)
    region = join[join["e_rel"] < e_rel_plot]       # plotting window (1 eV)

    rmse_or = sel.groupby(["phi1", "phi2"])["delta_e"].apply(
        lambda x: np.sqrt(np.mean(x**2))
    )
    worst = rmse_or.nlargest(3)
    best = rmse_or.nsmallest(1)
    selected = pd.concat([worst, best])
    selected_rmse = selected
    selected_keys = selected.index.tolist()

    def render(x_key, label, show_fit=False):
        fig, (ax, axerr) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        colors = plt.get_cmap("Set1")
        for k, key in enumerate(selected_keys):
            sub = region[(region["phi1"] == key[0]) & (region["phi2"] == key[1])]
            x = sub[x_key].to_numpy()
            erel = sub["e_rel"].to_numpy()
            fit_rel = (sub["e_model"] - sub["e_min"]).to_numpy()
            de = sub["delta_e"].to_numpy() * 1000
            color = colors(k % 9)
            ls = "--" if key in worst.index else "-"
            name = f"$\\phi_1$={key[0]}, $\\phi_2$={key[1]} " \
                   f"(RMSE={selected_rmse[key]:.1f} meV)"
            if show_fit:
                ax.plot(x, fit_rel, color=color, ls="-", lw=1.5, label=f"{name} (fit)")
                ax.plot(x, erel, color=color, ls="none", marker="o", ms=3,
                        label=f"$\\phi_1$={key[0]}, $\\phi_2$={key[1]} (data)")
            else:
                ax.plot(x, erel, color=color, ls=ls, lw=1.5, label=name)
            axerr.plot(x, de, color=color, ls=ls, lw=1.5,
                       label=f"$\\phi_1$={key[0]}, $\\phi_2$={key[1]}")
        ax.set_ylabel(r"$E - E(r_e)$ (eV)")
        ax.set_title(f"{label} — E−E(r$_e$) (left) and fit error (right), "
                     f"E−E(r$_e$)<1 eV (selected for <100 meV)")
        ax.legend(loc="best", fontsize=8)
        axerr.set_ylabel(r"$\Delta E$ (meV)")
        axerr.set_xlabel(label)
        axerr.legend(loc="best", fontsize=8)
        fig.tight_layout()
        os.makedirs("Figures", exist_ok=True)
        fig.savefig(out_shifted if x_key == "s" else out_abs)
        plt.close(fig)
        print("Saved", out_shifted if x_key == "s" else out_abs)

    render("s", "shifted distance $r - r_e$ (Å)")
    render("r", "distance $r$ (Å)", show_fit=True)


plot_top_rmse(
    df_ref, df_model,
    "Figures/oa_top_rmse_shifted.pdf",
    "Figures/oa_top_rmse_abs.pdf",
)
