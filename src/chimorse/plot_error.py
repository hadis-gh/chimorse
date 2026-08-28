"""Plot potential E(r) statistics and the Fourier-Morse fit error vs distance.

Supports any molecule/interaction/configuration via the CLI; all parameters
(interaction, harmonic ceilings, alpha fit, interpolation, energy range,
coarse/fine r spacing, weighting, and input paths) are set through command-line
arguments with sensible defaults.
"""
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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


def _tagged(out_dir, base, ext, tag):
    """Build an output path base.ext (optionally tagged) inside out_dir."""
    name = f"{base}_{tag}{ext}" if tag else f"{base}{ext}"
    return os.path.join(out_dir, name)


def _load_reference(molecule, interaction, data_file):
    """Load the reference data frame, from data_file (CSV) or the default loader."""
    if data_file:
        return pd.read_csv(data_file)
    return load_data(molecule, interaction, zero_zeta=True)

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
    ax1.set_title(f"{INTERACTION} interaction — {MOLECULE}: E(r) statistics vs. distance{title_suffix}")
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
    os.makedirs(os.path.dirname(out), exist_ok=True)
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
    # distinguish Max |error| (brown, dash-dot) from max E - E(re) (red, dotted)
    ax.plot(s, mae_pl, color="tab:purple", ls="-", lw=1.6, label="MAE (model-ref)")
    ax.plot(s, rmse_pl, color="tab:olive", ls="-", lw=1.6, label="RMSE")
    ax.plot(s, maxerr_pl, color="tab:brown", ls="-.", lw=1.4, label="Max |error|")
    ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("shifted distance $r - r_e$ (Å)")
    ax.set_ylabel("ΔE (meV)")
    ax.set_title(f"{INTERACTION} {MOLECULE}: MAE/RMSE vs. distance from equilibrium (E < 0)")

    # minor y-grid at 5 meV, major y-grid at 50 meV
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.grid(which="minor", axis="y", color="black", lw=0.3, alpha=0.6)
    ax.grid(which="major", axis="y", color="black", lw=0.8)

    axr = ax.twinx()
    axr.plot(s, emin_pl, color="tab:blue", ls="--", lw=1.2,
             label=r"$\min_{\phi_1,\phi_2} [E - E(r_e)]$")
    axr.plot(s, emax_pl, color="tab:red", ls=":", lw=1.2,
             label=r"$\max_{\phi_1,\phi_2} [E - E(r_e)]$")
    axr.set_ylabel(r"$E - E(r_e)$ (eV)")
    axr.grid(which="major", axis="y", color="gray", lw=0.6)

    ax.set_xlim(s.min(), s.max())
    ax.set_ylim(0, 300)
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = axr.get_legend_handles_labels()
    ax.legend(lines_l + lines_r, labels_l + labels_r, loc="upper right", ncol=2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)


def plot_shifted_rel_error(df_ref, df_model, out, csv_out=None,
                           ranges_csv_out=None, ranges_out=None,
                           e_floor=25e-3, e_uppers=None):
    """Relative error vs. shifted distance from equilibrium (fine grid, e < 0).

    The reference energy E is the energy above the equilibrium minimum,
    E = e_ref - e_min = E - E(r_e). Per-point relative error is
    |delta_e| / max(E, e_floor), i.e. delta_e/E for E >= e_floor and
    delta_e/e_floor for E < e_floor (e_floor = 25 meV ~ room-temperature).
    Shown as mean and max over orientations per fine-grid bin.

    csv_out:      plot data written to CSV before plotting (same rows plotted).
    ranges_csv_out / ranges_out: mean relative error over cumulative energy
    ranges E < e_uppers (meV), written to a file and plotted.
    """
    join = shifted_aligned(df_ref, df_model, e_lt_zero=True)
    join["E"] = join["e_rel"]
    join["rel_err"] = np.abs(join["delta_e"]) / np.maximum(join["E"], e_floor)

    g = join.groupby("s")["rel_err"].agg([np.mean, np.max])
    g.columns = ["rel_mean", "rel_max"]
    e_agg = join.groupby("s")["e_rel"].agg(["min", "max"])
    g = g.join(e_agg)

    s_min, s_max = np.floor(join["s"].min() / FINE_R) * FINE_R, np.ceil(join["s"].max() / FINE_R) * FINE_R
    s_grid = np.round(np.arange(s_min, s_max + FINE_R / 2, FINE_R), 4)

    def to_grid(values):
        arr = np.full(len(s_grid), np.nan)
        idx = np.round((values.index.to_numpy() - s_grid[0]) / FINE_R).astype(int)
        arr[idx] = values.to_numpy()
        return arr

    frame = pd.DataFrame(
        {
            "s": s_grid,
            "mean_rel_err": to_grid(g["rel_mean"]),
            "max_rel_err": to_grid(g["rel_max"]),
            "Emin_eV": to_grid(g["min"]),
            "Emax_eV": to_grid(g["max"]),
        }
    ).dropna()

    if csv_out is not None:
        frame.to_csv(csv_out, index=False)
        print("Saved", csv_out, f"({len(frame)} rows)")

    s = frame["s"].to_numpy()
    rel_mean = frame["mean_rel_err"].to_numpy()
    rel_max = frame["max_rel_err"].to_numpy()
    emin_pl = frame["Emin_eV"].to_numpy()
    emax_pl = frame["Emax_eV"].to_numpy()

    def render_panel(ax, axr, rel_ymax=None, e_ymax=None,
                      xlabel=False, title=None):
        """Plot the relative-error curves (left) and min/max E-E(re) (right)."""
        ax.plot(s, rel_mean, color="tab:purple", ls="-", lw=1.6, label="mean rel. error")
        ax.plot(s, rel_max, color="tab:brown", ls="-.", lw=1.4, label="max rel. error")
        ax.axhline(0, color="k", ls=":", lw=0.8)
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_ylabel(r"rel. error $|\Delta E|/\max(E{-}E(r_e),25\,\mathrm{meV})$")
        if title is not None:
            ax.set_title(title)
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.grid(which="minor", axis="y", color="black", lw=0.3, alpha=0.6)

        axr.plot(s, emin_pl, color="tab:blue", ls="--", lw=1.2,
                 label=r"$\min_{\phi_1,\phi_2} [E - E(r_e)]$")
        axr.plot(s, emax_pl, color="tab:red", ls=":", lw=1.2,
                 label=r"$\max_{\phi_1,\phi_2} [E - E(r_e)]$")
        axr.set_ylabel(r"$E - E(r_e)$ (eV)")
        axr.grid(which="major", axis="y", color="gray", lw=0.6)

        ax.set_xlim(s.min(), s.max())
        if rel_ymax is not None:
            ax.set_ylim(0, rel_ymax)
        if e_ymax is not None:
            axr.set_ylim(0, e_ymax)
        if xlabel:
            ax.set_xlabel("shifted distance $r - r_e$ (Å)")

        lines_l, labels_l = ax.get_legend_handles_labels()
        lines_r, labels_r = axr.get_legend_handles_labels()
        ax.legend(lines_l + lines_r, labels_l + labels_r, loc="upper right", ncol=2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axr1 = ax1.twinx()
    axr2 = ax2.twinx()

    render_panel(ax1, axr1,
                 title=f"{INTERACTION} {MOLECULE}: relative error vs. distance from equilibrium (E < 0)")

    # zoom panel: relative errors below 0.5 (left) and energies up to 200 meV (right)
    render_panel(ax2, axr2, rel_ymax=0.5, e_ymax=0.2,
                 xlabel=True,
                 title="zoom: rel. error ≤ 0.5, E − E(r_e) ≤ 200 meV")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)

    # ---- mean relative error over cumulative energy ranges E < e_uppers ----
    if e_uppers is None:
        e_uppers = [50] + list(range(100, 1001, 100))  # 50,100,200,300,...,1000 meV
    means = []
    for upper in e_uppers:
        mask = join["E"] < upper * 1e-3
        if mask.sum() > 0:
            means.append((upper, join["rel_err"][mask].mean()))
    range_df = pd.DataFrame(means, columns=["E_upper_meV", "mean_rel_err"])

    if ranges_csv_out is not None:
        range_df.to_csv(ranges_csv_out, index=False)
        print("Saved", ranges_csv_out, f"({len(range_df)} rows)")

    if ranges_out is not None:
        fgr, axr2 = plt.subplots(figsize=(7, 4))
        axr2.plot(range_df["E_upper_meV"], range_df["mean_rel_err"],
                  "o-", color="tab:green", lw=1.6)
        axr2.set_xlabel("upper energy bound $E$ (meV)")
        axr2.set_ylabel("mean relative error (ratio)")
        axr2.set_title(f"{INTERACTION}: mean rel. error vs. energy window; "
                       r"$|\Delta E|/\max(E{-}E(r_e),25\,\mathrm{meV})$")
        fgr.tight_layout()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fgr.savefig(ranges_out)
        plt.close(fgr)
        print("Saved", ranges_out)


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
    ax.set_title(f"{INTERACTION} {MOLECULE}: waterfall of aligned E(r), E < 0 ({n} orientations)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)


def plot_selected_series(df_ref, df_model, out, csv_out, log_out, e_floor=25e-3):
    """Plot E(r) (data=points, fit=lines) for orientations selected by largest
    relative E-error over several regions. The reference energy E is the
    energy above the equilibrium minimum, E = e_ref - e_min = E - E(r_e):
      R1: r < r_e and E > 0            -> max-max and max-mean
      R2: r > r_e                      -> max-max and max-mean
      R3: E < 100 meV                  -> max-max and max-mean
    Before plotting the selected curves are written to csv_out and the chosen
    orientations / error statistics to log_out.
    """
    join = shifted_aligned(df_ref, df_model, e_lt_zero=False)
    # true equilibrium distance per orientation (unrounded)
    re = extract_energy_minimums(df_ref, r_max=12)[["phi1", "phi2", "r"]] \
        .rename(columns={"r": "re"})
    join = join.merge(re, on=["phi1", "phi2"], how="left")
    join["E"] = join["e_rel"]
    join["rel_err"] = np.abs(join["delta_e"]) / np.maximum(join["E"], e_floor)

    join["R1"] = (join["r"] < join["re"]) & (join["E"] > 0)
    join["R2"] = join["r"] > join["re"]
    join["R3"] = join["E"] < 0.1  # 100 meV

    def pick(region_col, agg_name, label):
        sub = join.loc[join[region_col]]
        rel = sub.groupby(["phi1", "phi2"])["rel_err"].agg(agg_name)
        absg = sub.groupby(["phi1", "phi2"])["delta_e"].apply(
            lambda x: np.max(np.abs(x)) if agg_name == "max" else np.mean(np.abs(x))
        )
        key = rel.idxmax()
        return (label, key[0], key[1], rel[key], absg[key])

    sel = [
        pick("R1", "max", "R1<re,E>0 max-max"),
        pick("R1", "mean", "R1<re,E>0 max-mean"),
        pick("R2", "max", "R2>re max-max"),
        pick("R2", "mean", "R2>re max-mean"),
        pick("R3", "max", "R3<E<100meV max-max"),
        pick("R3", "mean", "R3<E<100meV max-mean"),
    ]

    # ---- write selected curves to CSV before plotting ----
    frames = []
    for label, phi1, phi2, relv, abserr in sel:
        sub = join[(join["phi1"] == phi1) & (join["phi2"] == phi2)].copy()
        sub["selection"] = label
        frames.append(sub[["selection", "phi1", "phi2", "r", "s",
                           "e_ref", "e_model", "delta_e", "rel_err"]])
    csv_df = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    csv_df.to_csv(csv_out, index=False)
    print("Saved", csv_out, f"({len(csv_df)} rows)")

    # ---- log chosen orientations and underlying error values ----
    lines = [
        "# phi1\tphi2\tselection\trelative_err_stat\tabsolute_err_stat_(eV)",
        "# relative_err_stat: max or mean |delta_e|/max(E-E(re),25meV) in the region",
        "# absolute_err_stat: max or mean |delta_e| (eV) in the region",
    ]
    for label, phi1, phi2, relv, abserr in sel:
        lines.append(f"{phi1}\t{phi2}\t{label}\t{relv:.6e}\t{abserr:.6e}")
    os.makedirs(os.path.dirname(log_out) or ".", exist_ok=True)
    with open(log_out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Saved", log_out)

    # ---- plot: data as points, fit as lines ----
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for k, (label, phi1, phi2, relv, abserr) in enumerate(sel):
        sub = csv_df[csv_df["selection"] == label]
        color = cmap(k % 10)
        ax.plot(sub["r"], sub["e_model"], color=color, ls="-", lw=1.5,
                label=f"{label}  (φ1={phi1}, φ2={phi2})")
        ax.plot(sub["r"], sub["e_ref"], color=color, ls="none", marker="o", ms=3)
    ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("E (eV)")
    ax.set_title(f"{INTERACTION}: selected orientations — fit (lines) vs data (points)")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)


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
        os.makedirs(os.path.dirname(out_shifted), exist_ok=True)
        fig.savefig(out_shifted if x_key == "s" else out_abs)
        plt.close(fig)
        print("Saved", out_shifted if x_key == "s" else out_abs)

    render("s", "shifted distance $r - r_e$ (Å)")
    render("r", "distance $r$ (Å)", show_fit=True)


def _parse_range(text):
    """Parse a comma-separated pair of floats into an (r_min, r_max) tuple."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected 'a,b', got {text!r}")
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid range {text!r}")


def _parse_harmonics(text):
    """Parse a 'EP:8,1;EA:8,1;OP:20,1;OA:20,1' string into a
    {interaction: (h_chi, h_psi)} dict (entries separated by ';')."""
    mapping = {}
    for token in text.split(";"):
        token = token.strip()
        if not token:
            continue
        inter, _, h = token.partition(":")
        if ":" not in token or not h:
            raise argparse.ArgumentTypeError(f"invalid harmonic entry {token!r}")
        chi, _, psi = h.partition(",")
        mapping[inter.strip().upper()] = (int(chi), int(psi))
    return mapping


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot potential E(r) statistics and the Fourier-Morse fit "
                    "error vs distance for any interaction/configuration."
    )
    parser.add_argument("--molecule", default="PA",
                        help="molecule name (default: PA)")
    parser.add_argument("--interaction", default="OA",
                        help="interaction type, e.g. OA/OP/EA/EP (default: OA)")
    parser.add_argument("--alpha-fit", dest="alpha_fit", action="store_true",
                        default=True,
                        help="fit per-orientation alpha via Fourier expansion "
                             "(default: on)")
    parser.add_argument("--no-alpha-fit", dest="alpha_fit", action="store_false",
                        help="use a fixed alpha (disables alpha fit)")
    parser.add_argument("--interpolate", dest="interpolate", action="store_true",
                        default=True,
                        help="off-grid harmonic interpolation of r_e (default: on)")
    parser.add_argument("--no-interpolate", dest="interpolate", action="store_false",
                        help="use discrete energy minimum for r_e")
    parser.add_argument(
        "--harmonics", type=_parse_harmonics,
        default="EP:8,1;EA:8,1;OP:20,1;OA:20,1",
        help="harmonic ceilings as 'EP:8,1;EA:8,1;OP:20,1;OA:20,1' "
             "(default: EP/EA 8,1; OP/OA 20,1)",
    )
    parser.add_argument("--energy-range", default="8.3,10.3", type=_parse_range,
                        help="restricted r range for stats, e.g. '8.3,10.3'")
    parser.add_argument("--coarse-r", default=0.1, type=float,
                        help="coarse r-sampling spacing in A (default: 0.1)")
    parser.add_argument("--fine-r", default=0.0125, type=float,
                        help="fine r-sampling spacing near minimum in A (default: 0.0125)")
    parser.add_argument("--fit-input", default=None,
                        help="CSV of a precomputed potential fit (df_model). "
                             "If omitted, a fit is computed inline with equal weights.")
    parser.add_argument("--data-file", default=None,
                        help="CSV of the reference data frame (df_ref). "
                             "If omitted, the default reference data for the "
                             "selected molecule/interaction is loaded.")
    parser.add_argument("--output-dir", default="Figures",
                        help="output directory (default: Figures)")
    parser.add_argument("--tag", default=None,
                        help="suffix appended to every output filename, e.g. "
                             "w_poisson_lam0.5 (default: w_equal when fitting inline)")
    parser.add_argument("--weight-func", default=None,
                        help="inline-fit alpha weighting, e.g. 'equal', "
                             "'poisson', 'gaussian', 'energy' (default: equal)")
    parser.add_argument("--weight-lam", default=None, type=float,
                        help="parameter for the inline weight function "
                             "(e.g. poisson lambda / gaussian sigma / energy exponent)")
    parser.add_argument("--weight-eps", default=1e-4, type=float,
                        help="floor for the energy weight (default: 1e-4)")
    args = parser.parse_args(argv)

    # ---- set module-level configuration used by the plotting functions ----
    global MOLECULE, INTERACTION, HARMONIC_CEILS, ALPHA_FIT
    global ENERGY_RANGE, COARSE_R, FINE_R
    MOLECULE = args.molecule
    INTERACTION = args.interaction
    HARMONIC_CEILS = args.harmonics
    ALPHA_FIT = args.alpha_fit
    ENERGY_RANGE = args.energy_range
    COARSE_R = args.coarse_r
    FINE_R = args.fine_r

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    molecule = load_molecule_info(MOLECULE)
    df_ref = _load_reference(molecule, INTERACTION, args.data_file)

    if args.fit_input:
        df_model = pd.read_csv(args.fit_input)
        tag = args.tag if args.tag is not None else ""
    else:
        fit_kwargs = dict(alpha_fit=ALPHA_FIT, interpolate=args.interpolate,
                          print_errors=False)
        if args.weight_func:
            from chimorse.fitting import make_weight_func
            fit_kwargs["weight_func"] = make_weight_func(
                args.weight_func, lam=args.weight_lam, eps=args.weight_eps
            )
        df_model = generate_fourier_morse_data(
            df_ref, molecule, INTERACTION, HARMONIC_CEILS, **fit_kwargs
        )
        tag = args.tag if args.tag is not None else "w_equal"

    # prefix every output with the interaction so different interactions do not collide
    pfx = INTERACTION.lower()
    _P = lambda base, ext: _tagged(out_dir, f"{pfx}_{base}", ext, tag)

    # ---------- statistics / bound-region ----------
    make_plot(df_ref, df_model, "", _P("E_statistics_and_error", ".pdf"))

    mask = df_ref["e"] <= 0
    df_ref_bound = df_ref[mask]
    keys = pd.MultiIndex.from_frame(df_ref_bound[["phi1", "phi2", "r"]].drop_duplicates())
    df_model_bound = df_model[
        pd.MultiIndex.from_frame(df_model[["phi1", "phi2", "r"]]).isin(keys)
    ].reset_index(drop=True)
    make_plot(
        df_ref_bound, df_model_bound, " (E ≤ 0 only)",
        _P("E_statistics_and_error_bound", ".pdf"),
        stats_file=_P("fit_error_stats", ".txt"),
    )

    # ---------- shifted / relative error / waterfall / selected series ----------
    plot_shifted_error(
        df_ref, df_model,
        _P("shifted_error", ".pdf"),
        csv_out=_P("shifted_error_data", ".csv"),
    )
    plot_waterfall(df_ref, df_model, _P("waterfall", ".pdf"))
    plot_shifted_rel_error(
        df_ref, df_model,
        _P("shifted_rel_error", ".pdf"),
        csv_out=_P("shifted_rel_error_data", ".csv"),
        ranges_csv_out=_P("rel_error_ranges", ".csv"),
        ranges_out=_P("rel_error_ranges", ".pdf"),
    )
    plot_selected_series(
        df_ref, df_model,
        _P("selected_series", ".pdf"),
        _P("selected_series_data", ".csv"),
        _P("selected_series_log", ".txt"),
    )

    # ---------- top-RMSE orientations ----------
    plot_top_rmse(
        df_ref, df_model,
        _P("top_rmse_shifted", ".pdf"),
        _P("top_rmse_abs", ".pdf"),
    )


if __name__ == "__main__":
    main()
