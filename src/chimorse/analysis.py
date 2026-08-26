"""
analysis.py
-----------
Geometric utilities and data-analysis routines for chiral interaction energy surfaces.
Includes screw-symmetry operations, energy-minimum extraction, and error metrics.
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------

def get_symm_chi(interaction):
    """Return 1 if interaction is chi-symmetric ('P'), else 0."""
    return 1 if interaction[1]=='P' else 0

# ----------------------------------------------------------------------

def get_screw_dir(interaction):
    """Return the screw direction (+1 or -1) from the interaction's first letter."""
    return 1 if interaction[0] in ('E', 'S') else -1

# ----------------------------------------------------------------------

def circular_distance_deg(a, b):
    """Circular distance on [0, 360)."""
    return np.abs((a - b + 180) % 360 - 180)

# ----------------------------------------------------------------------

def infer_screw_direction(df):
    """Infer screw direction (+1 or -1) from the stored chi coordinate."""
    required = {"phi1", "phi2", "chi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns required to infer screw direction: {sorted(missing)}")

    sample = df[["phi1", "phi2", "chi"]].dropna().tail(100)
    if sample.empty:
        raise ValueError("Cannot infer screw direction from an empty dataframe")

    chi_equal = (sample["phi1"] - sample["phi2"]) % 360
    chi_opposite = (sample["phi1"] + sample["phi2"]) % 360

    err_equal = circular_distance_deg(sample["chi"].to_numpy(), chi_equal.to_numpy())
    err_opposite = circular_distance_deg(sample["chi"].to_numpy(), chi_opposite.to_numpy())

    if np.all(err_equal < 1e-8):
        return 1
    if np.all(err_opposite < 1e-8):
        return -1

    raise ValueError("Cannot determine screw direction from chi")

# ----------------------------------------------------------------------

def extract_energy_minimums(df, r_max=12, interpolate=True, n_points=5):
    """
    Return the minimum-energy configuration for each angular orientation.

    If interpolate=True, estimate r_e and E_min continuously from a
    local quadratic fit around the lowest sampled radial point.
    """
    required = {"phi1", "phi2", "r", "e"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns required for energy minima: {sorted(missing)}"
        )

    data = df.loc[df["r"] <= r_max].copy()

    if data.empty:
        return data.reset_index(drop=True)

    group_cols = ["phi1", "phi2"]

    if "zeta" in data.columns:
        group_cols.append("zeta")

    rows = []

    for _, profile in data.groupby(group_cols, dropna=False):

        # Start from the actual sampled minimum row so that
        # phi1, phi2, chi, psi, zeta, etc. are preserved.
        idx = profile["e"].idxmin()
        row = profile.loc[idx].copy()

        if interpolate:
            re, e_min = _quadratic_energy_minimum(
                profile,
                n_points=n_points,
            )

            row["r"] = re
            row["e"] = e_min

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)

# ----------------------------------------------------------------------

def extract_energy_comparison(df_data, df_model):
    """Align model and reference on the same grid and compare E, D, and r_e."""
    use_zeta = "zeta" in df_data.columns or "zeta" in df_model.columns
    if use_zeta and not ("zeta" in df_data.columns and "zeta" in df_model.columns):
        raise ValueError("df_data and df_model must either both contain zeta or both omit it")

    grid_cols = ["phi1", "phi2"]
    if use_zeta:
        grid_cols.append("zeta")
    grid_cols.append("r")

    energy_cols = grid_cols + ["chi", "psi", "e"]

    model_subset = df_model.merge(
        df_data[grid_cols].drop_duplicates(),
        on=grid_cols,
        how="inner",
    ).copy()

    sorted_model = model_subset[energy_cols].sort_values(by=grid_cols).reset_index(drop=True)
    sorted_data = (
        df_data[energy_cols]
        .merge(model_subset[grid_cols].drop_duplicates(), on=grid_cols, how="inner")
        .sort_values(by=grid_cols)
        .reset_index(drop=True)
    )

    if len(sorted_data) != len(sorted_model):
        raise ValueError("Reference and model grids do not align one-to-one")

    df_min_model = extract_energy_minimums(sorted_model, r_max=12)
    df_min_data = extract_energy_minimums(sorted_data, r_max=12)

    key_cols = ["phi1", "phi2"]
    if use_zeta:
        key_cols.append("zeta")

    min_model = df_min_model[key_cols + ["e", "r"]].rename(
        columns={"e": "e_model", "r": "r_model"}
    )
    min_data = df_min_data[key_cols + ["e", "r"]].rename(
        columns={"e": "e_data", "r": "r_data"}
    )

    minima = min_data.merge(min_model, on=key_cols, how="inner")
    if len(minima) != len(min_data) or len(minima) != len(min_model):
        raise ValueError("Reference and model minima do not align one-to-one")

    E_data = sorted_data["e"].to_numpy()
    E_model = sorted_model["e"].to_numpy()
    D_data = minima["e_data"].to_numpy()
    D_model = minima["e_model"].to_numpy()
    re_data = minima["r_data"].to_numpy()
    re_model = minima["r_model"].to_numpy()

    return E_data, E_model, D_model, D_data, re_model, re_data

# ----------------------------------------------------------------------

def compute_near_equilibrium_energy_rmse(df_data, df_model, delta_r=1.0, r_max=12):
    r"""Compute the energy RMSE near the reference equilibrium distance.
    Only points satisfying
    :math:`|r - r_{e,\mathrm{ref}}| \leq \Delta r`
    for each angular configuration are included.
    """

    if delta_r <= 0:
        raise ValueError("delta_r must be positive")

    # Coordinates defining one angular configuration
    orientation_cols = ["phi1", "phi2"]

    use_zeta = "zeta" in df_data.columns or "zeta" in df_model.columns

    if use_zeta:
        if not ("zeta" in df_data.columns and "zeta" in df_model.columns):
            raise ValueError(
                "df_data and df_model must either both contain zeta "
                "or both omit it"
            )

        orientation_cols.append("zeta")

    # Find the reference equilibrium distance for each orientation
    df_min_data = extract_energy_minimums(df_data, r_max=r_max)

    re_reference = (df_min_data[orientation_cols + ["r"]].rename(columns={"r": "re_ref"}))

    # Align reference and model energies on the physical grid
    grid_cols = orientation_cols + ["r"]

    ref = (df_data[grid_cols + ["e"]].rename(columns={"e": "e_ref"}))

    model = (df_model[grid_cols + ["e"]].rename(columns={"e": "e_model"}))

    aligned = ref.merge(model, on=grid_cols, how="inner", validate="one_to_one")

    if aligned.empty:
        raise ValueError(
            "Reference and model data have no matching grid points"
        )

    # Attach the equilibrium distance belonging to each orientation.
    aligned = aligned.merge(
        re_reference,
        on=orientation_cols,
        how="left",
        validate="many_to_one",
    )

    if aligned["re_ref"].isna().any():
        raise ValueError(
            "Could not determine the reference equilibrium distance "
            "for every angular configuration"
        )

    # Keep only points close to equilibrium
    near_eq_mask = (np.abs(aligned["r"] - aligned["re_ref"]) <= delta_r)

    near_eq = aligned.loc[near_eq_mask]

    if near_eq.empty:
        raise ValueError(
            "No data points found inside the requested "
            f"near-equilibrium window ±{delta_r} Å"
        )

    # Energy RMSE
    delta_E = (near_eq["e_model"].to_numpy() - near_eq["e_ref"].to_numpy())

    return np.sqrt(np.mean(delta_E**2))


# ----------------------------------------------------------------------

def compute_energy_errors(df_data, df_model, print_errors=True, near_eq_delta_r=1.0):
    """ Compute RMSE and mean/max residuals for E, D, and r_e between
    model and reference. + RMSE using only points within ±near_eq_delta_r """
    print('='*200)
    (E_data, E_model, D_model, D_data, re_model, re_data) = extract_energy_comparison(df_data, df_model)

    Delta_E = E_model - E_data
    Delta_D = D_model - D_data
    Delta_re = re_model - re_data

    E_rmse_near_eq = compute_near_equilibrium_energy_rmse(df_data, df_model, delta_r=near_eq_delta_r)

    errors = {
        # Well depth
        # "global_D_residuals": np.mean(Delta_D),
        # "max_D_residuals": np.max(Delta_D),
        "D_rmse": np.sqrt(np.mean(Delta_D**2)),

        # Equilibrium distance
        # "global_re_residuals": np.mean(Delta_re),
        # "max_re_residuals": np.max(Delta_re),
        "re_rmse": np.sqrt(np.mean(Delta_re**2)),

        # Full energy grid
        # "global_E_residuals": np.mean(Delta_E),
        # "max_E_residuals": np.max(Delta_E),
        "E_rmse": np.sqrt(np.mean(Delta_E**2)),

        # Energy near equilibrium
        "E_rmse_near_eq": E_rmse_near_eq,
    }

    if print_errors:
        for key, value in errors.items():

            if (
                key.startswith("global_re")
                or key.startswith("max_re")
                or key.startswith("re_rmse")
            ):
                print(
                    f"{key:25s}: "
                    f"{value:.5e} Å"
                )

            else:
                print(
                    f"{key:25s}: "
                    f"{value * 1000:.5e} meV"
                )
    print('='*200)
    return errors

# ----------------------------------------------------------------------

def expand_by_screw_periodicity(df, screw_step, screw_dir):
    """Expand orientation minima by screw periodicity while preserving their values."""
    if screw_step <= 0:
        raise ValueError("screw_step must be positive")
    if screw_dir not in (-1, 1):
        raise ValueError("screw_dir must be +1 or -1")

    shifts = np.arange(0, 360, screw_step)
    frames = []

    for delta in shifts:
        tmp = df.copy()
        tmp["phi1"] = (tmp["phi1"] + delta) % 360
        tmp["phi2"] = (tmp["phi2"] + screw_dir * delta) % 360
        tmp["chi"] = (tmp["phi1"] - screw_dir * tmp["phi2"]) % 360
        tmp["psi"] = (tmp["phi1"] + screw_dir * tmp["phi2"]) % 360
        frames.append(tmp)

    out = pd.concat(frames, ignore_index=True)

    key_cols = ["phi1", "phi2", "chi", "psi"]
    if "zeta" in out.columns:
        key_cols.append("zeta")

    value_cols = [c for c in ("e", "r") if c in out.columns]
    if value_cols:
        return out.groupby(key_cols, as_index=False, dropna=False)[value_cols].mean()

    return out.drop_duplicates(subset=key_cols).reset_index(drop=True)

# ----------------------------------------------------------------------

def build_energy_table(df, piv, screw_step):
    """Build full energy grids in the chi-frame (tiled) and phi1-frame (rolled by screw periodicity) from a (phi2, chi) pivot table."""
    screw_dir = infer_screw_direction(df)
    n_repeats = round(360 / screw_step)

    col_step = float(piv.columns[1] - piv.columns[0])
    step_indices = round(screw_step / col_step)

    full_energy_chi = np.tile(piv.values, (n_repeats, 1))
    full_energy_phi2 = np.vstack([
        np.roll(piv.values, shift=i * screw_dir * step_indices, axis=1)
        for i in range(n_repeats)
    ])
    return full_energy_chi, full_energy_phi2

# ----------------------------------------------------------------------

def build_full_energy_table(df, piv, mode='chi', screw_step=None):
    """Build a full 360° periodic energy grid from a reduced pivot table."""
    if mode not in ("chi", "phi1", "phi"):
        raise ValueError(f"mode must be 'chi' or 'phi1', got {mode!r}")

    if screw_step is None:
        phi1_vals = np.sort(df["phi1"].dropna().unique())
        diffs = np.diff(phi1_vals)
        diffs = diffs[diffs > 1e-8]
        if len(diffs) == 0:
            raise ValueError("Cannot infer screw_step from phi1; pass it explicitly")
        screw_step = float(diffs.min())

    full_chi, full_phi1 = build_energy_table(df, piv, screw_step)
    return full_chi if mode == "chi" else full_phi1

# ----------------------------------------------------------------------

def expand_chi_psi_by_screw_periodicity(df, screw_step):
    """Expand the full data grid by screw periodicity without collapsing r values."""
    if screw_step <= 0:
        raise ValueError("screw_step must be positive")

    screw_dir = infer_screw_direction(df)
    shifts = np.arange(0, 360, screw_step)
    frames = []

    for delta in shifts:
        tmp = df.copy()
        tmp["phi1"] = (tmp["phi1"] + delta) % 360
        tmp["phi2"] = (tmp["phi2"] + screw_dir * delta) % 360
        tmp["chi"] = (tmp["phi1"] - screw_dir * tmp["phi2"]) % 360
        tmp["psi"] = (tmp["phi1"] + screw_dir * tmp["phi2"]) % 360
        frames.append(tmp)

    out = pd.concat(frames, ignore_index=True)

    coord_cols = [c for c in ("phi1", "phi2", "zeta", "r", "chi", "psi") if c in out.columns]
    value_cols = [c for c in out.columns if c not in coord_cols]

    if value_cols:
        return out.groupby(coord_cols, as_index=False, dropna=False)[value_cols].mean()

    return out.drop_duplicates(subset=coord_cols).reset_index(drop=True)

# ----------------------------------------------------------------------

def compute_local_morse_rmse(df, delta_r=0.5, r_max=12.0):
    """
    Irreducible near-equilibrium Morse radial-form error.

    For each angular configuration:
      1. locate the reference minimum,
      2. keep points within ±delta_r,
      3. independently fit D, re, alpha,
      4. compute the energy residuals in that same region.

    Returns one global RMSE in eV.
    """
    from scipy.optimize import curve_fit

    def morse(r, D, re, alpha):
        return D * (
            np.exp(-2.0 * alpha * (r - re))
            - 2.0 * np.exp(-alpha * (r - re))
        )

    data = df[df["r"] <= r_max].copy()

    group_cols = ["phi1", "phi2"]
    if "zeta" in data.columns:
        group_cols.append("zeta")

    residuals = []

    for _, profile in data.groupby(group_cols, dropna=False):

        profile = profile.sort_values("r")

        r_all = profile["r"].to_numpy()
        e_all = profile["e"].to_numpy()

        # Reference discrete minimum
        i_min = np.argmin(e_all)
        re0 = r_all[i_min]
        D0 = max(-e_all[i_min], 1e-8)

        # Near-equilibrium region only
        mask = np.abs(r_all - re0) <= delta_r

        r = r_all[mask]
        e = e_all[mask]

        # Need enough points to fit 3 parameters
        if len(r) < 4:
            continue

        try:
            popt, _ = curve_fit(
                morse,
                r,
                e,
                p0=[D0, re0, 1.2],
                bounds=(
                    [0.0, r.min(), 1e-6],
                    [np.inf, r.max(), np.inf],
                ),
                maxfev=10000,
            )

            e_fit = morse(r, *popt)
            residuals.extend(e_fit - e)

        except (RuntimeError, ValueError):
            continue

    if not residuals:
        raise ValueError("No Morse profiles could be fitted.")

    residuals = np.asarray(residuals)

    return np.sqrt(np.mean(residuals**2))

# ----------------------------------------------------------------------

def _quadratic_energy_minimum(profile, n_points=5):
    """
    Estimate the continuous minimum of one radial energy profile
    from a local quadratic fit around the lowest sampled point.

    Falls back to the discrete minimum if interpolation is not reliable.
    """
    profile = profile.sort_values("r")

    r = profile["r"].to_numpy(dtype=float)
    e = profile["e"].to_numpy(dtype=float)

    i_min = np.argmin(e)

    # Discrete fallback
    re_discrete = r[i_min]
    e_discrete = e[i_min]

    if len(r) < 3 or i_min == 0 or i_min == len(r) - 1:
        return re_discrete, e_discrete

    # Number of local points
    n_points = min(n_points, len(r))

    # Prefer an odd number
    if n_points % 2 == 0:
        n_points -= 1

    half = n_points // 2

    start = max(0, i_min - half)
    stop = min(len(r), i_min + half + 1)

    # Shift window if close to an edge
    if stop - start < n_points:
        if start == 0:
            stop = min(len(r), n_points)
        else:
            start = max(0, len(r) - n_points)

    r_local = r[start:stop]
    e_local = e[start:stop]

    if len(r_local) < 3:
        return re_discrete, e_discrete

    # Center r for numerical stability
    x = r_local - re_discrete

    a, b, c = np.polyfit(x, e_local, 2)

    # Must actually describe a minimum
    if a <= 0:
        return re_discrete, e_discrete

    x_min = -b / (2.0 * a)
    re = re_discrete + x_min

    # Reject extrapolation outside the fitted local interval
    if re < r_local.min() or re > r_local.max():
        return re_discrete, e_discrete

    e_min = a * x_min**2 + b * x_min + c

    return re, e_min