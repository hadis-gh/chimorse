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

def extract_energy_minimums(df, r_max=12):
    """Return the minimum-energy row for each physical angular configuration."""
    required = {"phi1", "phi2", "r", "e"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns required for energy minima: {sorted(missing)}")

    data = df.loc[df["r"] <= r_max].copy()
    if data.empty:
        return data.reset_index(drop=True)

    group_cols = ["phi1", "phi2"]
    if "zeta" in data.columns:
        group_cols.append("zeta")

    idx = data.groupby(group_cols, dropna=False)["e"].idxmin()
    return data.loc[idx].reset_index(drop=True)

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

def compute_energy_errors(df_data, df_model, print_errors=True):
    """Compute RMSE and mean/max residuals for E, D, and r_e between model and reference, optionally printing them."""
    E_data, E_model, D_model, D_data, re_model, re_data = extract_energy_comparison(df_data, df_model)

    Delta_E  = E_model - E_data
    Delta_D  = D_model - D_data          # in eV
    Delta_re = re_model - re_data        # in Å

    errors = {
        # full energy grid
        'global_E_residuals': np.mean(Delta_E),
        'max_E_residuals'   : np.max(Delta_E),
        'E_rmse'            : np.sqrt(np.mean(Delta_E**2)),

        # well depth (minimum energy)
        'global_D_residuals': np.mean(Delta_D),
        'max_D_residuals'   : np.max(Delta_D),
        'D_rmse'            : np.sqrt(np.mean(Delta_D**2)),

        # equilibrium distance
        'global_re_residuals': np.mean(Delta_re),
        'max_re_residuals'   : np.max(Delta_re),
        're_rmse'            : np.sqrt(np.mean(Delta_re**2)),
    }

    if print_errors:
        for k, v in errors.items():
            if k.startswith('global_re') or k.startswith('max_re') or k.startswith('re_rmse'):
                print(f"{k:25s}: {v:.5e} Å")
            else:
                print(f"{k:25s}: {v*1000:.5e} meV")

    return errors

# ----------------------------------------------------------------------

def expand_by_screw_periodicity(df, screw_step, screw_dir):
    """Expand surface by screw periodicity: phi1 -> phi1+delta, phi2 -> phi2+screw_dir*delta."""
    shifts = np.arange(0, 360, screw_step)
    frames = []

    for delta in shifts:
        tmp = df.copy()
        tmp['phi1'] = (tmp['phi1'] + delta) % 360
        tmp['phi2'] = (tmp['phi2'] + screw_dir * delta) % 360
        tmp['chi'] = (tmp['phi1'] - screw_dir * tmp['phi2']) % 360
        tmp['psi'] = (tmp['phi1'] + screw_dir * tmp['phi2']) % 360
        frames.append(tmp)

    out = pd.concat(frames, ignore_index=True)
    return out.groupby(['phi1', 'phi2', 'chi', 'psi'], as_index=False).agg({'e': 'mean', 'r': 'mean'})

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

def build_full_energy_table(df, piv, mode='chi'):
    """Build a full 360°-periodic energy grid from piv by tiling (mode='chi') or rolling by screw periodicity (mode='phi1')."""
    screw_dir = infer_screw_direction(df)
    phi_shift = 20
    n_repeats = 360 // phi_shift

    if mode == 'chi':
        full_energy = np.tile(piv.values, (n_repeats, 1))
    elif mode == 'phi1':
        full_energy = np.vstack([
            np.roll(piv.values, shift=i * screw_dir * phi_shift, axis=1)
            for i in range(n_repeats)
        ])
    else:
        ValueError(f"mode can be chi or phi1! {mode} is not valid here.")
    return full_energy

# ----------------------------------------------------------------------

def expand_chi_psi_by_screw_periodicity(df, screw_step):
    """Recompute chi/psi from phi1/phi2 and add psi-shifted copies (by 2*delta, delta stepping screw_step up to 180°) to fill the chi-psi torus."""
    screw_dir = infer_screw_direction(df)
    df = df.copy()
    df['chi'] = (df['phi1'] - screw_dir * df['phi2']) % 360
    df['psi'] = (df['phi1'] + screw_dir * df['phi2']) % 360
    return pd.concat([
        df.assign(psi=(df['psi'] + 2 * delta) % 360)
        for delta in np.arange(0, 180, screw_step)
    ], ignore_index=True)
