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
    """Infer screw direction from chi definition in data."""
    sample = df.tail(100)
    chi_minus = (sample['phi1'] - sample['phi2']) % 360
    chi_plus = (sample['phi1'] + sample['phi2']) % 360

    if (sample['chi'] == chi_minus).all(): return 1
    elif (sample['chi'] == chi_plus).all(): return -1
    else: raise ValueError("Cannot determine screw direction from chi")

# ----------------------------------------------------------------------

def extract_energy_minimums(df, r_max=12):
    """Extract rows at the minimum energy for each (chi, psi, class, ...) subject to r <= r_max."""
    cols_no_er = [c for c in df.columns if c not in ('e', 'r')]
    idx = df.groupby(cols_no_er)['e'].idxmin()
    df_lowest_e = df.loc[idx].reset_index(drop=True)
    return df_lowest_e[df_lowest_e['r'] <= r_max]

# ----------------------------------------------------------------------

def extract_energy_comparison(df_data, df_model):
    """Align model and reference on the same grid; return full energy vectors, well depths D, and equilibrium distances r_e for both."""
    cols_no_e = [c for c in df_data.columns if c not in ('e', 'psi', 'chi')]
    model_subset = df_model.merge(
        df_data[cols_no_e].drop_duplicates(),
        on=cols_no_e,
        how='inner'
    ).copy()

    sorted_model = model_subset.sort_values(by=cols_no_e)
    sorted_data  = df_data.sort_values(by=cols_no_e)

    df_min_model = extract_energy_minimums(sorted_model, r_max=12)
    df_min_data  = extract_energy_minimums(sorted_data,  r_max=12)

    key_cols = [c for c in df_min_data.columns if c not in ('e', 'r')]
    df_min_model = df_min_model.sort_values(by=key_cols)
    df_min_data  = df_min_data.sort_values(by=key_cols)

    D_model = df_min_model['e'].values       # in eV
    D_data  = df_min_data['e'].values
    re_model = df_min_model['r'].values      # in Å
    re_data  = df_min_data['r'].values

    E_data  = sorted_data['e'].values
    E_model = sorted_model['e'].values

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
