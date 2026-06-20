"""
fitting.py
----------
Curve-fitting routines for 1D potentials, alpha extraction,
Fourier-Morse model assembly, and coefficient pruning.
"""

import numpy as np
import pandas as pd
from itertools import product
from lmfit import Model
from scipy.optimize import curve_fit

from .analysis import extract_energy_minimums, get_symm_chi, get_screw_dir, compute_energy_errors
from .fourier import create_matrix_lsqt_2d, count_fourier_coeffs
from .models import (Morse_1D, MorseAnisotropic, MorseAnisotropicAlpha,
                    smooth_energy_profile, generate_model_df, evaluate_model_on_reference_grid)

# ----------------------------------------------------------------------

def print_modeling_information(molecule, interaction, harmonic_ceils):
    """Print molecule, interaction, harmonic orders, and the resulting Fourier coefficient counts."""
    h_chi, h_psi = harmonic_ceils[interaction]
    symm_chi = get_symm_chi(interaction)
    print(f"molecule                 : {molecule.name}")
    print(f"interaction              : {interaction}")
    print(f"harmonics (χ, ψ)         : {harmonic_ceils[interaction]}")
    print(f"n_coeffs before symmetry : {count_fourier_coeffs(h_chi, h_psi, symm_chi, molecule.screw_step, after_symm=False)}")
    print(f"n_coeffs                 : {count_fourier_coeffs(h_chi, h_psi, symm_chi, molecule.screw_step, after_symm=True)}")
    print('-'*30)

# ----------------------------------------------------------------------

def extract_reduced_coeffs(A, target, threshold=None, top_n=None, print_info=True):
    """Refit keeping only the largest-magnitude coefficients (by relative threshold or top_n); 
       return the zero-padded coefficient vector and kept indices."""
    if (threshold is None) == (top_n is None):
        raise ValueError("You must provide exactly one argument: either 'threshold' or 'top_n'.")

    full_coeff, *_ = np.linalg.lstsq(A, target, rcond=None)
    abs_coeffs = np.abs(full_coeff)

    if threshold is not None:
        threshold_var = threshold * np.max(abs_coeffs)
        keep_var = np.where(abs_coeffs > threshold_var)[0]
    else:
        keep_var = np.sort(np.argsort(abs_coeffs)[-top_n:])

    A_var_reduced = A[:, keep_var]
    reduced_coeff, *_ = np.linalg.lstsq(A_var_reduced, target, rcond=None)

    final_coeff = np.zeros(A.shape[1])
    final_coeff[keep_var] = reduced_coeff

    if print_info:
        print(f"Retained coeffs: {len(keep_var)} / {len(full_coeff)}")

    return final_coeff, keep_var

# ----------------------------------------------------------------------

def prune_by_magnitude(A, target, relative_thresholds, print_summary=False):
    """Run extract_reduced_coeffs over a list of thresholds; 
       return the resulting RMSE and coefficient-count lists."""
    full_coeff, *_ = np.linalg.lstsq(A, target, rcond=None)

    rmse_list = []
    n_coeff_list = []

    for threshold in relative_thresholds:
        reduced_coeff, keep = extract_reduced_coeffs(A, target, threshold, print_info=False)

        pred = A @ reduced_coeff
        rmse = np.sqrt(np.mean((target - pred) ** 2))

        rmse_list.append(rmse)
        n_coeff_list.append(len(keep))

        if print_summary:
            print(f"Threshold {threshold:.2e}: {len(keep):3d} / {len(full_coeff)} coeffs, "
                  f"RMSE: {rmse*1000:.4f} meV")

    return rmse_list, n_coeff_list

# ----------------------------------------------------------------------

def fit_Er_1D_curvefit(raw_data, phi_vals, potential_func, initial_params=None, fit_mode='fit'):
    """Fit potential_func to E(r) at the given (phi1, phi2) via curve_fit; 
       return the (r, e) curve from the fit or from initial_params."""
    df = raw_data
    Er = df.loc[(df['phi1'] == phi_vals[0]) & (df['phi2'] == phi_vals[1])].copy()
    popt, pcov = curve_fit(potential_func, Er.r, Er.e, p0=initial_params, method='trf')

    if fit_mode=='curve_fit':
        print(f'fitted params for {potential_func} model: {popt}')

    fitted_df = pd.DataFrame({
        'r': Er.r.values,
        'e': potential_func(Er.r.values, *popt) if fit_mode=='fit'else potential_func(Er.r.values, *initial_params)
    })
    return fitted_df

# ----------------------------------------------------------------------

def fit_Er_1D_lmfit(df, phi_vals, pot_model, init_params, fit_mode):
    """Fit pot_model to E(r) at the given (phi1, phi2) via lmfit; 
       return the (r, e) curve from the fit or from init_params."""
    y_model = Model(pot_model)
    if pot_model==Morse_1D:
        params = y_model.make_params(D=init_params[0], re=init_params[1], alpha=init_params[2])
    else:
        params = y_model.make_params(epsilon=init_params[0], sigma=init_params[1])
    Er = df.loc[(df['phi1'] == phi_vals[0]) & (df['phi2'] == phi_vals[1])].copy()
    if Er.empty:
        raise ValueError(f"No data found for phi1={phi_vals[0]}, phi2={phi_vals[1]}")
    result = y_model.fit(Er.e, params, r=Er.r)

    fitted_df = pd.DataFrame({
        'r': Er.r.values,
        'e': result.best_fit if fit_mode == 'fit' else pot_model(Er.r.values, *init_params)
    })
    return fitted_df

# ----------------------------------------------------------------------

def fit_alpha_morse(df, phi_vals, screw_dir, alpha_init=1.2, smooth_factor=None):
    """Fit the Morse alpha at fixed (phi1, phi2) with D and r_e fixed to the data minimum; 
       return a one-row DataFrame with chi, psi, D, re, alpha."""
    phi1, phi2 = phi_vals
    mask = (df['phi1'] == phi1) & (df['phi2'] == phi2)
    Er = df[mask].copy()

    e_vals = Er['e']
    if smooth_factor:
        r_smooth, e_vals = smooth_energy_profile(
            Er['r'].values,
            Er['e'].values,
            smooth_factor=smooth_factor
        )

    D = -e_vals.min()
    re = Er.loc[Er['e'].idxmin(), 'r']

    model = Model(Morse_1D)
    params = model.make_params(D=D, re=re, alpha=alpha_init)
    params['D'].vary = params['re'].vary = False
    result = model.fit(Er['e'], params, r=Er['r'])
    alpha = result.params['alpha'].value

    fitted_data = pd.DataFrame([{
        'phi1': phi1,
        'phi2': phi2,
        'chi': (phi1 - screw_dir * phi2) % 360,
        'psi': (phi1 + screw_dir * phi2) % 360,
        're': re,
        'D': D,
        'alpha': alpha
    }])
    return fitted_data

# ----------------------------------------------------------------------

def fit_alpha_values(df, interaction):
    """Run fit_alpha_morse over every (phi1, phi2) combination and 
       return the array of fitted alpha values."""
    screw_dir = get_screw_dir(interaction)
    phi1_range = df['phi1'].unique()
    phi2_range = df['phi2'].unique()

    results = [fit_alpha_morse(df, (phi1, phi2), screw_dir, smooth_factor=None)
               for phi1, phi2 in product(phi1_range, phi2_range)]

    return pd.concat(results, ignore_index=True)['alpha'].values

# ----------------------------------------------------------------------

def create_morse_model(df, molecule, interaction,
                       A, h_chi, h_psi, D_coeff, re_coeff, alpha_coeff=None):
    """Build a MorseAnisotropic (or MorseAnisotropicAlpha if alpha_coeff is given) model with its coefficient tuple."""
    if alpha_coeff is None:
        model = MorseAnisotropic(
            h_chi=h_chi,
            h_psi=h_psi,
            symm_chi=get_symm_chi(interaction),
            screw_step=molecule.screw_step,
            alpha=1.1
        )
        return model, (D_coeff, re_coeff)
    else:
        model = MorseAnisotropicAlpha(
            h_chi=h_chi,
            h_psi=h_psi,
            symm_chi=get_symm_chi(interaction),
            screw_step=molecule.screw_step,
        )
        return model, (D_coeff, re_coeff, alpha_coeff)

# ----------------------------------------------------------------------

def _parse_prune_arg(arg, keys=('D', 're', 'alpha')):
    """Normalize a pruning argument (None, dict, or scalar) into a {D, re, alpha} dict."""
    if arg is None:
        return {k: None for k in keys}
    if isinstance(arg, dict):
        return {k: arg.get(k, None) for k in keys}

    return {k: arg for k in keys}

# ----------------------------------------------------------------------

def generate_fourier_morse_data(df, molecule, interaction, harmonic_ceils,
                                alpha_fit=False, original_size=False, print_errors=True,
                                prune_model=False, prune_thresholds=None, prune_top_n=None):
    """Fit (and optionally prune) Fourier coefficients for D, r_e, and alpha, 
       then evaluate the resulting anisotropic Morse model."""
    print_modeling_information(molecule, interaction, harmonic_ceils)

    E_min_df = extract_energy_minimums(df, r_max=12)
    D, re = -E_min_df['e'], E_min_df['r']
    chi_rad, psi_rad = np.deg2rad(E_min_df['chi']), np.deg2rad(E_min_df['psi'])
    h_chi, h_psi = harmonic_ceils[interaction]
    symm_chi = get_symm_chi(interaction)

    A, labels = create_matrix_lsqt_2d(h_chi, h_psi, chi_rad, psi_rad, symm_chi, molecule.screw_step)

    thresh = _parse_prune_arg(prune_thresholds)
    top = _parse_prune_arg(prune_top_n)

    D_coeff, *_ = np.linalg.lstsq(A, D, rcond=None)
    re_coeff, *_ = np.linalg.lstsq(A, re, rcond=None)
    alpha_coeff = None

    if alpha_fit:
        alpha_vals = fit_alpha_values(df, interaction)
        alpha_coeff, *_ = np.linalg.lstsq(A, alpha_vals, rcond=None)

    if prune_model:
        D_coeff, keep_D = extract_reduced_coeffs(A, D, threshold=thresh['D'], top_n=top['D'])
        re_coeff, keep_re = extract_reduced_coeffs(A, re, threshold=thresh['re'], top_n=top['re'])
        if alpha_fit:
            alpha_coeff, keep_alpha = extract_reduced_coeffs(A, alpha_vals, threshold=thresh['alpha'], top_n=top['alpha'])

    r_values, phi1_values, phi2_values = [np.sort(df[col].unique()) for col in ['r', 'phi1', 'phi2']]

    model, coeffs = create_morse_model(df, molecule, interaction,
                                       A, h_chi, h_psi, D_coeff, re_coeff, alpha_coeff)

    df_model = generate_model_df(model, coeffs, r_values, phi1_values, phi2_values, interaction)
    if original_size:
        df_model = evaluate_model_on_reference_grid(model, df, coeffs)

    if print_errors:
        print('='*200)
        compute_energy_errors(df, df_model)

    return df_model
