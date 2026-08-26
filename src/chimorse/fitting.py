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
from scipy.optimize import curve_fit, brentq
from scipy.special import digamma, gammaln

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

def equal_weights(r, re=None):
    """Constant unit weights — every data point contributes equally.

    The re argument is accepted for signature compatibility with the other
    weight distributions but is unused.
    """
    return np.ones_like(r, dtype=float)


def gaussian_weights(r, re, sigma=1.0):
    """Gaussian weight distribution centered at the equilibrium distance r_e.

    Matches the historical weighting used for the alpha fit: points close to
    r_e are weighted most strongly, with a symmetric fall-off governed by sigma.
    """
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return np.sqrt(np.exp(-0.5 * ((r - re) / sigma) ** 2))


def _continuous_poisson_logw(x, lam):
    """Log of the continuous (Gamma) Poisson mass function, safe for x > -1."""
    return x * np.log(lam) - lam - gammaln(x + 1.0)


def _continuous_poisson_mode(lam):
    """Continuous mode of the Gamma-Poisson mass function (root of digamma)."""
    if lam < 1.0:
        return 0.0
    target = np.log(lam)
    hi = max(lam, 1.0)
    while digamma(hi + 1.0) < target:
        hi *= 2.0
    return brentq(lambda x: digamma(x + 1.0) - target, 0.0, hi)


def poisson_weights(r, re, lam=None):
    """Poisson weight distribution whose maximum sits at the equilibrium distance r_e.

    The Poisson shape is right-skewed (longer tail toward large r) and falls off
    steeply below r_e, so it suppresses residuals from the steep repulsive side of
    the potential without over-damping the large-r tail. ``lam`` sets the Poisson
    parameter (spread/skew); it defaults to r_e, which keeps the mode at r_e.
    """
    lam = float(re) if lam is None else float(lam)
    re = float(re)
    if lam <= 0 or re <= 0:
        raise ValueError("lam and re must be positive")

    mode = _continuous_poisson_mode(lam)
    # Shift the distribution so its mode maps exactly to r = re.
    x = (r - re) + mode
    x = np.asarray(x, dtype=float)

    logw = np.full_like(x, -np.inf)
    valid = x > -1.0
    logw[valid] = _continuous_poisson_logw(x[valid], lam)
    logw_mode = _continuous_poisson_logw(mode, lam)

    w = np.exp(logw - logw_mode)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    return w


def make_weight_func(name, sigma=1.0, lam=None):
    """Return a weight callable ``w(r, re)`` for a named weight distribution.

    name must be one of ``'equal'``, ``'gaussian'`` (center r_e, width sigma),
    or ``'poisson'`` (mode at r_e, parameter lam). The returned callable takes
    the radial coordinates and the equilibrium distance r_e, so the distribution
    is centred at the per-orientation r_e during the fit.
    """
    name = str(name).lower()
    if name == "equal":
        return lambda r, re=None: np.ones_like(r, dtype=float)
    if name == "gaussian":
        return lambda r, re: gaussian_weights(r, re, sigma=sigma)
    if name == "poisson":
        return lambda r, re: poisson_weights(r, re, lam=lam)
    raise ValueError(
        f"Unknown weight distribution {name!r}; expected one of "
        "{'equal', 'gaussian', 'poisson'}."
    )

# ----------------------------------------------------------------------

def fit_alpha_morse(df, phi_vals, screw_dir, alpha_init=1.2, smooth_factor=None,
                    interpolate=True, weight_func=equal_weights):
    """Fit the Morse alpha at fixed (phi1, phi2) with D and r_e fixed to the data minimum; 
       return a one-row DataFrame with chi, psi, D, re, alpha.

    weight_func is a callable ``w(r, re)`` mapping the radial coordinate array
    and the per-orientation equilibrium distance to per-point fit weights; it
    defaults to equal weighting (constant unit weights).
    """
    phi1, phi2 = phi_vals
    mask = (df['phi1'] == phi1) & (df['phi2'] == phi2)
    Er = df[mask].copy()

    minimum = extract_energy_minimums(Er, interpolate=interpolate).iloc[0]
    D = -minimum["e"]
    re = minimum["r"]

    weights = weight_func(Er['r'].to_numpy(), re=re)

    model = Model(Morse_1D)
    params = model.make_params(D=D, re=re, alpha=alpha_init)
    params['D'].vary = params['re'].vary = False
    result = model.fit(Er['e'], params, r=Er['r'], weights=weights)
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

def fit_alpha_values(df, interaction, interpolate=True,
                     weight_func=equal_weights):
    """Fit alpha values in the same angular order as the energy minima.

    weight_func (a callable ``w(r, re)``) is forwarded to fit_alpha_morse for
    each orientation.
    """
    screw_dir = get_screw_dir(interaction)

    E_min_df = extract_energy_minimums(df, r_max=12, interpolate=interpolate)

    results = [
        fit_alpha_morse(
            df,
            (row.phi1, row.phi2),
            screw_dir,
            smooth_factor=None,
            interpolate=interpolate,
            weight_func=weight_func,
        )
        for row in E_min_df.itertuples(index=False)
    ]

    return pd.concat(
        results,
        ignore_index=True
    )['alpha'].to_numpy()

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
                                alpha_fit=False, interpolate=True,
                                weight_func=equal_weights,
                                print_errors=True, near_eq_delta_r=.5,
                                prune_model=False, prune_thresholds=None, prune_top_n=None):
    """Fit (and optionally prune) Fourier coefficients for D, r_e, and alpha, 
       then evaluate the resulting anisotropic Morse model.

    weight_func (a callable ``w(r, re)``) sets the per-point fit weights for the
    per-orientation alpha fits (equal weights by default).
    """
    print_modeling_information(molecule, interaction, harmonic_ceils)

    E_min_df = extract_energy_minimums(df, r_max=12, interpolate=interpolate)
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
        alpha_vals = fit_alpha_values(df, interaction, interpolate=interpolate,
                                      weight_func=weight_func)
        alpha_coeff, *_ = np.linalg.lstsq(A, alpha_vals, rcond=None)

    if prune_model:
        D_coeff, keep_D = extract_reduced_coeffs(A, D, threshold=thresh['D'], top_n=top['D'])
        re_coeff, keep_re = extract_reduced_coeffs(A, re, threshold=thresh['re'], top_n=top['re'])
        if alpha_fit:
            alpha_coeff, keep_alpha = extract_reduced_coeffs(A, alpha_vals, threshold=thresh['alpha'], top_n=top['alpha'])

    r_values, phi1_values, phi2_values = [np.sort(df[col].unique()) for col in ['r', 'phi1', 'phi2']]

    model, coeffs = create_morse_model(df, molecule, interaction,
                                       A, h_chi, h_psi, D_coeff, re_coeff, alpha_coeff)

    df_model = evaluate_model_on_reference_grid(model, df, coeffs)

    if alpha_fit:
        chi_model_rad = np.deg2rad(df_model['chi'].to_numpy())
        psi_model_rad = np.deg2rad(df_model['psi'].to_numpy())

        A_model, _ = create_matrix_lsqt_2d(
            h_chi,
            h_psi,
            chi_model_rad,
            psi_model_rad,
            symm_chi,
            molecule.screw_step
        )

        df_model['alpha'] = A_model @ alpha_coeff

    compute_energy_errors(df, df_model, print_errors, near_eq_delta_r)

    return df_model