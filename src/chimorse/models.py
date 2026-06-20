"""
models.py
---------
Potential energy surface models: 1D potentials (Morse, Lennard-Jones) and
anisotropic Morse models with chi-psi Fourier-expanded parameters.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from .fourier import create_matrix_lsqt_2d

# ----------------------------------------------------------------------

def Morse_1D(r, D, re, alpha):
    """1D Morse potential D*(exp(-2*alpha*(r-re)) - 2*exp(-alpha*(r-re)))."""
    return D * (np.exp(-2 * alpha * (r - re)) - 2 * np.exp(-alpha * (r - re)))

# ----------------------------------------------------------------------

def LennardJones_1D(r, epsilon, sigma):
    """1D Lennard-Jones potential 4*epsilon*((sigma/r)^12 - (sigma/r)^6)."""
    ratio = sigma / r
    return 4 * epsilon * (ratio**12 - ratio**6)

# ----------------------------------------------------------------------

class MorseAnisotropic:
    """Anisotropic Morse potential with fixed alpha; D(chi,psi) and r_e(chi,psi) given by Fourier expansions."""
    def __init__(self, h_chi, h_psi, symm_chi, screw_step, alpha=1.1, ):
        self.h_chi = h_chi
        self.h_psi = h_psi
        self.symm_chi = bool(symm_chi)
        self.alpha = alpha
        self.screw_step = screw_step

    def _base_morse(self, r, D, re):
        """1D Morse potential at the fixed alpha."""
        a = self.alpha
        return D * (np.exp(-2 * a * (r - re)) - 2 * np.exp(-a * (r - re)))

    def _fourier_eval_2d(self, coeff, chi_rad, psi_rad):
        """Evaluate the chi-psi Fourier expansion with the given coefficients."""
        A, labels = create_matrix_lsqt_2d(
            self.h_chi,
            self.h_psi,
            chi_rad,
            psi_rad,
            symm_chi=self.symm_chi,
            screw_step=self.screw_step,
        )
        return A @ coeff

    def __call__(self, r, phi1, phi2, chi, psi, params, **_):
        """Evaluate the anisotropic Morse energy from Fourier-expanded D(chi,psi) and r_e(chi,psi)."""
        D_coeff, re_coeff = params

        chi_rad = np.deg2rad(chi)
        psi_rad = np.deg2rad(psi)

        D = self._fourier_eval_2d(D_coeff, chi_rad, psi_rad)
        re = self._fourier_eval_2d(re_coeff, chi_rad, psi_rad)

        return self._base_morse(r, D, re)

# ----------------------------------------------------------------------

class MorseAnisotropicAlpha:
    """Anisotropic Morse potential with D(chi,psi), alpha(chi,psi), and r_e(chi,psi) 
       all given by Fourier expansions."""
    def __init__(self, h_chi, h_psi, symm_chi, screw_step):
        self.h_chi = h_chi
        self.h_psi = h_psi
        self.symm_chi = bool(symm_chi)
        self.screw_step = screw_step

    def _base_morse(self, r, D, alpha, re):
        """1D Morse potential."""
        return D * (np.exp(-2 * alpha * (r - re)) - 2 * np.exp(-alpha * (r - re)))

    def _fourier_eval_2d(self, coeff, chi_rad, psi_rad):
        """Evaluate the chi-psi Fourier expansion with the given coefficients."""
        A, labels = create_matrix_lsqt_2d(
            self.h_chi,
            self.h_psi,
            chi_rad,
            psi_rad,
            symm_chi=self.symm_chi,
            screw_step=self.screw_step,
        )
        return A @ coeff

    def __call__(self, r, phi1, phi2, chi, psi, params, **_):
        """Evaluate the anisotropic Morse energy from Fourier-expanded D(chi,psi), r_e(chi,psi), and alpha(chi,psi)."""
        D_coeff, re_coeff, alpha_coeff = params

        chi_rad = np.deg2rad(chi)
        psi_rad = np.deg2rad(psi)

        D = self._fourier_eval_2d(D_coeff, chi_rad, psi_rad)
        re = self._fourier_eval_2d(re_coeff, chi_rad, psi_rad)
        alpha = self._fourier_eval_2d(alpha_coeff, chi_rad, psi_rad)

        return self._base_morse(r, D, alpha, re)

# ----------------------------------------------------------------------

def smooth_energy_profile(r, e, smooth_factor=0.01):
    """Sort by r and return a spline-smoothed E(r) curve with the given smoothing factor."""
    idx = np.argsort(r)

    r_sorted = r[idx]
    e_sorted = e[idx]

    spline = UnivariateSpline(
        r_sorted,
        e_sorted,
        s=smooth_factor * len(r_sorted)
    )

    e_smooth = spline(r_sorted)

    return r_sorted, e_smooth

# ----------------------------------------------------------------------

def generate_model_df(potential_func, params, r_range, phi1_range, phi2_range, interaction_type):
    """Evaluate potential_func over a (r, phi1, phi2) meshgrid and 
       return the resulting model DataFrame with chi/psi/e columns."""
    screw_dir=1 if interaction_type[0]=='E' else -1
    r_grid, phi1_grid, phi2_grid = np.meshgrid(r_range, phi1_range, phi2_range, indexing='ij')

    r_flat = r_grid.flatten()
    phi1_flat = phi1_grid.flatten()
    phi2_flat = phi2_grid.flatten()
    chi_flat = (phi1_flat - screw_dir * phi2_flat) % 360
    psi_flat = (phi1_flat + screw_dir * phi2_flat) % 360

    e_flat = potential_func(
        r=r_flat, phi1=phi1_flat, phi2=phi2_flat, chi=chi_flat, psi=psi_flat, params=params
    )

    df_model = pd.DataFrame({
        'phi1': phi1_flat,
        'phi2': phi2_flat,
        'r': r_flat,
        'e': e_flat,
        'chi': chi_flat,
        'psi':psi_flat
    })

    print("Model DataFrame shape:", df_model.shape)
    return df_model

# ----------------------------------------------------------------------

def evaluate_model_on_reference_grid(model, df_ref, params):
    """Evaluate model at the same (phi1, phi2, r, chi, psi) points as df_ref and return the resulting energies."""
    df_model = df_ref[['phi1', 'phi2', 'r', 'chi', 'psi']].copy()

    df_model['e'] = model(
        r=df_model['r'].to_numpy(),
        phi1=df_model['phi1'].to_numpy(),
        phi2=df_model['phi2'].to_numpy(),
        chi=df_model['chi'].to_numpy(),
        psi=df_model['psi'].to_numpy(),
        params=params
    )

    return df_model
