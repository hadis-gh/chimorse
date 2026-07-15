"""Tests for chimorse.models — 1D potentials and anisotropic Morse models."""

import numpy as np
import pandas as pd
import pytest

from chimorse.fourier import count_fourier_coeffs
from chimorse.models import (
    Morse_1D,
    LennardJones_1D,
    MorseAnisotropic,
    MorseAnisotropicAlpha,
    smooth_energy_profile,
    generate_model_df,
    evaluate_model_on_reference_grid,
)


def _const_coeffs(h_chi, h_psi, symm_chi, screw_step, value):
    """Coefficient vector whose Fourier expansion is the constant `value` everywhere."""
    n = count_fourier_coeffs(h_chi, h_psi, symm_chi, screw_step, after_symm=True)
    c = np.zeros(n)
    c[0] = value  # the constant ("1") term is the first column
    return c


def test_morse_minimum_at_re():
    D, re, alpha = 1.5, 9.0, 1.1
    assert Morse_1D(re, D, re, alpha) == pytest.approx(-D)
    assert Morse_1D(re - 0.5, D, re, alpha) > -D
    assert Morse_1D(re + 0.5, D, re, alpha) > -D


def test_lennard_jones_zero_at_sigma_and_min_depth():
    eps, sigma = 0.8, 3.4
    assert LennardJones_1D(sigma, eps, sigma) == pytest.approx(0.0, abs=1e-12)
    r_min = 2 ** (1 / 6) * sigma
    assert LennardJones_1D(r_min, eps, sigma) == pytest.approx(-eps, rel=1e-6)


def test_morse_anisotropic_constant_reduces_to_1d_morse():
    h_chi, h_psi, symm_chi, screw_step = 2, 1, 1, 20
    D, re, alpha = 1.2, 9.0, 1.1

    model = MorseAnisotropic(h_chi, h_psi, symm_chi, screw_step, alpha=alpha)
    D_coeff = _const_coeffs(h_chi, h_psi, symm_chi, screw_step, D)
    re_coeff = _const_coeffs(h_chi, h_psi, symm_chi, screw_step, re)

    r = np.array([8.0, 9.0, 10.0])
    chi = np.array([10.0, 80.0, 200.0])
    psi = np.array([30.0, 120.0, 300.0])

    out = model(r=r, phi1=None, phi2=None, chi=chi, psi=psi, params=(D_coeff, re_coeff))
    assert np.allclose(out, Morse_1D(r, D, re, alpha))


def test_morse_anisotropic_alpha_constant_reduces_to_1d_morse():
    h_chi, h_psi, symm_chi, screw_step = 1, 1, 0, 20
    D, re, alpha = 1.0, 9.0, 1.3

    model = MorseAnisotropicAlpha(h_chi, h_psi, symm_chi, screw_step)
    D_coeff = _const_coeffs(h_chi, h_psi, symm_chi, screw_step, D)
    re_coeff = _const_coeffs(h_chi, h_psi, symm_chi, screw_step, re)
    alpha_coeff = _const_coeffs(h_chi, h_psi, symm_chi, screw_step, alpha)

    r = np.array([8.5, 9.0, 9.5])
    chi = np.array([0.0, 90.0, 180.0])
    psi = np.array([45.0, 135.0, 225.0])

    out = model(r=r, phi1=None, phi2=None, chi=chi, psi=psi, params=(D_coeff, re_coeff, alpha_coeff))
    assert np.allclose(out, Morse_1D(r, D, re, alpha))


def test_generate_model_df_shape_and_columns():
    h_chi, h_psi, symm_chi, screw_step = 1, 1, 1, 20
    model = MorseAnisotropic(h_chi, h_psi, symm_chi, screw_step, alpha=1.1)
    D_coeff = _const_coeffs(h_chi, h_psi, symm_chi, screw_step, 1.0)
    re_coeff = _const_coeffs(h_chi, h_psi, symm_chi, screw_step, 9.0)

    r = np.array([8.0, 9.0, 10.0])
    phi1 = np.array([0.0, 20.0])
    phi2 = np.array([0.0, 40.0])

    df = generate_model_df(model, (D_coeff, re_coeff), r, phi1, phi2, "EP")

    assert len(df) == r.size * phi1.size * phi2.size
    assert {"phi1", "phi2", "r", "e", "chi", "psi"}.issubset(df.columns)
    assert np.all(np.isfinite(df["e"]))


def test_evaluate_on_reference_grid_matches_1d_morse():
    h_chi, h_psi, symm_chi, screw_step = 1, 1, 1, 20
    model = MorseAnisotropic(h_chi, h_psi, symm_chi, screw_step, alpha=1.1)
    D_coeff = _const_coeffs(h_chi, h_psi, symm_chi, screw_step, 1.0)
    re_coeff = _const_coeffs(h_chi, h_psi, symm_chi, screw_step, 9.0)

    df_ref = pd.DataFrame(
        {
            "phi1": [0, 0, 20],
            "phi2": [0, 0, 40],
            "r": [8.0, 9.0, 9.0],
            "chi": [0.0, 0.0, (20 - 40) % 360],
            "psi": [0.0, 0.0, (20 + 40) % 360],
        }
    )

    out = evaluate_model_on_reference_grid(model, df_ref, (D_coeff, re_coeff))

    assert len(out) == len(df_ref)
    assert "e" in out.columns
    assert np.allclose(out["e"].values, Morse_1D(df_ref["r"].values, 1.0, 9.0, 1.1))


def test_smooth_energy_profile_sorts_and_preserves_length():
    r = np.array([10.0, 8.0, 9.0, 11.0])
    e = np.array([0.0, -1.0, -0.8, 0.1])

    r_sorted, e_smooth = smooth_energy_profile(r, e, smooth_factor=0.0)

    assert np.all(np.diff(r_sorted) >= 0)
    assert len(r_sorted) == len(e_smooth) == r.size
