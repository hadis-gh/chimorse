"""Tests for chimorse.fourier — the chi-psi Fourier design matrix and RMSE helpers."""

import numpy as np
import pytest

from chimorse.fourier import (
    count_fourier_coeffs,
    create_matrix_lsqt_2d,
    create_matrix_lsqt_2d_sum,
    fit_fourier_rmse,
    compute_harmonic_rmse_grid,
)


@pytest.mark.parametrize(
    "h_chi,h_psi,symm_chi",
    [(1, 1, 1), (2, 1, 1), (3, 2, 0), (4, 3, 0), (2, 2, 1)],
)
def test_design_matrix_columns_match_coeff_count(h_chi, h_psi, symm_chi):
    """The number of design-matrix columns must equal count_fourier_coeffs(after_symm=True)."""
    rng = np.random.default_rng(0)
    n = 50
    chi = rng.uniform(0, 2 * np.pi, n)
    psi = rng.uniform(0, 2 * np.pi, n)
    screw_step = 20

    A, labels = create_matrix_lsqt_2d(h_chi, h_psi, chi, psi, symm_chi, screw_step)
    expected = count_fourier_coeffs(h_chi, h_psi, symm_chi, screw_step, after_symm=True)

    assert A.shape == (n, expected)
    assert len(labels) == expected


def test_constant_column_is_ones():
    chi = np.linspace(0, 2 * np.pi, 10)
    psi = np.linspace(0, 2 * np.pi, 10)
    A, labels = create_matrix_lsqt_2d(2, 1, chi, psi, symm_chi=1, screw_step=20)

    assert labels[0] == "1"
    assert np.allclose(A[:, 0], 1.0)


def test_create_matrix_shape_mismatch_raises():
    with pytest.raises(ValueError):
        create_matrix_lsqt_2d(1, 1, np.zeros(5), np.zeros(4), 1, 20)


def test_count_after_symm_formula():
    # chi-symmetric: 1 + h_chi + 2*h_psi + 2*h_chi*h_psi
    assert count_fourier_coeffs(3, 2, 1, 20, after_symm=True) == 1 + 3 + 2 * 2 + 2 * 3 * 2
    # non-symmetric: 1 + 2*h_chi + 2*h_psi + 4*h_chi*h_psi
    assert count_fourier_coeffs(3, 2, 0, 20, after_symm=True) == 1 + 2 * 3 + 2 * 2 + 4 * 3 * 2


def test_count_before_symm_uses_screw_step():
    # before symmetrization, h_psi is multiplied by k0 = round(360 / (2*screw_step))
    before = count_fourier_coeffs(2, 1, 1, 20, after_symm=False)
    after = count_fourier_coeffs(2, 1, 1, 20, after_symm=True)
    assert before > after


def test_fit_fourier_rmse_exact_recovery():
    """A target lying exactly in the basis span is fit with ~zero RMSE."""
    rng = np.random.default_rng(1)
    n = 200
    chi = rng.uniform(0, 2 * np.pi, n)
    psi = rng.uniform(0, 2 * np.pi, n)
    h_chi, h_psi, symm_chi, screw_step = 2, 1, 1, 20

    A, _ = create_matrix_lsqt_2d(h_chi, h_psi, chi, psi, symm_chi, screw_step)
    target = A @ rng.normal(size=A.shape[1])

    rmse = fit_fourier_rmse(chi, psi, target, h_chi, h_psi, symm_chi, screw_step)
    assert rmse < 1e-8


def test_harmonic_rmse_grid_shape_and_finite():
    rng = np.random.default_rng(2)
    n = 100
    chi = rng.uniform(0, 2 * np.pi, n)
    psi = rng.uniform(0, 2 * np.pi, n)
    target = rng.normal(size=n)

    grid = compute_harmonic_rmse_grid(
        chi, psi, target, h_chi_range=(1, 4), h_psi_max=3, symm_chi=1, screw_step=20
    )

    assert grid.shape == (4, 3)
    assert np.all(np.isfinite(grid))


def test_create_matrix_sum_shape():
    chi = np.linspace(0, 2 * np.pi, 30)
    psi = np.linspace(0, 2 * np.pi, 30)
    h_chi, h_psi = 2, 1

    A = create_matrix_lsqt_2d_sum(h_chi, h_psi, chi, psi, symm_chi=1, screw_step=20)

    # 1 (constant) + 2*h_psi (m=0 terms) + 2*h_chi*(2*h_psi+1) (m>=1 terms)
    expected = 1 + 2 * h_psi + 2 * h_chi * (2 * h_psi + 1)
    assert A.shape == (30, expected)
