"""Tests for chimorse.fitting — pruning, alpha extraction, model assembly, and the
end-to-end generate_fourier_morse_data driver."""

import numpy as np
import pandas as pd
import pytest

from chimorse.config import MoleculeInfo
from chimorse.fourier import create_matrix_lsqt_2d
from chimorse.models import Morse_1D, MorseAnisotropic, MorseAnisotropicAlpha
from chimorse.fitting import (
    extract_reduced_coeffs,
    prune_by_magnitude,
    _parse_prune_arg,
    create_morse_model,
    fit_alpha_morse,
    generate_fourier_morse_data,
    equal_weights,
    gaussian_weights,
    poisson_weights,
    energy_weights,
    make_weight_func,
)
from chimorse.analysis import extract_energy_minimums


def _basis(n=200, h_chi=2, h_psi=1, symm_chi=1, screw_step=20, seed=0):
    rng = np.random.default_rng(seed)
    chi = rng.uniform(0, 2 * np.pi, n)
    psi = rng.uniform(0, 2 * np.pi, n)
    A, _ = create_matrix_lsqt_2d(h_chi, h_psi, chi, psi, symm_chi, screw_step)
    return A, rng


def test_parse_prune_arg():
    assert _parse_prune_arg(None) == {"D": None, "re": None, "alpha": None}
    assert _parse_prune_arg(5) == {"D": 5, "re": 5, "alpha": 5}
    assert _parse_prune_arg({"D": 1}) == {"D": 1, "re": None, "alpha": None}


def test_extract_reduced_coeffs_requires_exactly_one_selector():
    A, rng = _basis()
    target = rng.normal(size=A.shape[0])
    with pytest.raises(ValueError):
        extract_reduced_coeffs(A, target)  # neither threshold nor top_n
    with pytest.raises(ValueError):
        extract_reduced_coeffs(A, target, threshold=0.1, top_n=5)  # both


def test_extract_reduced_coeffs_top_n_count():
    A, rng = _basis()
    target = rng.normal(size=A.shape[0])

    coeff, keep = extract_reduced_coeffs(A, target, top_n=5, print_info=False)

    assert coeff.shape == (A.shape[1],)
    assert len(keep) <= 5
    assert np.count_nonzero(coeff) <= 5


def test_extract_reduced_coeffs_exact_when_target_is_sparse():
    A, _ = _basis()
    true = np.zeros(A.shape[1])
    true[[0, 3, 7]] = [2.0, -1.5, 0.7]
    target = A @ true

    coeff, keep = extract_reduced_coeffs(A, target, threshold=1e-6, print_info=False)

    assert np.allclose(A @ coeff, target, atol=1e-8)


def test_prune_by_magnitude_returns_matching_lengths():
    A, rng = _basis()
    target = rng.normal(size=A.shape[0])
    thresholds = np.logspace(-4, 0, 6)

    rmse_list, n_coeff_list = prune_by_magnitude(A, target, thresholds)

    assert len(rmse_list) == len(n_coeff_list) == len(thresholds)
    assert np.all(np.isfinite(rmse_list))


def test_create_morse_model_dispatch():
    A, _ = _basis()
    mol = MoleculeInfo("TEST", 20, 0.0, "", "", "")
    zeros = np.zeros(A.shape[1])

    model, coeffs = create_morse_model(None, mol, "EP", A, 2, 1, zeros, zeros)
    assert isinstance(model, MorseAnisotropic)
    assert len(coeffs) == 2

    model_a, coeffs_a = create_morse_model(
        None, mol, "EP", A, 2, 1, zeros, zeros, alpha_coeff=zeros
    )
    assert isinstance(model_a, MorseAnisotropicAlpha)
    assert len(coeffs_a) == 3


def test_fit_alpha_morse_uses_interpolated_minimum():
    D_true, re_true, alpha_true = 1.0, 9.0, 1.25
    r = np.arange(7.0, 11.01, 0.25)
    e = Morse_1D(r, D_true, re_true, alpha_true)
    df = pd.DataFrame({"phi1": 0, "phi2": 0, "r": r, "e": e})

    out = fit_alpha_morse(df, (0, 0), screw_dir=1)
    minimum = extract_energy_minimums(df).iloc[0]

    assert out["D"].iloc[0] == pytest.approx(-minimum["e"])
    assert out["re"].iloc[0] == pytest.approx(minimum["r"])
    assert np.isfinite(out["alpha"].iloc[0])


def test_fit_alpha_morse_weights_change_result():
    """Different weight functions must yield different fitted alpha values, proving the
    weight_func callable is actually applied during the fit."""
    D_true, re_true, alpha_true = 1.0, 9.0, 1.25
    r = np.arange(7.0, 11.01, 0.25)  # grid includes re = 9.0
    e = Morse_1D(r, D_true, re_true, alpha_true)
    df = pd.DataFrame({"phi1": 0, "phi2": 0, "r": r, "e": e})

    out_equal = fit_alpha_morse(df, (0, 0), screw_dir=1)
    gauss = lambda rr, re=None, e=None, e_min=None: gaussian_weights(rr, re, sigma=0.3)
    out_weighted = fit_alpha_morse(df, (0, 0), screw_dir=1, weight_func=gauss)

    assert out_weighted["alpha"].iloc[0] != pytest.approx(out_equal["alpha"].iloc[0], rel=1e-2)


def test_equal_weights_are_constant_one():
    r = np.linspace(6.8, 11.5, 50)
    assert np.allclose(equal_weights(r), 1.0)


def test_gaussian_weights_peak_at_re():
    re = 9.125
    w_re = gaussian_weights(np.array([re]), re, sigma=0.7)
    assert w_re.max() == pytest.approx(1.0, abs=1e-6)
    # On a discrete grid the maximum is at the point nearest r_e.
    r = np.linspace(6.8, 11.5, 200)
    w = gaussian_weights(r, re, sigma=0.7)
    assert w.max() == pytest.approx(1.0, rel=1e-2)
    assert abs(r[np.argmax(w)] - re) < 0.05
    with pytest.raises(ValueError):
        gaussian_weights(r, re, sigma=0.0)


def test_poisson_weights_mode_at_re():
    r = np.linspace(6.8, 11.5, 300)
    re = 9.125
    for lam in (3.0, re, 15.0):
        w = poisson_weights(r, re, lam=lam)
        assert r[np.argmax(w)] == pytest.approx(re, abs=1e-2)
        assert w.max() == pytest.approx(1.0, abs=1e-6)
        assert np.isfinite(w).all()
        assert w.min() >= 0.0


def test_energy_weights_peak_at_minimum_and_floor_eps():
    r = np.linspace(7.0, 11.0, 200)
    re = 9.0
    # well-shaped binding energy: deep at re, rising (repulsive) below, ~0 tail above
    e = -1.5 * np.exp(-1.1 * (r - re) ** 2) + 3.0 * np.exp(-2.5 * (r - re)) * (np.exp(-2.5 * (r - re)) - 2)
    e_min = e.min()

    w = energy_weights(r, e, re=re, e_min=e_min)
    assert np.isfinite(w).all()
    assert (w >= 0.0).all()
    # deepest (most negative) energy region gets the largest weight
    imin = np.argmin(e)
    assert w[imin] == pytest.approx(w.max(), rel=1e-3)
    # repulsive (high-energy) points are suppressed to the sqrt(eps) floor
    irep = np.argmax(e)
    assert w[irep] == pytest.approx(np.sqrt(1e-4), rel=1e-3)
    # a negative eps is rejected
    with pytest.raises(ValueError):
        energy_weights(r, e, re=re, e_min=e_min, eps=-1.0)


def test_make_weight_func_dispatch():
    r = np.linspace(6.8, 11.5, 50)
    assert np.allclose(make_weight_func("equal")(r), 1.0)
    assert make_weight_func("gaussian", sigma=0.7)(np.array([9.0]), 9.0) == pytest.approx(1.0)
    assert make_weight_func("poisson", lam=5.0)(np.array([9.0]), 9.0) == pytest.approx(1.0, abs=1e-3)
    # energy weights tie to the energy values and use the configured eps
    re = 9.0
    e = -1.5 * np.exp(-1.0 * (r - re) ** 2)
    w = make_weight_func("energy", eps=1e-6)(r, re, e=e, e_min=e.min())
    assert np.isfinite(w).all()
    with pytest.raises(ValueError):
        make_weight_func("bogus")


def test_fit_alpha_morse_default_matches_equal_weights():
    """The default weight_func gives the same result as an explicit constant-1 weight."""
    D_true, re_true, alpha_true = 1.0, 9.0, 1.25
    r = np.arange(7.0, 11.01, 0.25)
    e = Morse_1D(r, D_true, re_true, alpha_true)
    df = pd.DataFrame({"phi1": 0, "phi2": 0, "r": r, "e": e})

    out_default = fit_alpha_morse(df, (0, 0), screw_dir=1)
    out_const = fit_alpha_morse(df, (0, 0), screw_dir=1,
                                weight_func=equal_weights)

    assert out_default["alpha"].iloc[0] == pytest.approx(out_const["alpha"].iloc[0], rel=1e-9)


def test_generate_fourier_morse_recovers_constant_potential():
    """End-to-end: an orientation-independent Morse surface is recovered to float precision.

    The alpha used here matches the fixed alpha=1.1 baked into MorseAnisotropic via
    create_morse_model, so the reconstructed energies must match the input.
    """
    D_true, re_true, alpha = 1.2, 9.0, 1.1
    screw_dir = 1
    phis = np.arange(0, 360, 40)
    r = np.arange(7.0, 11.01, 0.5)  # includes re = 9.0

    rows = []
    for p1 in phis:
        for p2 in phis:
            chi = (p1 - screw_dir * p2) % 360
            psi = (p1 + screw_dir * p2) % 360
            for rr in r:
                rows.append(
                    dict(
                        phi1=p1, phi2=p2, r=rr,
                        e=Morse_1D(rr, D_true, re_true, alpha),
                        chi=chi, psi=psi,
                    )
                )
    df = pd.DataFrame(rows)

    mol = MoleculeInfo("TEST", 20, 0.0, "", "", "")
    harmonic_ceils = {"EP": (2, 1)}

    df_model = generate_fourier_morse_data(
        df, mol, "EP", harmonic_ceils,
        alpha_fit=False, print_errors=False,
    )

    assert len(df_model) == len(df)
    minimum = extract_energy_minimums(df).iloc[0]

    e_expected = Morse_1D(
        df["r"].values,
        -minimum["e"],
        minimum["r"],
        alpha,
    )

    assert np.allclose(
        df_model["e"].values,
        e_expected,
        atol=1e-6,
    )
