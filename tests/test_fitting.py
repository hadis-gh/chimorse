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
)


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
    mol = MoleculeInfo("TEST", 20, 0.0)
    zeros = np.zeros(A.shape[1])

    model, coeffs = create_morse_model(None, mol, "EP", A, 2, 1, zeros, zeros)
    assert isinstance(model, MorseAnisotropic)
    assert len(coeffs) == 2

    model_a, coeffs_a = create_morse_model(
        None, mol, "EP", A, 2, 1, zeros, zeros, alpha_coeff=zeros
    )
    assert isinstance(model_a, MorseAnisotropicAlpha)
    assert len(coeffs_a) == 3


def test_fit_alpha_morse_recovers_known_alpha():
    D_true, re_true, alpha_true = 1.0, 9.0, 1.25
    r = np.arange(7.0, 11.01, 0.25)  # grid includes re = 9.0
    e = Morse_1D(r, D_true, re_true, alpha_true)
    df = pd.DataFrame({"phi1": 0, "phi2": 0, "r": r, "e": e})

    out = fit_alpha_morse(df, (0, 0), screw_dir=1)

    assert out["alpha"].iloc[0] == pytest.approx(alpha_true, rel=1e-3)
    assert out["D"].iloc[0] == pytest.approx(D_true, rel=1e-6)
    assert out["re"].iloc[0] == pytest.approx(re_true)


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

    mol = MoleculeInfo("TEST", 20, 0.0)
    harmonic_ceils = {"EP": (2, 1)}

    df_model = generate_fourier_morse_data(
        df, mol, "EP", harmonic_ceils,
        alpha_fit=False, print_errors=False,
    )

    assert len(df_model) == len(df)
    assert np.allclose(df_model["e"].values, df["e"].values, atol=1e-6)
