"""Tests for chimorse.analysis — symmetry helpers, minima extraction, error metrics."""

import numpy as np
import pandas as pd
import pytest

from chimorse.analysis import (
    get_symm_chi,
    get_screw_dir,
    circular_distance_deg,
    infer_screw_direction,
    extract_energy_minimums,
    compute_energy_errors,
)


@pytest.mark.parametrize("interaction,expected", [("EP", 1), ("EA", 0), ("OP", 1), ("OA", 0)])
def test_get_symm_chi(interaction, expected):
    assert get_symm_chi(interaction) == expected


@pytest.mark.parametrize("interaction,expected", [("EP", 1), ("EA", 1), ("OP", -1), ("OA", -1)])
def test_get_screw_dir(interaction, expected):
    assert get_screw_dir(interaction) == expected


@pytest.mark.parametrize("a,b,expected", [(10, 350, 20), (0, 180, 180), (350, 10, 20), (0, 0, 0)])
def test_circular_distance_deg(a, b, expected):
    assert circular_distance_deg(a, b) == pytest.approx(expected)


def _make_screw_df(sign):
    rng = np.random.default_rng(0)
    phi1 = rng.uniform(0, 360, 150)
    phi2 = rng.uniform(0, 360, 150)
    df = pd.DataFrame({"phi1": phi1, "phi2": phi2})
    if sign == 1:
        df["chi"] = (df["phi1"] - df["phi2"]) % 360
    else:
        df["chi"] = (df["phi1"] + df["phi2"]) % 360
    return df


def test_infer_screw_direction_plus():
    assert infer_screw_direction(_make_screw_df(1)) == 1


def test_infer_screw_direction_minus():
    assert infer_screw_direction(_make_screw_df(-1)) == -1


def test_infer_screw_direction_raises_when_ambiguous():
    df = _make_screw_df(1)
    df["chi"] = (df["chi"] + 5) % 360  # matches neither convention
    with pytest.raises(ValueError):
        infer_screw_direction(df)


def test_extract_energy_minimums_picks_min_per_orientation():
    rows = []
    for (p1, p2) in [(0, 0), (20, 40)]:
        for r, e in [(8, -0.5), (9, -1.0), (10, -0.7), (11, -0.6)]:
            rows.append(
                dict(phi1=p1, phi2=p2, chi=(p1 - p2) % 360, psi=(p1 + p2) % 360, r=r, e=e)
            )
    df = pd.DataFrame(rows)

    out = extract_energy_minimums(df, r_max=12)

    assert len(out) == 2
    assert set(out["r"]) == {9}
    assert out["e"].max() == pytest.approx(-1.0)


def test_extract_energy_minimums_respects_r_max():
    df = pd.DataFrame(
        [
            dict(phi1=0, phi2=0, chi=0, psi=0, r=9, e=-1.0),
            dict(phi1=0, phi2=0, chi=0, psi=0, r=15, e=-0.5),
        ]
    )
    out = extract_energy_minimums(df, r_max=12)
    assert list(out["r"]) == [9]


def test_compute_energy_errors_zero_for_identical_inputs():
    rows = []
    for (p1, p2) in [(0, 0), (20, 40), (40, 80)]:
        for r in [8, 9, 10, 11]:
            e = 0.1 * (r - 9) ** 2 - 1.0  # interior minimum at r = 9
            rows.append(
                dict(phi1=p1, phi2=p2, r=r, e=e, chi=(p1 - p2) % 360, psi=(p1 + p2) % 360)
            )
    df = pd.DataFrame(rows)

    errors = compute_energy_errors(df, df.copy(), print_errors=False)

    assert errors["E_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert errors["D_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert errors["re_rmse"] == pytest.approx(0.0, abs=1e-12)
