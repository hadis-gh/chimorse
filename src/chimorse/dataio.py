"""
dataio.py
---------
Data loading and pre-processing utilities.
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np
import json

from .analysis import get_screw_dir
from .fourier import create_fourier_terms_2d

# ----------------------------------------------------------------------

def expand_data(df, r_far=20):
    """Append a far-distance point (r=r_far, e=0) for each angle combination, for asymptotic anchoring."""
    angle_cols = [c for c in df.columns if c not in ('r', 'e')]
    combos = df[angle_cols].drop_duplicates()

    block = combos.copy()
    block['r'] = r_far
    block['e'] = 0.0

    df_expanded = pd.concat([df, block], ignore_index=True)
    df_expanded.sort_values(['phi1', 'phi2', 'r'], inplace=True)
    df_expanded.reset_index(drop=True, inplace=True)

    print(f"Data expanded: {len(df)} → {len(df_expanded)} rows")
    return df_expanded

# ----------------------------------------------------------------------

def load_data(molecule, interaction, zero_zeta=True):
    """Load raw E(phi1, phi2, z, r) data, derive chi/psi from phi1/phi2, and shift energy by -2*re_energy."""
    screw_dir = get_screw_dir(interaction)

    df = pd.read_csv(f'/home/hadis/chimorse/data/{molecule.path}/{molecule.name}/E_all_{interaction}.dat', sep='\t',
                     names=['phi1', 'phi2', 'z', 'r', 'e'])
    df['chi'] = (df['phi1'] - screw_dir * df['phi2']) % 360
    df['psi'] = (df['phi1'] + screw_dir * df['phi2']) % 360
    df['e'] -= molecule.re_energy * 2

    if zero_zeta:
        df = df[df['z'] == 0].drop(columns=['z']).copy()

    return df

# ======================================================================
# Export configuration to C++ Molecular Dynamics Engine
# ======================================================================
def export_chimorse_json(
    filepath,
    h_chi,
    h_psi,
    symm_chi,
    screw_step,
    D_coeff,
    re_coeff,
    alpha_coeff=None,
    fixed_alpha=None,
    cutoff=12.0,
):
    terms = create_fourier_terms_2d(
        h_chi=h_chi,
        h_psi=h_psi,
        symm_chi=symm_chi,
        screw_step=screw_step,
    )

    D_coeff = np.asarray(D_coeff, dtype=float)
    re_coeff = np.asarray(re_coeff, dtype=float)

    if len(D_coeff) != len(terms):
        raise ValueError("D coefficient size does not match basis size")

    if len(re_coeff) != len(terms):
        raise ValueError("re coefficient size does not match basis size")

    if alpha_coeff is not None:
        alpha_coeff = np.asarray(alpha_coeff, dtype=float)

        if len(alpha_coeff) != len(terms):
            raise ValueError("alpha coefficient size does not match basis size")

        alpha_model = {
            "type": "fourier",
            "coefficients": alpha_coeff.tolist(),
        }

    elif fixed_alpha is not None:
        alpha_model = {
            "type": "constant",
            "value": float(fixed_alpha),
        }

    else:
        raise ValueError("Provide either alpha_coeff or fixed_alpha")

    model = {
        "format": "chimorse_anisotropic_morse",
        "version": 1,

        "angle_units": "radian",

        "angle_definition": {
            "chi": "phi2 - phi1",
            "psi": "phi1 + phi2",
        },

        "morse": "D * (exp(-2*alpha*(r-re)) - 2*exp(-alpha*(r-re)))",

        "cutoff": float(cutoff),

        "basis_terms": terms,

        "parameters": {
            "D": {
                "type": "fourier",
                "coefficients": D_coeff.tolist(),
            },
            "re": {
                "type": "fourier",
                "coefficients": re_coeff.tolist(),
            },
            "alpha": alpha_model,
        },
    }

    with open(filepath, "w") as f:
        json.dump(model, f, indent=2)
