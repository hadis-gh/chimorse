"""
dataio.py
---------
Data loading and pre-processing utilities.
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np

from .analysis import get_screw_dir

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
def export_potential_params_json(
    filepath: str | Path, 
    molecule_name: str,
    interaction: str,
    h_chi: int, 
    h_psi: int, 
    symm_chi: bool, 
    screw_step: float, 
    D_coeff: np.ndarray, 
    re_coeff: np.ndarray, 
    alpha_coeff: np.ndarray = None, 
    fixed_alpha: float = None,
    basis_labels: list = None
):
    """
    Exports the Fourier-Morse coefficients and structural metadata to a JSON 
    configuration file for C++ integration.
    """
    data = {
        "metadata": {
            "model_name": "chimorse",
            "molecule": molecule_name,
            "interaction": interaction,
            "fourier_setup": {
                "h_chi": int(h_chi),
                "h_psi": int(h_psi),
                "symm_chi": bool(symm_chi),
                "screw_step": float(screw_step)
            }
        },
        "coefficients": {
            # .tolist() is required to serialize numpy arrays to JSON
            "D": D_coeff.tolist(),
            "re": re_coeff.tolist()
        }
    }

    if alpha_coeff is not None:
        data["coefficients"]["alpha"] = alpha_coeff.tolist()
        data["metadata"]["alpha_fit"] = True
    elif fixed_alpha is not None:
        data["metadata"]["fixed_alpha"] = float(fixed_alpha)
        data["metadata"]["alpha_fit"] = False

    if basis_labels is not None:
        data["metadata"]["basis_labels"] = basis_labels

    path_obj = Path(filepath)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path_obj, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Successfully exported potential parameters to: {path_obj.resolve()}")