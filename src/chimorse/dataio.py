"""
dataio.py
---------
Data loading and pre-processing utilities.
"""

import pandas as pd

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

    df = pd.read_csv(f'/home/hadis/paperSAM/chiral_morse/data/{molecule.path}/{molecule.name}/E_all_{interaction}.dat', sep='\t',
                     names=['phi1', 'phi2', 'z', 'r', 'e'])
    df['chi'] = (df['phi1'] - screw_dir * df['phi2']) % 360
    df['psi'] = (df['phi1'] + screw_dir * df['phi2']) % 360
    df['e'] -= molecule.re_energy * 2

    if zero_zeta:
        df = df[df['z'] == 0].drop(columns=['z']).copy()

    return df
