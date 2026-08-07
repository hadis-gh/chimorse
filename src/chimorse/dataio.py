"""
dataio.py
---------
Data loading and pre-processing utilities.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import get_screw_dir
from .fourier import create_fourier_terms_2d
from .config import MoleculeInfo

# ----------------------------------------------------------------------

def expand_data(df, r_far=20):
    """Append one asymptotic point (r=r_far, e=0) for each angular configuration."""
    config_cols = [c for c in ("phi1", "phi2", "zeta", "chi", "psi") if c in df.columns]
    if not {"phi1", "phi2"}.issubset(config_cols):
        raise ValueError("expand_data requires phi1 and phi2 columns")

    block = df[config_cols].drop_duplicates().copy()
    block["r"] = r_far
    block["e"] = 0.0

    # pair_energy is undefined for synthetic asymptotic points.
    if "pair_energy" in df.columns:
        block["pair_energy"] = np.nan

    df_expanded = pd.concat([df, block], ignore_index=True, sort=False)
    sort_cols = [c for c in ("phi1", "phi2", "zeta", "r") if c in df_expanded.columns]
    df_expanded.sort_values(sort_cols, inplace=True)
    df_expanded.reset_index(drop=True, inplace=True)

    print(f"Data expanded: {len(df)} → {len(df_expanded)} rows")
    return df_expanded

# ----------------------------------------------------------------------

def load_data(molecule: MoleculeInfo, 
              interaction: str, 
              zero_zeta: bool = True,
) -> pd.DataFrame:
    """Load pair energies and calculate helix-pair binding energies."""

    interaction = interaction.upper()

    if interaction not in molecule.interactions:
        raise ValueError(
            f"Invalid interaction {interaction!r} for {molecule.name}. "
            f"Valid interactions: {sorted(molecule.interactions)}"
        )

    screw_dir = get_screw_dir(interaction)

    data_path = molecule.data_file(interaction)

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Reference data not found at: {data_path}"
        )

    df = pd.read_csv(
        data_path,
        sep="\t",
        names=["phi1", "phi2", "zeta", "r", "pair_energy"],
    )

    df["chi"] = (df["phi1"] - screw_dir * df["phi2"]) % 360

    df["psi"] = (df["phi1"] + screw_dir * df["phi2"]) % 360

    # e: binding energy
    df["e"] = (
        df["pair_energy"] - 2.0 * molecule.re_energy
    )

    if zero_zeta:
        df = df[df['zeta'] == 0].drop(columns=['zeta']).copy()

    return df

# ======================================================================
# Export configuration to C++ Molecular Dynamics Engine
# ======================================================================

VALID_INTERACTIONS = {"EP", "EA", "OP", "OA"}


def export_chimorse_single_json(
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
    """
    Export one fitted chiMorse interaction model.

    The resulting file contains only interaction-specific information.
    Shared information, such as the angle convention and Morse equation,
    is added later by export_chimorse_combined_json().
    """

    if alpha_coeff is not None and fixed_alpha is not None:
        raise ValueError(
            "Provide alpha_coeff or fixed_alpha, not both."
        )

    if alpha_coeff is None and fixed_alpha is None:
        raise ValueError(
            "Provide either alpha_coeff or fixed_alpha."
        )

    terms = create_fourier_terms_2d(
        h_chi=h_chi,
        h_psi=h_psi,
        symm_chi=symm_chi,
        screw_step=screw_step,
    )

    D_coeff = np.asarray(D_coeff, dtype=float).reshape(-1)
    re_coeff = np.asarray(re_coeff, dtype=float).reshape(-1)

    number_of_terms = len(terms)

    if len(D_coeff) != number_of_terms:
        raise ValueError(
            "D coefficient size does not match basis size: "
            f"{len(D_coeff)} coefficients for {number_of_terms} terms."
        )

    if len(re_coeff) != number_of_terms:
        raise ValueError(
            "re coefficient size does not match basis size: "
            f"{len(re_coeff)} coefficients for {number_of_terms} terms."
        )

    if alpha_coeff is not None:
        alpha_coeff = np.asarray(
            alpha_coeff,
            dtype=float,
        ).reshape(-1)

        if len(alpha_coeff) != number_of_terms:
            raise ValueError(
                "alpha coefficient size does not match basis size: "
                f"{len(alpha_coeff)} coefficients for "
                f"{number_of_terms} terms."
            )

        alpha_model = {
            "type": "fourier",
            "coefficients": alpha_coeff.tolist(),
        }

    else:
        alpha_model = {
            "type": "constant",
            "value": float(fixed_alpha),
        }

    model = {
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

    filepath = Path(filepath)

    if filepath.suffix.lower() != ".json":
        filepath = filepath.with_suffix(".json")

    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("w", encoding="utf-8") as file:
        json.dump(model, file, indent=2)

    print(f"chiMorse interaction model written to: {filepath}")


# ----------------------------------------------------------------------


def export_chimorse_combined_json(
    target_file,
    source_directory,
    interactions=("EP", "EA", "OP", "OA"),
):
    """
    Combine individual EP, EA, OP, and OA interaction files into
    one chiMorse model file.

    Interaction labels:
        EP = equal handedness, parallel alignment
        EA = equal handedness, antiparallel alignment
        OP = opposite handedness, parallel alignment
        OA = opposite handedness, antiparallel alignment
    """

    source_directory = Path(source_directory)
    target_file = Path(target_file)

    if target_file.suffix.lower() != ".json":
        target_file = target_file.with_suffix(".json")

    normalized_interactions = [
        str(interaction).upper()
        for interaction in interactions
    ]

    if len(normalized_interactions) != len(
        set(normalized_interactions)
    ):
        raise ValueError(
            "The interaction list contains duplicate categories."
        )

    unknown_interactions = (
        set(normalized_interactions) - VALID_INTERACTIONS
    )

    if unknown_interactions:
        unknown_text = ", ".join(sorted(unknown_interactions))

        raise ValueError(
            f"Unknown interaction categories: {unknown_text}"
        )

    combined_model = {
        "format": "chimorse_multi_interaction",
        "version": 1,

        "angle_units": "radian",

        "angle_definition": {
            "chi": "phi1 - h * phi2",
            "psi": "phi1 + h * phi2",
        },

        "handedness_factor": {
            "equal": 1,
            "opposite": -1,
        },

        "morse": (
            "D * (exp(-2*alpha*(r-re)) "
            "- 2*exp(-alpha*(r-re)))"
        ),

        "interactions": {},
    }

    required_keys = {
        "cutoff",
        "basis_terms",
        "parameters",
    }

    for interaction in normalized_interactions:
        source_file = source_directory / f"{interaction}.json"

        if not source_file.is_file():
            raise FileNotFoundError(
                f"Missing interaction file: {source_file}"
            )

        with source_file.open("r", encoding="utf-8") as file:
            interaction_model = json.load(file)

        missing_keys = required_keys - interaction_model.keys()

        if missing_keys:
            missing_text = ", ".join(sorted(missing_keys))

            raise ValueError(
                f"{source_file} is missing required fields: "
                f"{missing_text}"
            )

        combined_model["interactions"][interaction] = {
            "cutoff": interaction_model["cutoff"],
            "basis_terms": interaction_model["basis_terms"],
            "parameters": interaction_model["parameters"],
        }

    target_file.parent.mkdir(parents=True, exist_ok=True)

    with target_file.open("w", encoding="utf-8") as file:
        json.dump(combined_model, file, indent=2)

    print(f"Combined chiMorse model written to: {target_file}")