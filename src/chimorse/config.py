"""
config.py
---------
Package-level constants, colormaps, and data structures shared across all modules.
"""

import json
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Union
from matplotlib.colors import ListedColormap

# ----------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------

PLOT_PARAMS = {
    'font.family': 'STIXGeneral',
    'mathtext.fontset': 'stix',
    'mathtext.default': 'regular',

    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,

    'figure.dpi': 200,
    'savefig.dpi': 300,

    'image.cmap': 'magma',
    'text.usetex': False,
}

PHI_COLORMAP = 'RdBu'

_COLOR_PALETTE_SPECS = {
    "EP": ("Blues_r", 0.65),
    "EA": ("Greens_r", 0.60),
    "OP": ("Purples_r", 0.55),
    "OA": ("Oranges_r", 0.50),
}

# ----------------------------------------------------------------------

def get_colors():
    """Return desaturated colormaps for each interaction type (EP, EA, OP, OA)."""
    return {
        key: ListedColormap(sns.color_palette(palette, n_colors=256, desat=desat))
        for key, (palette, desat) in _COLOR_PALETTE_SPECS.items()
    }

# ----------------------------------------------------------------------

INTERACTION_CMAPS = {'EP': 'Blues', 'EA': 'Greens', 'OP': 'Purples', 'OA': 'Oranges'}

# ----------------------------------------------------------------------

@dataclass(frozen=True)
class MoleculeInfo:
    """Metadata needed to locate and interpret one molecule dataset."""

    name: str
    screw_step: float
    re_energy: float
    data_dir: Path
    interactions: frozenset[str]
    files: Mapping[str, str]

    def data_file(self, interaction: str) -> Path:
        """Return the raw data file for an interaction type."""
        interaction = interaction.upper()
        if interaction not in self.interactions:
            raise ValueError(
                f"Invalid interaction {interaction!r} for {self.name}. "
                f"Valid interactions: {sorted(self.interactions)}"
            )
        return self.data_dir / self.files[interaction]


def load_molecule_info(
    molecule_id: str,
    metadata_path: Optional[Union[Path, str]] = None,
) -> MoleculeInfo:
    """Load dataset metadata from ``data/<molecule_id>/metadata.json``."""
    if metadata_path is None:
        metadata_path = REPO_ROOT / "data" / molecule_id / "metadata.json"
    else:
        metadata_path = Path(metadata_path)

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Dataset metadata not found at: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    molecule = metadata["molecule"]
    files = metadata["files"]
    interactions = frozenset(metadata["interactions"])

    missing_files = interactions - files.keys()
    if missing_files:
        raise ValueError(
            "metadata.json is missing file names for: "
            f"{sorted(missing_files)}"
        )

    return MoleculeInfo(
        name=molecule["id"],
        screw_step=float(molecule["screw_step_deg"]),
        re_energy=float(molecule["isolated_helix_energy_ev"]),
        data_dir=metadata_path.parent,
        interactions=interactions,
        files=MappingProxyType(dict(files)),
    )

# ----------------------------------------------------------------------

class FigureContext:
    """Resolve and create the output directory/file path for a given molecule and interaction."""
    def __init__(self, base, molecule, data_type, interaction):
        self.base = base
        self.molecule = molecule
        self.data_type = data_type
        self.interaction = interaction

    def dir(self):
        """Return (creating if needed) the directory base/data_type/molecule/interaction."""
        parts = [self.base, self.data_type, self.molecule, self.interaction]

        path = Path(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def file(self, name, ext):
        """Return dir()/name.ext."""
        return self.dir() / f"{name}.{ext}"
