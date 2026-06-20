"""
config.py
---------
Package-level constants, colormaps, and data structures shared across all modules.
"""

import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from matplotlib.colors import ListedColormap

# ----------------------------------------------------------------------

PLOT_PARAMS = {
    'font.family': 'serif',      # paper
    # 'font.family': 'Sans-serif',  # slide
    'font.serif': 'DejaVu Serif',
    'font.size': 11,
    'axes.titlesize': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'image.cmap': 'magma',
    'text.usetex': False,
    'mathtext.default': 'regular',
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

@dataclass
class MoleculeInfo:
    name: str
    screw_step: float
    re_energy: float
    path: str = ''

MOLECULES = {
    'PA' : MoleculeInfo('PA', 20, -6227.1749, ''),
    'PP1': MoleculeInfo('PP1', 36, -4592.68, ''),
}

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
