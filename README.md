<p align="center">
  <img src="docs/logo_chimorse.png" alt="ChiMorse logo" width="300"/>
</p>

<h1 align="center">ChiMorse</h1>

<p align="center">
  Compact, symmetry-constrained Fourier–Morse models for anisotropic pair interactions.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-%E2%89%A53.9-blue.svg" alt="Python ≥3.9"/>
  <img src="https://img.shields.io/badge/license-GPLv3%2B-blue.svg" alt="GPL-3.0-or-later"/>
  <img src="https://github.com/hadis-gh/chimorse/actions/workflows/test.yml/badge.svg" alt="Tests"/>
</p>

ChiMorse is a Python package for converting sampled, orientation-dependent pair-interaction energies into compact analytical potentials. It combines a radial Morse potential with symmetry-adapted Fourier expansions of the orientational dependence and provides tools for fitting, convergence analysis, coefficient pruning, model evaluation, visualization, and export.

The framework is formulated for anisotropic pair interactions in a two-dimensional surface geometry. The reference workflow included with the repository uses **α-polyalanine (αPA) helices** as a symmetry-rich demonstration case and fits SCC-DFTB interaction-energy data.

<p align="center">
  <img src="docs/workflow.pdf" alt="ChiMorse workflow from tabulated interaction data to a reduced analytical model" width="900"/>
</p>

## Why ChiMorse?

Dense multidimensional interaction tables retain detailed anisotropic energetics, but they require storage and interpolation during simulation. Simple analytical pair potentials are compact and efficient, but do not by themselves capture orientation dependence.

ChiMorse provides an intermediate representation that is:

- **Analytical** — the radial interaction is represented by a Morse potential.
- **Systematically refinable** — angular resolution is controlled by Fourier harmonic order.
- **Symmetry aware** — known periodicity and exchange symmetries can be imposed directly on the Fourier basis.
- **Reducible** — harmonic truncation and coefficient pruning provide controlled model compression.
- **Interpretable** — individual Fourier modes correspond to characteristic angular periodicities.
- **Simulation oriented** — fitted coefficients can be exported for evaluation in external simulation codes.

## Method at a glance

For each orientational configuration, ChiMorse represents the radial interaction as

$$
V(r;\chi,\psi)
= D(\chi,\psi)
\left[
\exp\{-2\alpha(\chi,\psi)[r-r_e(\chi,\psi)]\}
-2\exp\{-\alpha(\chi,\psi)[r-r_e(\chi,\psi)]\}
\right].
$$

The angular coordinates are

$$
\chi = \varphi_1-h\varphi_2,
\qquad
\psi = \varphi_1+h\varphi_2,
$$

where $h=\pm1$ specifies the relative screw direction. The Morse parameters $D$, $r_e$, and optionally $\alpha$ are represented by truncated two-dimensional Fourier expansions in $(\chi,\psi)$.

For the αPA demonstration, four interaction classes are distinguished by relative handedness and axial direction:

| Class | Relative handedness | Relative direction | $\chi$ exchange symmetry |
|:---:|---|---|:---:|
| `EP` | equal | parallel | yes |
| `EA` | equal | antiparallel | no |
| `OP` | opposite | parallel | yes |
| `OA` | opposite | antiparallel | no |

For the complete formulation, symmetry relations, convergence analysis, and validation, see the accompanying scientific paper listed under [Citation](#citation).

## Installation

ChiMorse is currently installed from source.

```bash
git clone https://github.com/hadis-gh/chimorse.git
cd chimorse
python -m pip install -e .
```

Python **3.9 or newer** is required. The main dependencies are NumPy, pandas, SciPy, Matplotlib, seaborn, and lmfit.

For development and testing:

```bash
python -m pip install -e ".[test]"
pytest -q
```

## Quick start

The reference αPA dataset is archived separately on Zenodo. ChiMorse can download it on demand and reuse the local copy on subsequent runs.

```python
from pathlib import Path

from chimorse.config import load_molecule_info
from chimorse.dataio import load_data
from chimorse.datasets import ensure_reference_data
from chimorse.fitting import generate_fourier_morse_data

# Download once; subsequent calls reuse the local copy.
data_dir = ensure_reference_data("PA", data_root=Path("data"))

# Read molecule-specific metadata from the downloaded dataset.
molecule = load_molecule_info(
    "PA",
    metadata_path=data_dir / "metadata.json",
)

# Load one interaction class.
interaction = "EP"
df = load_data(molecule, interaction, zero_zeta=True)

# Choose the retained Fourier harmonic orders for each class.
harmonic_ceils = {
    "EP": (8, 1),
    "EA": (8, 1),
    "OP": (20, 1),
    "OA": (20, 1),
}

# Fit the Fourier–Morse representation and evaluate it on the reference grid.
df_model = generate_fourier_morse_data(
    df,
    molecule,
    interaction,
    harmonic_ceils,
    alpha_fit=False,
    print_errors=True,
)
```

The downloader and data parser are intentionally separate: `ensure_reference_data()` handles external data acquisition, while `load_data()` operates only on local files.

For visualization, model reduction, and export workflows, use the example notebooks below.

## Example workflows

The notebooks are designed to illustrate successive parts of the workflow, but each notebook performs the same reference-data availability check and can therefore be opened independently.

| Notebook | Purpose |
|---|---|
| [`01_raw_visualization.ipynb`](examples/01_raw_visualization.ipynb) | Inspect raw radial curves, energy landscapes, and angular line cuts. |
| [`02_radial_fit_ER.ipynb`](examples/02_radial_fit_ER.ipynb) | Compare radial Morse and Lennard-Jones fits for selected orientations. |
| [`03_harmonic_convergence.ipynb`](examples/03_harmonic_convergence.ipynb) | Examine reconstruction error as Fourier harmonic resolution is increased. |
| [`04_fourier_morse.ipynb`](examples/04_fourier_morse.ipynb) | Fit and evaluate the full symmetry-adapted Fourier–Morse model. |
| [`05_pruned_fourier_morse.ipynb`](examples/05_pruned_fourier_morse.ipynb) | Reduce the fitted representation through coefficient pruning and assess the accuracy–compactness trade-off. |
| [`06_export_model_md.ipynb`](examples/06_export_model_md.ipynb) | Export fitted interaction models for use by an external molecular-dynamics implementation. |

## Reference data

The scientific reference data are **not stored in this Git repository**. Keeping code and reference data separate avoids duplicating an archived scientific dataset and makes the exact dataset version explicit.

The currently registered reference dataset is:

- **System:** α-polyalanine (`PA`)
- **Archive:** Zenodo
- **Version-specific DOI:** `10.5281/zenodo.21904448`
- **Local layout after download:** `data/PA/`

The dataset contains a `metadata.json` file together with the tabulated interaction files used by the examples. `load_molecule_info()` reads the molecule metadata, and `load_data()` reads the interaction file selected in that metadata.

A raw interaction table contains five tab-separated columns without a header:

```text
phi1    phi2    zeta    r    pair_energy
```

ChiMorse derives $(\chi,\psi)$ from $(\varphi_1,\varphi_2)$ and converts the pair energy to a binding energy using the isolated-helix reference energy stored in the dataset metadata.

See [Citation](#citation) for the dataset citation placeholder that will be finalized before release.

## Package layout

```text
src/chimorse/
├── analysis.py   # symmetry operations, minima extraction, error analysis
├── config.py     # molecule metadata, plotting configuration, shared constants
├── dataio.py     # local data parsing and model export utilities
├── datasets.py   # reference-dataset acquisition and checksum validation
├── fitting.py    # radial/Fourier–Morse fitting and coefficient pruning
├── fourier.py    # symmetry-adapted Fourier bases and design matrices
├── models.py     # Morse/Lennard-Jones and anisotropic potential models
└── plotting.py   # visualization and diagnostic plotting
```

The repository additionally contains:

```text
examples/          Jupyter workflows
tests/             automated tests
docs/              figures and documentation assets
.github/workflows/ continuous-integration configuration
```

## Using ChiMorse with another dataset

The αPA dataset is the current worked example, but the fitting code is organized around dataset metadata rather than a hard-coded molecule registry. A compatible dataset needs:

1. a `metadata.json` describing the molecule, available interaction classes, and data filenames; and
2. one tabulated interaction file for each declared interaction class.

The symmetry assumptions and harmonic basis must still be chosen consistently with the physical system. The present paper and examples specifically document the symmetry constraints used for the chiral αPA demonstration.

## Documentation

The Sphinx documentation source is available in [`docs/`](docs/index.rst) and includes installation and getting-started guides, a software-oriented description of the Fourier–Morse method, reference-data and example-workflow guidance, and an API reference generated from the package docstrings.

To build the documentation locally:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W -b html docs docs/_build/html
```

The worked notebooks in `examples/` remain the most detailed end-to-end scientific demonstrations.

## Citation

ChiMorse separates citation of the **software**, the **scientific method/paper**, and the **reference dataset**. Please cite the items relevant to your use.

### Software

The repository contains a [`CITATION.cff`](CITATION.cff) file. On GitHub, the repository's **Cite this repository** menu can be used to export the software citation metadata.

A persistent software archive/DOI will be added when the first GitHub release is deposited on Zenodo.

### Scientific paper

If your work uses the Fourier–Morse methodology or results described in the accompanying study, please also cite:

> Hadis Ghodrati, Sibylle Gemming, Florian Günther, and Jeffrey Kelling,  
> **“A Symmetry-Constrained Fourier–Morse Framework for Compact Anisotropic Interaction Potentials in Surface Self-Assembly.”**  
> *Publication details and DOI to be added when available.*

### Reference dataset

If you use the αPA SCC-DFTB reference data, please cite the archived dataset separately:

> **α-polyalanine interaction-energy dataset.**  
> Zenodo, version-specific DOI: `10.5281/zenodo.21904448`.  
> *Full dataset creator/title citation to be inserted from the finalized Zenodo metadata before release.*

## Reproducibility

The repository is organized so that the software and archived scientific data remain independently identifiable:

- the source code is versioned in Git;
- the reference dataset is retrieved from a version-specific Zenodo record;
- downloaded files are checksum-validated when checksum metadata are provided by Zenodo;
- automated tests run through GitHub Actions on supported Python versions; and
- release citation metadata are stored in `CITATION.cff`.

For a published analysis, record both the ChiMorse release/version and the dataset DOI used to generate the results.

## Contributing

Bug reports, reproducibility issues, and focused contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the repository's current contribution guidance.

For changes to the scientific model, please describe the physical assumption being changed and include or update tests where practical.

## License

ChiMorse is released under the **GNU General Public License v3.0 or later (GPL-3.0-or-later)**. See [`LICENSE`](LICENSE) for the full license text.

The external reference dataset is a separate research object and is subject to the licensing and citation terms of its Zenodo record.

## Acknowledgements

Development of ChiMorse was supported by the German Research Foundation (DFG), TRR-386, TP A4 and B2, project number 514664767.
