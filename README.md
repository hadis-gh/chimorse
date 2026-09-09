<p align="center">
  <img src="docs/images/logo_chimorse.png" alt="ChiMorse logo" width="45%"/>
</p>

<p align="center">
  <strong>From sampled anisotropic interaction data to compact analytical potentials.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-%E2%89%A53.9-blue.svg" alt="Python ≥3.9"/>
  <img src="https://img.shields.io/badge/license-GPLv3%2B-blue.svg" alt="GPL-3.0-or-later"/>
  <img src="https://github.com/hadis-gh/chimorse/actions/workflows/test.yml/badge.svg" alt="Tests"/>
  <img src="https://github.com/hadis-gh/chimorse/actions/workflows/docs.yml/badge.svg" alt="Documentation"/>
</p>

ChiMorse is a scientific Python toolkit for constructing compact analytical
representations of **sampled orientation-dependent pair interactions**. It
combines a radial Morse model with Fourier representations of the angular
dependence and provides tools for fitting, symmetry adaptation, convergence
analysis, coefficient pruning, validation, visualization, and export.

The current implementation uses one radial and two periodic orientational
coordinates. The accompanying α-polyalanine (αPA) system is a symmetry-rich
worked example rather than a restriction of the framework.

<p align="center">
  <img src="docs/images/workflow.png"
       alt="ChiMorse workflow from sampled interaction data to a reduced analytical model"
       width="1200"/>
</p>

<p align="center">
  <em>From sampled interaction data to a symmetry-adapted, validated, and
  optionally pruned analytical potential.</em>
</p>

## Contents

- [What ChiMorse does](#what-chimorse-does)
- [Demonstration at a glance](#demonstration-at-a-glance)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Example workflows](#example-workflows)
- [Using another interaction dataset](#using-chimorse-with-another-interaction-dataset)
- [Method in brief](#method-in-brief)
- [Software engineering and reproducibility](#software-engineering-and-reproducibility)
- [Documentation](#documentation)
- [Reference data and provenance](#reference-data-and-provenance)
- [Citation](#citation)

## What ChiMorse does

| Stage | Capability |
|---|---|
| **Input** | Load sampled radial interaction profiles over periodic orientations. |
| **Parameterization** | Extract local Morse parameters and represent their angular dependence with two-dimensional Fourier expansions. |
| **Symmetry** | Restrict the Fourier basis using known periodicity and interchange symmetries when available. |
| **Model reduction** | Select harmonic resolution from convergence and prune weak coefficients. |
| **Validation** | Compare reconstructed parameters and energies against the reference landscape. |
| **Output** | Evaluate the analytical potential and export fitted coefficients for external simulation codes. |

ChiMorse is intended for cases where a dense sampled interaction landscape
needs to be replaced by a compact, directly evaluable analytical model.

## Demonstration at a glance

The accompanying study demonstrates ChiMorse on four classes of rigid chiral
α-polyalanine helices (EP, EA, OP, and OA), corresponding to combinations of
relative handedness and axial alignment. Each class contains roughly
**300,000 sampled energy values**.

For the compact constant-α representation, symmetry adaptation reduces
the model to 54–246 Fourier coefficients before pruning and 49–122 afterward,
while the unpruned models reproduce near-equilibrium energies with RMSE values
of about **3.7–5.0 meV** within $2k_\mathrm{B}T$ at 300 K.

### Reference angular landscapes

The sampled $D(\chi,\psi)$ fields show the orientational structure that the
analytical model must reproduce; in this demonstration, the strongest
variation is along $\chi$, with a weaker screw-periodic modulation along
$\psi$.

<p align="center">
  <img src="docs/images/psi_chi_panel.png"
       alt="Reference well-depth landscapes D(chi, psi) and representative angular cuts for the four alpha-polyalanine interaction classes"
       width="800"/>
</p>

### Harmonic convergence

The Fourier resolution is selected from reconstruction-error convergence: the
chosen harmonic cutoffs lie near the onset of the RMSE plateau.

<p align="center">
  <img src="docs/images/harmonic_panel.png"
       alt="Reconstruction RMSE as a function of retained Fourier harmonic resolution for the four alpha-polyalanine interaction classes"
       width="800"/>
</p>

### Controlled coefficient pruning

After selecting the harmonic basis, weak Fourier coefficients can be removed
until further pruning begins to noticeably increase reconstruction error.

<p align="center">
  <img src="docs/images/pruning_panel.png"
       alt="Reconstruction RMSE versus number of retained Fourier coefficients during pruning for the four alpha-polyalanine interaction classes"
       width="800"/>
</p>

Together, these examples illustrate the central workflow: resolve the sampled
anisotropy, select sufficient harmonic complexity, and reduce the final model
without unnecessary loss of accuracy.

## Installation

ChiMorse is currently installed from source:

```bash
git clone https://github.com/hadis-gh/chimorse.git
cd chimorse
python -m pip install .
```

The package requires **Python 3.9 or newer**. The current continuous-integration
test matrix covers **Python 3.9–3.12**.

For development and testing:

```bash
python -m pip install -e ".[test]"
pytest -q
```

To run the notebooks:

```bash
python -m pip install -e ".[examples]"
```

## Quick start

The reference αPA dataset is archived separately on Zenodo. ChiMorse can
download it on demand and reuse the local copy on later runs.

```python
from pathlib import Path

from chimorse.config import load_molecule_info
from chimorse.dataio import load_data
from chimorse.datasets import ensure_reference_data
from chimorse.fitting import generate_fourier_morse_data

# Download the reference data once.
data_dir = ensure_reference_data("PA", data_root=Path("data"))

# Read dataset metadata.
molecule = load_molecule_info(
    "PA",
    metadata_path=data_dir / "metadata.json",
)

# Load one interaction class.
interaction = "EP"
df = load_data(molecule, interaction, zero_zeta=True)

# Harmonic cutoffs selected from convergence analysis.
harmonic_ceils = {
    "EP": (8, 1),
    "EA": (8, 1),
    "OP": (20, 1),
    "OA": (20, 1),
}

# Fit the compact Fourier–Morse representation and
# evaluate it on the reference grid.
df_model = generate_fourier_morse_data(
    df,
    molecule,
    interaction,
    harmonic_ceils,
    alpha_fit=False,
    print_errors=True,
)
```

`df_model` contains the reconstructed interaction on the reference grid and can
be passed to the plotting and error-analysis utilities.

## Example workflows

The notebooks form a progressive workflow from reference-data inspection to
model construction and export.

| Notebook | Purpose |
|---|---|
| [`01_raw_visualization.ipynb`](examples/01_raw_visualization.ipynb) | Inspect radial profiles, angular interaction landscapes, and representative cuts. |
| [`02_radial_fit_ER.ipynb`](examples/02_radial_fit_ER.ipynb) | Compare Morse and Lennard–Jones fits for selected radial profiles. |
| [`03_harmonic_convergence.ipynb`](examples/03_harmonic_convergence.ipynb) | Measure reconstruction error as Fourier harmonic resolution is increased. |
| [`04_fourier_morse.ipynb`](examples/04_fourier_morse.ipynb) | Fit and evaluate the symmetry-adapted Fourier–Morse model. |
| [`05_pruned_fourier_morse.ipynb`](examples/05_pruned_fourier_morse.ipynb) | Explore the accuracy–compactness trade-off through coefficient pruning. |
| [`06_export_model_md.ipynb`](examples/06_export_model_md.ipynb) | Export fitted coefficients for an external molecular-dynamics implementation. |
| [`07_weight_functions.ipynb`](examples/07_weight_functions.ipynb) | Compare weighting functions used in orientation-resolved $\alpha$ fitting. |
| [`08_fourier_morse_fit.ipynb`](examples/08_fourier_morse_fit.ipynb) | Run a configurable Fourier–Morse fitting workflow with selectable weighting and interpolation. |

The notebooks are committed without outputs to keep them lightweight and
reproducible. The figures above provide representative results for readers who
want a quick view of the workflow without executing the examples.

## Using ChiMorse with another interaction dataset

ChiMorse operates on the sampled interaction landscape rather than on the
method used to generate it. Compatible reference data may therefore originate
from electronic-structure calculations, classical models, other simulation
methods, or other sources of tabulated pair interactions.

The **current high-level workflow is not an arbitrary N-dimensional fitter**.
It assumes one radial coordinate and two periodic orientational coordinates.
The built-in αPA loader expects a four-column, tab-separated table without a
header:

```text
phi1    phi2    r    pair_energy
```

`load_data()` constructs the collective coordinates $\chi$ and $\psi$ and
converts the pair energy to the binding-energy column used by the fitting
workflow.

For another molecular or particle system, users must provide compatible
metadata and data and choose angular coordinates, harmonic resolution, and
symmetry restrictions appropriate to that system. The αPA screw and
interchange symmetries are a worked example, not a universal prescription.

## Method in brief

For each orientational configuration, ChiMorse represents the radial
interaction using a Morse potential whose parameters may depend on the angular
coordinates:

$$
V(r;\chi,\psi)=
D(\chi,\psi)
\left[
e^{-2\alpha(\chi,\psi)[r-r_e(\chi,\psi)]}-
2e^{-\alpha(\chi,\psi)[r-r_e(\chi,\psi)]}
\right].
$$

For the current helical formulation, the collective coordinates are

$$
\chi = \varphi_1-h\varphi_2,
\qquad
\psi = \varphi_1+h\varphi_2,
$$

with $h=+1$ for equal-handed pairs and $h=-1$ for opposite-handed pairs.

The parameter fields $D$, $r_e$, and optionally $\alpha$ are represented by
truncated two-dimensional Fourier expansions. When known symmetries are
available, incompatible Fourier terms are excluded before fitting. Model
complexity is then controlled through harmonic selection and coefficient
pruning.

For the full derivation, symmetry relations, and validation, see the
accompanying scientific manuscript.

## Software engineering and reproducibility

The repository follows a reusable scientific-software layout:

```text
src/chimorse/       installable Python package
tests/              automated unit/integration tests
examples/           reproducible Jupyter workflows
docs/               Sphinx user and API documentation
.github/workflows/  continuous integration and documentation deployment
```

Reference data are archived separately from the source code, dataset downloads
can be validated using checksum metadata when available, and fitted analytical
models can be exported as JSON for use in external simulation implementations.

## Documentation

The rendered documentation is available at
[hadis-gh.github.io/chimorse](https://hadis-gh.github.io/chimorse/).

It includes installation and getting-started guides, the software-oriented
method description, reference-data guidance, example workflows, and an API
reference generated from the package docstrings.

To build the documentation locally:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W -b html docs docs/_build/html
```

## Reference data and provenance

The αPA reference data are kept outside the Git repository and archived on
Zenodo for reproducible access.

- **System:** α-polyalanine (`PA`)
- **Archive:** Zenodo
- **Version-specific DOI:** [`10.5281/zenodo.21904448`](https://doi.org/10.5281/zenodo.21904448)
- **Local layout after download:** `data/PA/`

If you use these αPA interaction data in scientific work, please cite the study
in which the interaction model and dataset were originally developed:

> Hadis Ghodrati, Kevin Preis, Thi Ngoc Ha Nguyen, Christoph Tegenkamp,
> Sibylle Gemming, Jeffrey Kelling, and Florian Günther,  
> **“Simulation of Self-Assembled Monolayers of Polyalanine α-Helices:
> Development and Application of an Effective Potential for Film Structure
> Predictions.”**  
> *ACS Applied Materials & Interfaces* **18** (21), 30467–30479 (2026).  
> DOI: [`10.1021/acsami.6c01087`](https://doi.org/10.1021/acsami.6c01087)

The Zenodo record provides the archived data; the article above is the preferred
scientific citation for the αPA reference interaction.

## Citation

Please cite the resources relevant to how you use ChiMorse.

### Software

The repository contains [`CITATION.cff`](CITATION.cff), which GitHub can use to
export the software citation metadata.

> Hadis Ghodrati and Jeffrey Kelling, **ChiMorse Python package**.  
> DOI: [`10.5281/zenodo.22071766`](https://doi.org/10.5281/zenodo.22071766)

### Scientific manuscript

If your work uses the Fourier–Morse methodology described in the accompanying
study, please also cite:

> Hadis Ghodrati, Sibylle Gemming, Florian Günther, and Jeffrey Kelling,  
> **“A Symmetry-Constrained Fourier–Morse Framework for Compact Anisotropic Interaction Potentials.”**  
> *Manuscript/preprint details will be added when publicly available.*

### αPA reference interaction

If you use the αPA reference interaction data, please cite the originating
*ACS Applied Materials & Interfaces* article listed in
[Reference data and provenance](#reference-data-and-provenance). The Zenodo
record is provided as the reproducible data archive.

## Contributing

Bug reports, reproducibility issues, and focused contributions are welcome.
See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the current contribution
guidelines.

## License

ChiMorse is distributed under the
[GNU General Public License v3.0 or later](LICENSE).
