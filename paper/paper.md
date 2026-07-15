---
title: 'chimorse: Symmetry-constrained Fourier–Morse potentials for chiral helix–helix interactions'
tags:
  - Python
  - computational physics
  - molecular modeling
  - chirality
  - Morse potential
  - density-functional tight binding
authors:
  - name: Hadis Ghodrati
    orcid: 0009-0004-1451-8086
    corresponding: true
    affiliation: 1
  - name: Jeffrey Kelling
    orcid: 0000-0003-1761-2591
    affiliation: 1, 2
affiliations:
  - name: Institute of Physics, Technische Universität Chemnitz, 09107 Chemnitz, Germany
    index: 1
  - name: Institute of Radiation Physics, Helmholtz-Zentrum Dresden - Rossendorf, 01328 Dresden, Germany
    index: 2
date: 2026-07-15
bibliography: paper.bib
---

# Summary

`chimorse` is a Python package that converts tabulated quantum-chemical interaction
energies for pairs of chiral helical molecules into compact, smooth, analytic
potential-energy models. The distance dependence of the interaction is described by a
Morse potential [@Morse1929], while its well depth $D$, equilibrium distance $r_e$, and
(optionally) width $\alpha$ are allowed to vary with the relative orientation of the two
helices:

$$
V(r;\chi,\psi) = D(\chi,\psi)\left[e^{-2\alpha(r-r_e)} - 2\,e^{-\alpha(r-r_e)}\right].
$$

The orientation dependence is expressed as a Fourier expansion in two angular
coordinates, $\chi$ and $\psi$, chosen to reflect the chiral geometry of the pair, and
the exchange and screw symmetry of each interaction class restricts which Fourier terms
may appear—keeping the resulting models small and physically interpretable. The package
loads reference data, performs the least-squares fit, optionally sparsifies the model by
pruning small coefficients, evaluates the fitted potential on arbitrary grids, quantifies
errors against the reference, and provides a comprehensive set of plotting routines for
energy curves, energy surfaces, harmonic convergence, and model diagnostics.

# Statement of need

Atomistic simulations of self-assembling chiral systems, such as helical polypeptides,
require interaction potentials that are inexpensive to evaluate yet faithful to the
underlying electronic-structure energetics. First-principles methods such as
self-consistent-charge density-functional tight binding (SCC-DFTB) [@Elstner1998] provide
accurate pairwise energies, but only on discrete grids of geometries and at a cost that is
prohibitive for large-scale or long-time simulation. Conventional isotropic pair
potentials (for example, Morse or Lennard-Jones) reduce that cost but cannot represent the
strong dependence of helix–helix interactions on relative orientation and handedness.

Bridging this gap requires fitting an orientation-resolved analytic potential to reference
data while respecting the exchange and screw symmetries that distinguish chiral pairs—a
procedure that is easy to get subtly wrong and tedious to reproduce by hand. `chimorse`
provides a reusable, documented implementation of this workflow. It targets researchers in
computational chemistry, soft-matter physics, and molecular modeling who need transferable,
symmetry-consistent coarse-grained potentials derived from quantum-chemical data, and who
value a reproducible path from raw energy tables to a publishable model.

# State of the field

General-purpose curve-fitting and optimization libraries such as SciPy [@Virtanen2020] and
`lmfit` [@Newville2014] supply the numerical machinery for least-squares fitting but offer
no domain model for orientation-dependent chiral interactions. Force-field development
toolkits focus on standard bonded and non-bonded functional forms and do not provide a
symmetry-adapted angular basis for the relative configuration of two helices. To our
knowledge, no openly available package combines a Morse radial form with a
screw-symmetry-constrained Fourier expansion of its parameters tailored to chiral helix
pairs. `chimorse` fills this niche while building on the established scientific Python
stack: NumPy [@Harris2020], pandas [@McKinney2010], and Matplotlib [@Hunter2007].

# Software design

`chimorse` is organized as a small, layered package with a clear dependency order from
numerical primitives up to figures:

- **`config`** — molecule metadata, interaction definitions, colormaps, and output-path
  management.
- **`dataio`** — loading of the tabulated reference energies and derivation of the chiral
  coordinates $\chi = \varphi_1 - s\,\varphi_2$ and $\psi = \varphi_1 + s\,\varphi_2$,
  where $s = \pm 1$ is the screw direction.
- **`fourier`** — construction of the symmetry-adapted $\chi$–$\psi$ Fourier design matrix
  and counting of independent coefficients before and after symmetrization.
- **`analysis`** — screw-symmetry operations, extraction of per-orientation energy minima,
  screw-periodicity expansion, and error metrics.
- **`models`** — the 1D Morse and Lennard-Jones potentials and the anisotropic Morse model
  classes, in which $D$, $r_e$, and optionally $\alpha$ are Fourier-expanded.
- **`fitting`** — 1D radial fits, per-orientation extraction of $\alpha$, magnitude-based
  coefficient pruning, and the end-to-end driver `generate_fourier_morse_data`.
- **`plotting`** — energy curves, 2D and 3D energy surfaces, harmonic-convergence panels,
  correlation plots, and pruning diagnostics.

The package distinguishes four interaction classes by relative handedness (equal or
opposite) and relative direction (parallel or antiparallel): `EP`, `EA`, `OP`, and `OA`.
The symmetry of each class determines whether the expansion in $\chi$ is even (cosine-only)
or full, which is enforced directly in the basis construction. Models can be fitted at full
harmonic order or pruned to a sparse coefficient set, allowing the user to trade accuracy
against compactness in a controlled way. Five example Jupyter notebooks demonstrate the
complete workflow, from raw-data inspection and harmonic-convergence analysis to fitting the
full and pruned Fourier–Morse models and comparing them against the reference data.

# Research impact

`chimorse` was developed to model helix–helix interactions in $\alpha$-polyalanine from
SCC-DFTB reference data and underpins the modeling in an associated manuscript
[@Ghodrati2026]. By producing compact, symmetry-consistent analytic potentials from
quantum-chemical energy tables, it enables orientation-resolved interactions to be used in
larger-scale simulations of chiral molecular assemblies, and provides a reproducible
template that can be applied to other helical systems by adding the relevant molecule
metadata and reference data.

# Acknowledgements

We thank Prof. Florian Günther for providing the SCC-DFTB reference data used in the
development and validation of this software. The authors acknowledge funding by the German
Research Foundation (DFG), TRR-386, TP A4 and B2, project number 514664767.

# References