Data and example workflows
==========================

Reference dataset
-----------------

The scientific reference data are archived separately from the source code.
The registered demonstration dataset is the alpha-polyalanine (``PA``) record
with version-specific Zenodo DOI ``10.5281/zenodo.21904448``.

ChiMorse deliberately keeps three operations distinct:

#. :func:`chimorse.datasets.ensure_reference_data` acquires and validates the
   external dataset.
#. :func:`chimorse.config.load_molecule_info` reads dataset metadata.
#. :func:`chimorse.dataio.load_data` parses one local interaction table and
   derives the ChiMorse coordinates and binding energy.

This makes network access explicit and keeps the scientific parser usable with
local or independently archived datasets.

Expected dataset layout
-----------------------

After download, the reference data are organized as

.. code-block:: text

   data/
   └── PA/
       ├── metadata.json
       ├── E_all_EP.dat
       ├── E_all_EA.dat
       ├── E_all_OP.dat
       └── E_all_OA.dat

The exact data filenames are read from ``metadata.json`` rather than hard-coded
by ``load_data``. A raw interaction table has five tab-separated columns and no
header:

.. code-block:: text

   phi1    phi2    zeta    r    pair_energy

Using another dataset
---------------------

A compatible dataset needs a ``metadata.json`` describing the molecule,
interaction classes, and filenames, plus one interaction table for every class
listed in that metadata. The fitting code is not tied to the electronic-
structure method used to generate the reference energies.

The symmetry model *is* a physical assumption. A new molecular or particle
system therefore requires a justified mapping from its invariances to the
Fourier basis; the alphaPA screw and interchange restrictions are not a generic
default.

Example notebooks
-----------------

The notebooks form a progressive scientific workflow, but each performs the
same data-availability check and can be opened independently.

``01_raw_visualization.ipynb``
   Inspect raw radial profiles, angular energy landscapes, and representative
   cuts.
``02_radial_fit_ER.ipynb``
   Compare one-dimensional Morse and Lennard--Jones radial fits.
``03_harmonic_convergence.ipynb``
   Measure reconstruction error as the retained Fourier harmonic resolution is
   increased.
``04_fourier_morse.ipynb``
   Fit and assess the full symmetry-adapted Fourier--Morse representation.
``05_pruned_fourier_morse.ipynb``
   Reduce the representation by coefficient pruning and inspect the
   accuracy--compactness trade-off.
``06_export_model_md.ipynb``
   Export fitted model coefficients for an external molecular-dynamics
   implementation.

The notebooks live in the repository's ``examples/`` directory. For
reproducibility, use the version-specific reference dataset cited in
:doc:`citing` and record the ChiMorse release/tag used for your analysis.
