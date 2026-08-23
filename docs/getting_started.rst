Getting started
===============

This example takes one reference interaction class from data acquisition to a
fitted Fourier--Morse model. It intentionally keeps data acquisition separate
from data parsing: downloading is explicit, while ``load_data`` only reads
local files.

1. Acquire the reference dataset
--------------------------------

.. code-block:: python

   from pathlib import Path

   from chimorse.datasets import ensure_reference_data

   data_dir = ensure_reference_data("PA", data_root=Path("data"))

The first call downloads and validates the registered Zenodo files. Later calls
reuse the local copy when the dataset is complete.

2. Load molecule metadata and an interaction class
---------------------------------------------------

.. code-block:: python

   from chimorse.config import load_molecule_info
   from chimorse.dataio import load_data

   molecule = load_molecule_info(
       "PA",
       metadata_path=data_dir / "metadata.json",
   )

   interaction = "EP"
   df = load_data(molecule, interaction, zero_zeta=True)

The raw interaction table contains sampled rotation angles, axial offset,
intermolecular separation, and pair energy. ``load_data`` derives the
collective angular coordinates ``chi`` and ``psi`` and converts the pair energy
to the binding-energy column ``e`` using the isolated-helix reference energy
stored in the dataset metadata.

3. Choose the retained harmonics
--------------------------------

The demonstration uses class-specific harmonic ceilings selected from the
convergence analysis:

.. code-block:: python

   harmonic_ceils = {
       "EP": (8, 1),
       "EA": (8, 1),
       "OP": (20, 1),
       "OA": (20, 1),
   }

The two entries are the retained harmonic resolution in ``chi`` and the
symmetry-allowed ``psi`` family used by ChiMorse. See :doc:`method` and the
harmonic-convergence notebook before changing these values for a new system.

4. Fit and evaluate the model
-----------------------------

.. code-block:: python

   from chimorse.fitting import generate_fourier_morse_data

   df_model = generate_fourier_morse_data(
       df,
       molecule,
       interaction,
       harmonic_ceils,
       alpha_fit=False,
       print_errors=True,
   )

``df_model`` contains the model evaluated on the reference grid and can be used
with the plotting and error-analysis utilities. The worked notebooks show how
to inspect radial fits, choose harmonic orders, prune coefficients, compare
model/reference energies, and export coefficients for an external simulation
implementation.

Where to go next
----------------

* :doc:`method` explains the radial Morse representation, angular coordinates,
  symmetry-adapted Fourier basis, and reduction workflow.
* :doc:`data_examples` maps each notebook to a stage of the scientific
  workflow.
* :doc:`api/index` provides function- and class-level reference generated from
  the package docstrings.
