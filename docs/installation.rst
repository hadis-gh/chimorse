Installation
============

Requirements
------------

ChiMorse requires Python 3.9 or newer. Its runtime dependencies are declared in
``pyproject.toml`` and are installed automatically by ``pip``.

Install from a local clone
--------------------------

For development or for reproducing the example workflows, clone the repository
and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/hadis-gh/chimorse.git
   cd chimorse
   python -m pip install -e .

An editable installation means changes under ``src/chimorse/`` are immediately
visible to the installed package without reinstalling it.

Development installation
------------------------

To run the automated tests:

.. code-block:: bash

   python -m pip install -e ".[test]"
   pytest -q

To build this documentation locally:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   python -m sphinx -W -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` in a browser after the build completes.
The ``-W`` option makes documentation warnings fail the build, matching the
continuous-integration check.

Reference data
--------------

The alphaPA reference dataset is not bundled with the Python distribution.
ChiMorse downloads the registered version-specific Zenodo record on demand; see
:doc:`data_examples` and :func:`chimorse.datasets.ensure_reference_data`.
