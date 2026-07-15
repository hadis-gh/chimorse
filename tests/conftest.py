"""Shared pytest configuration.

Importing ``chimorse`` pulls in the plotting module, which imports
matplotlib/seaborn. Force a non-interactive backend so the test suite runs
in headless environments (e.g. CI) without a display.
"""

import matplotlib

matplotlib.use("Agg")
