"""Basic integrity checks for the example notebooks."""

import json
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples"
NOTEBOOKS = sorted(EXAMPLE_DIR.glob("*.ipynb"))


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_code_cells_are_valid_python(notebook_path):
    """Ensure every Python code cell is syntactically valid."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    for cell_index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        compile(source, f"{notebook_path.name}:cell-{cell_index}", "exec")
