# Configuration file for the Sphinx documentation builder.

project = "ChiMorse"
copyright = "2026, Hadis Ghodrati"
author = "Hadis Ghodrati"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

templates_path = []
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "furo"

html_static_path = ["_static"]
html_css_files = ["custom.css"]

autodoc_member_order = "bysource"