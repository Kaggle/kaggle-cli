# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))


def autodoc_skip_member(app, what, name, obj, skip, options):
    # 'what' is the type of the object (e.g., 'function', 'class')
    # 'name' is the name of the object (e.g., 'myfunction')
    # 'obj' is the Python object itself
    # 'skip' is a boolean indicating whether it's already marked to be skipped
    # 'options' are the autodoc options

    # Skip all CLI methods
    if name.endswith("_cli"):
        # Param what should be 'method' but is 'class'; obj is the function object.
        # https://github.com/sphinx-doc/sphinx/issues/6808
        return True

    return skip  # Return the original skip value for other members


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)


project = "kaggle-cli"
copyright = "2025, Kaggle"
author = "Kaggle"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/kaggle/models/*.py"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"

autoclass_content = "both"
myst_heading_anchors = 2
suppress_warnings = ["ref.unknown"]

# -- Options for sphinx.ext.apidoc -------------------------------------------
apidoc_module_dir = "../src/kaggle"
apidoc_output_dir = "source"
apidoc_excluded_paths = []
apidoc_separate_modules = True
