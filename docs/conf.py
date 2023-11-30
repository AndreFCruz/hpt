# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'hyperparameter-tuning'
copyright = '2022, AndreFCruz'
author = 'AndreFCruz'

# Import package version programmatically
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))
from hpt._version import __version__
release = __version__
version = __version__

# Copy examples folder to the documentation folder
import shutil
shutil.copytree(src="../examples", dst="examples", dirs_exist_ok=True)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    "sphinx_autodoc_typehints",  # needs to be AFTER napoleon
    "numpydoc",
    'sphinx_copybutton',
    # 'sphinx_autopackagesummary',
    'myst_parser',          # for rendering MD files
    'sphinx.ext.viewcode',
    'nbsphinx',             # for rendering jupyter notebooks
    # 'myst_nb',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autoclass_content = 'both'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_js_files = [
    'custom.js',    # custom JS file
]

# nbsphinx configuration
nbsphinx_execute = 'never'  # Set to 'always' if you want to execute the notebooks during the build process

# numpydoc configuration
numpydoc_show_class_members = False
