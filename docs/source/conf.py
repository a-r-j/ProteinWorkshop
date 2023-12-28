# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "Protein Workshop"
author = "Arian R. Jamasb"
release = "0.2.5"
copyright = f"{datetime.datetime.now().year}, {author}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
master_doc = "index"
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
    "sphinx_inline_tabs",
    "sphinxcontrib.gtagjs",
    "sphinxext.opengraph",
    "m2r2",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.napoleon",
    "sphinx_codeautolink",
    "sphinxcontrib.jquery",
    # "sphinx_autorun",
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
nbsphinx_allow_errors = True
nbsphinx_require_js_path = (
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"
)
nbsphinx_kernel_name = "graphein"
nbsphinx_execute = "never"

ogp_site_url = ""  # TODO
ogp_image = ""  # TODO

gtagjs_ids = [""]  # TODO


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "networkx": ("https://networkx.github.io/documentation/stable/", None),
    "nx": ("https://networkx.github.io/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "np": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pd": ("https://pandas.pydata.org/docs/", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "Sphinx": ("https://www.sphinx-doc.org/en/stable/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "torch_geometric": (
        "https://pytorch-geometric.readthedocs.io/en/latest/",
        None,
    ),
    "graphein": ("https://graphein.ai", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
}

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
mathjax2_config = {
    "tex2jax": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "processEscapes": True,
        "ignoreClass": "document",
        "processClass": "math|output_area",
    },
    "TeX": {
        "Macros": {
            "defeq": ":=",
            "vx": "\\mathbf{x}",
            "va": "\\mathbf{a}",
            "vb": "\\mathbf{b}",
            "vc": "\\mathbf{c}",
            "vd": "\\mathbf{d}",
            "ve": "\\mathbf{e}",
            "vf": "\\mathbf{f}",
            "vg": "\\mathbf{g}",
            "vh": "\\mathbf{h}",
            "vi": "\\mathbf{i}",
            "vj": "\\mathbf{j}",
            "vk": "\\mathbf{k}",
            "vl": "\\mathbf{l}",
            "vm": "\\mathbf{m}",
            "vn": "\\mathbf{n}",
            "vo": "\\mathbf{o}",
            "vp": "\\mathbf{p}",
            "vq": "\\mathbf{q}",
            "vr": "\\mathbf{r}",
            "vs": "\\mathbf{s}",
            "vt": "\\mathbf{t}",
            "vu": "\\mathbf{u}",
            "vv": "\\mathbf{v}",
            "vw": "\\mathbf{w}",
            "vy": "\\mathbf{y}",
            "vz": "\\mathbf{z}",
        }
    },
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
templates_path = ["_templates"]
exclude_patterns = ["_build"]
source_suffix = [".rst", ".md"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/workshop_icon.png"
html_title = f"{project} {release}"


def setup(app):
    app.add_js_file(
        "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"
    )
    app.add_js_file(
        "http://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"
    )
