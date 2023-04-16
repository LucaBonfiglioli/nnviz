# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "NNViz"
copyright = "2023, Luca Bonfiglioli"
author = "Luca Bonfiglioli"
release = "0.4.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_immaterial",
]

# Options for API Documentation
object_description_options = [
    (".*", dict(wrap_signatures_with_css=False)),
    ("py:.*", dict(wrap_signatures_with_css=True)),
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Automatically generate stub pages for each module
autosummary_generate = True

# Heading level depth for the table of contents
myst_heading_anchors = 3


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinx_immaterial"
html_static_path = ["_static"]
html_title = "NNViz Documentation"
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}
html_show_copyright = False


# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
# For the icon search for material icons on Google and look in this list
# https://gist.github.com/albionselimaj/14fabdb89d7893c116ee4b48fdfdc7ae
# if there's a valid code for your choice
html_theme_options = {
    "repo_url": "https://github.com/LucaBonfiglioli/nnviz",
    "repo_name": "LucaBonfiglioli/nnviz",
    "repo_type": "github",
    "icon": {
        "repo": "fontawesome/brands/git-alt",
        "edit": "material/file-edit-outline",
    },
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "red",
            "accent": "red",
            "toggle": {
                "icon": "material/weather-night",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "blue-grey",
            "accent": "red",
            "toggle": {
                "icon": "material/weather-sunny",
                "name": "Switch to light mode",
            },
        },
    ],
    "font": {
        "text": "Roboto",  # used for all the pages' text
        "code": "Roboto Mono",  # used for literal code blocks
    },
    # If False, expand all TOC entries
    "globaltoc_collapse": False,
    # If True, show hidden TOC entries
    "globaltoc_includehidden": False,
    # To investigate
    "features": [
        # The left sidebar will expand all collapsible subsections by default
        # "navigation.expand",
        # --------------------------------------------------------------
        # Top-level sections are rendered in a menu layer below the header for
        # viewports above 1220px, but remain as-is on mobile.
        # "navigation.tabs",
        # --------------------------------------------------------------
        # Navigation tabs will lock below the header and always remain
        # visible when scrolling down
        # "navigation.tabs.sticky",
        # --------------------------------------------------------------
        # A back-to-top button can be shown when the user, after scrolling down,
        # starts to scroll up again. It's rendered centered and just below the header
        # "navigation.top",
        # --------------------------------------------------------------
        # The URL in the address bar is automatically updated
        # with the active anchor as highlighted in the table of contents
        # "navigation.tracking",
        # --------------------------------------------------------------
        # If table of contents is enabled, it is always rendered as part
        # of the navigation sidebar on the left
        # "toc.integrate",
        # --------------------------------------------------------------
        # Top-level sections are rendered as groups in the sidebar
        # for viewports above 1220px, but remain as-is on mobile
        # "navigation.sections",
        # --------------------------------------------------------------
        # Clicks on all internal links will be intercepted and dispatched
        # via XHR without fully reloading the page
        "navigation.instant",
        # --------------------------------------------------------------
        # Header is automatically hidden when the user scrolls past
        # a certain threshold, leaving more space for content
        "header.autohide",
        # --------------------------------------------------------------
        # A back-to-top button can be shown when the user, after scrolling down,
        # starts to scroll up again. It's rendered centered and just below the header
        "navigation.top",
        # --------------------------------------------------------------
        # When anchor tracking is enabled, the URL in the address bar is automatically
        # updated with the active anchor as highlighted in the table of contents
        "navigation.tracking",
        # --------------------------------------------------------------
        # If a user clicks on a search result, the theme will highlight all
        # occurrences after following the link.
        "search.highlight",
        # --------------------------------------------------------------
        # A  share button is rendered next to the reset button, which allows to deep
        # link to the current search query and result
        "search.share",
        # --------------------------------------------------------------
    ],
    "toc_title": "In this page",
    "social": [
        {
            "icon": "fontawesome/brands/python",
            "link": "https://pypi.org/project/nnviz/",
            "name": "Check out NNViz on PyPi",
        },
    ],
    "version_dropdown": True,
    "toc_title_is_page_title": True,
}
