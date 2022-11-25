# Mock dependencies
import sys, os, unittest.mock
for name in ['jax', 'jax.numpy', 'haiku', 'flax', 'optax']:
  sys.modules[name] = unittest.mock.MagicMock()

# Work around MagicMock not preserving docstrings.
import jax, contextlib
class MockScope(contextlib.ContextDecorator):
  def __init__(self, fn=None, *args, **kwargs): self.fn = fn
  def __enter__(self): return self.fn
jax.named_scope = MockScope

# Import NinJax from repository
sys.path.insert(0, os.path.abspath('../ninjax'))
import ninjax as nj
sys.modules['nj'] = nj

project = 'Ninjax Docs'
copyright = '2022, Danijar Hafner'
author = 'Danijar Hafner'

master_doc = 'index'

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx_design',
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme_options = {
    'repository_url': 'https://github.com/danijar/ninjax',
    'use_repository_button': True,
    'use_issues_button': False,
    # 'prev_next_buttons_location': None,
    'show_navbar_depth': 1,
}

autodoc_default_options = {
    'member-order': 'bysource',
    'members': True,
    'undoc-members': True,
    # 'special-members': '__init__',
}

suppress_warnings = [
    'toc.circular',
]

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = ['style.css']
html_title = 'Ninjax'
html_logo = '_static/logo.png'
