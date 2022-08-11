import sys, os, mock
for name in ['jax', 'jax.numpy', 'haiku', 'flax', 'optax']:
  sys.modules[name] = mock.Mock()
sys.path.insert(0, os.path.abspath('../ninjax'))
import ninjax as nj
sys.modules['nj'] = nj

project = 'Ninjax'
copyright = '2022, Danijar Hafner'
author = 'Danijar Hafner'

master_doc = 'index'

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    'member-order': 'bysource',
    'members': True,
    'undoc-members': True,
    # 'special-members': '__init__',
}

suppress_warnings = [
    'toc.circular',
]

html_theme = 'sphinx_rtd_theme'
