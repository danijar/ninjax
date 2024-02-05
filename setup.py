import pathlib
import re
import setuptools


def parse_version(filename):
  text = (pathlib.Path(__file__).parent / filename).read_text()
  version = re.search(r"__version__ = '(.*)'", text).group(1)
  return version


setuptools.setup(
    name='ninjax',
    version=parse_version('ninjax/ninjax.py'),
    description='Flexible Modules for JAX',
    url='http://github.com/danijar/ninjax',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=['ninjax'],
    install_requires=['jax'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
