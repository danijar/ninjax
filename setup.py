import setuptools
import pathlib

import ninjax


setuptools.setup(
    name='ninjax',
    version=ninjax.__version__,
    description='General Modules for JAX',
    url='http://github.com/danijar/ninjax',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=['ninjax'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
