import setuptools
import pathlib


DESCRIPTION = (
    'Module system for JAX that offers full state access and allows to '
    'easily combine modules from other libraries'
)

setuptools.setup(
    name='ninjax',
    version='0.3.1',
    description=DESCRIPTION,
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
