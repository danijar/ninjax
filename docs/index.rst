.. currentmodule:: nj

.. raw:: html

  <h1>Ninjax API</h1>

General Modules for JAX


Basics
------

.. autosummary::
   pure
   rng
   Module
   Variable

.. autofunction:: pure
.. autofunction:: rng
.. autoclass:: Module
.. autoclass:: Variable


Transforms
----------

.. autosummary::
   grad
   jit
   pmap

.. autofunction:: grad
.. autofunction:: jit
.. autofunction:: pmap


Control Flow
------------

.. autosummary::
   cond
   scan

.. autofunction:: cond
.. autofunction:: scan


Advanced
--------

.. autosummary::
   state
   creating

.. autofunction:: state
.. autofunction:: creating

Integrations
------------

.. autosummary::
   HaikuModule
   FlaxModule
   OptaxModule

.. autoclass:: HaikuModule
.. autoclass:: FlaxModule
.. autoclass:: OptaxModule
