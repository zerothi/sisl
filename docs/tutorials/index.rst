.. _tutorials:

Tutorials
=========

These tutorials introduce sisl, from tight-binding Hamiltonians to electronic structure
calculations and the analysis of Siesta calculations.
To create geometries, start with the :ref:`quickstart tutorials <quickstart>`.

All examples assume this header::

   import numpy as np
   import sisl as si

to enable `numpy`_ and sisl.

Geometries and electronic structure
-----------------------------------

.. toctree::
   :maxdepth: 1

   tutorial_hamiltonian.ipynb
   tutorial_es_1.ipynb
   tutorial_es_2.ipynb
   tutorial_categories.ipynb
   tutorial_siesta_2_ahc.ipynb


Siesta/TranSiesta support
-------------------------

sisl was initiated around the `Siesta`_/`TranSiesta`_ code. The
sisl+TBtrans+TranSiesta tutorial `here <https://github.com/zerothi/ts-tbt-sisl-tutorial/>`_
is *very* educational.

If you use sisl as an analysis tool for Siesta, follow these tutorials:

.. toctree::
   :maxdepth: 1

   tutorial_siesta_1.ipynb
   tutorial_siesta_2.ipynb
