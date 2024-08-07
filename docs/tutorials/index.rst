
Tutorials
=========

sisl is shipped with these tutorials which introduce the basics.

All examples are assumed to have this in the header::

   import numpy as np
   import sisl as si

to enable `numpy`_ and sisl.

Below is a list of the current tutorials:

.. toctree::
   :maxdepth: 1

   tutorial_es_1.ipynb
   tutorial_es_2.ipynb
   tutorial_siesta_2_ahc.ipynb


Siesta/TranSiesta support
-------------------------

sisl was initiated around the `Siesta`_/`TranSiesta`_ code. And it may be *very* educational to
look at the sisl+TBtrans+TranSiesta tutorial located `here <https://github.com/zerothi/ts-tbt-sisl-tutorial/>`_.

If you plan on using sisl as an analysis tool for Siesta you are highly recommended to follow
these tutorials:

.. toctree::
   :maxdepth: 1

   tutorial_siesta_1.ipynb
   tutorial_siesta_2.ipynb


Deprecated tutorials
--------------------

Once everything has been moved to the IPython notebook notation, these tutorials will be
removed. Until that happens they are located here for consistency:

.. toctree::
   :maxdepth: 1

   Geometry creation -- part 1 <tutorial_01>
   Geometry creation -- part 2 <tutorial_02>
   Supercells <tutorial_04>
   Electronic structure setup -- part 1 <tutorial_05>
   Electronic structure setup -- part 2 <tutorial_06>
