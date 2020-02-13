.. _citing:

Citing sisl
===========

sisl is an open-source software package intended for the scientific community. It is
released under the LGPL-3 license.

You are encouraged to cite sisl you use it to produce scientific contributions.

The sisl citation can be found through Zenodo:

|zenodo|_

By citing sisl you are encouraging development and expoosing the software package.


Citing basic usage
------------------

If you are *only* using sisl as a post-processing tool and/or tight-binding calculations
you should cite this (Zenodo DOI):

.. code-block:: bash

    @misc{zerothi_sisl,
      author       = {Papior, Nick},
      title        = {sisl: v<fill-version>},
      year         = {2020},
      doi          = {10.5281/zenodo.597181},
      url          = {https://doi.org/10.5281/zenodo.597181}
    }


The `sgeom`, `sgrid` or `sdata` commands all print-out the above information in a suitable format:

.. code-block:: bash

    sgeom --cite
    sgrid --cite
    sdata --cite

which fill in the version for you, all yield the same output.


.. _citing-transport:
    
Citing transport backend
------------------------

When using sisl as tight-binding setup for Hamiltonians and dynamical matrices for
`TBtrans`_ and ``PHtrans`` you should cite these two DOI's:


.. code-block:: bash

    @misc{zerothi_sisl,
      author       = {Papior, Nick},
      title        = {sisl: v<fill-version>},
      year         = {2020},
      doi          = {10.5281/zenodo.597181},
      url          = {https://doi.org/10.5281/zenodo.597181}
    }

    @article{Papior2017,
      author = {Papior, Nick and Lorente, Nicol{\'{a}}s and Frederiksen, Thomas and Garc{\'{i}}a, Alberto and Brandbyge, Mads},
      doi = {10.1016/j.cpc.2016.09.022},
      issn = {00104655},
      journal = {Computer Physics Communications},
      month = {mar},
      number = {July},
      pages = {8--24},
      title = {{Improvements on non-equilibrium and transport Green function techniques: The next-generation transiesta}},
      volume = {212},
      year = {2017}
    }



.. |zenodo| image:: https://zenodo.org/badge/doi/10.5281/zenodo.597181.svg
.. _zenodo: http://dx.doi.org/10.5281/zenodo.597181
