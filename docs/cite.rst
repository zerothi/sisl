.. _citing:

Citing sisl
===========

sisl is an open-source software package intended for the scientific community. It is
released under the MPL-2 license.

You are encouraged to cite sisl when you use it to produce scientific contributions.

The sisl citation can be found through Zenodo: |zenodo|_

By citing sisl you are encouraging development and exposing the software package.

.. |zenodo| image:: https://zenodo.org/badge/doi/10.5281/zenodo.597181.svg
.. _zenodo: https://doi.org/10.5281/zenodo.597181


Citing basic usage
------------------

If you are *only* using sisl as a post-processing tool and/or tight-binding calculations
you should cite this (Zenodo DOI):

.. code-block:: bibtex

    @software{zerothi_sisl,
      author       = {Papior, Nick},
      title        = {sisl: v<fill-version>},
      year         = {2024},
      doi          = {10.5281/zenodo.597181},
      url          = {https://doi.org/10.5281/zenodo.597181}
    }


The `sgeom`, `sgrid` or `sdata` commands all print-out the above information in a suitable format:

.. code-block:: console

    sgeom --cite
    sgrid --cite
    sdata --cite

which fill in the version for you, all yield the same output.


.. _citing-transport:

Citing transport backend
------------------------

When using sisl as tight-binding setup for Hamiltonians and/or dynamical matrices for
`TBtrans`_ and/or ``PHtrans`` you should cite these two DOI's:


.. code-block:: bibtex

    @software{zerothi_sisl,
      author       = {Papior, Nick},
      title        = {sisl: v<fill-version>},
      year         = {2024},
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


If using real-space self-energies one should additionally cite:

.. code-block:: bibtex

    @article{papior2019,
      author = {Papior, Nick and Calogero, Gaetano and Leitherer, Susanne and Brandbyge, Mads},
      doi = {10.1103/physrevb.100.195417},
      number = {19},
      source = {Crossref},
      url = {https://doi.org/10.1103/physrevb.100.195417},
      volume = {100},
      journal = {Phys. Rev. B},
      publisher = {American Physical Society (APS)},
      title = {Removing all periodic boundary conditions: {Efficient} nonequilibrium Green's function calculations},
      issn = {2469-9950, 2469-9969},
      year = {2019},
      month = nov,
    }



.. _publications:

Publications using sisl
-----------------------

The `sisl` tool-suite has been used one way or the other in the listed
publications below.

Please help maintaining the list complete via a `pull request <pr_>`_ or
by writing an email to `nickpapior AT gmail.com <mailto:nickpapior@gmail.com>`_.


.. bibliography:: sisl_uses.bib
   :list: enumerated
   :all:
   :style: rev_year
   :labelprefix: U


arXiv publications
------------------

These publications are as far as we know in the review process.

- D. Weckbecker, M. Fleischmann, R. Gupta, W. Landgraf, S. Leitherer, O. Pankratov, S. Sharma, V. Meded, S. Shallcross,
  *Moiré ordered current loops in the graphene twist bilayer*,
  :doi:`1901.04712 <10.48550/arXiv.1901.04712>`

- Y. Guan, O.V. Yazyev,
  *Electronic transport in graphene with out-of-plane disorder*,
  :doi:`2210.16629 <10.48550/arXiv.2210.16629>`
