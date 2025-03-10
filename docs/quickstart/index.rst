.. _quickstart:

Quickstart
==========

.. toctree::
    :maxdepth: 1

    overview

Installation
------------

.. this document is heavily inspired by pandas
   All credit should go to the pandas contributors!

.. grid:: 1 2 2 2
    :gutter: 4

    .. grid-item-card:: Using conda?
        :columns: 12 12 6 6

        sisl can be installed with `Anaconda/Miniconda <conda-releases_>`_.

        ++++++++++++++++++++++

        .. code-block:: bash

            conda install -c conda-forge sisl

    .. grid-item-card:: Using pip?
        :columns: 12 12 6 6

        sisl can be installed via pip from `PyPI <pypi-releases_>`_.

        ++++

        .. code-block:: bash

            python3 -m pip install sisl

    .. grid-item-card:: Detailed installation instructions?
        :link: install
        :link-type: doc
        :columns: 12
        :padding: 1

        Check out the advanced installation page.

.. toctree::
    :maxdepth: 2
    :hidden:

    install

Tutorials
--------------------

.. nbgallery::
   :name: intro-tutorials-gallery

   intro_tutorials/01_geometry.ipynb
   intro_tutorials/02_geometry_orbitals.ipynb
   intro_tutorials/03_geometry_sile.ipynb
