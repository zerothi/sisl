.. sisl documentation main file, created by
   sphinx-quickstart on Wed Dec  2 19:55:34 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


| |pypi| |conda| |license| |zenodo|
| |discord| |buildstatus| |codecov| |python-versions|


.. module:: sisl

.. title:: sisl: Toolbox for electronic structure calculations
.. meta::
   :description: sisl is a tool to manipulate density functional
		 theory code input and/or output. It also implements tight-binding
		 tools to create and manipulate multi-orbital (non)-orthogonal basis sets.
   :keywords: LCAO, Siesta, TranSiesta, OpenMX, TBtrans,
	      VASP, GULP, BigDFT, DFTB+,
	      DFT,
	      tight-binding, electron, electrons, phonon, phonons


sisl: Toolbox for electronic structure calculations
===================================================

The Python library `sisl <https://github.com/zerothi/sisl>`_ was born out of a need to handle (create and read), manipulate and analyse output from DFT programs.
It was initially developed by Nick Papior (co-developer of `Siesta`_) as a side-project to `TranSiesta`_
and `TBtrans`_ to efficiently analyse TBtrans output for N-electrode calculations.

Since then it has expanded to accommodate a rich set of DFT code input/outputs.

.. grid:: 1 1 2 2
    :gutter: 2

    .. grid-item-card:: :fas:`person-running` -- Quick-start guides
        :link: quickstart/index
        :link-type: doc

        New to `sisl`? Quickly get started with basic tutorials
        and getting used to the basic details of `sisl`.

    .. grid-item-card::  :fas:`book-open` -- User guide

        .. The user guide provides a set of how-to's explaining
        .. more usages in greater details.

    .. grid-item-card::  :fas:`book-open` -- API reference
        :link: api/index
        :link-type: doc

        Already familiar with `sisl` and how to use it? Look through
        the API section to discover more functionality and details
        on the parameters and in-depth usage.

    .. grid-item-card:: :fab:`discord` -- Chat with us
        :link: https://discord.gg/5XnFXFdkv2
        :link-type: url

        Join our Discord channel to share scripts, get help, or simply hang
        out.


`sisl` is also part of the training material for a series of workshops hosted `here <workshop_>`_.


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting started

   introduction
   quickstart/index
   cite

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   tutorials/index
   visualization/index
   scripts/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Advanced usage

   api/index
   environment
   toolbox/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Development

   dev/index

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Extras

   math
   release
   other
   references



.. |buildstatus| image:: https://github.com/zerothi/sisl/actions/workflows/test.yaml/badge.svg?branch=main
   :target: https://github.com/zerothi/sisl/actions/workflows/test.yaml

.. |pypi| image:: https://badge.fury.io/py/sisl.svg
   :target: https://pypi.org/project/sisl

.. |license| image:: https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg
   :target: https://www.mozilla.org/en-US/MPL/2.0/

.. |conda| image:: https://anaconda.org/conda-forge/sisl/badges/version.svg
   :target: https://anaconda.org/conda-forge/sisl

.. |codecov| image:: https://codecov.io/gh/zerothi/sisl/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/zerothi/sisl

.. |zenodo| image:: https://zenodo.org/badge/doi/10.5281/zenodo.597181.svg
   :target: https://doi.org/10.5281/zenodo.597181

.. |discord| image:: https://img.shields.io/discord/742636379871379577.svg?label=&logo=discord&logoColor=ffffff&color=green&labelColor=red
   :target: https://discord.gg/5XnFXFdkv2

.. |codetriage| image:: https://www.codetriage.com/zerothi/sisl/badges/users.svg
   :target: https://www.codetriage.com/zerothi/sisl

.. |python-versions| image:: https://img.shields.io/pypi/pyversions/sisl.svg
   :target: https://pypi.org/project/sisl/
