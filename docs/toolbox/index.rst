.. _toc-toolbox:

.. module:: sisl_toolbox

Toolboxes
=========

Installing `sisl` will install two packages, `sisl` and `sisl_toolbox`.

The latter are user-contributed tools that are useful for end-users but are
very focused on solving a particular problem. In order to maintain a stable
and sufficiently clean API any `sisl` extensions that are thought to be specialized
would go into the `sisl_toolbox` suite.

Some toolboxes have a command-line-interface, see :ref:`script_stoolbox` for details.

Toolboxes should be imported directly.

.. code-block:: python

   import sisl_toolbox.siesta.atom


The implemented toolboxes are listed here:
   
.. toctree::
   :maxdepth: 1

   transiesta/ts_fft
   siesta/atom_plot
   btd/btd
