Basis optimization
==================

.. currentmodule:: sisl_toolbox.siesta.minimizer

Optimizing a basis set for SIESTA is a cumbersome task.
The goal of this toolbox is to allow users to **optimize a basis set with just one CLI command.**

The commands and their options can be accessed like:

.. code-block:: bash

   stoolbox basis --help

In summary, whenever one wants to optimize a basis set for a given system,
the first step is to create a directory with the input files to run the
calculation. This directory should contain, as usual:

- The ``.fdf`` files with all the input parameters for the calculation.
- The pseudopotentials (``.psf`` or ``.psml`` files).

Then, one can directly run the optimization:

.. code-block:: bash

   stoolbox basis optim --geometry input.fdf

with ``input.fdf`` being the input file for the calculation. This will use all the default
values. Since there are many possible tweaks, we invite you to read carefully the help
message from the CLI. Here we will just mention some important things that could go unnoticed.

**Basis enthalpy:** The quantity that is minimized is the basis enthalpy. This is :math:`H = E + pV`
with :math:`E` being the energy of the system, :math:`V` the volume of the basis and :math:`p` a "pressure" that
is defined in the fdf file with the ```BasisPressure`` table. This "pressure" penalizes bigger
basis, which result in more expensive calculations. It is the responsibility of the user to
set this value. As a rule of thumb, we recommend to set it to ``0.02 GPa`` for the first two
rows of the periodic table and ``0.2 GPa`` for the rest.

**The SIESTA command:** There is a ``--siesta-cmd`` option to specify the way of executing SIESTA. By default, it
is simply calling the ``siesta`` command, but you could set it for example to ``mpirun -n 4 siesta``
so that SIESTA is ran in parallel.

There is no problem with using this CLI in clusters inside a submitted job, for example.

**A custom basis definition:** It may happen that the conventional optimizable parameters as well as their lower and
upper bounds are not good for your case (e.g. you would like the upper bound for a cutoff
radius to be higher). In that case, you can create a custom ``--basis-spec``. The best way
to do it is by calling

.. code-block:: bash

   stoolbox basis build --geometry input.fdf

which will generate a yaml file with a basis specification that you can tweak manually.
Then, you can pass it directly to the optimization using the ``--config`` option:

.. code-block:: bash

   stoolbox basis optim --geometry input.fdf --config my_config.yaml

**Installing the optimizers:** The default optimizer is BADS (https://github.com/acerbilab/bads)
which is the one that we have found works best to optimize basis sets. The optimizer is however
not installed by default. You can install it using pip:

.. code-block:: bash

   pip install pybads

and the same would apply for other optimizers that you may want to use.

**Output:** The output that appears on the terminal is left to the particular optimizer.
However, sisl generates ``.dat`` files which contain information about each SIESTA execution.
These files contain one column for each variable being optimized and one column for the
metric to minimize.


Python API
----------

The functions that do the work are also usable in python code by importing them:

.. code-block:: python

   from sisl_toolbox.siesta.minimizer import optimize_basis, write_basis_to_yaml

Here is their documentation:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optimize_basis
   write_basis_to_yaml
