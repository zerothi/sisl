Minimizers
=================

.. currentmodule:: sisl_toolbox.siesta.minimizer

In `sisl_toolbox.siesta.minimizer` there is a collection of minimizers
that given some `variables`, a `runner` and a `metric` optimizes the
variables to minimize the metric.

These are the minimizer classes implemented:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseMinimize
   LocalMinimize
   DualAnnealingMinimize
   BADSMinimize
   ParticleSwarmsMinimize

For each of them, there is a subclass particularly tailored to optimize
SIESTA runs:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   MinimizeSiesta
   LocalMinimizeSiesta
   DualAnnealingMinimizeSiesta
   BADSMinimizeSiesta
   ParticleSwarmsMinimizeSiesta
