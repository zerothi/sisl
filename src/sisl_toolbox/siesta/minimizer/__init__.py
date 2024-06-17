# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

r""" Basis set optimizations using scipy.minimize

Developer: Nick Papior
Contact: GitHub development site
sisl-version: >=0.11.0

Sometimes it may be nice to minimize a metric based on the basis set for
a Siesta calculation.

This tool is based on work done by :cite:author:`Rivero2015` :cite:`Rivero2015` and Javier Junquera.

It is however very different in its use and what it can do.

- It is made to be easily extendable in order to fine-tune ones
  metric for optimization
- It allows combinations of metrics by allowing arithmetic on metric
  objects
- It uses open-source standards for the optimization by employing
  `scipy.optimize.minimize` or `scipy.optimize.dual_annealing`
- Configuration of basis-optimization variables are controlled via
  easily readable yaml files.
- Restart capabilities by reading previously created minimization data

Let us here create an example that minimizes the total energy by
weighing two different geometries of the same atomic specie.

It requires only 2 prepared directories which should
contain an executable ``run.sh`` file that runs Siesta using ``RUN.fdf`` and
writes to ``RUN.out``. In each of the directories a file called ``BASIS.fdf``
will be written that contains the ``PAO.Basis`` block, it is important
that this file is read by Siesta.

.. code:: python

    import numpy as np
    # import everything, just easier here
    from sisl_toolbox.siesta.basis import *
    import sisl

    siesta1 = SiestaRunner("geometry1")
    siesta2 = SiestaRunner("geometry2")
    runner = (siesta1 & siesta2)

    # Define the metric we wish to minimize.
    # This could essentially be anything depending on the basis set
    metric = (EnergyMetric(siesta1.absattr("stdout")) * 1.1 +
              EnergyMetric(siesta2.absattr("stdout")) * 0.9)

    # Use a local minimization technique, storing intermediate
    # data in local_etot.dat
    minimize = LocalMinimizeSiesta(runner, metric, out="local_etot.dat")

    # We need to define the basis of the atom we wish to minimize
    # Here we have an output file from a previous calculation
    # The only thing that is read is the PAO.Basis block and the *size* of the
    # basis. If the PAO.Basis block contains DZP data, then this is the basis set
    # used in the reference calculations.
    # The optimize.yaml contains the required data which creates the basis
    # used in the minimization procedure.
    basis = AtomBasis.from_yaml("optimize.yaml", "atom")

    # add variables (and skip parameters).
    # This 2nd step is required to disentangle the parameters in the
    # `basis` from the variables.
    for v in basis.get_variables("optimize.yaml", "atom"):
       minimize.add_variable(v)

    # We have found that for some cases normalized variables behave better
    # than non-normalized data. By default the Minimize class normalizes
    # variables to be in the range 0:10 which seems to work ok.
    # Please report back what seems to work the best.
    # For local minimizations we need the following:
    # - constraints
    # - bounds for variables
    # - displacements for Jacobian
    # - tolerance for the minimization
    constraints = minimize.get_constraints()
    eps = minimize.normalize("delta", with_offset=False)
    tolerance = 1e-5

    # now run minimization
    vopt = minimize.run(tol=tolerance, constraints=constraints,
                        options={"ftol": tolerance,
                                 "eps": eps,
                                 "finite_diff_rel_step": eps,
                                 "maxiter": 1000,
                                 "maxcor": 1000})

This far from covers everything here but should give some ideas of what to do.
There is also a class that optimizes the pseudopotential, but so far this is
not as tested as the other one.
"""
from ._atom_basis import *
from ._atom_pseudo import *
from ._metric import *
from ._metric_siesta import *
from ._minimize import *
from ._minimize_siesta import *
from ._runner import *
from ._variable import *
