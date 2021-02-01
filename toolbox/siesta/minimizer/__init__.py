r""" Basis set optimizations using scipy.minimize

Developer: Nick Papior
Contact: GitHub development site
sisl-version: >=0.11.0

Sometimes it may be nice to minimize a metric based on the basis set for
a Siesta calculation.

This tool is based on work done by Rivero et. al. [1]_ and Javier Junquera.

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

    # Define the metric we wish to minimize.
    # This could essentially be anything depending on the basis set
    metric = (TotalEnergyMetric(siesta1) * 1.1
              TotalEnergyMetric(siesta2) * 0.9)

    # Use a local minimization technique, storing intermediate
    # data in local_etot.dat
    minimize = LocalMinimizeBasis(metric, out="local_etot.dat")

    # We need to define the basis of the atom we wish to minimize
    # Here we have an output file from a previous calculation
    # The only thing that is read is the PAO.Basis block and the *size* of the
    # basis. If the PAO.Basis block contains DZP data, then this is the basis set
    # used in the reference calculations.
    minimize.set_basis(sisl.io.siesta.outSileSiesta("REF.out", 'r').read_basis_block())

    # Our configuration file (default.yaml) contain what to minimize
    # and what ranges are allowed.
    # See default.yaml for details
    minimize.add_yaml("default.yaml")

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
    bounds = minimize.normalize_bounds()
    eps = minimize.normalize("delta", with_offset=False)
    tolerance = 1e-5

    # now run minimization
    vopt = minimize.run(bounds=bounds, constraints=constraints, tol=tolerance,
                        options={"ftol": tolerance,
                                 "eps": eps,
                                 "finite_diff_rel_step": eps,
                                 "maxiter": 1000,
                                 "maxcor": 1000})

This far from covers everything here but should give some ideas of what to do.
There is also a class that optimizes the pseudopotential, but so far this is
not as tested as the other one.

References
----------
.. [1] P. Rivero, V.M. García-Suárez, D. Pereñiguez, K. Utt, Y. Yang, L. Bellaiche, K. Park, J. Ferrer, S. Barraza-Lopez, "Systematic pseudopotentials from reference eigenvalue sets for DFT calculations", Computational Materials Science, *98*, 372-389 (2015)
"""
from ._atom_pseudo import *
from ._atom_basis import *
from ._metric import *
from ._metric_siesta import *
from ._minimize import *
from ._minimize_siesta import *
from ._runner import *
from ._variable import *
