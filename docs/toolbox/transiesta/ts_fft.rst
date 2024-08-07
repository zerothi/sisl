
.. _toc-tool-ts-fft:

.. module:: sisl_toolbox.transiesta.poisson


TranSiesta Hartree correction for FFT Poisson solver
====================================================

`TranSiesta`_ calculations based on the NEGF formalism is based on solving the
Poisson equation for an open-boundary condition.
For 2 electrodes (left/right) a potential drop between the electrodes.

In `Siesta`_ the Poisson equation is solved using Fourier transforms which
implicitly assumes periodic boundary conditions. This is contrary to the
actual open boundary conditions which have fixed *and* different potentials at
the boundaries.

In 2-probe calculations this is trivially fixed by adding a ramp to the unit-cell
along the transport direction. This fixes the periodic boundary conditions.

General :math:`N` electrode calculation cannot easily fix the periodic boundary
conditions and this forces the users to intervene. The fdf option ``TS.Poisson``
should be used to supply a correction term to the Poisson solution that enforces
the boundary conditions imposed by the electrodes. However, creating such a correction
is not trivial.

This tool tries to remedy this by solving the Poisson equation with fixed boundaries
as defined by the user.


Command line tool
-----------------

It may be called using

.. code-block:: console

   stoolbox ts-fft --help

which gives a lot of instructions.

Here we list the required items needed to create a correction file compatible with
TranSiesta.

1. Supply a geometry from a previous `TBtrans`_ calculation
2. Manually define the applied bias' for each electrode
3. Define the shape of the grid, use values from Siesta output;
   search for line ``InitMesh: MESH = A x B x C``
4. Define boundary conditions for each of the simulation box 6 sides
5. Define the output file

All of these points are very important, while users should in particular take note
of point 4.

An example command line for a bulk system with transport along third lattice vector
would look something like:

.. code-block:: console

   stoolbox ts-fft --geometry siesta.TBT.nc -V Left 0.5 -V Right -0.5 \
	  --shape 300 200 100 --out fft-fix.TSV.nc \
	  --boundary-condition-c d d

Note the boundary conditions along the transport direction; here ``d`` means Dirichlet.

For systems where there is no periodicity one can use Neumann boundaries. For a 1D chain
one would do:

.. code-block:: console

   stoolbox ts-fft --geometry siesta.TBT.nc -V Left 0.5 -V Right -0.5 \
	  --shape 300 200 100 --out fft-fix.TSV.nc \
	  --boundary-condition-a n n \
	  --boundary-condition-b n n \
	  --boundary-condition-c d d

Which uses Dirichlet for transport direction, otherwise Neumann.

Better performance
^^^^^^^^^^^^^^^^^^

Sometimes the grid used is very large ``--shape``. In these case it may be beneficial
to solve the Poisson equation for a smaller shape, and then interpolate.
This is accomblished using both ``--pyamg-shape`` (Poisson solution) and ``--shape``
(output shape). An interpolation will be done after the solution step.


Method
------

The internal method solves the Poisson equation *twice* using a multigrid solver (`pyamg`).
First it fixes the potentials on all grid points touching the electrode atoms
(atomic radius controlled with ``--radius``). Then uses the solution to find
all boundary point values, fixes these and solves the Poisson equation again.
The resulting solution should *somewhat* match the boundary conditions of the calculation
and correct the FFT solution.

If you know the boundary conditions that fixes the FFT solution for your particular setup,
you are encouraged to use that instead.


.. autosummary::
   :toctree: generated/

   solve_poisson
