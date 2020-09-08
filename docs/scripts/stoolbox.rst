.. highlight:: bash

.. _script_stoolbox:

`stoolbox`
==========

sisl exposes user contributed tool-box command line interfaces through a common
interface.

The sisl toolbox is a separate package (shipped together with sisl) which contains
tools that are thought to be good tools for sisl users but are too specialized in
terms of usability to enter the sisl API.

We encourage users to contribute toolboxes to increase visibility and usability.

For a short description of the possible toolboxes do:

::

   stoolbox --help


Have an idea?
-------------

If you want to contribute or have an idea for a toolbox, feel free to open an issue `here <issue_>`_.

   
`stoolbox ts-poisson`
---------------------

This toolbox can create a solution of the Poisson equation for N-electrode calculations.

There are a set of required input-options for this command:

- ``--geometry *.TBT.nc`` this contains information such as buffer atoms, electrode atoms, number of electrodes and device atoms.
- ``--elec-V NAME V`` specificy the potential an electrode has, this should be specified for all electrodes.
- ``--shape NX NY NZ`` number of points in the TranSiesta simulation, read this off from the ``InitMesh`` line in the output
- ``--out FILE`` file to store the result in

The potentials you should give for the individual electrodes should match the ratio of the TranSiesta input.
So for a 2-probe calculation at :math:`\pm V/2` the command would be::

::

   stoolbox ts-poisson <other options> --elec-V Left 1. --elec-V Right -1.


If the ratio between the potentials is fixed for all simulations this is only necessary once since
TranSiesta can scale the potential for the correct bounds.

If the calculation takes a long time or uses too much memory one may try to do either, or both, of the
following:

- ``--pyamg-shape PX PY PZ`` use a grid size of this shape to solve the Poisson equation, once solved
  an interpolation to ``--shape`` will be performed to ensure the correct shape of the grid.
- ``--dtype f`` to use only floating point values (halves the memory).

These will speed up the calculation considerably.

Further details may be found by using the ``--help`` feature::

::

   stoolbox ts-poisson --help


.. highlight:: python
