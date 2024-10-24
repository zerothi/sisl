.. _other:

Related software
=================

One of `sisl`'s goals is an easy interaction between a variety of DFT simulations, much like `ASE`_ with
a high emphasis on `Siesta`_, while simultaneously providing the tools necessary to perform tight-binding
calculations.


However, `sisl` is far from the only Python package that implements simplistic tight-binding calculations.
We are currently aware of 3 established tight-binding methods used in literature (in random order):

- `PythTB <http://physics.rutgers.edu/pythtb/index.html>`__
- `kwant`_
- `pybinding <http://pybinding.site/>`__
- `ASE`_

`sisl`'s philosophy is drastically different in the sense that the Hamiltonian (and other
physical quantities described via matrices) is defined in matrix form. As for kwant and
pybinding the model is *descriptive* as *shapes* define the geometries.
Secondly, both kwant and pybinding are self-contained packages where all physics is handled by the
scripts them-selves, while sisl can calculate band-structures, but transport properties should be
off-loaded to `TBtrans`_.
