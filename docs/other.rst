.. _other:

Other resources
===============

One of sisl goals is an easy interaction between a variety of DFT simulations, much like `ASE`_ with
a high emphasis on `Siesta`_, while simultaneously providing the tools necessary to perform tight-binding
calculations.


However, sisl is far from the only Python package that implements simplistic tight-binding calculations.
Here a short introduction to some of the other methods is also highlighted. 

We are currently aware of 3 established tight-binding methods used in litterature (in random order):

- PythTB, see `here <http://physics.rutgers.edu/pythtb/index.html>`.
- kwant, see `here <https://kwant-project.org/>`
- pybinding, see `here <http://pybinding.site/>`

sisl's philosophy is drastically different in the sense that Hamiltonian's (and other
physical quantities described via matrices) are defined in matrix form. As for kwant and
pybinding the model is *descriptive* as *shapes* define the geometries.
Secondyl, both kwant and pybinding are self-contained packages where all physics is handled by the
scripts them-selves.
