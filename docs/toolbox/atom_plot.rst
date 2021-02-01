
.. _toc-tool-atom-plot:

Plotting `atom` output
======================

Creating pseudopotentials using `atom`_ will often require users to
post-analyze the output of the program.

Here we aid this visualization process by adding a command line interface
for plotting details of the output via a simple command.


Command line tool
-----------------

It may be called using

.. code:: bash

   stoolbox atom-plot <output-directory>

which by default will create 4 plots.

.. image:: atom_plot_output.png
   :align: center
   :width: 500
   :alt: Default `stoolbox atom-plot` output


It mimicks the shell commands provided by atom for plotting with some additional details.

By default the program will output details regarding the core-charge and valence charge
overlap (use full for determining whether core-corrections should be used).


.. code:: bash

   Total charge in atom: 82.00213
   Core-correction r_pc [2.0, 1.5, 1]: [0.8656676967, 0.9330918445, 1.0573356051] Bohr

Here it is for lead and the first line gives the total charge in the atom calculation.
Note that the deviation stems from the different integration schemes used in the toolbox
and in `atom`_.
The 2nd line prints details about the where the core charge is some factor larger than
the pseudo potential valence charge. In this case it prints the radii :math:`r_i` where
:math:`\rho_{\mathrm{core}}(r_i)/\rho_{\mathrm{valence}}(r_i)` equals :math:`2`, :math:`1.5` and :math:`1`.

The full overlap is printed in the title of the charge plot. It is calculated as:

.. math::

   \mathcal O = \int \mathrm{min}(\rho_{\mathrm{core}}(r), \rho_{\mathrm{valence}}(r)) \mathrm dr


Suggestions and improvements to this utility is most welcome!


