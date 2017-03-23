.. highlight:: bash

.. _script_sgrid:

`sgrid`
=======

The `sgrid` executable is a tool for manipulating a simulation grid and transforming
it into CUBE format for plotting 3D data in, *e.g.* VMD or XCrySDen.

Currently this is primarily intended for usage with SIESTA.

For a short help description of the possible uses do:

::
		
   sgrid --help

Here we list a few of the most frequent used commands.
Note that all commands are available via Python scripts and the `Grid` class.

Creating CUBE files
-------------------

The simplest usage is converting a grid file to CUBE file using

::
		
    sgrid Rho.grid.nc Rho.cube

which converts a SIESTA grid file of the electron density into a corresponding
CUBE file. The CUBE file writeout is implemented in `Cube`.

Conveniently CUBE files can accomodate geometries and species for inclusion in the 3D
plot and this can be added to the file via the ``--geometry`` flag, any geometry format
implemented in `sisl` are also compatible with `sgrid`.

::
		
   sgrid Rho.grid.nc --geometry RUN.fdf Rho.cube


the shorthand is ``-g``.
   
Grid differences
----------------

Often differences between two grids are needed. For this one can use the ``--diff`` flag which
takes one additional grid file for the difference. I.e.

::
		
   sgrid Rho.grid.nc[0] -g RUN.fdf --diff Rho.grid.nc[1] diff_up-down.cube

which takes the difference between the spin up and spin down in the same ``Rho.grid.nc`` file.

Reducing grid sizes
-------------------

Often grids are far too large in that only a small part of the full cell is needed to be studied.
One can remove certain parts of the grid after reading, before writing. This will greatly decrease
the output file *and* greatly speed-up the process as writing huge ASCII files is *extremely* time
consuming. There are two methods for reducing grids:

::
		
   sgrid <file> --sub x <pos|<frac>f>
   sgrid <file> --remove x [+-]<pos|<frac>f>

This needs an example, say the unit cell is an orthogonal unit-cell with side lengths 10x10x20 Angstrom.
To reduce the cell to a middle square of 5x5x5 Angstrom you can do:

::
		
   sgrid Rho.grid.nc --sub x 2.5:7.5 --sub y 2.5:7.5 --sub z 7.5:12.5 5x5x5.cube

note that the order of the reductions are made in the order of appearence. So *two* subsequent sub/remove
commands with the same direction will not yield the same final grid.
The individual commands can be understood via

  - ``--sub x 2.5:7.5``: keep the grid along the first cell direction above 2.5 Å and below 5 Å.
  - ``--sub y 2.5:7.5``: keep the grid along the second cell direction above 2.5 Å and below 5 Å.
  - ``--sub z 7.5:12.5``: keep the grid along the third cell direction above 7.5 Å and below 12.5 Å.

When one is dealing with fractional coordinates is can be convenient to use fractional grid operations.
The length unit for the position is *always* in Ångstrøm, unless an optional **f** is appended which
forces the unit to be in fractional position (must be between 0 and 1).

Averaging and summing
---------------------

Sometimes it is convenient to average or sum grids along cell directions:

::
		
   sgrid Rho.grid.nc --average x meanx.cube
   sgrid Rho.grid.nc --sum x sumx.cube

which takes the average or the sum along the first cell direction, respectively. Note that this results
in the number of partitions along that direction to be 1 (not all 3D software is capable of reading such a
CUBE file).


Advanced features
-----------------

The above operations are not the limited use of the `sisl` library. However, to accomblish more complex
things you need to manually script the actions using the `Grid` class and the methods available for that method.
For inspiration you can check the `sgrid` executable to see how the commands are used in the script.


.. highlight:: python
