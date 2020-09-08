.. highlight:: bash

.. _script_sgeom:
	       
`sgeom`
=======

The `sgeom` executable is a tool for reading and transforming general
coordinate formats to other formats, or alter them.

For a short help description of the possible uses do:

::

   sgeom --help


Here we list a few of the most frequent used commands.



Conversion
----------

The simplest usage is transforming from one format to another format.
`sgeom` takes at least two mandatory arguments, the first being the
input file format, and the second (and any third + argumets) the output
file formats

::
   
   sgeom <in> <out> [<out2>] [[<out3>] ...]

Hence to convert from an **fdf** Siesta input file to an **xyz** file
for plotting in a GUI program one can do this:

::
   
   sgeom RUN.fdf RUN.xyz

and the ``RUN.xyz`` file will be created.

Remark that the input file *must* be the first argument of `sgeom`.

    
Available formats
^^^^^^^^^^^^^^^^^

A great deal of different file formats are available. For details please see
`sisl.io` for a list of all implemented data-files. *Any* file that implements
the ``read_geometry``/``write_geometry`` methods will be usable by `sgeom`.


Advanced Features
-----------------

More advanced features are represented here.

The `sgeom` utility enables highly advanced creation of several geometry
structures by invocing the arguments *in order*.

I.e. if one performs:

::
   
   sgeom <in> --repeat 3 x repx3.xyz --repeat 3 y repx3_repy3.xyz

will read ``<in>``, repeat the geometry 3 times along the first unit-cell
vector, store the resulting geometry in ``repx3.xyz``. Subsequently it will repeat
the already repeated structure 3 times along the second unit-cell vector and store
the now ``3x3`` repeated structure as ``repx3_repy3.xyz``.

    
Repeating/Tiling structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^

One may use periodicity to create larger structures from a simpler structure.
This is useful for creating larger bulk structures.
To repeat a structure do

::
   
   sgeom <in> --repeat <int> [ax|yb|zc] <out>

which repeats the structure one atom at a time, ``<int>`` times, in the corresponding direction.
Note that ``x`` and ``a`` correspond to the same cell direction (the first).

To repeat the structure in *chunks* one can use the ``--tile`` option:

::
   
   sgeom <in> --tile <int> [ax|yb|zc] <out>

which results in the same structure as ``--repeat`` however with different atomic ordering.

Both tiling and repeating have the shorter variants:

::
   
   sgeom <in> -t[xyz] <int> -r[xyz] <int>

to ease the commands.

To repeat a structure 4 times along the *x* cell direction:

::
   
   sgeom RUN.fdf --repeat 4 x RUN4x.fdf
   sgeom RUN.fdf --repeat-x 4 RUN4x.fdf
   sgeom RUN.fdf --tile 4 x RUN4x.fdf
   sgeom RUN.fdf --tile-x 4 RUN4x.fdf

where all the above yields the same structure, albeit with different orderings.


Rotating structure
^^^^^^^^^^^^^^^^^^

To rotate the structure around certain cell directions one can do:

::
   
   sgeom <in> --rotate <angle> [ax|yb|zc] <out>

which rotates the structure around the origo with a normal vector along the
specified cell direction. The input angle is in degrees and *not* in radians.
If one wish to use radians append an ``r`` in the angle specification.

Again there are shorthand commands:

::

   sgeom <in> -R[xyz] <angle>


Combining command line arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    

All command line options may be used together. However, one should be aware that
the order of the command lines determine the order of operations.

If one starts by repeating the structure, then rotate it, then shift the structure,
it will be different from, shift the structure, then rotate, then repeat.

Be also aware that outputting structures are done *at the time in the command line order*.
This means one can store the intermediate steps while performing the entire operation:

::
   
   sgeom <in> --rotate <angle> --out <rotated> -tx 2 --out <rotate-tile-x> --ty 2 --out <rotate-tile-y>


.. highlight:: python
