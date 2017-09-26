.. highlight:: bash

.. _script_sdata:
	       
`sdata`
=======

The `sdata` executable is a tool for reading and performing actions
on *all* `sisl` file formats applicable.

Essentially it performs operations dependent on the file that is being
processed. If for instance the file contains any kind of `Geometry`
it allows the same operations as `sgeom`.

For a short help description of the possible uses do:

::
		
    sdata <in> --help

which shows a help dependent on which kind of file ``<in>`` is.

As the options for this utility depends on the input *file*, it
is not completely documented.



Files with `Geometry`
---------------------

If a file contains a `Geometry` one gets *all* the options like
`sgeom`. I.e. `sdata` is a generic form of the `sgeom` script.


Files with `Grid`
-----------------

If the file contains a `Grid` one gets *all* the options like
`sgrid`. I.e. `sdata` is a generic form of the `sgrid` script.


.. highlight:: python
