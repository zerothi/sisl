.. _io:

============
Input/Output
============

.. module:: sisl.io

Read and write files to many different formats used within
the atomic simulation community.

The sile framework
--------------------

`sisl` provides a unified interface to read and write files.
To distinguish a normal file from a `sisl` file, we use the term *sile*:

-------------------------------------

**sile**, noun (*sisl-file*)

*A supercharged file that has the capability to read and write
quantities related to atomic simulations.*

-------------------------------------

One can create a sile by simply doing:

.. code-block:: python

   import sisl
   sile = sisl.get_sile('myfile.extension')

Which will automatically determine the sile type based on the file extension.
Then, you might ask to read whatever the file contains. For example, if the
file contains a Hamiltonian, you can do:

.. code-block:: python

   hamiltonian = sile.read_hamiltonian()

Sometimes, the file extension is ambiguous as there are multiple file formats
that use the same extension. For this reason, `sisl` has other mechanisms at
your disposal. The current procedure for determining the file type is based on these
steps:

1. Extract the extension of the filename passed to `get_sile`, for instance
   ``hello.grid.nc`` will both examine extensions of ``grid.nc`` and ``nc``.
   This is necessary for leveled extensions.
2. Determine whether there is some specification in the file name ``hello.grid.nc{<specification>}``
   where ``<specification>`` can be:

   - ``contains=<name>`` (or simply ``<name>``) the class name must contain ``<name>``
   - ``endswith=<name>`` the class name must end with ``<name>``
   - ``startswith=<name>`` the class name must start with ``<name>``

   When a specification is used, only siles that obey that specification will be searched
   for the extension.
   This may be particularly useful when there are multiple codes using the same extension.
   For instance output files exists in several of the code bases.
3. Search all indexed (through `add_sile`) siles which obey the specification (all if no specifier)
   and collect all that matches the longest extension found.
4. If there is only 1 match, then return that class. If there are multiple `sisl` will try all ``read_*``
   methods and if all fail, then the sile will be removed from the eligible list.

You can find more information about the core functions and classes of the `sisl.io` module
here:

.. toctree::
   :maxdepth: 2

   basic

I/O from buffers and zip files
------------------------------

Sometimes, things are not as simple as reading/writing from/to the normal file system.
For example, you might have the files inside a zip file or you might have acces to the
file contents through a buffer (``io.StringIO``, ``io.BytesIO``...).

**All siles in `sisl` have the ability of reading and writing from buffers and zip files.**
In this case, there is no automatic detection of the sile type, so you will need to specify
the exact sile type you want to use. Following, we provide some examples:

Example for buffers:
^^^^^^^^^^^^^^^^^^^^

Writing a grid to a buffer:

.. code-block:: python

    import sisl
    import io

    geometry = sisl.geom.graphene()
    grid = sisl.Grid((2, 2, 2), geometry=geometry)
    grid[:] = np.random.random(grid.shape)

    buffer = io.BytesIO()

    nc = sisl.io.gridncSileSiesta(buffer, mode="wb")
    nc.write_grid(grid)

Reading a grid from a buffer:

.. code-block:: python

    import sisl
    import io

    buffer = ... # Get a buffer with the grid data

    grid = sisl.io.gridncSileSiesta(buffer).read_grid()

Example for zip files:
^^^^^^^^^^^^^^^^^^^^^^

Reading the Hamiltonian from a SIESTA run inside a zip file:

.. code-block:: python

    import sisl

    sisl.get_sile("/path/to/data.zip/run/RUN.fdf").read_hamiltonian()

Writing a Hamiltonian inside a zip file:

..  code-block:: python

    import sisl

    geom = sisl.geom.graphene()
    H = sisl.Hamiltonian(geom)

    H.write_hamiltonian("/path/to/data.zip/graphene.HSX")

By passing the path inside the zip file as a string, ``sisl`` will
automatically create the ``zipfile.ZipFile`` object and close it
when it is done. If you don't want the zip file to be closed, you can
create it externally and then pass a ``zipfile.Path`` to ``sisl``:

.. code-block:: python

    import sisl
    import zipfile

    # Create a zip file or append to an existing one
    zip_file = zipfile.ZipFile("myzipfile.zip", "a")
    # Define path inside the zip file
    H_path = zipfile.Path(zip_file, "graphene.HSX")

    geom = sisl.geom.graphene()
    H = sisl.Hamiltonian(geom)

    H.write_hamiltonian(H_path)

    # Now the zip file is not closed, it is up to you to close it
    # when you are done
    zip_file.close()

List of available siles
-----------------------

The above is a full list of all the available siles in `sisl`, and therefore
all the formats that you can read to and write from.

.. note::

   If you are missing the ability to read or write a specific file format in `sisl`,
   you can open an issue/PR. If you already have the code to read or write the file format,
   adding a new sile is easy, and it will make your main code compatible with
   other formats (e.g. other DFT codes) thanks to the unified interface.

.. toctree::
   :maxdepth: 2

   generic
   bigdft
   dftb
   fhiaims
   gulp
   openmx
   orca
   scaleup
   siesta
   tbtrans
   vasp
   wannier90
