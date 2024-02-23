
.. _toc-scripts:

Command line scripts
====================

`sisl` implements a set of command-line utitilies that enables easy interaction
with *all* the data files compatible with `sisl`.


.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: :fas:`shapes` -- Geometry manipulation
      :link: sgeom
      :link-type: doc

      Process files containing geometries.

      Manipulate, by extending/removing atoms and write to
      a large variety of file formats.

      Can be used as a file format conversion tool.

   .. grid-item-card:: :fas:`table-cells` -- Grid manipulation
      :link: sgrid
      :link-type: doc

      Process files containing grid-data.

      Manipulation of the grids, by subtracting/reducing grids.

      Can be used as a file format conversion tool.

   .. grid-item-card:: :fas:`toolbox` -- Generic data manipulator
      :link: sdata
      :link-type: doc

      Click here for details.

   .. grid-item-card:: :fas:`toolbox` -- Toolboxes
      :link: stoolbox
      :link-type: doc

      Toolbox interfaces that can be extended with user-defined
      functionality.

.. toctree::
   :hidden:
   :maxdepth: 1

   sgeom
   sgrid
   stoolbox
   sdata
