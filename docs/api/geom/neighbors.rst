.. _geom-neighbors:

Finding neighbors
=================

.. currentmodule:: sisl.geom

`sisl` implements an **algorithm to find neighbors in a geometry**. It has two main properties:

:fas:`truck-fast` **It is very fast**. It can compute all the neighbors for a system of 3 million atoms in around 6 seconds.

:fas:`maximize` **It scales linearly** with the number of atoms in the system.

The algorithm is based on the **partition of space into bins**. This partition limits the search for
neighbors of a certain atom only within the bins adjacent to the bin the atom is in.

Since the bins must be created only once, there is a `NeighborFinder` class which **on initialization
creates the bin grid**. Once the finder is created, you can query it for neighbors as many times
as you want. You can ask for all neighbors or only the neighbors of atoms that you are interested in.

.. autosummary::
   :toctree: generated/

   NeighborFinder
   FullNeighborList
   UniqueNeighborList
   PartialNeighborList
   AtomNeighborList
   CoordsNeighborList
   CoordNeighborList


Neighbor lists
--------------

Once created, a neighbor finder can be queried to get the neighbors. It will return a neighbor list.
Depending on which type of query you make to the neighbor finder, it will return a different type of
neighbor list.

The following table summarizes the properties of each type of neighbor list:

+----------------------+-------------------------+---------------------+-----------------------+
| Class                | Neighbors for           | :math:`I < J`       |  Items                |
+======================+=========================+=====================+=======================+
| `FullNeighborList`   | All atoms               | :fas:`circle-xmark` |  `AtomNeighborList`   |
+----------------------+-------------------------+---------------------+-----------------------+
| `UniqueNeighborList` | All atoms               | :fas:`circle-check` |         N/A           |
+----------------------+-------------------------+---------------------+-----------------------+
| `PartialNeighborList`| Selected atoms          | :fas:`circle-xmark` |   `AtomNeighborList`  |
+----------------------+-------------------------+---------------------+-----------------------+
| `AtomNeighborList`   | One atom                | :fas:`circle-xmark` |         N/A           |
+----------------------+-------------------------+---------------------+-----------------------+
| `CoordsNeighborList` | Coordinates in space    | :fas:`circle-xmark` |   `CoordNeighborList` |
+----------------------+-------------------------+---------------------+-----------------------+
| `CoordNeighborList`  | Coordinate in space     | :fas:`circle-xmark` |         N/A           |
+----------------------+-------------------------+---------------------+-----------------------+

Where:

- :math:`I < J` indicates whether the list only contains one direction of the interaction, i.e. by omitting
  the transposed interaction.
- ``Items`` indicates the type that you get when you iterate over indices (e.g. ``neighbors[0]``) the list.
