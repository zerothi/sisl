.. _geom-neighbors:

*****************
Finding neighbors
*****************

.. module:: sisl.geom

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
   PointsNeighborList
   PointNeighborList


Neighbor lists
================

Once created, a neighbor finder can be queried to get the neighbors. It will return a neighbor list.
Depending on which type of query you make to the neighbor finder, it will return a different type of
neighbor list.

The following table summarizes the properties of each type of neighbor list:

+----------------------+--------------------+--------------+-----------------------+
| Class                | Neighbors for      |   `i` < `j`  |  Items                |
+======================+====================+==============+=======================+
| `FullNeighborList`   | All atoms          |      No      |  `AtomNeighborList`   |
+----------------------+--------------------+--------------+-----------------------+
| `UniqueNeighborList` | All atoms          |      Yes     |         N/A           |
+----------------------+--------------------+--------------+-----------------------+
| `PartialNeighborList`| Selected atoms     |      No      |   `AtomNeighborList`  |
+----------------------+--------------------+--------------+-----------------------+
| `AtomNeighborList`   | One atom           |      No      |         N/A           |
+----------------------+--------------------+--------------+-----------------------+
| `PointsNeighborList` | Points in space    |      No      |   `PointNeighborList` |
+----------------------+--------------------+--------------+-----------------------+
| `PointNeighborList`  | One point in space |      No      |         N/A           |
+----------------------+--------------------+--------------+-----------------------+

Where:

- `i < j` indicates whether the list only contains one direction of the interaction, i.e. by omitting
  the transposed interaction.
- `Items` indicates the type that you get when you iterate or index (e.g. ``neighbors[0]``) the list.
