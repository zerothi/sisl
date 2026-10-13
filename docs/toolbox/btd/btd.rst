
.. _toc-tool-btd:

.. module:: sisl_toolbox.btd


Block-tri-diagonal inversion algorithms
=======================================

Enables block-tri-diagonal inversion algorithms directly in Python.

The Green function of a device coupled to any number of semi-infinite
electrodes is calculated by `DeviceGreen`. It gives access to transmissions,
spectral functions, scattering states, scattering matrices and eigenchannels.
The device may be created from a TranSiesta/TBtrans calculation
(`DeviceGreen.from_fdf` and `DeviceGreen.from_tbt`), or directly in Python
from any `~sisl.Hamiltonian`, as shown below. TBtrans is *not* required.


Quick start
-----------

A one-dimensional chain with an impurity, coupled to two semi-infinite chains:

.. code-block:: python

   import numpy as np
   import sisl
   from sisl_toolbox.btd import BlochElectrode, DeviceGreen

   # electrode: a single orbital per unit cell, transport along a
   chain = sisl.Geometry(
       [0, 0, 0], sisl.Atom(1, R=1.1), lattice=sisl.Lattice([1, 10, 10], nsc=[3, 1, 1])
   )
   H_elec = sisl.Hamiltonian(chain)
   H_elec.construct([(0.1, 1.1), (0.0, -1.0)])

   # device: 10 cells of the electrode, with an impurity in the middle
   H = H_elec.tile(10, 0)
   H[5, 5] = 0.5

   # the bulk electrodes extend along -a (left) and +a (right)
   left = BlochElectrode(sisl.RecursiveSI(H_elec, "-A"), name="Left")
   right = BlochElectrode(sisl.RecursiveSI(H_elec, "+A"), name="Right")

   # place the electrodes on the first and last atom of the device
   G = DeviceGreen(H, [(left, [0]), (right, [9])])
   E = np.linspace(-2.5, 2.5, 11)
   T = [G.transmission(e, "Left", "Right") for e in E]

The following concepts are needed to set up a calculation:

Electrode
   An `Electrode` calculates the self-energy of a semi-infinite lead.
   `BlochElectrode` wraps a `~sisl.physics.RecursiveSI` (or a GF file),
   `RealSpaceElectrode` wraps a `~sisl.physics.RealSpaceSE` (electrodes that
   are semi-infinite in more than one direction).

Semi-infinite direction
   The sign tells to which side the bulk electrode extends, i.e. *away* from the device.
   A left electrode (at the start of the device along :math:`\mathbf a`) is ``"-A"``,
   a right electrode is ``"+A"``.

Electrode placement
   Each electrode is passed together with the indices of ``H`` its self-energy
   is placed on: atoms (default), ``{"atoms": idx}`` or ``{"orbitals": idx}``.
   The indices must be in the same order as the electrode orbitals, including
   Bloch expansions (equivalent to ``H_elec.tile`` along :math:`\mathbf a`, :math:`\mathbf b`,
   :math:`\mathbf c`). Typically they are the outermost layer of the device on the
   electrode side.
   `DeviceGreen` warns when the electrode does not seem to be placed correctly,
   i.e. a wrong semi-infinite direction, an electrode not at the edge of the device,
   or differently ordered orbitals.

Couplings along the semi-infinite directions
   The self-energies replace the periodic couplings of ``H`` along the
   semi-infinite directions; they are removed by `DeviceGreen` (on a copy of ``H``),
   no need to call ``H.set_nsc(...)``.

Imaginary part :math:`\eta`
   The device region uses ``eta`` for real energies, which defaults to the
   smallest electrode :math:`\eta` divided by 10. The electrodes use their
   own value, e.g. ``sisl.RecursiveSI(H_elec, "-A", eta=1e-4)``.

Bulk
   An optional third element, ``(left, [0], True)``, replaces the device
   Hamiltonian on the electrode indices by the bulk electrode Hamiltonian
   (`Electrode.bulk_matrix`).

Block-tri-diagonal partitioning
   By default the device region is all of ``H``, and it is partitioned
   automatically (``permute="gps"``), see `btd_partition`. Pass a `PermuteBTD`,
   or the device indices, to control the device region.


Downfolding
-----------

Parts of ``H`` may be removed from the device region by downfolding the
electrode self-energies through them with `DownfoldElectrode`. This reduces
the size of the device region, e.g. when only the scattering region is of interest.
`DownfoldElectrode` takes the electrode, ``H``, the indices of the electrode in
``H`` and the downfolding region (including the electrode):

.. code-block:: python

   from sisl_toolbox.btd import DownfoldElectrode

   # continued from the 1D chain
   left_down = DownfoldElectrode(left, H, [0], [0, 1, 2])
   right_down = DownfoldElectrode(right, H, [9], [7, 8, 9])
   # downfolded electrodes know their placement in the device region
   G = DeviceGreen(H, [left_down, right_down], [3, 4, 5, 6])

By default (``bulk=True``) the electrode indices in the downfolding region use
the bulk electrode Hamiltonian, including the electrode :math:`\eta`. The
broadening matrices of the downfolded self-energies include the absorption
from all imaginary parts in the downfolding region, so results may differ
by :math:`\mathcal O(\eta)` from a calculation without downfolding.


Bloch expansion and k-points
----------------------------

Electrodes with periodic directions are integrated over k-points, and may
be Bloch expanded to fit the device cell:

.. code-block:: python

   graphene = sisl.geom.graphene()
   H_elec = sisl.Hamiltonian(graphene)
   H_elec.construct([(0.1, 1.44), (0.0, -2.7)])

   # device: 2 cells along b (Bloch expanded electrode) and 6 along a
   H = H_elec.tile(2, 1).tile(6, 0)
   no = H_elec.no * 2  # orbitals per electrode layer
   left = BlochElectrode(sisl.RecursiveSI(H_elec, "-A"), bloch=(1, 2, 1), name="Left")
   right = BlochElectrode(sisl.RecursiveSI(H_elec, "+A"), bloch=(1, 2, 1), name="Right")
   G = DeviceGreen(
       H,
       [
           (left, {"orbitals": np.arange(no)}),
           (right, {"orbitals": np.arange(no) + 5 * no}),
       ],
   )
   bz = sisl.MonkhorstPack(H, [1, 11, 1])
   T = sum(w * G.transmission(0.5, "Left", "Right", k=k) for k, w in zip(bz.k, bz.weight))


Reference
---------

.. autosummary::
   :toctree: generated/

   BlockArray
   BlockArrayIndexer
   Electrode
   BlochElectrode
   DownfoldElectrode
   RealSpaceElectrode
   DeviceGreen
   PermuteBTD
   btd_partition
   downfold_partition
