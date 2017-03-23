File formats
============

`sisl` implements a generic interface for interacting with many different file
formats. When using the :doc:`command line utilities<scripts>` all these files
are accepted as input for, especially :ref:`script_sdata` while only those which
contains geometries (`Geometry`) may be used with :ref:`script_sgeom`.

In `sisl` any file is named a `Sile` to destinguish it from `File`.

Here is a list of the currently supported file-formats with the file-endings
defining the file format:

``xyz``
   `XYZSile` file format, generic file format for many geometry viewers.

``cube``
   `CUBESile` file format, real-space grid file format (also contains geometry)

``xsf``
   `XSFSile` file format, XCrySDen_ file format

``ham``
   `HamiltonianSile` file format, native file format for `sisl`

``dat``
   `TableSile` for tabular data

Below there is a list of file formats especially targetting a variety of DFT codes.

* BigDFT_
  File formats inherent to BigDFT_:

  ``ascii``
      `ASCIISileBigDFT` input file for BigDFT, currently only implements geometry

* SIESTA_
  File formats inherent to SIESTA_:

  ``fdf``
      `fdfSileSiesta` input file for SIESTA

  ``bands``
      `bandsSileSiesta` contains the band-structure output of SIESTA, with
      :ref:`script_sdata` one may plot this file using the command-line.

  ``out``
      `outSileSiesta` output file of SIESTA, currently this may be used to
      query certain elements from the output, such as the final geometry, etc.

  ``grid.nc``
      `gridncSileSiesta` real-space grid files of SIESTA. This `Sile` allows
      reading the NetCDF_ output of SIESTA for the real-space quantities, such
      as, electrostatic potential, charge density, etc.

  ``nc``
      `ncSileSiesta` generic output file of SIESTA (only `>=4.1`).
      This `Sile` may contain *all* real-space grids, Hamiltonians, density matrices, etc.
  
  ``TSHS``
      `TSHSSileSiesta` contains the Hamiltonian (read to get a `Hamiltonian` instance)
      and overlap matrix from a TranSIESTA_ run.

  ``TBT.nc``
      `tbtncSileSiesta` is the output file of TBtrans_ which contains all transport
      related quantities.

  ``TBT.AV.nc``
      `tbtavncSileSiesta` is the **k**-averaged equivalent of `tbtncSileSiesta`,
      this may be generated using `sdata siesta.TBT.nc --tbt-av<script_sdata>`.

  ``XV``
      `XVSileSiesta` is the currently runned geometry in SIESTA.

* VASP_
  File formats inherent to VASP:

  ``POSCAR``
      `POSCARSileVASP` contains the geometry of the VASP run.

  ``CONTCAR``
      `CONTCARSileVASP` is the continuation geometries from VASP.

* Wannier90_
  File formats inherent to Wannier90:

  ``win``
      `winSileW90` is the seed file for Wannier90. From this one may read the `Geometry`
      or the `Hamiltonian` if it has been output by Wannier90.
