# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once
we hit release version 1.0.0.


## [0.15.0] - YYYY-MM-DD

### Added
- `units` to `read_*` for some `Sile`s, #726
- "Hz", "MHz", "GHz", "THz", and "invcm" as valid energy units, #725
- added `read_gtensor` and `read_hyperfine_coupling` to `txtSileORCA`, #722
- enabled `AtomsArgument` and `OrbitalsArgument` to accept `bool` for *all* or *none*
- enabled `winSileWannier90.read_hamiltonian` to read the ``_tb.dat`` files
- `atoms` argument to `DensityMatrix.spin_align` to align a subset of atoms
  (only diagonal elements between the atoms orbitals)
- added an efficient neighbor finder, #393
- enabled reading DFTB+ output Hamiltonian and overlap matrices, #579
- `bond_order` for `DensityMatrix` objects, #507
- better error messages when users request quantities not calculated by Siesta/TBtrans
- functional programming of the basic sisl classes
  Now many of the `Geometry|Lattice|Grid.* manipulation routines which
  returns new objects, are subjected to dispatch methods.
  E.g.

      sisl.tile(geometry, 2, axis=1)
      geometry.tile(2, axis=1)

  will call the same method. The first uses a dispatch method, and a `SislError`
  will be raised if the dispatch argument is not implemented.
- `SparseCSR.toarray` to comply with array handling (equivalent to `todense`)
- enabled `Grid.to|new` with the most basic stuff
  str|Path|Grid|pyamg
- `Shape.translate`, to easily translate entire shape constructs, #655
- Creation of chiral GNRs (`kind=chiral` in `sisl.geom.nanoribbon`/`sisl.geom.graphene_nanoribbon`
  as well as `sisl.geom.cgnr`)
- Creation of [n]-triangulenes (`sisl.geom.triangulene`)
- added `offset` argument in `Geometry.add_vacuum` to enable shifting atomic coordinates
- A new `AtomicMatrixPlot` to plot sparse matrices, #668

### Fixed
- documentation links to external resources
- fixed `chgSileVASP.read_grid` for spinful calculations
- `txtSileOrca.info.no` used a wrong regex, added a test
- raises error when requesting isosurface for complex valued grids, #709
- some attributes associated with `Sile.info.*` will now warn instead of raising information
- reading matrices from HSX files with *weird* labels, should now work (*fingers-crossed*)
- `Atom(Z="1000")` will now correctly work, #708
- `AtomUnknown` now also has a default mass of 1e40
- changed `read_force_constant` to `read_hessian`, the old methods are retained with
  deprecation warnings.
- `pdosSileSiesta` plotting produced wrong spin components for NC/SOC
- `tqdm` changed API in 2019, `eta=True` in Notebooks should now work
- `SparseCSR` ufunc handling, in some corner cases could the dtype casting do things
  wrongly.
- fixed corner cases where the `SparseCSR.diags(offsets=)` would add elements
  in non-existing elements
- some cases of writing orthogonal matrices to TSHS/nc file formats #661
- `BDOS` from TBtrans calculations now returns the full DOS of all (Bloch-expanded)
  atoms
- `Lattice` objects now issues a warning when created with 0-length vectors
- HSX file reads should respect input geometry arguments
- enabled slicing in matrix assignments, #650
- changed `Shape.volume()` to `Shape.volume`
- growth direction for zigzag heteroribbons
- `BandStructure` points can now automatically add the `nsc == 1` axis as would
  be done for assigning matrix elements (it fills with 0's).

### Removed
- removed `Selector` and `TimeSelector`, they were never used internally

### Changed
- renamed `stdoutSileVASP` to `outcarSileVASP`, #719
- deprecated scale_atoms in favor of scale_basis in `Geometry.scale`
- changed default number of eigenvalues calculated in sparse `eigsh`, from 10 to 1
- `stdoutSileSiesta.read_*` now defaults to read the *next* entry, and not the last
- `stdoutSileSiesta.read_*` changed MD output functionality, see #586 for details
- `AtomNeighbours` changed name to `AtomNeighbor` to follow #393
- changed method name `spin_squared` to `spin_contamination`
- removed `Lattice.translate|move`, they did not make sense, and so their
  usage should be deferred to `Lattice.add` instead.
- `vacuum` is now an optional parameter for all ribbon structures
- enabled `array_fill_repeat` with custom axis, to tile along specific
  dimensions
- Importing `sisl.viz` explicitly is no longer needed, as it will be lazily
  loaded whenever it is required.


## [0.14.3] - 2023-11-07

### Added
- Creation of honeycomb flakes (`sisl.geom.honeycomb_flake`,
  `sisl.geom.graphene_flake`), #636
- added `Geometry.as_supercell` to create the supercell structure,
  thanks to @pfebrer for the suggestion
- added `Lattice.to` and `Lattice.new` to function the same
  as `Geometry`, added Lattice.to["Cuboid"]
- added `Atom.to`, currently only `to.Sphere()`
- enabled `Geometry.to|new.Sile(...)`
- added logging in some modules, to be added in more stuff to allow easier
  debugging.
- marked all `toSphere|toEllipsoid|...` as deprecated
- a simple extensionable method to add `Sile.info.<attr>` by exposing
  attributes through an object on each class.
  The _info_attributes_ contains a list of attributes that can be
  discovered while reading ascii files see #509

### Fixed
- fixed cases where `Geometry.close` would not catch all neighbours, #633

### Changed
- sisl now enforces the black style
- `Lattice` now holds the boundary conditions (not `Grid`), see #626
- Some siles exposed certain properties containing basic information
  about the content, say number of atoms/orbitals etc.
  These will be moved to `sile.info.<attr>` instead to reduce
  the number of methods exposed on each sile.


## [0.14.2] - 2023-10-04

### Fixed
- problems in the sisl.viz module got fixed

### Changed
- xarray is now a full dependency (this also implies pandas)


## [0.14.1] - 2023-09-28



## [0.14.0] - 2023-09-28

### Added
- added SISL_UNIT_SIESTA to select between legacy or codata2018 units (since Siesta 5)
  New default is codata2018, may create inconsistencies until Siesta 5 is widely adopted.
- added --remove to sgeom for removing single atoms
- added a EllipticalCylinder as a new shape
- added basis-enthalpy to the stdoutSiestaSile.read_energy routine
- added `read_trajectory` to read cell vectors, atomic positions, and forces from VASP OUTCAR
- slicing io files multiple output (still WIP), see #584 for details
  Intention is to have all methods use this method for returning
  multiple values, it should streamline the API.
- allowed xyz files to read Origin entries in the comment field
- allowed sile specifiers to be more explicit:
     - "hello.xyz{contains=<name>}" equivalent to "hello.xyz{<name>}"
     - "hello.xyz{startswith=<name>}" class name should start with `<name>`
     - "hello.xyz{endswith=<name>}" class name should end with `<name>`
        This is useful for defining a currently working code:

            SISL_IO_DEFAULT=siesta

- added environment variable ``SISL_IO_DEFAULT`` which appends a sile specifier
  if not explicitly added. I.e. ``get_sile("hello.xyz")`` is equivalent to
  ``get_sile("hello.xyz{os.environ["SISL_IO_DEFAULT"]}"``.
  Fixes #576
- added a context manager for manipulating the global env-vars in temporary
  locations. ``with sisl_environ(SISL_IO_DEFAULT=...)``
- enabled `Geometry.append|prepend` in `sgeom` command (reads other files)
- added `fdfSileSiesta.write_brillouinzone` to easily write BandLines to the fdf output,
  see #141
- added `aniSileSiesta` for MD output of Siesta, #544
- `mdSileOpenMX` for MD output of OpenMX
- `Atoms.formula` to get a chemical formula, currently only Hill notation
- unified the index argument for reading Grids, `read_grid`, this influences
  Siesta and VASP grid reads.
- `sisl.mixing`:
  - `AndersonMixer` enables the popular and very simple linear-like mixer
  - `StepMixer` allows switching between different mixers, for instance this
     enabled restart capabilities among other things.
  - Enabled composite mixers (simple math with mixers)
- `BrillouinZone.merge` allows simple merging of several objects, #537

### Changed
- updated the viz module, #476
- allowing ^ negation in order arguments for siles
- internal change to comply with scipy changes, use issparse instead
  of spmatrix, see #598
- netCDF4 is now an optional dependency, #595
- interface for Sparse*.nonzero(), arguments suffixed with 's'
- `stdoutSileVASP` will not accept `all=` arguments
- `stdoutSileVASP.read_energy` returns as default the next item (no longer the last)
- `txtSileOrca` will not accept `all=` arguments, see #584
- `stdoutSileOrca` will not accept `all=` arguments, see #584
- `xyzSile` out from sisl will now default to the extended xyz file-format
  Explicitly adding the nsc= value makes it compatible with other exyz
  file formats and parseable by sisl, this is an internal change
- default of `Geometry.translate2uc`, now only periodic axes are
  default to be moved
- all out files have been renamed to stdout to clarify they are
  user determined output file names, suggestion by @tfrederiksen
- bumped Python requirement to >=3.8
- orbitals `R` arguments will now by default determine the minimal radii
  that contains 99.99% of the function integrand. The argument now
  accepts values -1:0 which is a fraction of the integrand that the function
  should contain, a positive value will explicitly set the range #574
- Added printout of the removed couplings in the `RecursiveSI`
- `SuperCell` class is officially deprecated in favor of `Lattice`, see #95 for details
  The old class will still be accessible and usable for some time (at least a year)
- Enabled EigenState.wavefunction(grid) to accept grid as the initialization of
	the grid argument, so one does not need to produce the `Grid` on before-hand
- ``Geometry.rotate(only=)`` to ``(what=)``, this is to unify the interfaces across, #541
  Also changed the default value to be "xyz" if atoms is Not none
- ``tbtncSileTBtrans(only=)`` arguments are changed to (what=) #541
- `SelfEnergy.scattering_matrix` is changed to `SelfEnergy.broadening_matrix`
  ince the scattering matrix is an S-matrix usage.
  Also changed `se2scat` to `se2broadening` #529
- allow `BrillouinZone` initialization with scalar weights for all k-points #537
- `Geometry.swapaxes` and `SuperCell.swapaxes`, these are now more versatile by
	allowing multiple swaps in a single run, #539
- deprecated `set_sc`
- internal build-system is changed to `scikit-build-core`, the `distutils` will be
  deprecated in Python>=3.12 so it was a needed change.
  This resulted in a directory restructuring.


### Fixed
- fixed Mulliken calculations for polarized calculations due to missing copy, #611
- fixed single argument `ret_isc=True` of `close`, #604 and #605
- tiling Grid now only possible for commensurate grids (grid.lattice % grid.geometry.lattice)
- rare cases for non-Gamma calculations with actual Gamma matrices resulted
  in crashes #572
- `MonkhorstPack.replace` now checks for symmetry k-points if the BZ is using
  trs. Additionally the displacements are moved to the primitive point before
  comparing, this partly fixed #568
- spin-orbit Hamiltonians in `RealSpaceSE` and `RealSpaceSI`, fixes #567
- ufunc reductions on `SparseGeometry` where `axis` arguments reduces
  dimensionality
- interaction with pymatgen
- `fdfSileSiesta.includes` would fail when empty lines were present, #555
  fixed and added test
- Documentation now uses global references
- `Geometry.swapaxes` would not swap latticevector cartesian coordinates, #539


### toolbox.btd
#### Added
- calculation of scattering matrices


## [0.13.0] - 2023-1-18

### Added
- `Geometry.apply` apply functions to slices of data depending on the geometry
- enabled Gaussian and Slater type orbitals, #463
  Please give feedback!
- deltancSileTBtrans.merge allowing easy merging of several delta
  siles, #513
- implemented reading of output files from ORCA, #500
- HydrogenicOrbital is added for simple handling of 1-valence electron
  orbitals, #499
- Bohr radius to constants
- enabled ASCII siles to read from file-handles and buffers, #484
- enabled unit specification for lengths in cube-files
- added `kwargs` passed to eigenstate functions in `berry_phase`
  and `conductivity`
- ensured that non-orthogonal `transform` will copy over overlap matrix
  in case the matrix is only touching the non-overlap elements
- enabled dictionary entries for the `Atoms` initialization
  in place of `atoms` argument. Both in the list-like entry, or
  as the only argument.

### Fixed
- rare compiler bug, #512
- `within_inf` with periodic arguments, #511
- reading TranSiesta data from outSileSiesta
- regression from 80f27b05, reading version 0 HSX content, #492
- delta-files (netCDF) would always have diagonal components,
  this has now been removed since it only needs the elements with
  values
- Siesta sparse matrices could in some cases set wrong diagonal
  components
- too large energies in Siesta files could result in crash, #482
- orbital quantum numbers from HSX file was wrong in v1, #462
- corrected sign for spin-Y direction, `PDOS`, `spin_moment`, #486
- RealSpaceSI for right semi-infinite directions, #475
- tbtrans files now have a separate entry in the documentation

### Changed
- removed all deprecated routines, #495
- oplist now can do in-place operations on generators
- significant performance improvement for COOP calculations,
  thanks to Susanne Leitherer for discovering the issue
- changed argument order of ElectronState.COP
- index ordering of spin and coordinate quantities are now changed to
  have these as the first indices. This ensures consistency across
  return types and allows easier handling.
  Note that non-polarized PDOS calculations now has an extra dimension
  for coherence with non-colinear spin.  (see #501)
- ensured all units are now CODATA-2018 values
- `cell_length` changed to `cell2length` with new axes argument
- enabled orbitals up to the h-shell, #491
- swapped order of `honeycomb` (`graphene` derivatives)
  lattice vectors, to ensure the vectors are following right-hand-rule, #488
- changed DIIS solver to assume the matrix is symmetric (it is)
- tbtncSileTBtrans and its derivates has changed, drastically.
  This will accommodate changes related to #477 and #478.
  Now `*_transmission` refers to energy resolved transmissions
  and `*_current` reflects bias-window integrated quantities.
  The defaults and argument order has changed drastically, so
  users should adapt their scripts depending on `sisl` version.
  A check can be made, `if sisl.__version_tuple__[:3] >= (0, 13, 0):`
- To streamline argument order the `*_ACO[OH]P` routines have changed
  `elec` and `E` argument order. This makes them compatible with
  `orbital_transmission` etc.


## [0.12.2] - 2022-5-2

### Added
- enabled parsing geometry.in files from FHIaims
- added `batched_indices` for memory-reduced location of array values
- enabled manifold extractions `sisl.physics.yield_manifolds`
- enabled center of mass for periodic systems (chooses *best* COM)
- enabled returning the overlap matrix from `berry_phase`
- added `rocksalt` @tfrederiksen
- slab geometry creations, `fcc_slab`, `bcc_slab` and `rocksalt_slab` @tfrederiksen
- added `Geometry.translate2uc` to shift everything into the unit-cell @tfrederiksen
- added `Geometry.unrepeat` to reverse `repeat` calls (and to `sgeom`)
- added `SparseGeometry.unrepeat` to reverse `repeat` calls

### Fixed
- enabled reading HSX file version 1, #432
- major performance boost for reading GULP FC files
- cleaned mixing methods and decoupled the History and Mixers
- incorrect handling of `atoms` argument in `Geometry.center` calls

### Changed
- State*.outer corrected to the same interface as State*.inner
- all `sisl.geom` geometries are now calling `optimize_nsc` if needed
- `SparseGeometry.cut` -> `SparseGeometry.untile`
  - much faster
  - many more checks to warn about wrong usage
  - `cut` is now deprecated (removed in 0.13)
  - changed the --cut flag in `sgeom` to `--untile`, deprecated flag
- enabled in/out arguments to tbt siles (easier to remember meaning)


## [0.12.1] - 2022-2-10

### Added
- return spin moment from SCF output files of Siesta
- read_fermi_level to siesta.PDOS files

### Fixed
- MacOS builds
- `sdata` handling of siesta.PDOS* files, much more versatily now
- masking import of xarray
- Fixes to sisl.viz module related to 3.10 and other details


## [0.12.0] - 2022-1-28
### Added
- Geometry.sub_orbital is added
- BrillouinZone.volume enables easy calculation of volumes for BZ integrals
- State.sub|remove are now allowed to be done inplace
- State.derivative can now correctly calculate 1st and 2nd order derivatives #406
- Enabled discontinuity jumps in band-structures (pass points as None)
- COOP and COHP calculations for eigenstates
- inverse participation ration calculations (with arbitrary q)
- origin point for mirror functionality (Geometry)
- degenerate_dir for `velocity` directions
- `State.remove` complementary to `State.sub`
- copying Dispatchers for subclasses.
- dispatchers to `Shape`
- `Spin.spinor` to get number of spinor components
- `sc` argument to `xyzSile.read_geometry` for user defined cells
- tiling a State object, #354 and #355
- replacing atoms in SparseOrbital geometries #139
- direction now accepts `abc` and `xyz` keywords to retrieve vectors depending on direction input.
- replacing atoms in SparseOrbital geometries #139
- reading from STRUCT_* files (Siesta input/output) #308
- reading the SuperCell block from fdf
- reading PAO.Basis blocks from both out and fdf files, almost complete functionality #90
- generic `transform` method for matrix transformations
- doing ufunc.reduce on SparseCSR matrices; *wrong* values for e.g. np.prod, generally be **CAUTIOUS** with reduction operations
- transposing a SparseCSR matrix
- added pymatgen conversion (Geometry.to/new.pymatgen)
- atom indexing by shapes #337

### Fixed
- `sub_orbital` allows lists of orbitals
- `berry_phase` now works for non-orthogonal basis sets (uses Lowdin transformation)
  This may require sufficiently small dk for accurateness.
- `degenerate` argument for `conductivity` to enable decoupling of states
- BandStructure.lineark now always starts from 0
- reading coordinates from siesta.out when bands are calculated #362
- complex warning for spin_moment #360 and #363
- partially fixed #102 (`wavefunction` for `fxyz` outside box, related to #365 and how origin is interpreted in the code
- non-collinear PDOS plotting
- improvement for BandStructure setup, arguments more stringent
- several fixes for `sisl.viz`; #368, #376 and #382
- empty array handlings in `_sanitize_*` #370
- ensured AtomicOrbital can be instantiated without specifying m (default to 0)
- fixed bug when copying orbitals
- fixed reading atomic labels in xsf files #402
- fixed hpc parameters #403

### Changed
- order of arguments for `nanoribbon` it was not consistent with the others
- removed cell argument in `Geometry.sub`
- removed `Sile.exist`, refer to `Sile.file` which always will be a `pathlib.Path` instance
- `berry_phase` now uses the gauge=R convention, the code became much simpler
- `BrillouinZone.parametrize` function changed interface to allow more dimensions
- `EigenStateElectron.inner` does not use the overlap matrix by default, norm2 is for
  exactly this behaviour
- changed license to MPLv2 makes toolboxes easier to contribute under different license
- renamed origo to origin, see #365
- default parallel calculations are disabled
- changed `State.align_*` routines to align `self` rather than `other`
- doc fixes for recommending `python -m pip`

### Removed
- removed keywords align for State.inner|outer, manually use `align` if required
- removed method `State.expectation`

### toolbox.btd
#### Added
- calculation of scattering states and eigenchannels
- multiple variants of scattering state methods


## [0.11.0] - 2021-2-17

- **Major addition**: plotly backend for plotting and interaction with
  output. This is still a work in progress made by Pol Febrer.
  Many thanks to @pfebrer!

- Added unzip argument to BZ.apply methods to unzip multiple
  return values, also added documentation to reflect this

- Fixed reading data-arrays from Siesta-PDOS files

- Enabled minimization method for basis information and pseudo generation

- Enabled plotting grids using the command-line

- Bug in how non-colinear matrices are dealt with, now fixed
  Thanks to Xe Hu for discovering this.

- Allowed reading the geometry for supercell HSX files
  Atomic coordinates and nsc are determined from xij arrays

- Basic implementation of Hermitian construct.
  It now ensures a correct Hermitian matrix for simple cases

- Added more return from close/within, supercell offsets
  may be queried (ret_isc)

- Added more transposing functionality for spin matrices

- Fixed wfsxSileSiesta returning proper k-points if a geometry
  is passed (i.e. reduced k-points). Otherwise warns users

- Huge performance increase for finalizing very large structures

- Fixed writing %block in fdf files

- Enabled reading Fermi level from VASP DOSCAR files

- Cleaned siesta and VASP reading of completed jobs, #287

- added Geometry.new allowing easy type-lookups to convert to Geometry
  e.g. Geometry.new("RUN.fdf") and Geometry.new(ase_atoms) automatically
  figures out which method to call and how to interpret the objects.
  added Geometry.to allowing easy type-lookups to convert to other objects
  #282

- enabled calculating supercell matrices with phases, format=sc:<format>
  returns in supercell matrix form (no, no_s)

- removed support for int and long as matrix types, only float/complex

- Enabled `sgrid` to write tables of data

- Merged spin_orbital_moment(deleted) and spin_moment with
  optional argument project

- Enabled orbital resolved velocities

- Added outSileSiesta.read_energy to read final energies in a property-dict
  (works both as a property (`energy.fermi`) and a dictionary (`energy["fermi"]`)

- Ensured ghost atoms in Siesta are handled with separate
  class, AtomGhost, #249

- Using `si.RealspaceSI` with `unfold=(1,1,1)` no longer results in `nsc` on
    the given surface hamiltonian being set to `(1,1,1)`.

- Added calculation of isosurfaces, #246

- Added `sisl.WideBandSE` for self-energies with constant
  diagonals

- Enabled more user control over categories, #242

- Improved interpolation function for Grid's, and also
  added filters

- Bugfix for periodic directions for ASE conversion, #231

- Fixed tuples for `_sanitize_atoms`, #233

- Fixed reading correct unit from deltanc files, #234

- Enabled berry-phase calculations for NC+SOC, #235

- Added tiling to Grid, #238

- Added Atoms.group_data which nicely splits an array holding
  orbital information into atomic contributions (a list since
  each sub-list may be unequal in length)

- Many small bug-fixes and performance improvements


## [0.10.0] - 2020-6-9

- Exposing sisl_toolbox as a module for external contributions
  Now stuff contributed from 3rd parties can easily be included
  in a toolbox which is a separate module.

- Changed asarray (as*) methods for SparseGeometry
  Now we have a dispatch class which enables one
  to store the behaviour as variables and then post-process

- Using `*.geom` or `geometry.atom` is now deprecated, use
  `*.geometry` and `geometry.atoms` instead (respectively)

- Added spin-rotation for density matrices, this will
  enable sisl to manipulate DM and write them for
  Siesta calculations

- Enabled all numpy.ufuncs (np.exp(H))

- Added nanoribbons construction (@tfrederiksen)

- Internal change to pathlib for files and paths

- Added velocity calculations for NC+SOC Hamiltonians

- Sparse pattern transposes of non-full matrices, fixed bug

- Changed Geometry.sort to be more diverse (this may break old code)
  This new way of sorting is way more flexible and allows very fine
  control, fixes #191, #197

- Added a bilayer geometry which can create twisted bilayers #181, #186

- Enabled VASP `*CAR` files to write/read dynamic specifications #185

- Enabled `xarray.DataArray` returning from BrillouinZone objects #182

- Several improvements to outSileSiesta.read_scf #174, #180

- A huge performance increase for data extraction in tbtncSileTbtrans
  (thanks to Gaetano Calogero for finding the bottleneck)

- Added preliminary usage of Mixers, primarily intented for extending
  sisl operations where SCF are used (may heavily change).

- Lots of small bug-fixes

- Now sisl is Python >=3.6 only, #162

This release was helped by the following committers (THANKS):

- Thomas Frederiksen
- Pol Febrer
- Jonas Lundholm Bertelsen
- Bernhard Kretz


## [0.9.8] - 2020-2-10

- fixed #160 by removing all(?) TRS k-points in a Monkhorst Pack grid

- fixed repeat for SparseGeometryOrbital #161

- changed lots of places for einsum in electron.py for increased performance

- added AHC conductivity calculations `conductivity` (not tested)

- added Berry curvature calculations `berry_flux` (not tested)

- added Overlap class to directly use overlap matrices (without having a
  second matrix).

- fixed geometry align issue when reading geometries from Siesta output #153

- fixed pickling a sparse matrix #150

- Fixed TSV.nc write-out for grid files (see poisson_explicit.py)

- Fixed fermi level calculation for non-polarized calculations

- Reverted Fermi calculation routine for more stable implementation

- fixed DynamiclMatrix reading for number of atoms not divisable by 4 #145

A huge thanks to Jonas L. B. for fixes, suggestions etc.


## [0.9.7] - 2019-9-26

- Bug-fix for reading geometries in outSiesta

- Enabled reading the fermi level from the output, fixes #126

- Enabled Siesta STM and STS output

- Fixed an inheritance issue in `axsfSile` which meant it was unusable until
  now

- Maintenance fix for looping sparse matrices.
  Now the default is to loop the sparse non-zero elements.
  If one wishes to loop all atoms/orbitals one should use `iter_orbitals()`
  NOTE: This *may* break some codes if they used loops on sparse matrices

- Fixed reading VASP CAR files with constraints (thanks to T. Frederiksen)

- Added `overlap` method to `Geometry` to find overlapping atoms
  between two geometries.

- Added Siesta LDOS charge handling

- Changed edges method to not exclude it-self by default.
  This is because it is not intuitive given the default exclude=None

  Note: this may break compatibility with other software/scripts.

- Added mulliken charge calculations and orbital angular momentum
  for SO DM, fixes #136

- Fixed reading overlap matrix in conjunction with DM from fdf-sile

- Performance increase for the real-space self-energy calculations

- Fixed transposing of the spin-box for NC and SO matrices

- Enabled TRS handler for SO matrices, fixes #125

- Enabled better b-casting assignments for sparse-matrices, fixes #134

- Upgraded documentation to a layout that obeys numpydoc

- Fixed reading ASE xyz outputs, thanksto JL. Bertelsen,

- Fixed a typo in fdf reading onlyS, thanks to JL. Bertelsen, #135

- Enabled reading arbitrary self-energy by requesting an energy and k-point
  from TSGF files.

- Upgraded handling of TBT.*.nc files to conform with the >=Siesta-4.1-b5
  releases where all files contain the same device + electrode meta-data.

- Deprecated TBTGFSileTBtrans (use tbtgfSileTBtrans instead)

- Forced align=False in inner such that users should take care of this

- Added align_norm to swap states such that they more or less
  correspond to the same band (which should have a closer residual
  for on-site coefficients).

- Removed norm2 and made norm equal to norm2 for states. This is
  the more natural thing, besides. Doing norm() ** 0.5 shouldn't be
  too much of a problem.


## [0.9.6] - 2019-6-18

- Officially added real-space self-energy calculations

- Cleaned TBT vs. PHT for class name structures

- Bugfix for reading MD output from Siesta out-files #130

- Bugfix for tbtse files when requesting pivoting indices using this
  combination ``in_device=True, sort=False`` which in most cases
  return wrong indices, thanks to J. Bertelsen for bug-find!

- Added several routines for retrieving transposed coupling elements.
  When having connections `i -> j` it may be beneficial to easily get
  the transposed connection `j -> i` by taking into account the
  supercell. `Geometry.a2transpose` enables this functionality making
  construct functions much simpler when having edges/boundaries.

- Bug-fix for reading white-space prefixed keywords in XSF files, #127

- Performance increase for self-energy calculations for very small
  systems

- Huge memory reduction for `Geometry.o2a` with very large system

- Enabled pickling on `BrillouinZone` objects

- Added `spin_moment` to `Hamiltonian`

- Removed ``rotate[abc]`` methods since they were cluttering the name-space
  Codes should simply replace with:

     >>> geometry.rotate(angle, geometry.cell[{012}, :], *)

  for the same effect.

- Finally removed deprecated `write_geom` from the API

- Enabled calculation of ``<S^2>`` for spin-polarized calculations, this
  may be used for calculating spin-contaminations

- added checks for `SparseCSR` to disallow out-of-bounds keys

- Bug fixed for reading POSCAR files from VASP (only when multiple species are
  used in a non-ordered fashion)

- added `sisl` command line utility, it is exactly the same as `sdata`

- Enabled pickling sparse matrices, this allows dask usage of sparse matrices

- Performance increase for sparse matrix handling

- Fixed a problem with Fortran IO + Jupyter notebooks, now the file-handles
  are re-used if a code block is terminated before closing the file

- Added `SparseOrbital` `append` + `transpose`
  This enables appending Hamiltonian's (append) and makes hermiticity
  checks possible (transpose)

- Enabled complex averaged calculations using `oplist`
  The `oplist` object is a container allowing inter-element operations

      >>> l1 = oplist([0, 1])
      >>> l2 = oplist([2, 3])
      >>> l = l1 + l2
      >>> print(l)
      [2, 4]

  This is extremely handy for `BrillouinZone.asaverage`/`assum` when calculating
  multiple values using `eigenstate` objects.

- Added reflection calculation to `tbtncSileTBtrans`

- Added more distribution functions (step and heaviside)

- Removed numpy deprecated class numpy.matrix, now everything is array

- Removed possibility of using `kavg=list(...)` due to complexity, now single
  `kavg` requests are *not* k-averaged.

- Bugfix in calculating `shot_noise`, `noise_power` and `fano` factors in `tbtncSileSiesta`
  They were only correct for Gamma-point calculations

- Fixed `*.EIG` `sdata` processing when using `--dos`

- Fixed reading geometries from grids from VASP (grid values were correct)

- Toolboxes:

  * Added a toolbox to calculate the Poisson solution for arbitrary
    electrodes for TranSiesta


## [0.9.5] - 2018-11-12

- Fixed temperature for phonon output pht*nc files

- Added tbtprojncSileTBtrans sile for analyzing projected transmissions

- Removed deprecated dhSileTBtrans

- Bug fix for binary grid files with Siesta and also reads using fdf-files

- Changed default self-energy eta values to 1e-4 eV

- Added Zak-phase calculations (thanks to T. Frederiksen)

- Updated lots of State methods

- added Bloch expansion class which can expand any method

- self-energy calculations:
  - Much faster
  - enabled left/right self-energies in one method

- fixed AtomicOrbital copies

- enabled TSGF reads

- Added noise-power calculations for TBT.nc files

- Fixed TBT.SE.nc files, units and scattering matrix retrieval

- added more VASP files


## [0.9.4] - 2018-8-4

- Fixes for the GULP dynamical matrix reads

- Enabled preliminary reads of OpenMX input file

- Enabled DOS calculation for the eigenvalue files

- Added Berry-phase calculation for orthogonal basis sets

- Added velocity calculation of electronic eigenstates

- Enabled effective mass tensor in electronic eigenstates (un-tested)

- High performance increase by moving stuff to Cython.

- Added Siesta interaction tutorials

- Added orthogonality checks when reading sparse matrices

- Lots of fixes for the fdf-file

- Added Mulliken calculation in DensityMatrix/EnergyDensityMatrix

- Enabled reading phonons from FC files

- Added named-groups which enables accessing groups of atoms by names.

      Geometry['Hello'] = [2, 3, 4]

- Changed Hessian to DynamicalMatrix to clarify the units

- Added new units class to handle complex units.

- Enabled a Phonon class to calculate group velocities of phonons, DOS and PDOS,
  displacements

- Bug-fixes for Siesta binary writes, now the supercell format is *always*
  Siesta compliant.

- Enabled replacing k-points in MonkhorstPack grids.

- Enabled calculation of band-velocities from eigenstates

- Made better progress-bars. Using eta= now relies on tqdm
  It is however still an optional dependency.

- Fixed Gamma-point periodic wavefunction storage.
  Creating grids with wave-functions is fully functional
  for arbitrarily big supercells.

- BrillouinZone objects:

  - Renamed PathBZ to BandStructure

  - Renamed MonkhorstPackBZ to MonkhorstPack

  - Enabled MonkhorstPack symmetry. This will reduce the number of
    k-points to roughly half (note symmetry is by default *on*)

  - Forced MonkhorstPack to create a k-grid which is Gamma centered

- Shapes (backwards compatibility broken)

  - Complete re-write of Shapes

  - Skewed Cuboids, Ellipsoids

  - Set combinations of Shapes (unions, difference sets, etc.)

- Grid

  - Enabled Grid.index for shapes.

  - Fixed grid initialization to create grid spacings fixed by a real.
    I.e. the voxel spacing.


        >>> Grid([10, 10, 10]) # 10 points per lattice vector
        >>> Grid(0.1) # 0.1 Angstrom spacing

  - Enabled plotting wavefunctions on grids.

  - Enabled plotting charge density on grids.

- Enabled tqdm usage for progressbar. It is fast and easy to use
  and a small requirement. (still optional)

- Added intrinsic Sisl exceptions which will be used throughout
  (at some point)

- Removed deprecated TightBinding class (use Hamiltonian instead)

- Added many SislWarning raises which are used to notify the user of
  potentially important things (say if sisl knows there should be a unit
  associated but it couldn't find it).

- Added TSDE file reading in sisl.

- Siesta reading of grid-related data is now much *smarter*. It will
  try and recognize the units of the data so the units become sisl
  intrinsics (Ry -> eV, Bohr -> Ang, etc.).
  This means that typically one does not need to do manual unit-conversion.
  There are however a few cases where sisl cannot figure out the
  units. Particularly if the files are renamed.

- Added a new class EigenSystem which holds information regarding
  eigenvalues and eigenvectors.

  - Currently an EigenState class is also enabled which can currently
    be used to calculate wavefunctions, DOS, PDOS and more to come.

- Fixed lots of bugs in fdf-reading quantities.
  Now one is also able to read Hamiltonian and other physical
  quantities from the fdf-object directly. There is pre-defined
  orders of which files to read from if there are multiple files
  eligeble.

  Reading the geometry now defaults to the fdf file, but one can query
  the output files by a boolean.

- Enabled PDOS calculations for the Hamiltonian. Together
  with the MonkhorstPack class one can easily calculate
  k-averaged PDOS quantities.

- Fixed cube reading/writing of multi-column data.

- Added siesta PDOS xml parsing, currently this is only scriptable
  but it manages easy extraction of quantities without the PDOSXML utility.
  This also enables retrieving the PDOS as an xarray.DataArray.

- Fixed a bug in writing XV files (only for -100/-200 species)

- TBtrans / TBT.nc file:

  - Added TBT.SE.nc file to enable easy extraction of self-energies
    from TBtrans

  - Added COOP and COHP extraction to the TBT.nc files.

  - Added DM and ADM extraction to the TBT.nc files.

  - Reorganized the TBtrans netcdf files (internal changes only)

  - Added shot-noise calculation (and Fano factor). Currently un-tested!

- Several added files


## [0.9.3] - 2018-8-4


## [0.9.2] - 2017-10-25

- Various minor bug-fixes


## [0.9.1] - 2017-10-23

- Fixed scaling of bond-currents in case 'all' is used, makes comparison
  with '+' and '-' easier.

- Updated defaults in bond_current to '+' such that only forward
  going electrons are captured.

- Updated defaults in vector_current to '+' such that only forward
  going electrons are captured.


## [0.9.0] - 2017-10-16

- Enabled reading a tabular data-file

- Lots of updates to the spin-class. It should now be more coherent.

- Added rij and Rij to the sparse_geometry classes to extract orbital or
  atomic distance matrices (returing the same sparsity pattern).

- Renamed `which` keyword in `Geometry.center` to `what`

- Added uniq keyword to o2a for better handling of orbitals -> atoms.

- Fixed a performance bottleneck issue related to the `scipy.linalg.solve`
  routine which was changed since 0.19.0.

- Changed internal testing scheme to `pytest`

- Lots of bug-fixes here and there

- Geometry files used in the command-line has updated these arguments:

   - tile
   - repeat
   - rotate

  The order of the arguments are interchanged to be similar to the
  scripting capabilities.

  Also fixed an issue related to moving atoms into the unit-cell.

- Enabled deleting supercell elements of a sparse Geometry. This
  will come in handy when calculating the self-energies and Green
  functions. I.e. Hamiltonian.set_nsc(...) will truncate entries
  based on the new supercell.

- Preliminary testing of reading Siesta binary output (.RHO, .VT, etc.)

- Added parsing the Siesta EIG file (easy plotting, reading in Python)

- Changed interface for BrillouinZone objects.
  Now a BrillouinZone accepts any object which has cell/rcell entries.
  Any function call on the BrillouinZone object will transfer the call to the
  passed object and evaluate that function for all k-points in the BrillouinZone.

- sisl.io.siesta.tbtrans

  * Added current calculator to TBT.nc sile to calculate the current as TBtrans
    does it (this requires the latest commit in SIESTA which defines the
    chemical potential and electronic structure of *all* electrodes).

  * Bug-fixes for TBT.nc sile, the bond-currents for multi-orbital systems
    were in some cases wrong.

  * Huge performance increase for TBT.nc data processing. Now the majority
    of routines are based on array-indexing, rather than sparse loops.

  * Changed the DOS retrieval functions to be more flexible. The default is
    now to return the summed DOS across the selected atoms.

  * Added a TBTGFSileSiesta which enables one to create _external_ self-energies
    to be read in by TBtrans (complete electrode control).

  * Added `deltancSileSiesta` as a replacement for `dHncSileSiesta`, TBtrans 4.1b4
    will have two delta terms, dH (adds to bond-currents) and dSigma (does not
    add to bond-currents).

  * BEWARE, lots of defaults has changed in this release.

- Hamiltonian.tile is now even faster, only utilizing
  intrinsic numpy array functionality.

- Greatly speeded up Hamiltonian.remove/sub functions.
  Now there are no for-loops in the remove/sub routines which
  will greatly increase performance.
  It will now be much faster to generate the Hamiltonian for
  a small reference cell, tile/repeat it, remove atoms.


## [0.8.5] - 2017-7-21

- Added the following routines:

  * `SuperCell.fit` routine to determine a new supercell object
    such that a given set of coordinates are all within AND
    periodic in the new supercell.
  * `SuperCell.parallel` to check whether two objects have parallel
    latticevectors.
  * `Geometry.distance` returns a list of distances from a given
    set of atoms. I.e. to determine a set of distances required for
    a subsequent close call. This routine can also be used to group
    neighbouring atoms in a common fashion.
  * `Geometry.optimize_nsc` loops all atoms and minimizes `nsc` in case
    one is not sure of the interaction range.
  * `Hamiltonian.shift` enables the shift of the entire electronic structure
    Fermi-level.
  * Added new flag to `Hamiltonian.Hk` routines
    ``format={'csr', 'array', 'dense', ...}``
    to ensure a consistent return of the data-type.

- Bug fix for dHncSileSiesta for multiple levels.

- Performance boost for the sub and remove functions for the
  Hamiltonian objects. Instead of creating the geometry first,
  it may now be much faster to generate the small Hamiltonian,
  tile -> repeat -> sub -> remove.

- Performance boost for the tile and repeat functions for the
  Hamiltonian objects. They are now the preferred method for creating
  large systems.

- Bug fixed when having extremely long atomic ranges and using tile/repeat.
  The number of supercells was too large.
  It did not affect anything, but it was inconsistent.

- Enabled reading the density matrix and energy density matrix from siesta.

- Addition of a PerformanceSelector class which enables a dynamic
  selection of the best routine.

  Currently this is enabled in the SparseOrbitalBZ class where
  constructing a matrix @ k can be done in numerous ways.

- Bug fixed in supercell specification of the Hamiltonian:

      >>> H[io, jo, (-1, 0, 0)]

  now works in all cases.

- Spin-orbit H(k) has been enabled

- Fixed reading the <>.nc file from SIESTA, the non-zero elements count was
  wrong.

- Now H(k) has been tested for non-colinear and spin-orbit coupling and
  one can now use sisl to perform non-colinear and spin-orbit coupling
  calculations.

- API change, all dR keywords has been changed to R for consistency and
  reduction of ambiguity.
  Also the `Atoms.dR` is now referred to as `Atoms.maxR()` to indicate
  its meaning.

  This may break old scripts if one use the `dR` keyword in arguments.


## [0.8.4] - 2017-6-11

- Added BrillouinZone class to easily create BrillouinZone plots etc.
  When calculating the eigenspectrum of a Hamiltonian one may pass
  the BrillouinZone object instead of the k-point to retrieve all
  eigenvalues for the k-points in the BrillouinZone object.
  Say for a PathBZ one can now easily retrieve the band-structure.

- Enabled specification of Hamiltonian connections across supercells via
  a tuple index (as the last index):

      >>> H[io, jo, (-1, 0, 0)]

  Thus connecting orbital `io` and `jo` across the -1 first lattice vector

- Enabled tbtrans files to attach a geometry (to get correct species).

- API change of:

      read/write_geom => read/write_geometry
      read/write_sc => read/write_supercell
      read/write_es => read/write_hamiltonian

  Moved `quantity` to `physics`.

- Enabled slice deletion in `SparseCSR`

  Enabled eliminate_zeros() to remove unneeded values.

- Added ScaleUp compatibility. sisl now acceps ScaleUp files which is
  a 2nd principles code for large scale calculations using Wannier
  functions.

- Added Hamiltonian.sub/remove/tile for easy extension of Hamiltonian
  without having to construct the larger geometries.
  This should speed up the creation of really large structures
  as one may then simply "update" the Hamiltonian elements subsequently.


## [0.8.3] - 2017-4-5

- Fixed bug in __write_default (should have been _write_default)

- API change in `close` functions, now ret_coord => ret_xyz,
  ret_dist => ret_rij

- Added `SparseCSR` math operations work on other `SparseCSR` matrices
  Thus one may now do:

      >>> a, b = SparseCSR(...), SparseCSR(...)
      >>> aMb, aPb = a * b, a + b

  Which makes many things much easier.
  If this is used, you are encouraged to assert that the math is correct.
  Currently are the routines largely untested. Assistance is greatly appreciated
  in creating tests.

- Geometries now _always_ create a supercell. This was not the case when
  an atom with no defined orbital radius was used. Now this returns a
  supercell with 10 A of vacuum along each Cartesian direction.


## [0.8.2] - 2017-3-31

- Fixed reading _hr.dat from Wannier90, now the band-structure of
  SrTiO3 (Junquera's test example) is correct.

- Speeded up tbtrans.py analyzing methods enourmously by introducing
  faster sparse iterators. Now one can easily perform data-analysis on
  systems in excess of 10.000 atoms very fast.

- Added the TBT.AV.nc file which is meant to be created by `sisl` from
  the TBT.nc files (i.e. create the k-averaged output).
  This enables users to run tbtrans, create the k-averaged output, and
  then delete the old file to heavily reduce disk-usage.

  An example:

      tbtrans RUN.fdf > TBT.out
      sdata siesta.TBT.nc --tbt-av
      rm siesta.TBT.nc

  after this `siesta.TBT.AV.nc` exists will all k-averaged quantites.
  If one is not interested in k-resolved quantities this may be very interesting.

- Updated the TBT.nc sile for improved readability.

- Easier script data-extraction from TBT.nc files due to easier conversion
  between atomic indices and pivoting orbitals.

  For this:

  * a2p
    returns the pivoting indices for the given atoms (complete set)
  * o2p
    returns the pivoting indices for the given orbitals

  * Added `atom` keyword for retrieving DOS for a given set of atoms

  * `sdata` and `TBT.nc` files now enable the creation of the TBT.AV.nc file
    which is the k-averaged file of TBT.nc

- Faster bond-current algorithms (faster iterator)

- Initial template for TBT.Proj files for sdata processing

- Geometry:

  * Enabled multiplying geometries with integers to emulate `repeat` or
    `tile` functions:

        >>> geometry * 2 == geometry.tile(2, 0).tile(2, 1).tile(2, 2)
        >>> geometry * [2, 1, 2] == geometry.tile(2, 0).tile(2, 2)
        >>> geometry * [2, 2] == geometry.tile(2, 2)
        >>> geometry * ([2, 1, 2], 'repeat') == geometry.repeat(2, 0).repeat(2, 2)
        >>> geometry * ([2, 1, 2], 'r') == geometry.repeat(2, 0).repeat(2, 2)
        >>> geometry * ([2, 0], 'r') == geometry.repeat(2, 0)
        >>> geometry * ([2, 2], 'r') == geometry.repeat(2, 2)

    This may be considered an advanced feature but useful nonetheless.

  * Enabled "adding" geometries in a similar way as multiplication
    I.e. the following applies:

        >>> A + B == A.add(B)
        >>> A + (B, 1) == A.append(B, 1)
        >>> A + (B, 2) == A.append(B, 2)
        >>> (A, 1) + B == A.prepend(B, 1)

  * Added `origo` and `atom` argument to rotation functions. Previously this could be
    accomblished by:

        rotated = geometry.move(-origo).rotate(...).move(origo)

    while now it is:

        rotated = geometry.rotate(..., origo=origo)

    The origo argument may also be a single integer in which case the rotation
    is around atom `origo`.

    Lastly the `atom` argument enables only rotating a sub-set of atoms.

  * Geometry[..] is now calling axyz if `..` is pure indices, if it is
    a `slice` it does not work with super-cell indices

  * Added `rij` functions to the Geometry for retrieving distances
    between two atoms (`orij` for orbitals)

  * Renamed iter_linear to iter

  * Added argument to iter_species for only looping certain atomic indices

  * Added iter_orbitals which returns an iterator with atomic _and_ associated
    orbitals.
    The orbitals are with respect to the local orbital indices on the given atom

    ```
    >>> for ia, io in Geometry.iter_orbitals():
    >>>     Geometry.atom[ia].R[io]
    ```

    works, while

    ```
    >>> for ia, io in Geometry.iter_orbitals(local=False):
    >>>     Geometry.atom[ia].R[io]
    ```

    does not work because `io` is globally defined.

  * Changed argument name for `coords`, `atom` instead of the
    old `idx`.

  * Renamed function `axyzsc` to `axyz`

- SparseCSR:

  * Added `iter_nnz(i=None)` which loops on sparse elements connecting to
    row `i` (or default to loop on all rows and columns).

  * `ispmatrix` to iterate through a `scipy.sparse.*_matrix` (and the `SparseCSR`
    matrix).

- Hamiltonian:

  * Added `iter_nnz` which is the `Hamiltonian` equivalent of `SparseCSR.iter_nnz`.
    It enables explicit looping on atomic couplings, or orbital couplings.
    I.e. one may specify a subset of atoms or orbitals to loop over.

  * Preliminary implementation of the non-collinear spin-case. Needs testing.


## [0.8.1] - 2017-2-23

- Fix a bug when reading non-Gamma TSHS files, now the
  supercell information is correct.

- tbtncSileSiesta now distinguishes between:
  electronic_temperature [K] and kT [eV]
  where the units are not the same.

- Fixed TBT_DN.nc TBT_UP.nc detection as a `Sile`

- Added information printout for the TBT.nc files

       sdata siesta.TBT.nc --info

  will print out what information is contained in the file.

- `Atoms` overhauled with a lot of the utility routines
  inherent to the `Geometry` object.
  It is now much faster to perform operations on this
  object.

- The FDF sile now allows setting and retrieving variables
  from the fdf file. Hence one may now set specific
  fdf flags via:

       sdata RUN.fdf --set SolutionMethod Transiesta

- Changed default output precision for TXT files to .8f.
  Additionally one may use flag `--format` in `sgeom` to
  define the precision.

- `Shape` have been added. There are now several Shapes
  which may be used to easily find atoms within a given Shape.
  This should in principle allow construction of very complex Shapes
  and easier construction of complex Hamiltonians


## [0.8.0] - 2017-1-7

This release introduces many API changes and a much more stream-lined
interface for interacting with sisl.

You are heavily encouraged to update your distribution.

Here is a compressed list of changes:

- sdata is now an input AND output dependent command.
  It first reads the input and output files, in a first run, then
  it determines the options for the given set of files.
  Secondly, the sdata command uses "position dependent" options.
  This means that changing the order of options may change the output.
- tbtncSile

  * Correct vector currents (for xsf files)
  * bug-fix for Gamma-only calculations
  * returned DOS is now correctly in 1/eV (older versions returned 1/Ry)
  * fixed sdata atomic[orbital] ranges such that, e.g. `--atom [1-2][3-5]`
    (for atom 1 and 2 and only orbitals 3, 4 and 5 on those atoms.)
  * DOS queries now has an extra argument (E) which returns only for the
    given energy.
  * When storing tables in sdata this now adds information regarding
    each column at the top (instead of at the bottom).
    Furthermore, the information is more descriptive

- Changed all `square` named arguments to `orthogonal`
- Added nsc field to xyz files (to retain number of supercells)
- Added `move` function for geometry (same as translate)
- Added `prepend` function, equivalent to `append`, but adding the
  atoms in the beginning instead of the end
- Fixed many bugs related to the use of Python-ranges (as opposed to numpy ranges)
- SparseCSR now enables operations:

      a = SparseCSR(...)
      a = a * 2 + 2

  is now viable. This enables easy scaling, translation etc. using the
  sparse matrix format (very handy for magnetic fields).
- Enabled `del` for SparseCSR, i.e. `del SparseCSR(..)[0, 1]` will
  remove the element, completely.
- Enabled reading of the TSHS file from SIESTA 4.1, now we may easily interact
  with SIESTA.
- Moved version.py to info.py
- Moved scripts to `entry_points`, this makes scripts intrinsic in the module
  and one may import and use the commands as their command-line equivalents.
- Hamiltonian.construct now takes a single argument which is the function
  for the inner loop.
  The old behaviour may be achieved by doing either:

      >>> func = Hamiltonian.create_construct(R, param)
      >>> Hamiltonian.construct(func)

  or

      >>> Hamiltonian.construct((R, param))

- The atoms contained in the Geometry are now not duplicated in case of many
  similar Atom objects. This should reduce overhead and increase throughput.
  However, the efficiency is not optimal yet.
- Added many more tests, thus further stabilizing sisl

  I would really like help with creating more tests!
  Please help if you can!

<!--
# Local Variables:
# mode: text
# comment-column: 0
# tab-width: 2
# End:
-->
