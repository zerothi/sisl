# sisl #

[![Build Status](https://travis-ci.org/zerothi/sisl.svg?branch=master)](https://travis-ci.org/zerothi/sisl)
[![DOI for citation](https://zenodo.org/badge/doi/10.5281/zenodo.597181.svg)](http://dx.doi.org/10.5281/zenodo.597181)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Join the chat at https://gitter.im/sisl-tool/Lobby](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/sisl-tool/Lobby)
<!--- [![Documentation on RTD](https://readthedocs.org/projects/docs/badge/?version=latest)](http://sisl.readthedocs.io/en/latest/) -->
[![Install sisl using PyPI](https://badge.fury.io/py/sisl.svg)](https://badge.fury.io/py/sisl)
[![Install sisl using conda](https://anaconda.org/conda-forge/sisl/badges/version.svg)](https://anaconda.org/conda-forge/sisl)
[![Checkout sisl code coverage](https://codecov.io/gh/zerothi/sisl/branch/master/graph/badge.svg)](https://codecov.io/gh/zerothi/sisl)
[![sisl Codacy](https://api.codacy.com/project/badge/Grade/8b0a94eba3ec4434a676883b40a08850)](https://www.codacy.com/app/nickpapior/sisl?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=zerothi/sisl&amp;utm_campaign=Badge_Grade)
[![Donate money to support development of sisl](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=NGNU2AA3JXX94&lc=DK&item_name=Papior%2dCodes&item_number=codes&currency_code=EUR&bn=PP%2dDonationsBF%3abtn_donate_SM%2egif%3aNonHosted)

The [API documentation][sisl-api] can be found [here][sisl-api].

The sisl toolbox provides a simple API for manipulating, constructing and creating tight-binding matrices 
in a standard and uniform way.  
Secondly, it provides easy interfaces for creating and calculating band-structures of
simple tight-binding models as well as interfacing to more advanced DFT utilities.

sisl may also be used together with the [ASE][ase] environment.

sisl provides an interface to [TBtrans][tbtrans] and enables the calculation of
transport using the non-equilibrium Green function method and easily allows calculation of tight-binding
systems of millions of atoms.

## Downloading and installation ##

Installing sisl using PyPi or Conda is the easiest:

    pip install sisl
    pip install sisl[analysis] # also installs tqdm and xarray
    # or
    conda install -c conda-forge sisl

If performing a manual installation, these packages are required:

   - A C- and fortran-compiler
   - __six__
   - __numpy__ (1.10 or later)
   - __scipy__ (0.18 or later)
   - __netCDF4__
   - __setuptools__
   - __pyparsing__
   - __matplotlib__, encouraged optional dependency
   - __tqdm__, encouraged optional dependency
   - __xarray__, optional dependency

You are encouraged to also install `matplotlib` and `tqdm` for plotting utilities and
progress-bar support.

Subsequently manual installation may be done using this command:

    python setup.py install --prefix=<prefix>

If trying to install without root access, you may be required to use this command:

    python setup.py install --user --prefix=<prefix>


### Testing installation ###

After installation it may be a good idea to check that the tests shipped with sisl pass.
To do this the `pytest` package is necessary. Then run

    pytest --pyargs sisl

which will run the shipped sisl test-suite.

## Usage ##

If used to produce scientific contributions, please use this [DOI][doi] for citation. It is recommend to specify the version of sisl in combination of this citation:

    @misc{zerothi_sisl,
      author       = {Papior, Nick R.},
      title        = {sisl: v<fill-version>},
      year         = {2018},
      doi          = {10.5281/zenodo.597181},
      url          = {https://doi.org/10.5281/zenodo.597181}
    }


### Scripts ###

sisl contains a utility to easily convert geometries from existing files
to other formats. After installation the executable `sgeom` is available which
enables the conversion between all formats accessible as `Sile` objects.

To convert a SIESTA FDF file to `xyz` _and_ an `XV` file one does

    sgeom siesta.fdf geom.xyz geom.XV

Try `sgeom -h` for additional features such as repeating the structure.


### Geometry manipulation ###

sisl contains a class for manipulating geometries in a consistent and easy
way. Without using any other feature this enables you to easily create and
manipulate structures in a consistent manner. 

For instance to create a huge graphene flake

    sq3h  = 3.**.5 * 0.5
    sc = SuperCell(np.array([[1.5, sq3h,  0.],
                             [1.5,-sq3h,  0.],
                             [ 0.,   0., 10.]],np.float64) * 1.42,
                             nsc=[3,3,1])
    gr = Geometry(np.array([[ 0., 0., 0.],
                            [ 1., 0., 0.]],np.float64) * 1.42,
                  atom=Atom(Z=6, R=1.42), sc=sc)
    huge = gr.tile(100, axis=0).tile(100, axis=1)

Which results in a 20,000 atom big graphene flake.

Several basic geometries are intrinsically available

    from sisl.geom import *

    # Graphene basic unit-cell
    g = graphene()
    # SC crystal structure
    g = sc(<lattice constant>, <Atom>)
    # BCC crystal structure
    g = bcc(<lattice constant>, <Atom>)
    # FCC crystal structure
    g = fcc(<lattice constant>, <Atom>)
    # HCP crystal structure
    g = hcp(<lattice constant>, <Atom>)

The `Graphene`, `BCC`, `FCC` and `HCP` structures can be created in
an orthogonal unit-cell by adding the flag `orthogonal=True` in the call:

    g = graphene(orthogonal=True)

#### IO-manipulation ####

sisl employs a variety of IO interfaces for managing different physical quantities.
A large variety of files describing the geometry (atomic positions and species) are
the main part of the IO routines.

All text files can also be read from their gzipped file formats with transparency.

All file formats in sisl are called a _Sile_ (sisl file). This small difference
prohibits name clashes with other implementations.

To read geometries from content you may do

    import sisl
    geom = sisl.Geometry.read('file.xyz')

which will read the geometry in `file.xyz` and return a `Geometry` object.

If you want to read several different objects from a single file you should
use the specific `get_sile` routine to retrieve the `Sile` object:

    import sisl
    fxyz = sisl.get_sile('file.xyz')

which returns an `xyzSile` file object that enables reading the geometry in
`file.xyz`. Subsequently you may read the geometry and obtain a geometry object
using

    geom = fxyz.read_geometry()

The above two methods are equivalent.

Even though these are hard coded you can easily extend your own file format

    sisl.add_sile(<file ending>, <SileObject>)

for instance the `xyzSile` is hooked using:

    sisl.add_sile('xyz', xyzSile, case=False)

which means that `sisl.get_sile` understands files `*.xyz` and `*.XYZ` files as
an `xyzSile` object. You can put whatever file-endings here and classes to retain API
compatibility. See the `sisl.io` package for more information. Note that a call to
`add_sile` with an already existing file ending results in overwriting the initial
meaning of that file object.

__NOTE__: if you know the file is in _xyz_ file format but the ending is erroneous, you can force the `xyzSile` by instantiating using that class

    sisl.io.xyzSile(<filename>)

which disregards the ending check. 

### Tight-binding ###

To create a tight-binding model you _extend_ a geometry to a `Hamiltonian` class which
contains the required sparse pattern.

To create the nearest neighbour tight-binding model for graphene you simply do

    # Create nearest-neighbour tight-binding
    # graphene lattice constant 1.42
    R = ( 0.1 , 1.5 )

    # Ensure that graphene has supercell connections
    gr.sc.set_nsc([3, 3, 1])
    tb = Hamiltonian(gr)
    for ia in tb.geom:
        idx_a = tb.close(ia, R=R)
        tb[ia,idx_a[0]] = 0. # on-site
        tb[ia,idx_a[1]] = -2.7 # nearest neighbour

at this point you have the tight-binding model for graphene and you can easily create
the Hamiltonian using this construct:

    Hk = tb.Hk(k=[0., 0.5, 0])

which returns the Hamiltonian in the `scipy.sparse.csr_matrix`
format. To calculate the dispersion you diagonalize and plot the eigenvalues

    import matplotlib.pyplot as plt
    klist = ... # dispersion curve
    eigs = np.empty([len(klist), tb.no])
    for ik, k in enumerate(klist):
        eigs[ik,:] = tb.eigh(k, eigvals_only=True)
        # Or equivalently:
        #   Hk = tb.Hk(k)
        #   eigs[ik,:] = sli.eigh(Hk.todense(), eigvals_only=True)
     for i in range(tb.no):
        plt.plot(eigs[:,i])

Very large tight-binding models are notoriously slow to create, however, sisl
implements a much faster method to loop over huge geometries

    for ias, idxs in tb.geom.iter_block(iR = 10):
        for ia in ias:
	        idx_a = tb.geom.close(ia, R, idx = idxs)
	        tb[ia,idx_a[0]] = 0.
            tb[ia,idx_a[1]] = -2.7

which accomplishes the same thing, but at much faster execution. `iR` should be a
number such that `tb.geom.close(<any index>,R = tb.geom.maxR() * iR)` is approximately
1,000 atoms.

The above example is for the default orthogonal Hamiltonian. However, sisl is
not limited to orthogonal basis functions. To construct the same example using
explicit overlap matrix the following procedure is necessary:

    # Create nearest-neighbour tight-binding
    # graphene lattice constant 1.42
    R = ( 0.1 , 1.5 )

    tb = Hamiltonian(gr, orthogonal=False)
    for ia in tb.geom:
        idx_a = tb.close(ia, R)
        tb.H[ia,idx_a[0]] = 0.
        tb.S[ia,idx_a[0]] = 1.
        tb.H[ia,idx_a[1]] = -2.7
        tb.S[ia,idx_a[1]] = 0. # still orthogonal (fake overlap matrix)
    Hk = tb.Hk(k=[0., 0.5, 0])
    Sk = tb.Sk(k=[0., 0.5, 0])
    eigs = sli.eigh(Hk.todense(), Sk.todense(), eigvals_only=True)


## Contributions, issues and bugs ##

I would advice any users to contribute as much feedback and/or PRs to further
maintain and expand this library.

Please do not hesitate to contribute!

If you find any bugs please form a [bug report/issue][issue].

If you have a fix please consider adding a [pull request][pr].

## License ##

The sisl license is [LGPL][lgpl], please see the LICENSE file.


<!---
Links to external and internal sites.
-->
[sisl@git]: https://github.com/zerothi/sisl
[sisl-api]: https://zerothi.github.io/sisl
[issue]: https://github.com/zerothi/sisl/issues
[tbtrans]: https://launchpad.net/siesta
[doi]: http://dx.doi.org/10.5281/zenodo.597181
[pr]: https://github.com/zerothi/sisl/pulls
[lgpl]: http://www.gnu.org/licenses/lgpl.html
[ase]: https://wiki.fysik.dtu.dk/ase/

<!---
Local variables for emacs to turn on flyspell-mode
% Local Variables:
%   mode: flyspell
%   tab-width: 4
%   indent-tabs-mode: nil
% End:
-->

