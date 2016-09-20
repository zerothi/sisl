# sisl #

[![Build Status](https://travis-ci.org/zerothi/sisl.svg?branch=master)](https://travis-ci.org/zerothi/sisl) [![DOI](https://zenodo.org/badge/19941/zerothi/sisl.svg)](https://zenodo.org/badge/latestdoi/19941/zerothi/sisl) [![codecov](https://codecov.io/gh/zerothi/sisl/branch/master/graph/badge.svg)](https://codecov.io/gh/zerothi/sisl) [![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=NGNU2AA3JXX94&lc=DK&item_name=Papior%2dCodes&item_number=codes&currency_code=EUR&bn=PP%2dDonationsBF%3abtn_donate_SM%2egif%3aNonHosted)


The [API documentation][sisl-api] can be found [here][sisl-api].

The sisl toolbox provides a simple API for manipulating, constructing and creating tight-binding matrices 
in a standard and uniform way.

It provides easy interfaces for creating and calculating band-structures of
simple tight-binding models as well as interfacing to more advanced DFT
programs.

sisl also enables an easy interface for [ASE][ase].

## Usage ##

If used to produce scientific contributions, please use the DOI for citation.
Press the DOI link at the top of this page and select

    Cite as 

in the right side of the zenodo webpage. Select your citation style.

### Scripts ###

sisl contain a utility to easily convert geometries from existing files
to other formats. After installing the executable `sgeom` is available which
enables the conversion between all formats accessible as `Sile` objects.

To convert a SIESTA FDF file to `xyz` _and_ an `XV` file one does

    sgeom siesta.fdf geom.xyz geom.XV

When doing complex geometry complexes one can use piping to do consecutive
manipulations of the geometry, for instance to first repeat, then rotate
a structure one could do:

    sgeom -rx 2 siesta.fdf | sgeom -Rx 90a -o rep_rot.fdf

Note that `-o` is needed as `sgeom` otherwise does not now the difference
between piping and input-file names.

Try `sgeom -h` for additional features such as repeating the structure.


### Geometry manipulation ###

sisl contain a class for manipulating geometries in a consistent and easy
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
                  atoms=Atom(Z=6, R=1.42), sc=sc)
    huge = gr.tile(100, axis=0).tile(100, axis=1)

Which results in a 20000 atom big graphene flake.

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
a square unit-cell by adding the flag `square=True` in the call:

    g = graphene(square=True)

#### IO-manipulation ####

sisl employs a variety of IO interfaces for managing geometries.

The hard-coded file formats are:

1. ___xyz___, standard coordinate format
 Note that the the _xyz_ file format does not _per see_ contain the cell size.
 The `XYZSile` writes the cell information in the `xyz` file comment section (2nd line). Hence if the file was written with sisl you retain the cell information.
2. ___gout___, reads geometries from GULP output
3. ___nc___, reads/writes NetCDF4 files created by SIESTA
4. ___TBT.nc___/___PHT.nc___, reads NetCDF4 files created by TBtrans/PHtrans
5. ___tb___, intrinsic file format for geometry/tight-binding models
6. ___fdf___, SIESTA native format
7. ___XV___, SIESTA coordinate format with velocities
8. ___POSCAR___/___CONTCAR___, VASP coordinate format
9. ___ASCII___, BigDFT coordinate format
10. ___win___, Wannier90 Hamiltonian and Wannier centres
11. ___xsf___, XCrySDen file format


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

which returns an `XYZSile` file object that enables reading the geometry in
`file.xyz`. Subsequently you may read the geometry and obtain a geometry object
using

    geom = fxyz.read_geom()

The above two methods are equivalent.

Even though these are hard coded you can easily extend your own file format

    sisl.add_sile(<file ending>, <SileObject>)

for instance the `XYZSile` is hooked using:

    sisl.add_sile('xyz', XYZSile, case=False)

which means that `sisl.get_sile` understands files `*.xyz` and `*.XYZ` files as
an `XYZSile` object. You can put whatever file-endings here and classes to retain API
compatibility. See the `sisl.io` package for more information. Note that a call to
`add_sile` with an already existing file ending results in overwriting the initial
meaning of that file object.

__NOTE__: if you know the file is in _xyz_ file format but the ending is erroneous, you can force the `XYZSile` by instantiating using that class

    sisl.XYZSile(<filename>)

which disregards the ending check. 

### Tight-binding ###

To create a tight-binding model you _extend_ a geometry to a `Hamiltonian` class which
contains the required sparse pattern.

To create the nearest neighbour tight-binding model for graphene you simply do

    # Create nearest-neighbour tight-binding
    # graphene lattice constant 1.42
    dR = ( 0.1 , 1.5 )
    on = 0. 
    nn = -0.5

    # Ensure that graphene has supercell connections
    gr.sc.set_nsc([3,3,1])
    tb = Hamiltonian(gr)
    for ia in tb.geom:
        idx_a = tb.close(ia, dR=dR)
        tb[ia,idx_a[0]] = on
        tb[ia,idx_a[1]] = nn

at this point you have the tight-binding model for graphene and you can easily create
the Hamiltonian using this construct:

    Hk = tb.Hk(k=[0.,0.5,0])

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
implement a much faster method to loop over huge geometries

    for ias, idxs in tb.geom.iter_block(iR = 10):
        for ia in ias:
	        idx_a = tb.geom.close(ia, dR = dR, idx = idxs)
	        tb[ia,idx_a[0]] = on
            tb[ia,idx_a[1]] = nn

which accomplishes the same thing, but at much faster execution. `iR` should be a
number such that `tb.geom.close(<any index>,dR = tb.geom.dR * iR)` is approximately
1000 atoms.

The above example is for the default orthogonal Hamiltonian. However, sisl is
not limited to orthogonal basis functions. To construct the same example using
explicit overlap matrix the following procedure is necessary:

    # Create nearest-neighbour tight-binding
    # graphene lattice constant 1.42
    dR = ( 0.1 , 1.5 )
    on = ( 0. , 1.)
    nn = (-0.5, 0.) # still orthogonal (but with fake overlap)

    tb = Hamiltonian(gr, ortho=False)
    for ia in tb.geom:
        idx_a = tb.close(ia, dR=dR)
        tb[ia,idx_a[0]] = on
        tb[ia,idx_a[1]] = nn
    Hk = tb.Hk(k=[0.,0.5,0])
    Sk = tb.Sk(k=[0.,0.5,0])
    eigs = sli.eigh(Hk.todense(), Sk.todense(), eigvals_only=True)



## Downloading and installation ##

Installing sisl requires the following packages:

   - __six__
   - __numpy__
   - __scipy__
   - __netCDF4__, this module is only required if you need interface to construct
    the transport tight-binding model for `TBtrans`

Installing sisl is as any simple Python package

    python setup.py install --prefix=<prefix>


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
[sisl-api]: http://zerothi.github.io/sisl/index.html
[issue]: https://github.com/zerothi/sisl/issues
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

