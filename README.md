# sids #

[![Build Status](https://travis-ci.org/zerothi/sids.svg?branch=master)](https://travis-ci.org/zerothi/sids)

The system incentive for dense simulation toolbox provides a simple
toolbox for manipulating, constructing and creating tight-binding matrices 
in a standard and uniform way.

It provides easy interfaces for creating and calculating band-structures of
simple tight-binding models as well as interfacing to more advanced DFT
programs.

## Usage ##

### Geometry manipulation ###

sids contain a class for manipulating geometries in a consistent and easy
way.

For instance to create a huge graphene flake

	sq3h  = 3.**.5 * 0.5
    gr = Geometry(cell=np.array([[1.5, sq3h,  0.],
                                 [1.5,-sq3h,  0.],
                                 [ 0.,   0., 10.]],np.float) * 1.42,
                  xyz=np.array([[ 0., 0., 0.],
                                [ 1., 0., 0.]],np.float) * 1.42,
                  atoms = Atom(Z=6,R = 1.42), nsc = [3,3,1])
    huge = gr.tile(100,axis=0).tile(100,axis=1)

Which results in a 20000 atom big graphene flake.


### Tight-binding ###

To create a tight-binding model you _extend_ a geometry to a `TightBinding` class which
contains the required sparse pattern.

To create the nearest neighbour tight-binding model for graphene you simply do

    # Create nearest-neighbour tight-binding
    # graphene lattice constant 1.42
    dR = ( 0.1 , 1.5 )
	on = (0.,1.)
    nn = (-0.5,0.)

	# Ensure that graphene has supercell connections
	gr.set_supercell([3,3,1])
    tb = TightBinding(gr)
    for ia in tb.geom:
        idx_a = tb.close_all(ia,dR=dR)
        tb[ia,idx_a[0]] = on
        tb[ia,idx_a[1]] = nn

at this point you have the tight-binding model for graphene and you can easily create
the Hamiltonian using this construct:

    H, S = tb.tocsr(k=[0.,0.5,0])

which returs the Hamiltonian and the overlap matrices in the `scipy.sparse.csr_matri`
format. To calculate the dispersion you diagonalize and plot the eigenvalues

	import matplotlib.pyplot as plt
    klist = ... # dispersion curve
	eigs = np.empty([len(klist),tb.no])
	for ik,k in enumerate(klist):
		H, S = tb.tocsr(k)
		eigs[ik,:] = sli.eigh(H.todense(),S.todense(),eigvals_only=True)
	for i in range(tb.no):
		plt.plot(eigs[:,i])


## Downloading and installation ##

Installing sids requires the following packages:

   - __numpy__
   - __scipy__
   - __netCDF4__, this module is only required if you need interface to construct
    the transport tight-binding model for `TBtrans`


## Contributions, issues and bugs ##

I would advice any users to contribute as much feedback and/or PRs to further
maintain and expand this library.

Please do not hesitate to contribute!

If you find any bugs please form a [bug report/issue][issue].

If you have a fix please consider adding a [pull request][pr].


## License ##

The sids license is [LGPL][lgpl], please see the LICENSE file.


<!---
Links to external and internal sites.
-->
[sids@git]: https://github.com/zerothi/sids
[sids-doc]: http://github.com/zerothi/sids
[issue]: https://github.com/zerothi/sids/issues
[pr]: https://github.com/zerothi/sids/pulls
[lgpl]: http://www.gnu.org/licenses/lgpl.html

