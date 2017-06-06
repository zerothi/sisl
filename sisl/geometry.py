"""
Geometry class to retain the atomic structure.
"""
from __future__ import print_function, division

# To check for integers
import warnings
from numbers import Integral, Real
from collections import deque
from six import string_types
from math import acos, pi
from itertools import product

import numpy as np

from ._help import _str
from ._help import _range as range
from ._help import array_fill_repeat, ensure_array, ensure_dtype
from ._help import isiterable, isndarray
from .utils import *
from .quaternion import Quaternion
from .supercell import SuperCell, SuperCellChild
from .atom import Atom, Atoms
from .shape import Shape, Sphere, Cube
from .sparse import SparseCSR

__all__ = ['Geometry', 'sgeom']


class Geometry(SuperCellChild):
    """ Holds atomic information, coordinates, species, lattice vectors

    The `Geometry` class holds information regarding atomic coordinates,
    the atomic species, the corresponding lattice-vectors.

    It enables the interaction and conversion of atomic structures via
    simple routine methods.

    All lengths are assumed to be in units of Angstrom, however, as
    long as units are kept same the exact units are irrespective.

    Examples
    --------

    An atomic lattice consisting of Hydrogen atoms.
    An atomic square lattice of Hydrogen atoms

     >>> xyz = [[0, 0, 0],
                [1, 1, 1]]
     >>> sc = SuperCell([2,2,2])
     >>> g = Geometry(xyz,Atom['H'],sc)

    The following estimates the lattice vectors from the
    atomic coordinates, although possible, it is not recommended
    to be used.

     >>> xyz = [[0, 0, 0],
                [1, 1, 1]]
     >>> g = Geometry(xyz,Atom['H'])

    Attributes
    ----------
    na : int
        number of atoms, ``len(self)``
    xyz : ndarray
        atomic coordinates
    atom : array_like, `Atom`
        the atomic objects associated with each atom
    sc : `SuperCell`
        the supercell describing the periodicity of the
        geometry
    no: int
        total number of orbitals in the geometry
    dR : float np.max([a.dR for a in self.atom])
        maximum orbital range

    Parameters
    ----------
    xyz : ``array_like``
        atomic coordinates
        ``xyz[i,:]`` is the atomic coordinate of the i'th atom.
    atom : ``array_like``
        atomic species retrieved from the `PeriodicTable`
    sc : ``SuperCell``
        the unit-cell describing the atoms in a periodic
        super-cell
    """

    def __init__(self, xyz, atom=None, sc=None):

        # Create the geometry coordinate
        self.xyz = np.copy(np.asarray(xyz, dtype=np.float64))
        self.xyz.shape = (-1, 3)

        # Default value
        if atom is None:
            atom = Atom('H')

        # Create the local Atoms object
        self._atom = Atoms(atom, na=self.na)

        # Get total number of orbitals
        orbs = self.atom.orbitals

        # Create local first
        firsto = np.append(np.array(0, np.int32), orbs)
        self.firsto = np.cumsum(firsto)

        self.__init_sc(sc)

    def __init_sc(self, sc):
        """ Initializes the supercell by *calculating* the size if not supplied

        If the supercell has not been passed we estimate the unit cell size
        by calculating the bond-length in each direction for a square
        Cartesian coordinate system.
        """
        # We still need the *default* super cell for
        # estimating the supercell
        self.set_supercell(sc)

        if sc is not None:
            return

        # First create an initial guess for the supercell
        # It HAS to be VERY large to not interact
        closest = self.close(0, dR=(0., 0.4, 5.))[2]
        if len(closest) < 1:
            # We could not find any atoms very close,
            # hence we simply return and now it becomes
            # the users responsibility

            # We create a molecule box with +10 A in each direction
            m, M = np.amin(self.xyz, axis=0), np.amax(self.xyz, axis=0) + 10.
            self.set_supercell(M-m)
            return

        sc_cart = np.zeros([3], np.float64)
        cart = np.zeros([3], np.float64)
        for i in range(3):
            # Initialize cartesian direction
            cart[i] = 1.

            # Get longest distance between atoms
            max_dist = np.amax(self.xyz[:, i]) - np.amin(self.xyz[:, i])

            dist = self.xyz[closest, :] - self.xyz[0, :][None, :]
            # Project onto the direction
            dd = np.abs(np.dot(dist, cart))

            # Remove all below .4
            tmp_idx = np.where(dd >= .4)[0]
            if len(tmp_idx) > 0:
                # We have a success
                # Add the bond-distance in the Cartesian direction
                # to the maximum distance in the same direction
                sc_cart[i] = max_dist + np.amin(dd[tmp_idx])
            else:
                # Default to LARGE array so as no
                # interaction occurs (it may be 2D)
                sc_cart[i] = max(10., max_dist)
            cart[i] = 0.

        # Re-set the supercell to the newly found one
        self.set_supercell(sc_cart)

    @property
    def atom(self):
        """ Atoms for the geometry (`Atoms` object) """
        return self._atom

    # Backwards compatability (do not use)
    atoms = atom

    @property
    def dR(self):
        """ Maximum orbital range of the atoms """
        return np.amax(self.atom.dR)

    @property
    def na(self):
        """ Number of atoms in geometry """
        return self.xyz.shape[0]

    @property
    def na_s(self):
        """ Number of supercell atoms """
        return self.na * self.n_s

    def __len__(self):
        """ Number of atoms in geometry """
        return self.na

    @property
    def no(self):
        """ Number of orbitals """
        return self.atom.no

    @property
    def no_s(self):
        """ Number of supercell orbitals """
        return self.no * self.n_s

    @property
    def orbitals(self):
        """ List of orbitals per atom """
        return self.atom.orbitals

    ## End size of geometry

    @property
    def lasto(self):
        """ The first orbital on the corresponding atom """
        return self.firsto[1:] - 1

    def __getitem__(self, atom):
        """ Geometry coordinates (allows supercell indices)"""
        if isinstance(atom, Integral):
            return self.axyz(atom)

        elif isinstance(atom, slice):
            if atom.stop is None:
                atom = atom.indices(self.na)
            else:
                atom = atom.indices(self.na_s)
            return self.axyz(np.arange(atom[0], atom[1], atom[2]))

        elif atom is None:
            return self.axyz()

        elif isinstance(atom, tuple):
            return self[atom[0]][:, atom[1]]

        elif atom[0] is None:
            return self.axyz()[:, atom[1]]

        return self.axyz(atom)

    def rij(self, ia, ja):
        """ Distance between atom ``ia`` and ``ja``, atoms are expected to be in super-cell indices

        Returns the distance between two atoms:

        .. math ::
            r\\_{ij} = |r\\_j - r\\_i|

        Parameters
        ----------
        ia : int or array_like
           atomic index of first atom
        ja : int or array_like
           atomic indices
        """
        xi = self.axyz(ia)
        xj = self.axyz(ja)

        if isinstance(ja, Integral):
            return ((xj[0] - xi[0]) ** 2. + (xj[1] - xi[1]) ** 2 + (xj[2] - xi[2]) ** 2) ** .5
        elif np.all(xi.shape == xj.shape):
            return np.sqrt(np.sum((xj - xi) ** 2., axis=1))

        return np.sqrt(np.sum((xj - xi[None, :]) ** 2., axis=1))

    def orij(self, io, jo):
        """ Return distance between orbital ``io`` and ``jo``, orbitals are expected to be in super-cell indices

        Returns the distance between two orbitals:

        .. math ::
            r\\_{ij} = |r\\_j - r\\_i|

        Parameters
        ----------
        io : ``int``
           orbital index of first orbital
        jo : ``int``, ``array_like``
           orbital indices
        """
        return self.rij(self.o2a(io), self.o2a(jo))

    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads geometry from the `Sile` using `Sile.read_geometry`

        Parameters
        ----------
        sile : `Sile`, str
            a `Sile` object which will be used to read the geometry
            if it is a string it will create a new sile using `get_sile`.
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            return sile.read_geometry(*args, **kwargs)
        else:
            return get_sile(sile).read_geometry(*args, **kwargs)

    def write(self, sile, *args, **kwargs):
        """ Writes geometry to the `Sile` using `sile.write_geometry`

        Parameters
        ----------
        sile : ``Sile``, ``str``
            a `Sile` object which will be used to write the geometry
            if it is a string it will create a new sile using `get_sile`
        *args, **kwargs:
            Any other args will be passed directly to the
            underlying routine
        """

        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            sile.write_geometry(self, *args, **kwargs)
        else:
            get_sile(sile, 'w').write_geometry(self, *args, **kwargs)

    def __repr__(self):
        """ Representation of the object """
        s = '{{na: {0}, no: {1}, species:\n '.format(self.na, self.no)
        s += repr(self.atom).replace('\n', '\n ')
        return (s[:-2] + ',\n nsc: [{1}, {2}, {3}], dR: {0}\n}}\n'.format(self.dR, *self.nsc)).strip()

    def iter(self):
        """
        Returns an iterator for atoms ranges.

        This iterator is the same as:

          >>> for ia in range(len(self)):
          >>>    <do something>
        or equivalently
          >>> for ia in self:
          >>>    <do something>
        """
        for ia in range(len(self)):
            yield ia

    __iter__ = iter

    def iter_species(self, atom=None):
        """
        Returns an iterator over all atoms and species as a tuple in this geometry

        >>> for ia, a, idx_specie in self.iter_species():

        with ``ia`` being the atomic index, ``a`` the `Atom` object, `idx_specie`
        is the index of the specie

        Parameters
        ----------
        atom : ``int``, ``array_like``
           only loop on the given atoms, default to all atoms
        """
        if atom is None:
            for ia in self:
                yield ia, self.atom[ia], self.atom.specie[ia]
        else:
            for ia in ensure_array(atom):
                yield ia, self.atom[ia], self.atom.specie[ia]

    def iter_orbitals(self, atom=None, local=True):
        """
        Returns an iterator over all atoms and their associated orbitals

         >>> for ia, io in self.iter_orbitals():

        with ``ia`` being the atomic index, ``io`` the associated orbital index on atom ``ia``.
        Note that ``io`` will start from ``0``.

        Parameters
        ----------
        atom : `int`, `array_like`
           only loop on the given atoms, default to all atoms
        local : `bool=True`
           whether the orbital index is the global index, or the local index relative to 
           the atom it resides on.
        """
        if atom is None:
            for ia in self:
                ia1 = self.firsto[ia]
                ia2 = self.lasto[ia] + 1
                for io in range(ia2 - ia1):
                    if local:
                        yield ia, io
                    else:
                        yield ia, io + ia1
        else:
            for ia in ensure_array(atom):
                ia1 = self.firsto[ia]
                ia2 = self.lasto[ia] + 1
                for io in range(ia2 - ia1):
                    if local:
                        yield ia, io
                    else:
                        yield ia, io + ia1

    def iR(self, na=1000, iR=20, dR=None):
        """ Return an integer number of maximum radii (`self.dR`) which holds approximately `na` atoms

        Parameters
        ----------
        na : ``int``
           number of atoms within the radius
        iR : ``int``
           initial ``iR`` value, which the sphere is estitametd from
        dR : ``float``
           the value used for atomic range (defaults to ``self.dR``)
        """
        ia = np.random.randint(len(self) - 1)

        # default block iterator
        if dR is None:
            dR = self.dR

        # Number of atoms in within 20 * dR
        naiR = len(self.close(ia, dR=dR * iR))

        # Convert to na atoms spherical radii
        iR = int(4 / 3 * np.pi * dR ** 3 / naiR * na)

        return iR

    def iter_block_rand(self, iR=10, dR=None, atom=None):
        """ Perform the *random* block-iteration by randomly selecting the next center of block """

        # We implement yields as we can then do nested iterators
        # create a boolean array
        na = len(self)
        not_passed = np.empty(na, dtype='b')
        if atom is not None:
            # Reverse the values
            not_passed[:] = False
            not_passed[atom] = True
        else:
            not_passed[:] = True

        # Figure out how many we need to loop on
        not_passed_N = np.sum(not_passed)

        if dR is None:
            dR = self.dR
        # The boundaries (ensure complete overlap)
        dR = np.array([iR - 0.975, iR + .025]) * dR

        where = np.where
        append = np.append

        # loop until all passed are true
        while not_passed_N > 0:

            # Take a random non-passed element
            all_true = where(not_passed)[0]

            # Shuffle should increase the chance of hitting a
            # completely "fresh" segment, thus we take the most
            # atoms at any single time.
            # Shuffling will cut down needed iterations.
            np.random.shuffle(all_true)
            idx = all_true[0]
            del all_true

            # Now we have found a new index, from which
            # we want to create the index based stuff on

            # get all elements within two radii
            all_idx = self.close(idx, dR=dR)

            # Get unit-cell atoms
            all_idx[0] = self.sc2uc(all_idx[0], uniq=True)
            # First extend the search-space (before reducing)
            all_idx[1] = self.sc2uc(append(all_idx[1], all_idx[0]), uniq=True)

            # Only select those who have not been runned yet
            all_idx[0] = all_idx[0][where(not_passed[all_idx[0]])[0]]
            if len(all_idx[0]) == 0:
                raise ValueError('Internal error, please report to the developers')

            # Tell the next loop to skip those passed
            not_passed[all_idx[0]] = False
            # Update looped variables
            not_passed_N -= len(all_idx[0])

            # Now we want to yield the stuff revealed
            # all_idx[0] contains the elements that should be looped
            # all_idx[1] contains the indices that can be searched
            yield all_idx[0], all_idx[1]

        if np.any(not_passed):
            raise ValueError('Error on iterations. Not all atoms has been visited.')

    def iter_block_shape(self, shape=None, iR=10, atom=None):
        """ Perform the *grid* block-iteration by looping a grid """

        # We implement yields as we can then do nested iterators
        # create a boolean array
        na = len(self)
        not_passed = np.empty(na, dtype='b')
        if atom is not None:
            # Reverse the values
            not_passed[:] = False
            not_passed[atom] = True
        else:
            not_passed[:] = True

        # Figure out how many we need to loop on
        not_passed_N = np.sum(not_passed)

        dR = self.dR
        if shape is None:
            # we default to the Cube shapes
            dS = (Cube(dR * (iR - 1.975)),
                  Cube(dR * (iR + 0.025)))
        else:
            dS = tuple(shape)
            if len(dS) == 1:
                dS += dS[0].expand(dR)
        if len(dS) != 2:
            raise ValueError('Number of Shapes *must* be one or two')

        # Now create the Grid
        # convert the radius to a square Grid
        # We do this by examining the x, y, z coordinates
        xyz_m = np.min(self.xyz, axis=0)
        xyz_M = np.max(self.xyz, axis=0)
        dxyz = xyz_M - xyz_m

        # Retrieve the internal diameter
        ir = dS[0].displacement

        # Figure out number of segments in each iteration
        # (minimum 1)
        ixyz = np.array(np.ceil(dxyz / ir + 0.0001), np.int32)

        # Calculate the steps required for each iteration
        for i in [0, 1, 2]:
            dxyz[i] = dxyz[i] / ixyz[i]

            # Correct the initial position to offset the initial displacement
            # so that we are at the border.
            xyz_m[i] += min(dxyz[i], ir[i]) / 2

            if xyz_m[i] > xyz_M[i]:
                # This is the case where one of the cell dimensions
                # is far too great.
                # In this case ixyz[i] should be 1
                xyz_m[i] = (xyz_M[i] - xyz_m[i]) / 2

        # Shorthand function
        where = np.where
        append = np.append

        # Now we loop in each direction
        for x, y, z in product(range(ixyz[0]),
                               range(ixyz[1]),
                               range(ixyz[2])):

            # Create the new center
            center = xyz_m + [x * dxyz[0], y * dxyz[1], z * dxyz[2]]
            # Correct in case the iteration steps across the maximum
            center = where(center < xyz_M, center, xyz_M)
            dS[0].set_center(center[:])
            dS[1].set_center(center[:])

            # Now perform the iteration
            # get all elements within two radii
            all_idx = self.within(dS)

            # Get unit-cell atoms
            all_idx[0] = self.sc2uc(all_idx[0], uniq=True)
            # First extend the search-space (before reducing)
            all_idx[1] = self.sc2uc(append(all_idx[1], all_idx[0]), uniq=True)

            # Only select those who have not been runned yet
            all_idx[0] = all_idx[0][where(not_passed[all_idx[0]])[0]]
            if len(all_idx[0]) == 0:
                continue

            # Tell the next loop to skip those passed
            not_passed[all_idx[0]] = False
            # Update looped variables
            not_passed_N -= len(all_idx[0])

            # Now we want to yield the stuff revealed
            # all_idx[0] contains the elements that should be looped
            # all_idx[1] contains the indices that can be searched
            yield all_idx[0], all_idx[1]

        if np.any(not_passed):
            print(where(not_passed)[0])
            print(np.sum(not_passed), len(self))
            raise ValueError('Error on iterations. Not all atoms has been visited.')

    def iter_block(self, iR=10, dR=None, atom=None, method='rand'):
        """
        Returns an iterator for performance critical looping.

        NOTE: This requires that dR has been set correctly as the maximum interaction range.

        I.e. the loop would look like this:

        >>> for ias, idxs in Geometry.iter_block():
        >>>    for ia in ias:
        >>>        idx_a = dev.close(ia, dR = dR, idx = idxs)

        This iterator is intended for systems with more than 1000 atoms.

        Remark that the iterator used is non-deterministic, i.e. any two iterators need
        not return the same atoms in any way.

        Parameters
        ----------
        atom : ``array_like``
            enables only effectively looping a subset of the full geometry
        iR  : ``int`` (`10`)
            the number of ``dR`` ranges taken into account when doing the iterator
        dR  : ``float``, (`self.dR`)
            enables overwriting the local dR quantity.
        method : ``str`` (`'rand'`)
            select the method by which the block iteration is performed. 
            Possible values are:
             `rand`: a spherical object is constructed with a random center according to the internal atoms
             `sphere`: a spherical equispaced shape is constructed and looped
             `cube`: a cube shape is constructed and looped

        Returns two lists with [0] being a list of atoms to be looped and [1] being the atoms that
        need searched.
        """
        method = method.lower()
        if method == 'rand' or method == 'random':
            for ias, idxs in self.iter_block_rand(iR, dR, atom):
                yield ias, idxs
        else:
            if dR is None:
                dR = self.dR

            # Create shapes
            if method == 'sphere':
                dS = (Sphere(dR * (iR - 0.975)),
                      Sphere(dR * (iR + 0.025)))
            elif method == 'cube':
                dS = (Cube(dR * (2 * iR - 0.975)),
                      Cube(dR * (2 * iR + 0.025)))

            for ias, idxs in self.iter_block_shape(dS):
                yield ias, idxs

    def copy(self):
        """
        Returns a copy of the object.
        """
        return self.__class__(np.copy(self.xyz),
                              atom=self.atom.copy(), sc=self.sc.copy())

    def sub(self, atom, cell=None):
        """
        Returns a subset of atoms from the geometry.

        Indices passed *MUST* be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atom  : ``array_like``
            indices of all atoms to be removed.
        cell   : ``array_like``, ``SuperCell`` (`self.cell`)
            the new associated cell of the geometry
        """
        atms = self.sc2uc(atom)
        if cell is None:
            return self.__class__(self.xyz[atms, :],
                                  atom=self.atom.sub(atms), sc=self.sc.copy())
        return self.__class__(self.xyz[atms, :],
                              atom=self.atom.sub(atms), sc=cell)

    def cut(self, seps, axis, seg=0, rtol=1e-4, atol=1e-4):
        """
        Returns a subset of atoms from the geometry by cutting the
        geometry into ``seps`` parts along the direction ``axis``.
        It will then _only_ return the first cut.

        This will effectively change the unit-cell in the ``axis`` as-well
        as removing ``self.na/seps`` atoms.
        It requires that ``self.na % seps == 0``.

        REMARK: You need to ensure that all atoms within the first
        cut out region are within the primary unit-cell.

        Doing ``geom.cut(2, 1).tile(2, 1)``, could for symmetric setups,
        be equivalent to a no-op operation. A ``UserWarning`` will be issued
        if this is not the case.

        Parameters
        ----------
        seps  : ``int``
            number of times the structure will be cut.
        axis  : ``int``
            the axis that will be cut
        seg : ``int`` (`0`)
            returns the i'th segment of the cut structure
            Currently the atomic coordinates are not translated,
            this may change in the future.
        rtol : (tolerance for checking tiling, see ``numpy.allclose``)
        atol : (tolerance for checking tiling, see ``numpy.allclose``)
        """
        if self.na % seps != 0:
            raise ValueError(
                'The system cannot be cut into {0} different ' +
                'pieces. Please check your geometry and input.'.format(seps))
        # Truncate to the correct segments
        lseg = seg % seps
        # Cut down cell
        sc = self.sc.cut(seps, axis)
        # List of atoms
        n = self.na // seps
        off = n * lseg
        new = self.sub(np.arange(off, off + n), cell=sc)
        if not np.allclose(new.tile(seps, axis).xyz, self.xyz,
                           rtol=rtol, atol=atol):
            st = 'The cut structure cannot be re-created by tiling'
            st += '\nThe difference between the coordinates can be altered using rtol, atol'
            warnings.warn(st, UserWarning)
        return new

    def remove(self, atom):
        """
        Remove atom from the geometry.

        Indices passed *MUST* be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atom  : array_like
            indices of all atoms to be removed.
        """
        atom = self.sc2uc(atom)
        atom = np.setdiff1d(np.arange(self.na), atom, assume_unique=True)
        return self.sub(atom)

    def tile(self, reps, axis):
        """
        Returns a geometry tiled, i.e. copied.

        The atomic indices are retained for the base structure.

        Parameters
        ----------
        reps  : ``int``
           number of tiles (repetitions)
        axis  : ``int``
           direction of tiling, 0, 1, 2 according to the cell-direction

        Examples
        --------
        >>> geom = Geometry(cell=[[1.,0,0],[0,1.,0.],[0,0,1.]],xyz=[[0,0,0],[0.5,0,0]])
        >>> g = geom.tile(2,axis=0)
        >>> print(g.xyz)
        [[ 0.   0.   0. ]
         [ 0.5  0.   0. ]
         [ 1.   0.   0. ]
         [ 1.5  0.   0. ]]
        >>> g = geom.tile(2,0).tile(2,axis=1)
        >>> print(g.xyz)
        [[ 0.   0.   0. ]
         [ 0.5  0.   0. ]
         [ 1.   0.   0. ]
         [ 1.5  0.   0. ]
         [ 0.   1.   0. ]
         [ 0.5  1.   0. ]
         [ 1.   1.   0. ]
         [ 1.5  1.   0. ]]

        """
        # We need a double copy as we want to re-calculate after
        # enlarging cell
        sc = self.sc.copy()
        sc.cell[axis, :] *= reps
        # Only reduce the size if it is larger than 5
        if sc.nsc[axis] > 3 and reps > 1:
            sc.nsc[axis] -= 2
        sc = sc.copy()
        # Pre-allocate geometry
        # Our first repetition *must* be with
        # the later coordinate
        # Copy the entire structure
        xyz = np.tile(self.xyz, (reps, 1))
        # Single cell displacements
        dx = np.dot(np.arange(reps)[:, None], self.cell[axis, :][None, :])
        # Correct the unit-cell offsets
        xyz[0:self.na * reps, :] += np.repeat(dx, self.na, axis=0)
        # Create the geometry and return it (note the smaller atoms array
        # will also expand via tiling)
        return self.__class__(xyz, atom=self.atom.tile(reps), sc=sc)

    def repeat(self, reps, axis):
        """
        Returns a geometry repeated, i.e. copied in a special way.

        The atomic indices are *NOT* retained for the base structure.

        The expansion of the atoms are basically performed using this
        algorithm:

        >>> ja = 0
        >>> for ia in range(self.na):
        >>>     for id,r in args:
        >>>        for i in range(r):
        >>>           ja = ia + cell[id,:] * i

        This method allows to utilise Bloch's theorem when creating
        Hamiltonian parameter sets for TBtrans.

        For geometries with a single atom this routine returns the same as
        `tile`.

        It is adviced to only use this for electrode Bloch's theorem
        purposes as `tile` is faster.

        Parameters
        ----------
        reps  : ``int``
           number of repetitions
        axis  : ``int``
           direction of repetition, 0, 1, 2 according to the cell-direction

        Examples
        --------
        >>> geom = Geometry(cell=[[1.,0,0],[0,1.,0.],[0,0,1.]],xyz=[[0,0,0],[0.5,0,0]])
        >>> g = geom.repeat(2,axis=0)
        >>> print(g.xyz)
        [[ 0.   0.   0. ]
         [ 1.   0.   0. ]
         [ 0.5  0.   0. ]
         [ 1.5  0.   0. ]]
        >>> g = geom.repeat(2,0).repeat(2,1)
        >>> print(g.xyz)
        [[ 0.   0.   0. ]
         [ 1.   0.   0. ]
         [ 0.   1.   0. ]
         [ 1.   1.   0. ]
         [ 0.5  0.   0. ]
         [ 1.5  0.   0. ]
         [ 0.5  1.   0. ]
         [ 1.5  1.   0. ]]

        """
        # Figure out the size
        sc = self.sc.copy()
        sc.cell[axis, :] *= reps
        # Only reduce the size if it is larger than 5
        if sc.nsc[axis] > 3 and reps > 1:
            sc.nsc[axis] -= 2
        sc = sc.copy()
        # Pre-allocate geometry
        na = self.na * reps
        xyz = np.zeros([na, 3], np.float64)
        dx = np.dot(np.arange(reps)[:, None], self.cell[axis, :][None, :])
        # Start the repetition
        ja = 0
        for ia in range(self.na):
            # Single atom displacements
            # First add the basic atomic coordinate,
            # then add displacement for each repetition.
            xyz[ja:ja + reps, :] = self.xyz[ia, :][None, :] + dx[:, :]
            ja += reps
        # Create the geometry and return it
        return self.__class__(xyz, atom=self.atom.repeat(reps), sc=sc)

    def __mul__(self, m):
        """ Implement easy repeat function 

        Parameters
        ----------
        m : int or tuple or list or (tuple, str) or (list, str)
           a tuple/list may be of length 2 or 3. A length of 2 corresponds
           to tuple[0] == *number of multiplications*, tuple[1] is the
           axis.
           A length of 3 corresponds to each of the directions.
           An optional string may be used to specify the `tile` or `repeat` function.
           The default is the `tile` function.

        Examples
        --------

        >>> geometry * 2 == geometry.tile(2, 0).tile(2, 1).tile(2, 2)
        >>> geometry * [2, 1, 2] == geometry.tile(2, 0).tile(2, 2)
        >>> geometry * [2, 2] == geometry.tile(2, 2)
        >>> geometry * ([2, 1, 2], 'repeat') == geometry.repeat(2, 0).repeat(2, 2)
        >>> geometry * ([2, 1, 2], 'r') == geometry.repeat(2, 0).repeat(2, 2)
        >>> geometry * ([2, 0], 'r') == geometry.repeat(2, 0)
        >>> geometry * ([2, 2], 'r') == geometry.repeat(2, 2)

        """

        # Reverse arguments in case it is on the LHS
        if not isinstance(self, Geometry):
            return m * self

        # Simple form
        if isinstance(m, Integral):
            return self * [m, m, m]

        # Error in argument, fall-back
        if len(m) == 1:
            return self * m[0]

        # Look-up table
        method_tbl = {'r': 'repeat',
                  'repeat': 'repeat',
                  't': 'tile',
                  'tile': 'tile'}
        method = 'tile'

        # Determine the type
        if len(m) == 2:
            # either
            #  (r, axis)
            #  ((...), method
            if isinstance(m[1], _str):
                method = method_tbl[m[1]]
                m = m[0]

        if len(m) == 1:
            #  r
            m = m[0]
            g = self.copy()
            for i in range(3):
                g = getattr(g, method)(max(m, 1), i)

        elif len(m) == 2:
            #  (r, axis)
            g = getattr(self, method)(max(m[0], 1), m[1])

        elif len(m) == 3:
            #  (r, r, r)
            g = self.copy()
            for i in range(3):
                g = getattr(g, method)(max(m[i], 1), i)

        else:
            raise ValueError('Multiplying a geometry has received a wrong argument')

        return g

    __rmul__ = __mul__

    def rotatea(self, angle, origo=None, atom=None, only='abc+xyz', radians=False):
        """ Rotate around first lattice vector, see ``rotate`` """
        return self.rotate(angle, self.cell[0, :], origo, atom, only, radians)

    def rotateb(self, angle, origo=None, atom=None, only='abc+xyz', radians=False):
        """ Rotate around second lattice vector, see ``rotate`` """
        return self.rotate(angle, self.cell[1, :], origo, atom, only, radians)

    def rotatec(self, angle, origo=None, atom=None, only='abc+xyz', radians=False):
        """ Rotate around third lattice vector, see ``rotate`` """
        return self.rotate(angle, self.cell[2, :], origo, atom, only, radians)

    def rotate(self, angle, v, origo=None, atom=None, only='abc+xyz', radians=False):
        """
        Rotates the geometry, in-place by the angle around the vector

        Per default will the entire geometry be rotated, such that everything
        is aligned as before rotation.

        However, by supplying ``only='abc|xyz'`` one can designate which
        part of the geometry that will be rotated.

        Parameters
        ----------
        angle : ``float``
             the angle in radians of which the geometry should be rotated
        v     : ``array_like`` [3]
             the normal vector to the rotated plane, i.e.
             v = [1,0,0] will rotate the ``yz`` plane
        origo : int or array_like, [0, 0, 0]
             the origin of rotation. Anything but [0, 0, 0] is equivalent
             to a `self.move(-origo).rotate(...).move(origo)`.
             If this is an `int` it corresponds to the atomic index.
        atom : int or array_like
             only rotate the given atomic indices, if not specified, all
             atoms will be rotated.
        only  : ('abc+xyz'), str, optional
             which coordinate subject should be rotated,
             if ``abc`` is in this string the cell will be rotated
             if ``xyz`` is in this string the coordinates will be rotated
        """
        if origo is None:
            origo = [0., 0., 0.]
        elif isinstance(origo, Integral):
            origo = self.axyz(origo)
        origo = ensure_array(origo, np.float64)

        if not atom is None:
            # Only rotate the unique values
            atom = self.sc2uc(atom, uniq=True)

        # Ensure the normal vector is normalized...
        vn = np.copy(np.asarray(v, dtype=np.float64)[:])
        vn /= np.sum(vn ** 2) ** .5

        # Prepare quaternion...
        q = Quaternion(angle, vn, radians=radians)
        q /= q.norm()

        # Rotate by direct call
        if 'abc' in only:
            sc = self.sc.rotate(angle, vn, radians=radians, only=only)
        else:
            sc = self.sc.copy()

        # Copy
        xyz = np.copy(self.xyz)

        if 'xyz' in only:
            # subtract and add origo, before and after rotation
            xyz[atom, :] = q.rotate(xyz[atom, :] - origo[None, :]) + origo[None, :]

        return self.__class__(xyz, atom=self.atom.copy(), sc=sc)

    def rotate_miller(self, m, v):
        """ Align Miller direction along ``v``

        Rotate geometry and cell such that the Miller direction
        points along the Cartesian vector ``v``.
        """
        # Create normal vector to miller direction and cartesian
        # direction
        cp = np.array([m[1] * v[2] - m[2] * v[1],
                       m[2] * v[0] - m[0] * v[2],
                       m[0] * v[1] - m[1] * v[0]], np.float64)
        cp /= np.sum(cp**2) ** .5

        lm = np.array(m, np.float64)
        lm /= np.sum(lm**2) ** .5
        lv = np.array(v, np.float64)
        lv /= np.sum(lv**2) ** .5

        # Now rotate the angle between them
        a = acos(np.sum(lm * lv))
        return self.rotate(a, cp)

    def move(self, v, atom=None, cell=False):
        """ Translates the geometry by ``v``

        One can translate a subset of the atoms by supplying ``atom``.

        Returns a copy of the structure translated by ``v``.
        """
        g = self.copy()
        if atom is None:
            g.xyz[:, :] += np.asarray(v, g.xyz.dtype)[None, :]
        else:
            g.xyz[ensure_array(atom), :] += np.asarray(v, g.xyz.dtype)[None, :]
        if cell:
            g.set_supercell(g.sc.translate(v))
        return g
    translate = move

    def swap(self, a, b):
        """ Returns a geometry with swapped atoms

        This can be used to reorder elements of a geometry.
        """
        a = ensure_array(a)
        b = ensure_array(b)
        xyz = np.copy(self.xyz)
        xyz[a, :] = self.xyz[b, :]
        xyz[b, :] = self.xyz[a, :]
        return self.__class__(xyz, atom=self.atom.swap(a, b), sc=self.sc.copy())

    def swapaxes(self, a, b, swap='cell+xyz'):
        """ Returns geometry with swapped axis

        If ``swapaxes(0,1)`` it returns the 0 and 1 values
        swapped in the ``cell`` variable.

        Parameters
        ----------
        a : ``int``
           axes 1, swaps with ``b``
        b : ``int``
           axes 2, swaps with ``a``
        swap : ``str`` (`"cell+xyz"`)
           decide what to swap, if `"cell"` is in `swap` then
           the cell axis are swapped.
           if `"xyz"` is in `swap` then
           the xyz (Cartesian) axis are swapped.
           Both may be in `swap`.
        """
        xyz = np.copy(self.xyz)
        if 'xyz' in swap:
            xyz[:, a] = self.xyz[:, b]
            xyz[:, b] = self.xyz[:, a]
        if 'cell' in swap:
            sc = self.sc.swapaxes(a, b)
        else:
            sc = self.sc.copy()
        return self.__class__(xyz, atom=self.atom.copy(), sc=sc)

    def center(self, atom=None, which='xyz'):
        """ Returns the center of the geometry

        By specifying ``which`` one can control whether it should be:

        * ``xyz|position``: Center of coordinates (default)
        * ``mass``: Center of mass
        * ``cell``: Center of cell

        Parameters
        ----------
        atom : ``array_like``
            list of atomic indices to find center of
        which : ``str``
            determine whether center should be of 'cell', mass-centered ('mass'),
            or absolute center of the positions.
        """
        if 'cell' in which:
            return self.sc.center()
        if atom is None:
            g = self
        else:
            g = self.sub(ensure_array(atom))
        if 'mass' in which:
            mass = self.mass
            return np.dot(mass, g.xyz) / np.sum(mass)
        if not ('xyz' in which or 'position' in which):
            raise ValueError(
                'Unknown which, not one of [xyz,position,mass,cell]')
        return np.mean(g.xyz, axis=0)

    def append(self, other, axis):
        """
        Appends structure along ``axis``. This will automatically
        add the ``self.cell[axis,:]`` to all atomic coordiates in the
        ``other`` structure before appending.

        The basic algorithm is this:

        >>> oxa = other.xyz + self.cell[axis,:][None,:]
        >>> self.xyz = np.append(self.xyz,oxa)
        >>> self.cell[axis,:] += other.cell[axis,:]

        NOTE: The cell appended is only in the axis that
        is appended, which means that the other cell directions
        need not conform.

        Parameters
        ----------
        other : ``Geometry``, ``SuperCell``
            Other geometry class which needs to be appended
            If a ``SuperCell`` only the super cell will be extended
        axis  : ``int``
            Cell direction to which the ``other`` geometry should be
            appended.
        """
        if isinstance(other, SuperCell):
            # Only extend the supercell.
            xyz = np.copy(self.xyz)
            atom = self.atom.copy()
            sc = self.sc.append(other, axis)
        else:
            xyz = np.append(self.xyz,
                            self.cell[axis, :][None, :] + other.xyz,
                            axis=0)
            atom = self.atom.append(other.atom)
            sc = self.sc.append(other.sc, axis)
        return self.__class__(xyz, atom=atom, sc=sc)

    def prepend(self, other, axis):
        """
        Prepends structure along ``axis``. This will automatically
        add the ``self.cell[axis,:]`` to all atomic coordiates in the
        ``other`` structure before prepending.

        The basic algorithm is this:

        >>> oxa = other.xyz
        >>> self.xyz = np.append(oxa, self.xyz + other.cell[axis,:][None,:])
        >>> self.cell[axis,:] += other.cell[axis,:]

        NOTE: The cell prepended is only in the axis that
        is prependend, which means that the other cell directions
        need not conform.

        Parameters
        ----------
        other : ``Geometry``, ``SuperCell``
            Other geometry class which needs to be prepended
            If a ``SuperCell`` only the super cell will be extended
        axis  : ``int``
            Cell direction to which the ``other`` geometry should be
            prepended
        """
        if isinstance(other, SuperCell):
            # Only extend the supercell.
            xyz = np.copy(self.xyz)
            atom = self.atom.copy()
            sc = self.sc.prepend(other, axis)
        else:
            xyz = np.append(other.xyz,
                            self.xyz + other.cell[axis, :][None, :],
                            axis=0)
            atom = self.atom.prepend(other.atom)
            sc = self.sc.append(other.sc, axis)
        return self.__class__(xyz, atom=atom, sc=sc)

    def add(self, other):
        """
        Adds atoms (as is) from the ``other`` geometry.
        This will not alter the cell vectors.

        Parameters
        ----------
        other : ``Geometry``
            Other geometry class which is added
        """
        xyz = np.append(self.xyz, other.xyz, axis=0)
        sc = self.sc.copy()
        return self.__class__(xyz, atom=self.atom.add(other.atom), sc=sc)

    def __add__(a, b):
        """ Implement easy merging of two geometries

        Parameters
        ----------
        a, b : Geometry or tuple or list
           when adding a Geometry with a Geometry it defaults to using `add` function
           with the LHS retaining the cell-vectors.
           a tuple/list may be of length 2 with the first element being a Geometry and the second
           being an integer specifying the lattice vector where it is appended.
           One may also use a `SuperCell` instead of a `Geometry` which behaves similarly.

        Examples
        --------

        >>> A + B == A.add(B)
        >>> A + (B, 1) == A.append(B, 1)
        >>> A + (B, 2) == A.append(B, 2)
        >>> (A, 1) + B == A.prepend(B, 1)

        """

        if isinstance(a, Geometry):
            if isinstance(b, Geometry):
                return a.add(b)
            return a.append(b[0], b[1])
        elif isinstance(b, Geometry):
            return a.prepend(b[0], b[1])

        raise ValueError('Arguments for adding (add/append/prepend) are incorrect')

    __radd__ = __add__

    def attach(self, s_idx, other, o_idx, dist='calc', axis=None):
        """ Attaches another ``Geometry`` at the `s_idx` index with respect to `o_idx` using different methods.

        Parameters
        ----------
        dist : ``array_like``, ``float``, ``str`` (`'calc'`)
           the distance (in `Ang`) between the attached coordinates. 
           If `dist` is `arraylike it should be the vector between
           the atoms;
           if `dist` is `float` the argument `axis` is required
           and the vector will be calculated along the corresponding latticevector;
           else if `dist` is `str` this will correspond to the
           `method` argument of the ``Atom.radius`` class of the two 
           atoms. Here `axis` is also required. 
        axis : ``int``
           specify the direction of the lattice vectors used.
           Not used if `dist` is an array-like argument.
        """
        if isinstance(dist, Real):
            # We have a single rational number
            if axis is None:
                raise ValueError("Argument `axis` has not been specified, please specify the axis when using a distance")

            # Now calculate the vector that we should have
            # between the atoms
            v = self.cell[axis, :]
            v = v / (v[0]**2 + v[1]**2 + v[2]**2) ** .5 * dist

        elif isinstance(dist, string_types):
            # We have a single rational number
            if axis is None:
                raise ValueError("Argument `axis` has not been specified, please specify the axis when using a distance")

            # This is the empirical distance between the atoms
            d = self.atom[s_idx].radius(dist) + other.atom[o_idx].radius(dist)
            if isinstance(axis, Integral):
                v = self.cell[axis, :]
            else:
                v = np.array(axis)

            v = v / (v[0]**2 + v[1]**2 + v[2]**2) ** .5 * d

        else:
            # The user *must* have supplied a vector
            v = np.array(dist)

        # Now create a copy of the other geometry
        # so that we move it...
        # Translate to origo, then back to position in new cell
        o = other.translate(-other.xyz[o_idx] + self.xyz[s_idx] + v)

        # We do not know how to handle the lattice-vectors,
        # so we will do nothing...
        return self.add(o)

    def reverse(self, atom=None):
        """ Returns a reversed geometry

        Also enables reversing a subset of the atoms.
        """
        if atom is None:
            xyz = self.xyz[::-1, :]
        else:
            atom = ensure_array(atom)
            xyz = np.copy(self.xyz)
            xyz[atom, :] = self.xyz[atom[::-1], :]
        return self.__class__(xyz, atom=self.atom.reverse(atom), sc=self.sc.copy())

    def mirror(self, plane, atom=None):
        """ Mirrors the structure around the center of the atoms """
        g = self.copy()
        lplane = ''.join(sorted(plane.lower()))
        if lplane == 'xy':
            g.xyz[:, 2] *= -1
        elif lplane == 'yz':
            g.xyz[:, 0] *= -1
        elif lplane == 'xz':
            g.xyz[:, 1] *= -1
        return self.__class__(g.xyz, atom=g.atom, sc=self.sc.copy())

    def insert(self, atom, geom):
        """ Inserts other atoms right before index

        We insert the ``geom`` `Geometry` before `atom`.
        Note that this will not change the unit cell.

        Parameters
        ----------
        atom : ``int``
           the index at which atom the other geometry is inserted
        geom : ``Geometry``
           the other geometry to be inserted
        """
        xyz = np.insert(self.xyz, atom, geom.xyz, axis=0)
        atoms = self.atom.insert(atom, geom.atom)
        return self.__class__(xyz, atom=atoms, sc=self.sc.copy())

    @property
    def fxyz(self):
        """ Returns geometry coordinates in fractional coordinates """
        return np.linalg.solve(self.cell.T, self.xyz.T).T

    def axyz(self, atom=None, isc=None):
        """ Return the atomic coordinates in the supercell of a given atom.

        The `Geometry[...]` slicing is calling this function with appropriate options.

        Examples
        --------

        >>> geom = Geometry(cell=1., xyz=[[0,0,0],[0.5,0,0]])
        >>> print(geom.axyz(isc=[1,0,0])
        [[ 1.   0.   0. ]
         [ 1.5  0.   0. ]]

        >>> geom = Geometry(cell=1., xyz=[[0,0,0],[0.5,0,0]])
        >>> print(geom.axyz(0))
        [ 1.   0.   0. ]

        Parameters
        ----------
        atom : int or array_like
          atom(s) from which we should return the coordinates, the atomic indices
          may be in supercell format.
        isc   : ``array_like``, ``[0,0,0]``
            Returns the atomic coordinates shifted according to the integer
            parts of the cell.
        """
        if atom is None and isc is None:
            return self.xyz

        # If only atom has been specified
        if isc is None:
            # get offsets from atomic indices (note that this will be per atom)
            isc = self.a2isc(atom)
            offset = self.sc.offset(isc)
            return self.xyz[self.sc2uc(atom), :] + offset

        elif atom is None:
            offset = self.sc.offset(isc)
            return self.xyz[:, :] + offset[None, :]

        # Neither of atom, or isc are `None`, we add the offset to all coordinates
        offset = self.sc.offset(isc)
        if isinstance(atom, Integral):
            return self.axyz(atom) + offset

        return self.axyz(atom) + offset[None, :]

    def scale(self, scale):
        """ Scale coordinates and unit-cell to get a new geometry with proper scaling

        Parameters
        ----------
        scale : ``float``
           the scale factor for the new geometry (lattice vectors, coordinates
           and the atomic radii are scaled).
        """
        xyz = self.xyz * scale
        atom = self.atom.scale(scale)
        sc = self.sc.scale(scale)
        return self.__class__(xyz, atom=atom, sc=sc)

    def within_sc(self, shapes, isc=None,
                  idx=None, idx_xyz=None,
                  ret_xyz=False, ret_rij=False):
        """
        Calculates which atoms are close to some atom or point
        in space, only returns so relative to a super-cell.

        This returns a set of atomic indices which are within a
        sphere of radius ``dR``.

        If dR is a tuple/list/array it will return the indices:
        in the ranges:
        >>> ( x <= dR[0] , dR[0] < x <= dR[1], dR[1] < x <= dR[2] )

        Parameters
        ----------
        shapes  : ``Shape``, ``list of Shape``
            A list of increasing shapes that define the extend of the geometric
            volume that is searched.
            It is vital that:
               shapes[0] in shapes[1] in shapes[2] ...
        isc       : ``array_like`` (`[0, 0, 0]`)
            The super-cell which the coordinates are checked in.
        idx       : ``array_like`` (`None`)
            List of atoms that will be considered. This can
            be used to only take out a certain atoms.
        idx_xyz : ``array_like`` (`None`)
            The atomic coordinates of the equivalent ``idx`` variable (``idx`` must also be passed)
        ret_xyz : ``bool`` (`False`)
            If true this method will return the coordinates
            for each of the couplings.
        ret_rij : ``bool`` (`False`)
            If true this method will return the distance
            for each of the couplings.
        """

        # Ensure that `shapes` is a list
        if isinstance(shapes, Shape):
            shapes = [shapes]
        nshapes = len(shapes)

        # Convert to actual array
        if idx is not None:
            if not isndarray(idx):
                idx = ensure_array(idx)
        else:
            # If idx is None, then idx_xyz cannot be used!
            # So we force it to None
            idx_xyz = None

        # Get shape centers
        off = shapes[-1].center[:]
        # Get the supercell offset
        soff = self.sc.offset(isc)[:]

        # Get atomic coordinate in principal cell
        if idx_xyz is None:
            xa = self[idx, :] + soff[None, :]
        else:
            # For extremely large systems re-using the
            # idx_xyz is faster than indexing
            # a very large array
            # However, this idx_xyz should not
            # be offset by any supercell
            xa = idx_xyz[:, :] + soff[None, :]

        # Get indices and coordinates of the largest shape
        # The largest part of the calculation are to calculate
        # the content in the largest shape.
        ix, xa = shapes[-1].iwithin(xa, return_sub=True)

        if idx is None:
            # This is because of the pre-check of the distance checks
            idx = ix
        else:
            idx = idx[ix]

        if len(xa) == 0:
            # Quick return if there are no entries...

            ret = [[np.empty([0], np.int32)] * nshapes]
            rc = 0
            if ret_xyz:
                rc = rc + 1
                ret.append([np.empty([0, 3], np.float64)] * nshapes)
            if ret_rij:
                rd = rc + 1
                ret.append([np.empty([0], np.float64)] * nshapes)

            if nshapes == 1:
                if ret_xyz and ret_rij:
                    return [ret[0][0], ret[1][0], ret[2][0]]
                elif ret_xyz or ret_rij:
                    return [ret[0][0], ret[1][0]]
                return ret[0][0]
            if ret_xyz or ret_rij:
                return ret
            return ret[0]

        # Calculate distance
        if ret_rij:
            d = np.sum((xa - off[None, :]) ** 2, axis=1) ** .5

        # Create the initial lists that we will build up
        # Then finally, we will return the reversed lists

        # Quick return
        if nshapes == 1:
            ret = [[idx]]
            if ret_xyz:
                ret.append([xa])
            if ret_rij:
                ret.append([d])
            if ret_xyz or ret_rij:
                return ret
            return ret[0]

        # TODO Check that all shapes coincide with the following shapes

        # Now we create a list of indices which coincide
        # in each of the shapes
        # Do a reduction on each of the list elements
        ixS = []
        cum = np.array([], idx.dtype)
        for i, s in enumerate(shapes):
            x = s.iwithin(xa)
            if i > 0:
                x = np.setdiff1d(x, cum, assume_unique=True)
            # Update elements to remove in next loop
            cum = np.append(cum, x)
            ixS.append(x)

        # Do for the first shape
        ret = [[ensure_array(idx[ixS[0]])]]
        rc = 0
        if ret_xyz:
            rc = rc + 1
            ret.append([xa[ixS[0], :]])
        if ret_rij:
            rd = rc + 1
            ret.append([d[ixS[0]]])
        for i in range(1, nshapes):
            ret[0].append(ensure_array(idx[ixS[i]]))
            if ret_xyz:
                ret[rc].append(xa[ixS[i], :])
            if ret_rij:
                ret[rd].append(d[ixS[i]])

        if ret_xyz or ret_rij:
            return ret
        return ret[0]

    def close_sc(self, xyz_ia, isc=None,
                 dR=None,
                 idx=None, idx_xyz=None,
                 ret_xyz=False, ret_rij=False):
        """
        Calculates which atoms are close to some atom or point
        in space, only returns so relative to a super-cell.

        This returns a set of atomic indices which are within a
        sphere of radius ``dR``.

        If dR is a tuple/list/array it will return the indices:
        in the ranges:
        >>> ( x <= dR[0] , dR[0] < x <= dR[1], dR[1] < x <= dR[2] )

        Parameters
        ----------
        xyz_ia    : ``coordinate``, ``int``
            Either a point in space or an index of an atom.
            If an index is passed it is the equivalent of passing
            the atomic coordinate ``close_sc(self.xyz[xyz_ia,:])``.
        isc       : ``array_like``, (`[ 0, 0, 0]`)
            The super-cell which the coordinates are checked in.
        dR        : ``float``, ``array_like`` (`None`)
            The radii parameter to where the atomic connections are found.
            If ``dR`` is an array it will return the indices:
            in the ranges:
               ``( x <= dR[0] , dR[0] < x <= dR[1], dR[1] < x <= dR[2] )``
            If a single float it will return:
               ``x <= dR``
        idx       : ``array_like`` (`None`)
            List of atoms that will be considered. This can
            be used to only take out a certain atoms.
        idx_xyz : ``array_like`` (`None`)
            The atomic coordinates of the equivalent ``idx`` variable (``idx`` must also be passed)
        ret_xyz : ``bool`` (`False`)
            If true this method will return the coordinates
            for each of the couplings.
        ret_rij : ``bool`` (`False`)
            If true this method will return the distance
            for each of the couplings.
        """

        # Common numpy used functions (reduces function look-ups)
        where = np.where
        log_and = np.logical_and
        fabs = np.fabs

        if dR is None:
            dR = np.array([self.dR], np.float64)
        elif not isndarray(dR):
            dR = ensure_array(dR, np.float64)

        # Maximum distance queried
        max_dR = dR[-1]

        # Convert to actual array
        if idx is not None:
            if not isndarray(idx):
                idx = ensure_array(idx)
        else:
            # If idx is None, then idx_xyz cannot be used!
            idx_xyz = None

        if isinstance(xyz_ia, Integral):
            off = self.xyz[xyz_ia, :]
        elif not isndarray(xyz_ia):
            off = ensure_array(xyz_ia, np.float64)
        else:
            off = xyz_ia

        # Calculate the complete offset
        foff = self.sc.offset(isc)[:] - off[:]

        # Get atomic coordinate in principal cell
        if idx_xyz is None:
            dxa = self[idx, :] + foff[None, :]
        else:
            # For extremely large systems re-using the
            # idx_xyz is faster than indexing
            # a very large array
            dxa = idx_xyz[:, :] + foff[None, :]

        # Immediately downscale by easy checking
        # This will reduce the computation of the vector-norm
        # which is the main culprit of the time-consumption
        # This abstraction will _only_ help very large
        # systems.
        # For smaller ones this will actually be a slower
        # method...
        # TODO should we abstract the methods dependent on size?
        ix = log_and.reduce(fabs(dxa[:, :]) <= max_dR, axis=1)

        if idx is None:
            # This is because of the pre-check of the
            # distance checks
            idx = where(ix)[0]
        else:
            idx = idx[ix]
        dxa = dxa[ix, :]

        # Create default return
        ret = [[np.empty([0], np.int32)] * len(dR)]
        i = 0
        if ret_xyz:
            i += 1
            rc = i
            ret.append([np.empty([0, 3], np.float64)] * len(dR))
        if ret_rij:
            i += 1
            rc = i
            ret.append([np.empty([0], np.float64)] * len(dR))

        if len(dxa) == 0:
            # Quick return if there are
            # no entries...
            if len(dR) == 1:
                if ret_xyz and ret_rij:
                    return [ret[0][0], ret[1][0], ret[2][0]]
                elif ret_xyz or ret_rij:
                    return [ret[0][0], ret[1][0]]
                return ret[0][0]
            if ret_xyz or ret_rij:
                return ret
            return ret[0]

        # Retrieve all atomic indices which are closer
        # than our delta-R
        # The linear algebra norm function could be used, but it
        # has a lot of checks, hence we do it manually
        #xaR = np.linalg.norm(dxa,axis=-1)
        # It is faster to do a single multiplacation than
        # a sqrt of MANY values
        # After having reduced the dxa array, we may then
        # take the sqrt
        max_dR = max_dR * max_dR
        xaR = dxa[:, 0]**2 + dxa[:, 1]**2 + dxa[:, 2]**2
        ix = np.where(xaR <= max_dR)[0]

        # Reduce search space and correct distances
        d = xaR[ix] ** .5
        if ret_xyz:
            xa = dxa[ix, :] + off[None, :]
        del xaR, dxa  # just because these arrays could be very big...

        # Check whether we only have one range to check.
        # If so, we need not reduce the index space
        if len(dR) == 1:
            ret = [idx[ix]]
            if ret_xyz:
                ret.append(xa)
            if ret_rij:
                ret.append(d)
            if ret_xyz or ret_rij:
                return ret
            return ret[0]

        if np.any(np.diff(dR) < 0.):
            raise ValueError(('Proximity checks for several quantities '
                              'at a time requires ascending dR values.'))

        # The more neigbours you wish to find the faster this becomes
        # We only do "one" heavy duty search,
        # then we immediately reduce search space to this subspace
        tidx = where(d <= dR[0])[0]
        ret = [[ensure_array(idx[ix[tidx]])]]
        i = 0
        if ret_xyz:
            rc = i + 1
            i += 1
            ret.append([xa[tidx]])
        if ret_rij:
            rd = i + 1
            i += 1
            ret.append([d[tidx]])
        for i in range(1, len(dR)):
            # Search in the sub-space
            # Notice that this sub-space reduction will never
            # allow the same indice to be in two ranges (due to
            # numerics)
            tidx = where(log_and(dR[i - 1] < d, d <= dR[i]))[0]
            ret[0].append(ensure_array(idx[ix[tidx]]))
            if ret_xyz:
                ret[rc].append(xa[tidx])
            if ret_rij:
                ret[rd].append(d[tidx])

        if ret_xyz or ret_rij:
            return ret
        return ret[0]

    def bond_correct(self, ia, atom, method='calc'):
        """ Corrects the bond between `ia` and the `atom`.

        Corrects the bond-length between atom `ia` and `atom` in such
        a way that the atomic radius is preserved.
        I.e. the sum of the bond-lengths minimizes the distance matrix.

        Only atom `ia` is moved.

        Parameters
        ----------
        ia : ``int``
            The atom to be displaced according to the atomic radius
        atom : ``array_like``, ``int``
            The atom(s) from which the radius should be reduced.
        method : ``str``, ``float``
            If str will use that as lookup in `Atom.radius`.
            Else it will be the new bond-length.
        """

        # Decide which algorithm to choose from
        if isinstance(atom, Integral):
            # a single point
            algo = atom
        elif len(atom) == 1:
            algo = atom[0]
        else:
            # signal a list of atoms
            algo = -1

        if algo >= 0:

            # We have a single atom
            # Get bond length in the closest direction
            # A bond-length HAS to be below 10
            idx, c, d = self.close(ia, dR=(0.1, 10.), idx=algo,
                                   ret_xyz=True, ret_rij=True)
            i = np.argmin(d[1])
            # Convert to unitcell atom (and get the one atom)
            idx = self.sc2uc(idx[1][i])
            c = c[1][i]
            d = d[1][i]

            # Calculate the bond vector
            bv = self.xyz[ia, :] - c

            try:
                # If it is a number, we use that.
                rad = float(method)
            except:
                # get radius
                rad = self.atom[idx].radius(method) \
                      + self.atom[ia].radius(method)

            # Update the coordinate
            self.xyz[ia, :] = c + bv / d * rad

        else:
            raise NotImplementedError(
                'Changing bond-length dependent on several lacks implementation.')

    def within(self, shapes,
            idx=None, idx_xyz=None,
            ret_xyz=False, ret_rij=False):
        """
        Returns supercell atomic indices for all atoms connecting to ``xyz_ia``

        This heavily relies on the `close_sc` method.

        Note that if a connection is made in a neighbouring super-cell
        then the atomic index is shifted by the super-cell index times
        number of atoms.
        This allows one to decipher super-cell atoms from unit-cell atoms.

        Parameters
        ----------
        shapes : ``Shape``, list of ``Shape``s
        idx     : ``array_like`` (`None`)
            List of indices for atoms that are to be considered
        idx_xyz : ``array_like`` (`None`)
            The atomic coordinates of the equivalent ``idx`` variable (``idx`` must also be passed)
        ret_xyz : ``bool`` (`False`)
            If true this method will return the coordinates
            for each of the couplings.
        ret_rij : ``bool`` (`False`)
            If true this method will return the distances from the ``xyz_ia``
            for each of the couplings.
        """

        # Ensure that `shapes` is a list
        if isinstance(shapes, Shape):
            shapes = [shapes]
        nshapes = len(shapes)

        # Get global calls
        # Is faster for many loops
        concat = np.concatenate

        ret = [[np.empty([0], np.int32)] * nshapes]
        i = 0
        if ret_xyz:
            c = i + 1
            i += 1
            ret.append([np.empty([0, 3], np.float64)] * nshapes)
        if ret_rij:
            d = i + 1
            i += 1
            ret.append([np.empty([0], np.float64)] * nshapes)

        ret_special = ret_xyz or ret_rij

        for s in range(self.n_s):
            na = self.na * s
            sret = self.within_sc(shapes, self.sc.sc_off[s, :],
                                  idx=idx, idx_xyz=idx_xyz,
                                  ret_xyz=ret_xyz, ret_rij=ret_rij)
            if not ret_special:
                # This is to "fake" the return
                # of a list (we will do indexing!)
                sret = [sret]

            if isinstance(sret[0], list):
                # we have a list of arrays (nshapes > 1)
                for i, x in enumerate(sret[0]):
                    ret[0][i] = concat((ret[0][i], x + na), axis=0)
                    if ret_xyz:
                        ret[c][i] = concat((ret[c][i], sret[c][i]), axis=0)
                    if ret_rij:
                        ret[d][i] = concat((ret[d][i], sret[d][i]), axis=0)
            elif len(sret[0]) > 0:
                # We can add it to the list (nshapes == 1)
                # We add the atomic offset for the supercell index
                ret[0][0] = concat((ret[0][0], sret[0] + na), axis=0)
                if ret_xyz:
                    ret[c][0] = concat((ret[c][0], sret[c]), axis=0)
                if ret_rij:
                    ret[d][0] = concat((ret[d][0], sret[d]), axis=0)

        if nshapes == 1:
            if ret_xyz and ret_rij:
                return [ret[0][0], ret[1][0], ret[2][0]]
            elif ret_xyz or ret_rij:
                return [ret[0][0], ret[1][0]]
            return ret[0][0]

        if ret_special:
            return ret

        return ret[0]

    def close(self, xyz_ia, dR=None,
            idx=None, idx_xyz=None,
            ret_xyz=False, ret_rij=False):
        """
        Returns supercell atomic indices for all atoms connecting to ``xyz_ia``

        This heavily relies on the `close_sc` method.

        Note that if a connection is made in a neighbouring super-cell
        then the atomic index is shifted by the super-cell index times
        number of atoms.
        This allows one to decipher super-cell atoms from unit-cell atoms.

        Parameters
        ----------
        xyz_ia : coordinate/index
            Either a point in space or an index of an atom.
            If an index is passed it is the equivalent of passing
            the atomic coordinate `close_sc(self.xyz[xyz_ia,:])`.
        dR      : (None), float/tuple of float
            The radii parameter to where the atomic connections are found.
            If ``dR`` is an array it will return the indices:
            in the ranges:

            >>> ``( x <= dR[0] , dR[0] < x <= dR[1], dR[1] < x <= dR[2] )``

            If a single float it will return:

            >>> ``x <= dR``

        idx     : (None), array_like
            List of indices for atoms that are to be considered
        idx_xyz : (None), array_like
            The atomic coordinates of the equivalent ``idx`` variable (``idx`` must also be passed)
        ret_xyz : (False), boolean
            If true this method will return the coordinates
            for each of the couplings.
        ret_rij : (False), boolean
            If true this method will return the distances from the ``xyz_ia``
            for each of the couplings.
        """
        if dR is None:
            dR = self.dR
        dR = ensure_array(dR, np.float64)

        # Get global calls
        # Is faster for many loops
        concat = np.concatenate

        ret = [[np.empty([0], np.int32)] * len(dR)]
        i = 0
        if ret_xyz:
            c = i + 1
            i += 1
            ret.append([np.empty([0, 3], np.float64)] * len(dR))
        if ret_rij:
            d = i + 1
            i += 1
            ret.append([np.empty([0], np.float64)] * len(dR))

        ret_special = ret_xyz or ret_rij

        for s in range(self.n_s):
            na = self.na * s
            sret = self.close_sc(xyz_ia,
                self.sc.sc_off[s, :], dR=dR,
                idx=idx, idx_xyz=idx_xyz,
                ret_xyz=ret_xyz, ret_rij=ret_rij)

            if not ret_special:
                # This is to "fake" the return
                # of a list (we will do indexing!)
                sret = [sret]

            if isinstance(sret[0], list):
                # we have a list of arrays (len(dR) > 1)
                for i, x in enumerate(sret[0]):
                    ret[0][i] = concat((ret[0][i], x + na), axis=0)
                    if ret_xyz:
                        ret[c][i] = concat((ret[c][i], sret[c][i]), axis=0)
                    if ret_rij:
                        ret[d][i] = concat((ret[d][i], sret[d][i]), axis=0)
            elif len(sret[0]) > 0:
                # We can add it to the list (len(dR) == 1)
                # We add the atomic offset for the supercell index
                ret[0][0] = concat((ret[0][0], sret[0] + na), axis=0)
                if ret_xyz:
                    ret[c][0] = concat((ret[c][0], sret[c]), axis=0)
                if ret_rij:
                    ret[d][0] = concat((ret[d][0], sret[d]), axis=0)

        if len(dR) == 1:
            if ret_xyz and ret_rij:
                return [ret[0][0], ret[1][0], ret[2][0]]
            elif ret_xyz or ret_rij:
                return [ret[0][0], ret[1][0]]
            return ret[0][0]

        if ret_special:
            return ret

        return ret[0]

    # Hence ``close_all`` has exact meaning
    # but ``close`` is shorten and retains meaning
    close_all = close

    def a2o(self, ia, all=False):
        """
        Returns an orbital index of the first orbital of said atom.
        This is particularly handy if you want to create
        TB models with more than one orbital per atom.

        Note that this will preserve the super-cell offsets.

        Parameters
        ----------
        ia : ``array_like``
             Atomic indices
        all : ``bool``  (`False`)
             `False`, return only the first orbital corresponding to the atom,
             `True`, returns list of the full atom
        """
        if not all:
            ia = np.asarray(ia)
            return self.firsto[ia % self.na] + (ia // self.na) * self.no
        ia = np.asarray(ia, np.int32)
        ob = self.a2o(ia)
        oe = self.a2o(ia + 1)

        # Create ranges
        if isinstance(ob, Integral):
            return np.arange(ob, oe, dtype=np.int32)

        # Several ranges
        o = np.empty([np.sum(oe - ob)], np.int32)
        n = 0
        narange = np.arange
        for i in range(len(ob)):
            o[n:n + oe[i] - ob[i]] = narange(ob[i], oe[i], dtype=np.int32)
            n += oe[i] - ob[i]
        return o

    def o2a(self, io, uc=False):
        """
        Returns an atomic index corresponding to the orbital indicies.

        This is a particurlaly slow algorithm due to for-loops.

        Note that this will preserve the super-cell offsets.

        Parameters
        ----------
        io: ``array_like``
             List of indices to return the atoms for
        """
        if isinstance(io, Integral):
            return np.argmax(io % self.no <= self.lasto) + (io // self.no) * self.na
        iio = np.asarray(io) % self.no
        a = np.array([np.argmax(i <= self.lasto) for i in iio], np.int32)
        return a + (iio // self.no) * self.na

    def sc2uc(self, atom, uniq=False):
        """ Returns atom from super-cell indices to unit-cell indices, possibly removing dublicates """
        atom = ensure_dtype(atom)
        if uniq:
            return np.unique(atom % self.na)
        return atom % self.na
    asc2uc = sc2uc

    def osc2uc(self, orbs, uniq=False):
        """ Returns orbitals from super-cell indices to unit-cell indices, possibly removing dublicates """
        orbs = ensure_dtype(orbs)
        if uniq:
            return np.unique(orbs % self.no)
        return orbs % self.no

    def a2isc(self, ia):
        """
        Returns the super-cell index for a specific/list atom

        Returns a vector of 3 numbers with integers.
        """
        idx = ensure_dtype(ia) // self.na
        return self.sc.sc_off[idx, :]

    # This function is a bit weird, it returns a real array,
    # however, there should be no ambiguity as it corresponds to th
    # offset and "what else" is there to query?
    def a2sc(self, a):
        """
        Returns the super-cell offset for a specific atom
        """
        return self.sc.offset(self.a2isc(a))

    def o2isc(self, io):
        """
        Returns the super-cell index for a specific orbital.

        Returns a vector of 3 numbers with integers.
        """
        idx = ensure_dtype(io) // self.no
        return self.sc.sc_off[idx, :]

    def o2sc(self, o):
        """
        Returns the super-cell offset for a specific orbital.
        """
        return self.sc.offset(self.o2isc(o))

    @classmethod
    def fromASE(cls, aseg):
        """ Returns geometry from an ASE object.

        Parameters
        ----------
        aseg : ASE ``Atoms`` object which contains the following routines:
            ``get_atomic_numbers``, ``get_positions``, ``get_cell``.
            From those methods a `sisl` object will be created.
        """
        Z = aseg.get_atomic_numbers()
        xyz = aseg.get_positions()
        cell = aseg.get_cell()
        # Convert to sisl object
        return cls(xyz, atom=Z, sc=cell)

    def toASE(self):
        """ Returns the geometry as an ASE ``Atoms`` object """
        from ase import Atoms
        return Atoms(symbols=self.atom.tolist(), positions=self.xyz.tolist(),
                     cell=self.cell.tolist())

    @property
    def mass(self):
        """ Returns the mass of all atoms as an array """
        return self.atom.mass

    def __eq__(self, other):
        if not isinstance(other, Geometry):
            return False
        same = self.sc == other.sc
        same = same and np.allclose(self.xyz, other.xyz)
        same = same and np.all(self.atom == other.atom)
        return same

    def __ne__(self, other):
        return not (self == other)

    def sparserij(self, dtype=np.float64, na_iR=1000, method='rand'):
        """ Return the sparse matrix with all distances in the matrix

        The sparse matrix will only be defined for the elements which have
        orbitals overlapping with other atoms.

        Parameters
        ----------
        dtype : numpy.dtype, numpy.float64
           the data-type of the sparse matrix
        na_iR : int, 1000
           number of atoms within the sphere for speeding
           up the `iter_block` loop.
        method : str, 'rand'
           see `iter_block` for details

        Returns
        -------
        SparseCSR
           sparse matrix with all rij elements
        """
        rij = SparseCSR((self.na, self.na_s), nnzpr=20, dtype=dtype)

        # Get dR
        dR = (0.1, self.dR)
        iR = self.iR(na_iR)

        # Do the loop
        for ias, idxs in self.iter_block(iR=iR, method=method):

            # Get all the indexed atoms...
            # This speeds up the searching for
            # coordinates...
            idxs_xyz = self[idxs, :]

            # Loop the atoms inside
            for ia in ias:
                idx, r = self.close(ia, dR=dR, idx=idxs, idx_xyz=idxs_xyz, ret_rij=True)
                rij[ia, idx[1]] = r[1]

        return rij

    # Create pickling routines
    def __getstate__(self):
        """ Returns the state of this object """
        d = self.sc.__getstate__()
        d['xyz'] = self.xyz
        d['atom'] = self.atom.__getstate__()
        return d

    def __setstate__(self, d):
        """ Re-create the state of this object """
        sc = SuperCell([1, 1, 1])
        sc.__setstate__(d)
        atoms = Atoms()
        atoms.__setstate__(d['atom'])
        self.__init__(d['xyz'], atom=atoms, sc=sc)

    @classmethod
    def _ArgumentParser_args_single(cls):
        """ Returns the options for `Geometry.ArgumentParser` in case they are the only options """
        return {'limit_arguments': False,
                'short': True,
                'positional_out': True,
            }

    # Hook into the Geometry class to create
    # an automatic ArgumentParser which makes actions
    # as the options are read.
    @dec_default_AP("Manipulate a Geometry object in sisl.")
    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Create and return a group of argument parsers which manipulates it self `Geometry`. 

        Parameters
        ----------
        parser: ``ArgumentParser`` (`None`)
           in case the arguments should be added to a specific parser. It defaults
           to create a new.
        limit_arguments: ``bool`` (`True`)
           If `False` additional options will be created which are similar to other options.
           For instance `--repeat-x` which is equivalent to `--repeat x`.
        short: ``bool`` (`False`)
           Create short options for a selected range of options
        positional_out: ``bool`` (`False`)
           If `True`, adds a positional argument which acts as --out. This may be handy if only the geometry is in the argument list.
        """
        limit_args = kwargs.get('limit_arguments', True)
        short = kwargs.get('short', False)

        def opts(*args):
            if short:
                return args
            return [args[0]]

        # We limit the import to occur here
        import argparse

        # The first thing we do is adding the geometry to the NameSpace of the
        # parser.
        # This will enable custom actions to interact with the geometry in a
        # straight forward manner.
        d = {
            "_geometry": self.copy(),
            "_stored_geometry": False,
        }
        namespace = default_namespace(**d)

        # Create actions
        class Format(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                ns._geom_fmt = value[0]
        p.add_argument(*opts('--format'), action=Format, nargs=1, default='.8f',
                   help='Specify output format for coordinates.')

        class MoveOrigin(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                ns._geometry.xyz[:, :] -= np.amin(ns._geometry.xyz, axis=0)[None, :]
        p.add_argument(*opts('--origin', '-O'), action=MoveOrigin, nargs=0,
                   help='Move all atoms such that one atom will be at the origin.')

        class MoveCenterOf(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                xyz = ns._geometry.center(which='xyz')
                ns._geometry = ns._geometry.translate(ns._geometry.center(which=value) - xyz)
        p.add_argument(*opts('--center-of', '-co'), choices=['mass', 'xyz', 'position', 'cell'],
                       action=MoveCenterOf,
                       help='Move coordinates to the center of the designated choice.')

        class MoveUnitCell(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                if value in ['translate', 'tr', 't']:
                    # Simple translation
                    tmp = np.amin(ns._geometry.xyz, axis=0)
                    # Find the smallest distance from the first atom
                    _, d = ns._geometry.close(0, dR=(0.1, 20.), ret_rij=True)
                    d = np.amin(d[1]) / 2
                    ns._geometry = ns._geometry.translate(-tmp + np.array([d, d, d]))
                elif value in ['mod']:
                    # Change all coordinates using the reciprocal cell
                    rcell = ns._geometry.rcell / (2. * np.pi)
                    idx = np.abs(np.array(np.dot(ns._geometry.xyz, rcell), np.int32))
                    # change supercell
                    nsc = np.amax(idx * 2 + 1, axis=0)
                    ns._geometry.set_nsc(nsc)
                    # Change the coordinates
                    for ia in ns._geometry:
                        ns._geometry.xyz[ia, :] = ns._geometry.axyz(isc=idx[ia, :], atom=ia)
        p.add_argument(*opts('--unit-cell', '-uc'), choices=['translate', 'tr', 't', 'mod'],
                       action=MoveUnitCell,
                       help='Moves the coordinates into the unit-cell by translation or the mod-operator')

        # Rotation
        class Rotation(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                # Convert value[0] to the direction
                d = direction(values[0])
                # The rotate function expects degree
                ang = angle(values[1], radians=False, in_radians=False)
                if d == 0:
                    v = [1, 0, 0]
                elif d == 1:
                    v = [0, 1, 0]
                elif d == 2:
                    v = [0, 0, 1]
                ns._geometry = ns._geometry.rotate(ang, v)
        p.add_argument(*opts('--rotate', '-R'), nargs=2, metavar=('DIR', 'ANGLE'),
                       action=Rotation,
                       help='Rotate geometry around given axis. ANGLE defaults to be specified in degree. Prefix with "r" for input in radians.')

        if not limit_args:
            class RotationX(argparse.Action):

                def __call__(self, parser, ns, value, option_string=None):
                    # The rotate function expects degree
                    ang = angle(value, radians=False, in_radians=False)
                    ns._geometry = ns._geometry.rotate(ang, [1, 0, 0])
            p.add_argument(*opts('--rotate-x', '-Rx'), metavar='ANGLE',
                           action=RotationX,
                           help='Rotate geometry around first cell vector. ANGLE defaults to be specified in degree. Prefix with "r" for input in radians.')

            class RotationY(argparse.Action):

                def __call__(self, parser, ns, value, option_string=None):
                    # The rotate function expects degree
                    ang = angle(value, radians=False, in_radians=False)
                    ns._geometry = ns._geometry.rotate(ang, [0, 1, 0])
            p.add_argument(*opts('--rotate-y', '-Ry'), metavar='ANGLE',
                           action=RotationY,
                           help='Rotate geometry around second cell vector. ANGLE defaults to be specified in degree. Prefix with "r" for input in radians.')

            class RotationZ(argparse.Action):

                def __call__(self, parser, ns, value, option_string=None):
                    # The rotate function expects degree
                    ang = angle(value, radians=False, in_radians=False)
                    ns._geometry = ns._geometry.rotate(ang, [0, 0, 1])
            p.add_argument(*opts('--rotate-z', '-Rz'), metavar='ANGLE',
                           action=RotationZ,
                           help='Rotate geometry around third cell vector. ANGLE defaults to be specified in degree. Prefix with "r" for input in radians.')

        # Reduce size of geometry
        class ReduceSub(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                # Get atomic indices
                rng = lstranges(strmap(int, value))
                ns._geometry = ns._geometry.sub(rng)
        p.add_argument(*opts('--sub', '-s'), metavar='RNG',
                       action=ReduceSub,
                       help='Removes specified atoms, can be complex ranges.')

        class ReduceCut(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                d = direction(values[0])
                s = int(values[1])
                ns._geometry = ns._geometry.cut(s, d)
        p.add_argument(*opts('--cut', '-c'), nargs=2, metavar=('DIR', 'SEPS'),
                       action=ReduceCut,
                       help='Cuts the geometry into `seps` parts along the unit-cell direction `dir`.')

        # Swaps atoms
        class AtomSwap(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                # Get atomic indices
                a = lstranges(strmap(int, value[0]))
                b = lstranges(strmap(int, value[1]))
                if len(a) != len(b):
                    raise ValueError('swapping atoms requires equal number of LHS and RHS atomic ranges')
                ns._geometry = ns._geometry.swap(a, b)
        p.add_argument(*opts('--swap'), metavar=('A', 'B'), nargs=2,
                       action=AtomSwap,
                       help='Swaps groups of atoms (can be complex ranges). The groups must be of equal length.')

        # Add an atom
        class AtomAdd(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                # Create an atom from the input
                g = Geometry([float(x) for x in values[0].split(',')], atom=Atom(values[1]))
                ns._geometry = ns._geometry.add(g)
        p.add_argument(*opts('--add'), nargs=2, metavar=('COORD', 'Z'),
                       action=AtomAdd,
                       help='Adds an atom, coordinate is comma separated (in Ang). Z is the atomic number.')

        # Translate
        class Translate(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                # Create an atom from the input
                if ',' in values[0]:
                    xyz = [float(x) for x in values[0].split(',')]
                else:
                    xyz = [float(x) for x in values[0].split()]
                ns._geometry = ns._geometry.translate(xyz)
        p.add_argument(*opts('--translate', '-t'), nargs=1, metavar='COORD',
                       action=Translate,
                       help='Translates the coordinates via a comma separated list (in Ang).')

        # Periodicly increase the structure
        class PeriodRepeat(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                d = direction(values[0])
                r = int(values[1])
                ns._geometry = ns._geometry.repeat(r, d)
        p.add_argument(*opts('--repeat', '-r'), nargs=2, metavar=('DIR', 'TIMES'),
                       action=PeriodRepeat,
                       help='Repeats the geometry in the specified direction.')

        if not limit_args:
            class PeriodRepeatX(argparse.Action):

                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.repeat(int(value), 0)
            p.add_argument(*opts('--repeat-x', '-rx'), metavar='TIMES',
                           action=PeriodRepeatX,
                           help='Repeats the geometry along the first cell vector.')

            class PeriodRepeatY(argparse.Action):

                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.repeat(int(value), 1)
            p.add_argument(*opts('--repeat-y', '-ry'), metavar='TIMES',
                           action=PeriodRepeatY,
                           help='Repeats the geometry along the second cell vector.')

            class PeriodRepeatZ(argparse.Action):

                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.repeat(int(value), 2)
            p.add_argument(*opts('--repeat-z', '-rz'), metavar='TIMES',
                           action=PeriodRepeatZ,
                           help='Repeats the geometry along the third cell vector.')

        class PeriodTile(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                d = direction(values[0])
                r = int(values[1])
                ns._geometry = ns._geometry.tile(r, d)
        p.add_argument(*opts('--tile'), nargs=2, metavar=('DIR', 'TIMES'),
                       action=PeriodTile,
                       help='Tiles the geometry in the specified direction.')

        if not limit_args:
            class PeriodTileX(argparse.Action):

                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.tile(int(value), 0)
            p.add_argument(*opts('--tile-x', '-tx'), metavar='TIMES',
                           action=PeriodTileX,
                           help='Tiles the geometry along the first cell vector.')

            class PeriodTileY(argparse.Action):

                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.tile(int(value), 1)
            p.add_argument(*opts('--tile-y', '-ty'), metavar='TIMES',
                           action=PeriodTileY,
                           help='Tiles the geometry along the second cell vector.')

            class PeriodTileZ(argparse.Action):

                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.tile(int(value), 2)
            p.add_argument(*opts('--tile-z', '-tz'), metavar='TIMES',
                           action=PeriodTileZ,
                           help='Tiles the geometry along the third cell vector.')

        # Print some common information about the
        # geometry (to stdout)
        class PrintInfo(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                # We fake that it has been stored...
                ns._stored_geometry = True
                print(ns._geometry)
        p.add_argument(*opts('--info'), nargs=0,
                       action=PrintInfo,
                       help='Print, to stdout, some regular information about the geometry.')

        class Out(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                if value is None:
                    return
                if len(value) == 0:
                    return
                # If the vector, exists, we should write it
                kwargs = {}
                if hasattr(ns, '_geom_fmt'):
                    kwargs['fmt'] = ns._geom_fmt
                if hasattr(ns, '_vector'):
                    v = getattr(ns, '_vector')
                    if getattr(ns, '_vector_scale', True):
                        v /= np.max((v[:, 0]**2 + v[:, 1]**2 + v[:, 2]**2) ** .5)
                    kwargs['data'] = v
                ns._geometry.write(value[0], **kwargs)
                # Issue to the namespace that the geometry has been written, at least once.
                ns._stored_geometry = True
        p.add_argument(*opts('--out', '-o'), nargs=1, action=Out,
                       help='Store the geometry (at its current invocation) to the out file.')

        # If the user requests positional out arguments, we also add that.
        if kwargs.get('positional_out', False):
            p.add_argument('out', nargs='*', default=None, action=Out,
                           help='Store the geometry (at its current invocation) to the out file.')

        # We have now created all arguments
        return p, namespace


def sgeom(geom=None, argv=None, ret_geometry=False):
    """ Main script for sgeom script. 

    This routine may be called with `argv` and/or a `Sile` which is the geometry at hand.

    Parameters
    ----------
    geom : ``Geometry``, ``BaseSile``
       this may either be the geometry, as-is, or a `Sile` which contains
       the geometry.
    argv : list of ``str``
       the arguments passed to sgeom
    ret_geometry : ``bool`` (`False`)
       whether the function should return the geometry
    """
    import sys
    import os.path as osp
    import argparse

    from sisl.io import get_sile, BaseSile

    # The geometry-file *MUST* be the first argument
    # (except --help|-h)

    # We cannot create a separate ArgumentParser to retrieve a positional arguments
    # as that will grab the first argument for an option!

    # Start creating the command-line utilities that are the actual ones.
    description = """
This manipulation utility is highly advanced and one should note that the ORDER of
options is determining the final structure. For instance:

   {0} geom.xyz --repeat x 2 --repeat y 2

is NOT equivalent to:

   {0} geom.xyz --repeat y 2 --repeat x 2

This may be unexpected but enables one to do advanced manipulations.

Additionally, in between arguments, one may store the current state of the geometry
by writing to a standard file.

   {0} geom.xyz --repeat y 2 geom_repy.xyz --repeat x 2 geom_repy_repx.xyz

will create two files:
   geom_repy.xyz
will only be repeated 2 times along the second lattice vector, while:
   geom_repy_repx.xyz
will be repeated 2 times along the second lattice vector, and then the first
lattice vector.
    """.format(osp.basename(sys.argv[0]))

    if argv is not None:
        if len(argv) == 0:
            argv = ['--help']
    elif len(sys.argv) == 1:
        # no arguments
        # fake a help
        argv = ['--help']
    else:
        argv = sys.argv[1:]

    # Ensure that the arguments have pre-pended spaces
    argv = cmd.argv_negative_fix(argv)

    p = argparse.ArgumentParser('Manipulates geometries from any Sile.',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=description)

    # First read the input "Sile"
    if geom is None:
        argv, input_file = cmd.collect_input(argv)
        try:
            geom = get_sile(input_file).read_geometry()
        except:
            geom = Geometry([0, 0, 0])

    elif isinstance(geom, Geometry):
        # Do nothing, the geometry is already created
        argv = ['fake.xyz'] + argv
        pass

    elif isinstance(geom, BaseSile):
        try:
            geom = sile.read_geometry()
            # Store the input file...
            input_file = geom.file
        except Exception as E:
            geom = Geometry([0, 0, 0])
        argv = ['fake.xyz'] + argv

    # Do the argument parser
    p, ns = geom.ArgumentParser(p, **geom._ArgumentParser_args_single())

    # Now the arguments should have been populated
    # and we will sort out if the input options
    # is only a help option.
    try:
        if not hasattr(ns, '_input_file'):
            setattr(ns, '_input_file', input_file)
    except:
        pass

    # Now try and figure out the actual arguments
    p, ns, argv = cmd.collect_arguments(argv, input=False,
                                        argumentparser=p,
                                        namespace=ns)

    # We are good to go!!!
    args = p.parse_args(argv, namespace=ns)
    g = args._geometry

    if not args._stored_geometry:
        # We should write out the information to the stdout
        # This is merely for testing purposes and may not be used for anything.
        print('Cell:')
        for i in (0, 1, 2):
            print('  {0:10.6f} {1:10.6f} {2:10.6f}'.format(*g.cell[i, :]))
        print('SuperCell:')
        print('  {0:d} {1:d} {2:d}'.format(*g.nsc))
        print(' {:>10s} {:>10s} {:>10s}  {:>3s}'.format('x', 'y', 'z', 'Z'))
        for ia in g:
            print(' {1:10.6f} {2:10.6f} {3:10.6f}  {0:3d}'.format(g.atom[ia].Z,
                                                                  *g.xyz[ia, :]))

    if ret_geometry:
        return g
    return 0
