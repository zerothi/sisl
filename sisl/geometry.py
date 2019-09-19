from __future__ import print_function, division

# To check for integers
from numbers import Integral, Real
from six import string_types
from math import acos
from itertools import product

import numpy as np
from numpy import ndarray, int32, bool_
from numpy import dot, square, sqrt
from numpy import floor, ceil

from . import _plot as plt
from . import _array as _a
from ._math_small import is_ascending
from ._indices import indices_in_sphere_with_dist, indices_le, indices_gt_le
from ._indices import list_index_le
from .messages import info, warn, SislError
from ._help import _str
from ._help import _range as range
from ._help import isndarray
from .utils import default_ArgumentParser, default_namespace, cmd, str_spec
from .utils import angle, direction
from .utils import lstranges, strmap, array_arange
from .utils.mathematics import fnorm
from .quaternion import Quaternion
from .supercell import SuperCell, SuperCellChild
from .atom import Atom, Atoms
from .shape import Shape, Sphere, Cube
from ._namedindex import NamedIndex

__all__ = ['Geometry', 'sgeom']


class Geometry(SuperCellChild):
    """ Holds atomic information, coordinates, species, lattice vectors

    The `Geometry` class holds information regarding atomic coordinates,
    the atomic species, the corresponding lattice-vectors.

    It enables the interaction and conversion of atomic structures via
    simple routine methods.

    All lengths are assumed to be in units of Angstrom, however, as
    long as units are kept same the exact units are irrespective.

    .. code:: python

       >>> square = Geometry([[0.5, 0.5, 0.5]], Atom(1),
       ...                   sc=SuperCell([1, 1, 10], nsc=[3, 3, 1]))
       >>> print(square)
       Geometry{na: 1, no: 1,
        Atoms{species: 1,
         Atom{H, Z: 1, mass(au): 1.00794, maxR: -1.00000,
          Orbital{R: -1.00000, q0: 0.0}
         }: 1,
        },
        maxR: -1.00000,
        SuperCell{volume: 1.0000e+01, nsc: [3 3 1]}
       }


    Attributes
    ----------
    na
    xyz : numpy.ndarray
        atomic coordinates
    atoms
    orbitals
    sc : SuperCell
        the supercell describing the periodicity of the
        geometry
    no
    n_s : int
        total number of supercells in the supercell
    no_s : int
        total number of orbitals in the geometry times number of supercells

    Parameters
    ----------
    xyz : array_like
        atomic coordinates
        ``xyz[i, :]`` is the atomic coordinate of the i'th atom.
    atom : array_like or Atoms
        atomic species retrieved from the `PeriodicTable`
    sc : SuperCell
        the unit-cell describing the atoms in a periodic
        super-cell

    Examples
    --------

    An atomic cubic lattice of Hydrogen atoms

    >>> xyz = [[0, 0, 0],
    ...        [1, 1, 1]]
    >>> sc = SuperCell([2,2,2])
    >>> g = Geometry(xyz, Atom('H'), sc)

    The following estimates the lattice vectors from the
    atomic coordinates, although possible, it is not recommended
    to be used.

    >>> xyz = [[0, 0, 0],
    ...        [1, 1, 1]]
    >>> g = Geometry(xyz, Atom('H'))

    See Also
    --------
    Atoms : contained atoms `self.atoms`
    Atom : contained atoms are each an object of this
    """

    def __init__(self, xyz, atom=None, sc=None, names=None):

        # Create the geometry coordinate
        # We need flatten to ensure a copy
        self.xyz = _a.asarrayd(xyz).flatten().reshape(-1, 3)

        # Default value
        if atom is None:
            atom = Atom('H')

        # Create the local Atoms object
        self._atoms = Atoms(atom, na=self.na)

        # Assign a group specifier
        if isinstance(names, NamedIndex):
            self._names = names.copy()
        else:
            self._names = NamedIndex(names)

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
        closest = self.close(0, R=(0., 0.4, 5.))[2]
        if len(closest) < 1:
            # We could not find any atoms very close,
            # hence we simply return and now it becomes
            # the users responsibility

            # We create a molecule box with +10 A in each direction
            m, M = np.amin(self.xyz, axis=0), np.amax(self.xyz, axis=0) + 10.
            self.set_supercell(M-m)
            return

        sc_cart = _a.zerosd([3])
        cart = _a.zerosd([3])
        for i in range(3):
            # Initialize cartesian direction
            cart[i] = 1.

            # Get longest distance between atoms
            max_dist = np.amax(self.xyz[:, i]) - np.amin(self.xyz[:, i])

            dist = self.xyz[closest, :] - self.xyz[0, :][None, :]
            # Project onto the direction
            dd = np.abs(dot(dist, cart))

            # Remove all below .4
            tmp_idx = (dd >= .4).nonzero()[0]
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
    def atoms(self):
        """ Atoms for the geometry (`Atoms` object) """
        return self._atoms

    # Backwards compatability (do not use)
    atom = atoms

    @property
    def names(self):
        """ The named index specifier """
        return self._names

    @property
    def q0(self):
        """ Total initial charge in this geometry (sum of q0 in all atoms) """
        return self.atoms.q0.sum()

    @property
    def mass(self):
        """ The mass of all atoms as an array """
        return self.atoms.mass

    def maxR(self, all=False):
        """ Maximum orbital range of the atoms """
        return self.atoms.maxR(all)

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
        return self.atoms.no

    @property
    def no_s(self):
        """ Number of supercell orbitals """
        return self.no * self.n_s

    @property
    def firsto(self):
        """ The first orbital on the corresponding atom """
        return self.atoms.firsto

    @property
    def lasto(self):
        """ The last orbital on the corresponding atom """
        return self.atoms.lasto

    @property
    def orbitals(self):
        """ List of orbitals per atom """
        return self.atoms.orbitals

    ## End size of geometry

    def __setitem__(self, atom, value):
        """ Specify geometry coordinates """
        if isinstance(atom, _str):
            self.names.add_name(atom, value)
        elif isinstance(value, _str):
            self.names.add_name(value, atom)

    def __getitem__(self, atom):
        """ Geometry coordinates (allows supercell indices) """
        if isinstance(atom, (Integral, _str)):
            return self.axyz(atom)

        elif isinstance(atom, slice):
            if atom.stop is None:
                atom = atom.indices(self.na)
            else:
                atom = atom.indices(self.na_s)
            return self.axyz(np.arange(atom[0], atom[1], atom[2], dtype=np.int32))

        elif atom is None:
            return self.axyz()

        elif isinstance(atom, tuple):
            return self[atom[0]][..., atom[1]]

        return self.axyz(atom)

    def _sanitize_atom(self, atom):
        """ Converts an `atom` to index under given inputs

        `atom` may be one of the following:

        - boolean array -> nonzero()[0]
        - name -> self._names[name]
        """
        if isinstance(atom, str):
            return self.names[atom]
        elif isinstance(atom, ndarray) and atom.dtype == bool_:
            return np.flatnonzero(atom)
        # We shouldn't .ravel() since the calling routine may expect
        # a 0D vector.
        atom = _a.asarrayi(atom)
        if atom.ndim > 1:
            raise ValueError('Indexing geometries with a multi-dimensional array is not supported, ensure 0D or 1D arrays.')
        return atom

    def _sanitize_orb(self, orbital):
        """ Converts an `orbital` to index under given inputs

        `orbital` may be one of the following:

        - boolean array -> nonzero()[0]
        """
        if isinstance(orbital, ndarray) and orbital.dtype == bool_:
            return np.flatnonzero(orbital)
        # We shouldn't .ravel() since the calling routine may expect
        # a 0D vector.
        orbital = _a.asarrayi(orbital)
        if orbital.ndim > 1:
            raise ValueError('Indexing geometries with a multi-dimensional array is not supported, ensure 0D or 1D arrays.')
        return orbital

    def as_primary(self, na_primary, ret_super=False):
        """ Try and reduce the geometry to the primary unit-cell comprising `na_primary` atoms

        This will basically try and find the tiling/repetitions required for the geometry to only have
        `na_primary` atoms in the unit cell.

        Parameters
        ----------
        na_primary : int
           number of atoms in the primary unit cell
        ret_super : bool, optional
           also return the number of supercells used in each direction

        Returns
        -------
        Geometry
             the primary unit cell
        SuperCell
             the tiled supercell numbers used to find the primary unit cell (only if `ret_super` is true)

        Raises
        ------
        SislError
             If the algorithm fails.
        """
        na = len(self)
        if na % na_primary != 0:
            raise ValueError(self.__class__.__name__ + '.as_primary requires the number of atoms to be divisable by the '
                            'total number of atoms.')

        n_supercells = len(self) // na_primary
        if n_supercells == 1:
            # Return a copy of self
            return self.copy()

        # Now figure out the repetitions along each direction
        fxyz = self.fxyz
        # Move to 0
        fxyz -= fxyz.min(0)
        # Shift a little bit in to account for inaccuracies.
        fxyz += (0.5 - (fxyz.max(0) - fxyz.min(0)) / 2).reshape(1, -1) * 0.01

        # Default guess to 1 along all directions
        supercell = _a.onesi(3)

        n_bin = n_supercells
        while n_bin > 1:

            # Create bins
            bins = np.linspace(0, 1, n_bin + 1)

            # Loop directions where we need to check
            for i in (supercell == 1).nonzero()[0]:

                # A histogram should yield an equal splitting for each bins
                # if the geometry is a n_bin repetition along the i'th direction.
                # Hence if diff == 0 for all elements we have a match.
                diff_bin = np.diff(np.histogram(fxyz[:, i], bins)[0])

                if diff_bin.sum() == 0:
                    supercell[i] = n_bin
                    if np.product(supercell) > n_supercells:
                        # For geometries with more than 1 atom in the primary unit cell
                        # we can get false positives (each layer can be split again)
                        # We will search again the max-value supercell
                        i_max = supercell.argmax()
                        n_bin = supercell[i_max]
                        supercell[i_max] = 1

            # Quick escape if hit the correct number of supercells
            if np.product(supercell) == n_supercells:
                break

            n_bin -= 1

        # Check that the number of supercells match
        if np.product(supercell) != n_supercells:
            raise SislError(self.__class__.__name__ + '.as_primary could not determine the optimal supercell.')

        # Cut down the supercell (TODO this does not correct the number of supercell connections!)
        sc = self.sc.copy()
        for i in range(3):
            sc = sc.cut(supercell[i], i)

        # Now we need to find the atoms that are in the primary cell
        # We do this by finding all coordinates within the primary unit-cell
        fxyz = dot(self.xyz, sc.icell.T)
        # Move to 0 and shift in 0.05 Ang in each direction
        fxyz -= fxyz.min(0)

        # Find minimal distance in each direction
        sc_idx = (supercell > 1).nonzero()[0]
        min_fxyz = _a.zerosd(3)
        for i in sc_idx:
            s_fxyz = np.sort(fxyz[:, i])
            min_fxyz[i] = s_fxyz[(s_fxyz < 1e-4).nonzero()[0][-1] + 1]
        fxyz += min_fxyz * 0.05

        # Find all fractional indices that are below 1
        ind = np.logical_and.reduce(fxyz < 1., axis=1).nonzero()[0]

        geom = self.sub(ind)
        geom.set_supercell(sc)
        if ret_super:
            return geom, supercell
        return geom

    def reorder(self):
        """ Reorders atoms according to first occurence in the geometry

        Notes
        -----
        This is an in-place operation.
        """
        self._atoms = self.atoms.reorder(in_place=True)

    def reduce(self):
        """ Remove all atoms not currently used in the ``self.atoms`` object

        Notes
        -----
        This is an in-place operation.
        """
        self._atoms = self.atoms.reduce(in_place=True)

    def rij(self, ia, ja):
        r""" Distance between atom `ia` and `ja`, atoms can be in super-cell indices

        Returns the distance between two atoms:

        .. math::
            r_{ij} = |r_j - r_i|

        Parameters
        ----------
        ia : int or array_like
           atomic index of first atom
        ja : int or array_like
           atomic indices
        """
        R = self.Rij(ia, ja)

        if len(R.shape) == 1:
            return (R[0] ** 2. + R[1] ** 2 + R[2] ** 2) ** .5

        return fnorm(R)

    def Rij(self, ia, ja):
        r""" Vector between atom `ia` and `ja`, atoms can be in super-cell indices

        Returns the vector between two atoms:

        .. math::
            R_{ij} = r_j - r_i

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
            return xj[:] - xi[:]
        elif np.allclose(xi.shape, xj.shape):
            return xj - xi

        return xj - xi[None, :]

    def orij(self, io, jo):
        r""" Distance between orbital `io` and `jo`, orbitals can be in super-cell indices

        Returns the distance between two orbitals:

        .. math::
            r_{ij} = |r_j - r_i|

        Parameters
        ----------
        io : int or array_like
           orbital index of first orbital
        jo : int or array_like
           orbital indices
        """
        return self.rij(self.o2a(io), self.o2a(jo))

    def oRij(self, io, jo):
        r""" Vector between orbital `io` and `jo`, orbitals can be in super-cell indices

        Returns the vector between two orbitals:

        .. math::
            R_{ij} = r_j - r_i

        Parameters
        ----------
        io : int or array_like
           orbital index of first orbital
        jo : int or array_like
           orbital indices
        """
        return self.Rij(self.o2a(io), self.o2a(jo))

    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads geometry from the `Sile` using `Sile.read_geometry`

        Parameters
        ----------
        sile : Sile, str or pathlib.Path
            a `Sile` object which will be used to read the geometry
            if it is a string it will create a new sile using `get_sile`.

        See Also
        --------
        write : writes a `Geometry` to a given `Sile`/file
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            return sile.read_geometry(*args, **kwargs)
        else:
            with get_sile(sile) as fh:
                return fh.read_geometry(*args, **kwargs)

    def write(self, sile, *args, **kwargs):
        """ Writes geometry to the `Sile` using `sile.write_geometry`

        Parameters
        ----------
        sile : Sile, str or pathlib.Path
            a `Sile` object which will be used to write the geometry
            if it is a string it will create a new sile using `get_sile`
        *args, **kwargs:
            Any other args will be passed directly to the
            underlying routine

        See Also
        --------
        read : reads a `Geometry` from a given `Sile`/file
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            sile.write_geometry(self, *args, **kwargs)
        else:
            with get_sile(sile, 'w') as fh:
                fh.write_geometry(self, *args, **kwargs)

    def __str__(self):
        """ str of the object """
        s = self.__class__.__name__ + '{{na: {0}, no: {1},\n '.format(self.na, self.no)
        s += str(self.atom).replace('\n', '\n ')
        if len(self.names) > 0:
            s += ',\n ' + str(self.names).replace('\n', '\n ')
        return (s + ',\n maxR: {0:.5f},\n {1}\n}}'.format(self.maxR(), str(self.sc).replace('\n', '\n '))).strip()

    def iter(self):
        """ An iterator over all atomic indices

        This iterator is the same as:

        >>> for ia in range(len(self)):
        ...    <do something>

        or equivalently

        >>> for ia in self:
        ...    <do something>

        See Also
        --------
        iter_species : iterate across indices and atomic species
        iter_orbitals : iterate across atomic indices and orbital indices
        """
        for ia in range(len(self)):
            yield ia

    __iter__ = iter

    def iter_species(self, atom=None):
        """ Iterator over all atoms (or a subset) and species as a tuple in this geometry

        >>> for ia, a, idx_specie in self.iter_species():
        ...     isinstance(ia, int) == True
        ...     isinstance(a, Atom) == True
        ...     isinstance(idx_specie, int) == True

        with ``ia`` being the atomic index, ``a`` the `Atom` object, ``idx_specie``
        is the index of the specie

        Parameters
        ----------
        atom : int or array_like, optional
           only loop on the given atoms, default to all atoms

        See Also
        --------
        iter : iterate over atomic indices
        iter_orbitals : iterate across atomic indices and orbital indices
        """
        if atom is None:
            for ia in self:
                yield ia, self.atoms[ia], self.atoms.specie[ia]
        else:
            for ia in self._sanitize_atom(atom).ravel():
                yield ia, self.atoms[ia], self.atoms.specie[ia]

    def iter_orbitals(self, atom=None, local=True):
        r"""
        Returns an iterator over all atoms and their associated orbitals

        >>> for ia, io in self.iter_orbitals():

        with ``ia`` being the atomic index, ``io`` the associated orbital index on atom ``ia``.
        Note that ``io`` will start from ``0``.

        Parameters
        ----------
        atom : int or array_like, optional
           only loop on the given atoms, default to all atoms
        local : bool, optional
           whether the orbital index is the global index, or the local index relative to
           the atom it resides on.

        Yields
        ------
        ia
           atomic index
        io
           orbital index

        See Also
        --------
        iter : iterate over atomic indices
        iter_species : iterate across indices and atomic species
        """
        if atom is None:
            if local:
                for ia, IO in enumerate(zip(self.firsto, self.lasto + 1)):
                    for io in range(IO[1] - IO[0]):
                        yield ia, io
            else:
                for ia, IO in enumerate(zip(self.firsto, self.lasto + 1)):
                    for io in range(IO[0], IO[1]):
                        yield ia, io
        else:
            atom = self._sanitize_atom(atom).ravel()
            if local:
                for ia, io1, io2 in zip(atom, self.firsto[atom], self.lasto[atom] + 1):
                    for io in range(io2 - io1):
                        yield ia, io
            else:
                for ia, io1, io2 in zip(atom, self.firsto[atom], self.lasto[atom] + 1):
                    for io in range(io1, io2):
                        yield ia, io

    def iR(self, na=1000, iR=20, R=None):
        """ Return an integer number of maximum radii (``self.maxR()``) which holds approximately `na` atoms

        Parameters
        ----------
        na : int, optional
           number of atoms within the radius
        iR : int, optional
           initial `iR` value, which the sphere is estitametd from
        R : float, optional
           the value used for atomic range (defaults to ``self.maxR()``)

        Returns
        -------
        int
            number of radius needed to contain `na` atoms. Minimally 2 will be returned.
        """
        ia = np.random.randint(len(self))

        # default block iterator
        if R is None:
            R = self.maxR()
        if R < 0:
            raise ValueError(self.__class__.__name__ + ".iR unable to determine a number of atoms within a sphere with negative radius, is maxR() defined?")

        # Number of atoms within 20 * R
        naiR = max(1, len(self.close(ia, R=R * iR)))

        # Convert to na atoms spherical radii
        iR = int(4 / 3 * np.pi * R ** 3 / naiR * na)

        return max(2, iR)

    def iter_block_rand(self, iR=20, R=None, atom=None):
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

        if iR < 2:
            raise SislError(self.__class__.__name__ + '.iter_block_rand too small iR!')

        if R is None:
            R = self.maxR()
        # The boundaries (ensure complete overlap)
        R = np.array([iR - 0.975, iR + .025]) * R

        append = np.append

        # loop until all passed are true
        while not_passed_N > 0:

            # Take a random non-passed element
            all_true = not_passed.nonzero()[0]

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
            all_idx = self.close(idx, R=R)
            # Get unit-cell atoms
            all_idx[0] = self.sc2uc(all_idx[0], unique=True)
            # First extend the search-space (before reducing)
            all_idx[1] = self.sc2uc(append(all_idx[1], all_idx[0]), unique=True)

            # Only select those who have not been runned yet
            all_idx[0] = all_idx[0][not_passed[all_idx[0]].nonzero()[0]]
            if len(all_idx[0]) == 0:
                raise SislError('Internal error, please report to the developers')

            # Tell the next loop to skip those passed
            not_passed[all_idx[0]] = False
            # Update looped variables
            not_passed_N -= len(all_idx[0])

            # Now we want to yield the stuff revealed
            # all_idx[0] contains the elements that should be looped
            # all_idx[1] contains the indices that can be searched
            yield all_idx[0], all_idx[1]

        if np.any(not_passed):
            print(not_passed.nonzero()[0])
            print(np.sum(not_passed), len(self))
            raise SislError(self.__class__.__name__ + '.iter_block_rand error on iterations. Not all atoms have been visited.')

    def iter_block_shape(self, shape=None, iR=20, atom=None):
        """ Perform the *grid* block-iteration by looping a grid """

        # We implement yields as we can then do nested iterators
        # create a boolean array
        na = len(self)
        if atom is not None:
            not_passed = np.zeros(na, dtype=bool)
            # Reverse the values
            not_passed[atom] = True
        else:
            not_passed = np.ones(na, dtype=bool)

        # Figure out how many we need to loop on
        not_passed_N = np.sum(not_passed)

        if iR < 2:
            raise SislError(self.__class__.__name__ + '.iter_block_shape too small iR!')

        R = self.maxR()
        if shape is None:
            # we default to the Cube shapes
            dS = (Cube(R * (iR - 1.975)),
                  Cube(R * (iR + 0.025)))
        else:
            if isinstance(shape, Shape):
                dS = (shape,)
            else:
                dS = tuple(shape)
            if len(dS) == 1:
                dS += (dS[0].expand(R + 0.01), )
        if len(dS) != 2:
            raise ValueError(self.__class__.__name__ + '.iter_block_shape, number of Shapes *must* be one or two')

        # Now create the Grid
        # convert the radius to a square Grid
        # We do this by examining the x, y, z coordinates
        xyz_m = np.amin(self.xyz, axis=0)
        xyz_M = np.amax(self.xyz, axis=0)
        dxyz = xyz_M - xyz_m

        # Currently iterating different shapes only works for
        # Sphere and Cube
        for s in dS:
            if not isinstance(s, (Cube, Sphere)):
                raise ValueError(self.__class__.__name__ + '.iter_block_shape currently only works for '
                                 'Cube or Sphere objects. Please change sources.')

        # Retrieve the internal diameter
        if isinstance(dS[0], Cube):
            ir = dS[0].edge_length
        elif isinstance(dS[0], Sphere):
            ir = [dS[0].radius * 0.5 ** 0.5 * 2] * 3
        elif isinstance(dS[0], Shape):
            # Convert to spheres (which probably should be cubes for performance)
            dS = [s.toSphere() for s in dS]
            # Now do the same with spheres
            ir = [dS[0].radius * 0.5 ** 0.5 * 2] * 3

        # Figure out number of segments in each iteration
        # (minimum 1)
        ixyz = _a.arrayi(ceil(dxyz / ir + 0.0001))

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
        for x, y, z in product(range(ixyz[0]), range(ixyz[1]), range(ixyz[2])):

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
            all_idx[0] = self.sc2uc(all_idx[0], unique=True)
            # First extend the search-space (before reducing)
            all_idx[1] = self.sc2uc(append(all_idx[1], all_idx[0]), unique=True)

            # Only select those who have not been runned yet
            all_idx[0] = all_idx[0][not_passed[all_idx[0]].nonzero()[0]]
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
            print(not_passed.nonzero()[0])
            print(np.sum(not_passed), len(self))
            raise SislError(self.__class__.__name__ + '.iter_block_shape error on iterations. Not all atoms have been visited.')

    def iter_block(self, iR=20, R=None, atom=None, method='rand'):
        """ Iterator for performance critical loops

        NOTE: This requires that `R` has been set correctly as the maximum interaction range.

        I.e. the loop would look like this:

        >>> for ias, idxs in self.iter_block():
        ...    for ia in ias:
        ...        idx_a = self.close(ia, R = R, idx = idxs)

        This iterator is intended for systems with more than 1000 atoms.

        Remark that the iterator used is non-deterministic, i.e. any two iterators need
        not return the same atoms in any way.

        Parameters
        ----------
        iR : int, optional
            the number of `R` ranges taken into account when doing the iterator
        R : float, optional
            enables overwriting the local R quantity. Defaults to ``self.maxR()``
        atom : array_like, optional
            enables only effectively looping a subset of the full geometry
        method : {'rand', 'sphere', 'cube'}
            select the method by which the block iteration is performed.
            Possible values are:

             `rand`: a spherical object is constructed with a random center according to the internal atoms
             `sphere`: a spherical equispaced shape is constructed and looped
             `cube`: a cube shape is constructed and looped

        Returns
        -------
        numpy.ndarray
            current list of atoms currently searched
        numpy.ndarray
            atoms that needs searching
        """
        if iR < 2:
            raise SislError(self.__class__.__name__ + '.iter_block too small iR!')

        method = method.lower()
        if method == 'rand' or method == 'random':
            for ias, idxs in self.iter_block_rand(iR, R, atom):
                yield ias, idxs
        else:
            if R is None:
                R = self.maxR()

            # Create shapes
            if method == 'sphere':
                dS = (Sphere(R * (iR - 0.975)),
                      Sphere(R * (iR + 0.025)))
            elif method == 'cube':
                dS = (Cube(R * (2 * iR - 0.975)),
                      Cube(R * (2 * iR + 0.025)))

            for ias, idxs in self.iter_block_shape(dS):
                yield ias, idxs

    def copy(self):
        """ A copy of the object. """
        g = self.__class__(np.copy(self.xyz), atom=self.atoms.copy(), sc=self.sc.copy())
        g._names = self.names.copy()
        return g

    def overlap(self, other, eps=0.1, offset=(0., 0., 0.), offset_other=(0., 0., 0.)):
        """ Calculate the overlapping indices between two geometries

        Find equivalent atoms (in the primary unit-cell only) in two geometries.
        This routine finds which atoms have the same atomic positions in `self` and `other`.

        Note that this will return duplicate overlapping atoms if one atoms lies within `eps`
        of more than 1 atom in `other`.

        Parameters
        ----------
        other : Geometry
           Geometry to compare with `self`
        eps : float, optional
           atoms within this distance will be considered *equivalent*
        offset : list of float, optional
           offset for `self.xyz` before comparing
        offset_other : list of float, optional
           offset for `other.xyz` before comparing

        Examples
        --------
        >>> gr22 = sisl.geom.graphene().tile(2, 0).tile(2, 1)
        >>> gr44 = gr22.tile(2, 0).tile(2, 1)
        >>> offset = np.array([0.2, 0.4, 0.4])
        >>> gr22 = gr22.translate(offset)
        >>> idx = np.arange(len(gr22))
        >>> np.random.shuffle(idx)
        >>> gr22 = gr22.sub(idx)
        >>> idx22, idx44 = gr22.overlap(gr44, offset=-offset)
        >>> assert idx22 == np.arange(len(gr22))
        >>> assert idx44 == idx

        Returns
        -------
        idx_self : numpy.ndarray of int
             indices in `self` that are equivalent with `idx_other`
        idx_other : numpy.ndarray of int
             indices in `other` that are equivalent with `idx_self`
        """
        s_xyz = self.xyz + (_a.arrayd(offset) - _a.arrayd(offset_other)).reshape(1, 3)
        idx_self = []
        self_append = idx_self.append
        idx_other = []
        other_append = idx_other.append

        for ia, xyz in enumerate(s_xyz):
            idx = other.close_sc(xyz, R=(eps,))
            for ja in idx:
                self_append(ia)
                other_append(ja)
        return _a.arrayi(idx_self), _a.arrayi(idx_other)

    def sort(self, axes=(2, 1, 0)):
        """ Return an equivalent geometry by sorting the coordinates according to the order of axis

        Examples
        --------
        >>> idx = np.lexsort((self.xyz[:, i] for i in axes))
        >>> new = self.sub(idx)

        Parameters
        ----------
        axes : tuple, optional
           sorting axes (note the last element has highest precedence)

        Returns
        -------
        Geometry
            sorted geometry
        """
        axes = _a.arrayi(axes).ravel()
        idx = np.lexsort(tuple((self.xyz[:, i] for i in axes)))
        return self.sub(idx)

    def optimize_nsc(self, axis=None, R=None):
        """ Optimize the number of supercell connections based on ``self.maxR()``

        After this routine the number of supercells may not necessarily be the same.

        This is an in-place operation.

        Parameters
        ----------
        axis : int or array_like, optional
           only optimize the specified axis (default to all)
        R : float, optional
           the maximum connection radius for each atom
        """
        if axis is None:
            axis = [0, 1, 2]
        else:
            axis = _a.asarrayi(axis).ravel()

        if R is None:
            R = self.maxR()
        if R < 0:
            R = 0.00001
            warn(self.__class__.__name__ +
                 ".optimize_nsc could not determine the radius from the "
                 "internal atoms (defaulting to zero radius).")

        ic = self.icell
        nrc = 1 / fnorm(ic)
        idiv = floor(np.maximum(nrc / (2 * R), 1.1)).astype(np.int32, copy=False)
        imcell = ic * idiv.reshape(-1, 1)

        # We know this is the maximum
        nsc = self.nsc.copy()
        # We need to subtract one to ensure we are not taking into account
        # too big supercell connections.
        # I don't think we need anything other than this.
        # However, until I am sure that this wouldn't change, regardless of the
        # cell. I will keep it.
        Rimcell = R * fnorm(imcell)[axis]
        nsc[axis] = (floor(Rimcell) + ceil(Rimcell % 0.5 - 0.5)).astype(np.int32)
        # Since for 1 it is not sure that it is a connection or not, we limit the search by
        # removing it.
        nsc[axis] = np.where(nsc[axis] > 1, nsc[axis], 0)
        for i in axis:
            # Initialize the isc for this direction
            # (note we do not take non-orthogonal directions
            #  into account)
            isc = _a.zerosi(3)
            isc[i] = nsc[i]
            # Initialize the actual number of supercell connections
            # along this direction.
            prev_isc = isc[i]
            while prev_isc == isc[i]:
                # Try next supercell connection
                isc[i] += 1
                for ia in self:
                    idx = self.close_sc(ia, isc=isc, R=R)
                    if len(idx) > 0:
                        prev_isc = isc[i]
                        break

            # Save the reached supercell connection
            nsc[i] = prev_isc * 2 + 1

        self.set_nsc(nsc)

        return nsc

    def sub(self, atom, cell=None):
        """ Create a new `Geometry` with a subset of this `Geometry`

        Indices passed *MUST* be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atom : int or array_like
            indices/boolean of all atoms to be removed
        cell   : array_like or SuperCell, optional
            the new associated cell of the geometry (defaults to the same cell)

        See Also
        --------
        SuperCell.fit : update the supercell according to a reference supercell
        remove : the negative of this routine, i.e. remove a subset of atoms
        """
        atom = self.sc2uc(atom)
        if cell is None:
            return self.__class__(self.xyz[atom, :],
                                  atom=self.atoms.sub(atom), sc=self.sc.copy())
        return self.__class__(self.xyz[atom, :],
                              atom=self.atoms.sub(atom), sc=cell)

    def cut(self, seps, axis, seg=0, rtol=1e-4, atol=1e-4):
        """ A subset of atoms from the geometry by cutting the geometry into `seps` parts along the direction `axis`.

        This will effectively change the unit-cell in the `axis` as-well
        as removing ``self.na/seps`` atoms.
        It requires that ``self.na % seps == 0``.

        REMARK: You need to ensure that all atoms within the first
        cut out region are within the primary unit-cell.

        Doing ``geom.cut(2, 1).tile(2, 1)``, could for symmetric setups,
        be equivalent to a no-op operation. A ``UserWarning`` will be issued
        if this is not the case.

        This method may be regarded as the opposite of `tile`.

        Parameters
        ----------
        seps : int
            number of times the structure will be cut.
        axis : int
            the axis that will be cut
        seg : int, optional
            returns the i'th segment of the cut structure
            Currently the atomic coordinates are not translated,
            this may change in the future.
        rtol : (tolerance for checking tiling, see `numpy.allclose`)
        atol : (tolerance for checking tiling, see `numpy.allclose`)

        Examples
        --------
        >>> g = sisl.geom.graphene()
        >>> gxyz = g.tile(4, 0).tile(3, 1).tile(2, 2)
        >>> G = gxyz.cut(2, 2).cut(3, 1).cut(4, 0)
        >>> np.allclose(g.xyz, G.xyz)
        True

        See Also
        --------
        tile : opposite method of this
        """
        if self.na % seps != 0:
            raise ValueError(self.__class__.__name__ + '.cut '
                             'cannot be cut into {0} different '.format(seps) +
                             'pieces. Please check your geometry and input.')
        # Truncate to the correct segments
        lseg = seg % seps
        # Cut down cell
        sc = self.sc.cut(seps, axis)
        # List of atoms
        n = self.na // seps
        off = n * lseg
        new = self.sub(_a.arangei(off, off + n), cell=sc)
        if not np.allclose(new.tile(seps, axis).xyz, self.xyz, rtol=rtol, atol=atol):
            st = 'The cut structure cannot be re-created by tiling'
            st += '\nThe difference between the coordinates can be altered using rtol, atol'
            warn(st)
        return new

    def remove(self, atom):
        """ Remove atoms from the geometry.

        Indices passed *MUST* be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atom : int or array_like
            indices/boolean of all atoms to be removed

        See Also
        --------
        sub : the negative of this routine, i.e. retain a subset of atoms
        """
        if isinstance(atom, ndarray) and atom.dtype == bool_:
            atom = np.flatnonzero(atom)
        elif isinstance(atom, str):
            atom = self.names[atom]
        atom = self.sc2uc(atom)
        atom = np.delete(_a.arangei(self.na), atom)
        return self.sub(atom)

    def tile(self, reps, axis):
        """ Tile the geometry to create a bigger one

        The atomic indices are retained for the base structure.

        This method allows to utilise Bloch's theorem when creating
        Hamiltonian parameter sets for TBtrans.

        Tiling and repeating a geometry will result in the same geometry.
        The *only* difference between the two is the final ordering of the atoms.

        Parameters
        ----------
        reps : int
           number of tiles (repetitions)
        axis : int
           direction of tiling, 0, 1, 2 according to the cell-direction

        Examples
        --------
        >>> geom = Geometry([[0, 0, 0], [0.5, 0, 0]], sc=1.)
        >>> g = geom.tile(2,axis=0)
        >>> print(g.xyz) # doctest: +NORMALIZE_WHITESPACE
        [[0.   0.   0. ]
         [0.5  0.   0. ]
         [1.   0.   0. ]
         [1.5  0.   0. ]]
        >>> g = geom.tile(2,0).tile(2,axis=1)
        >>> print(g.xyz) # doctest: +NORMALIZE_WHITESPACE
        [[0.   0.   0. ]
         [0.5  0.   0. ]
         [1.   0.   0. ]
         [1.5  0.   0. ]
         [0.   1.   0. ]
         [0.5  1.   0. ]
         [1.   1.   0. ]
         [1.5  1.   0. ]]

        See Also
        --------
        repeat : equivalent but different ordering of final structure
        cut : opposite method of this
        """
        if reps < 1:
            raise ValueError(self.__class__.__name__ + '.tile() requires a repetition above 0')

        sc = self.sc.tile(reps, axis)

        # Our first repetition *must* be with
        # the former coordinate
        xyz = np.tile(self.xyz, (reps, 1))
        # We may use broadcasting rules instead of repeating stuff
        xyz.shape = (reps, self.na, 3)
        nr = _a.arangei(reps)
        nr.shape = (reps, 1)
        for i in range(3):
            # Correct the unit-cell offsets along `i`
            xyz[:, :, i] += nr * self.cell[axis, i]
        xyz.shape = (-1, 3)

        # Create the geometry and return it (note the smaller atoms array
        # will also expand via tiling)
        return self.__class__(xyz, atom=self.atoms.tile(reps), sc=sc)

    def repeat(self, reps, axis):
        """ Create a repeated geometry

        The atomic indices are *NOT* retained from the base structure.

        The expansion of the atoms are basically performed using this
        algorithm:

        >>> ja = 0
        >>> for ia in range(self.na):
        ...     for id,r in args:
        ...        for i in range(r):
        ...           ja = ia + cell[id,:] * i

        This method allows to utilise Bloch's theorem when creating
        Hamiltonian parameter sets for TBtrans.

        For geometries with a single atom this routine returns the same as
        `tile`.

        Tiling and repeating a geometry will result in the same geometry.
        The *only* difference between the two is the final ordering of the atoms.

        Parameters
        ----------
        reps : int
           number of repetitions
        axis : int
           direction of repetition, 0, 1, 2 according to the cell-direction

        Examples
        --------
        >>> geom = Geometry([[0, 0, 0], [0.5, 0, 0]], sc=1)
        >>> g = geom.repeat(2,axis=0)
        >>> print(g.xyz) # doctest: +NORMALIZE_WHITESPACE
        [[0.   0.   0. ]
         [1.   0.   0. ]
         [0.5  0.   0. ]
         [1.5  0.   0. ]]
        >>> g = geom.repeat(2,0).repeat(2,1)
        >>> print(g.xyz) # doctest: +NORMALIZE_WHITESPACE
        [[0.   0.   0. ]
         [0.   1.   0. ]
         [1.   0.   0. ]
         [1.   1.   0. ]
         [0.5  0.   0. ]
         [0.5  1.   0. ]
         [1.5  0.   0. ]
         [1.5  1.   0. ]]

        See Also
        --------
        tile : equivalent but different ordering of final structure
        """
        if reps < 1:
            raise ValueError(self.__class__.__name__ + '.repeat() requires a repetition above 0')

        sc = self.sc.repeat(reps, axis)

        # Our first repetition *must* be with
        # the former coordinate
        xyz = np.repeat(self.xyz, reps, axis=0)
        # We may use broadcasting rules instead of repeating stuff
        xyz.shape = (self.na, reps, 3)
        nr = _a.arangei(reps)
        nr.shape = (1, reps)
        for i in range(3):
            # Correct the unit-cell offsets along `i`
            xyz[:, :, i] += nr * self.cell[axis, i]
        xyz.shape = (-1, 3)

        # Create the geometry and return it
        return self.__class__(xyz, atom=self.atoms.repeat(reps), sc=sc)

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
        >>> geometry = Geometry([0.] * 3, sc=[1.5, 3, 4])
        >>> geometry * 2 == geometry.tile(2, 0).tile(2, 1).tile(2, 2)
        True
        >>> geometry * [2, 1, 2] == geometry.tile(2, 0).tile(2, 2)
        True
        >>> geometry * [2, 2] == geometry.tile(2, 2)
        True
        >>> geometry * ([2, 1, 2], 'repeat') == geometry.repeat(2, 0).repeat(2, 2)
        True
        >>> geometry * ([2, 1, 2], 'r') == geometry.repeat(2, 0).repeat(2, 2)
        True
        >>> geometry * ([2, 0], 'r') == geometry.repeat(2, 0)
        True
        >>> geometry * ([2, 2], 'r') == geometry.repeat(2, 2)
        True

        See Also
        --------
        tile : specific method to enlarge the geometry
        repeat : specific method to enlarge the geometry
        """
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
                g = getattr(g, method)(m, i)

        elif len(m) == 2:
            #  (r, axis)
            g = getattr(self, method)(m[0], m[1])

        elif len(m) == 3:
            #  (r, r, r)
            g = self.copy()
            for i in range(3):
                g = getattr(g, method)(m[i], i)

        else:
            raise ValueError('Multiplying a geometry has received a wrong argument')

        return g

    __rmul__ = __mul__

    def angle(self, atom, dir=(1., 0, 0), ref=None, rad=False):
        r""" The angle between atom `atom` and the direction `dir`, with possibility of a reference coordinate `ref`

        The calculated angle can be written as this

        .. math::
            \alpha = \arccos \frac{(\mathrm{atom} - \mathrm{ref})\cdot \mathrm{dir}}
            {|\mathrm{atom}-\mathrm{ref}||\mathrm{dir}|}

        and thus lies in the interval :math:`[0 ; \pi]` as one cannot distinguish orientation without
        additional vectors.

        Parameters
        ----------
        atom : int or array_like
           indices/boolean of all atoms where angles should be calculated on
        dir : str, int or array_like, optional
           the direction from which the angle is calculated from, default to ``x``
        ref : int or array_like, optional
           the reference point from which the vectors are drawn, default to origo
        rad : bool, optional
           whether the returned value is in radians
        """
        xi = self.axyz(atom)
        if isinstance(dir, (_str, Integral)):
            dir = self.cell[direction(dir), :]
        else:
            dir = _a.asarrayd(dir)
        # Normalize so we don't have to have this in the
        # below formula
        dir /= fnorm(dir)
        # Broad-casting
        dir.shape = (1, -1)

        if ref is None:
            pass
        elif isinstance(ref, Integral):
            xi -= self.axyz(ref)[None, :]
        else:
            xi -= _a.asarrayd(ref)[None, :]
        nx = sqrt(square(xi).sum(1))
        ang = np.where(nx > 1e-6, np.arccos((xi * dir).sum(axis=1) / nx), 0.)
        if rad:
            return ang
        return np.degrees(ang)

    def rotate(self, angle, v, origo=None, atom=None, only='abc+xyz', rad=False):
        """ Rotate geometry around vector and return a new geometry

        Per default will the entire geometry be rotated, such that everything
        is aligned as before rotation.

        However, by supplying ``only = 'abc|xyz'`` one can designate which
        part of the geometry that will be rotated.

        Parameters
        ----------
        angle : float
             the angle in degrees to rotate the geometry. Set the ``rad``
             argument to use radians.
        v     : array_like
             the normal vector to the rotated plane, i.e.
             v = [1,0,0] will rotate the ``yz`` plane
        origo : int or array_like, optional
             the origin of rotation. Anything but [0, 0, 0] is equivalent
             to a `self.move(-origo).rotate(...).move(origo)`.
             If this is an `int` it corresponds to the atomic index.
        atom : int or array_like, optional
             only rotate the given atomic indices, if not specified, all
             atoms will be rotated.
        only : {'abc+xyz', 'xyz', 'abc'}
             which coordinate subject should be rotated,
             if ``abc`` is in this string the cell will be rotated
             if ``xyz`` is in this string the coordinates will be rotated
        rad : bool, optional
             if ``True`` the angle is provided in radians (rather than degrees)

        See Also
        --------
        Quaternion : class to rotate
        """
        if origo is None:
            origo = [0., 0., 0.]
        elif isinstance(origo, Integral):
            origo = self.axyz(origo)
        origo = _a.asarrayd(origo)

        if not atom is None:
            # Only rotate the unique values
            atom = self.sc2uc(atom, unique=True)

        # Ensure the normal vector is normalized... (flatten == copy)
        vn = _a.asarrayd(v).flatten()
        vn /= fnorm(vn)

        # Rotate by direct call
        if 'abc' in only:
            sc = self.sc.rotate(angle, vn, rad=rad, only=only)
        else:
            sc = self.sc.copy()

        # Copy
        xyz = np.copy(self.xyz)

        if 'xyz' in only:
            # Prepare quaternion...
            q = Quaternion(angle, vn, rad=rad)
            q /= q.norm()
            # subtract and add origo, before and after rotation
            xyz[atom, :] = q.rotate(xyz[atom, :] - origo[None, :]) + origo[None, :]

        return self.__class__(xyz, atom=self.atoms.copy(), sc=sc)

    def rotate_miller(self, m, v):
        """ Align Miller direction along ``v``

        Rotate geometry and cell such that the Miller direction
        points along the Cartesian vector ``v``.
        """
        # Create normal vector to miller direction and cartesian
        # direction
        cp = _a.arrayd([m[1] * v[2] - m[2] * v[1],
                        m[2] * v[0] - m[0] * v[2],
                        m[0] * v[1] - m[1] * v[0]])
        cp /= fnorm(cp)

        lm = _a.arrayd(m)
        lm /= fnorm(lm)
        lv = _a.arrayd(v)
        lv /= fnorm(lv)

        # Now rotate the angle between them
        a = acos(np.sum(lm * lv))
        return self.rotate(a, cp)

    def move(self, v, atom=None, cell=False):
        """ Translates the geometry by `v`

        One can translate a subset of the atoms by supplying `atom`.

        Returns a copy of the structure translated by `v`.

        Parameters
        ----------
        v : array_like
             the vector to displace all atomic coordinates
        atom : int or array_like, optional
             only displace the given atomic indices, if not specified, all
             atoms will be displaced
        cell : bool, optional
             If True the supercell also gets enlarged by the vector
        """
        g = self.copy()
        if atom is None:
            g.xyz[:, :] += np.asarray(v, g.xyz.dtype)[None, :]
        else:
            g.xyz[self._sanitize_atom(atom).ravel(), :] += np.asarray(v, g.xyz.dtype)[None, :]
        if cell:
            g.set_supercell(g.sc.translate(v))
        return g
    translate = move

    def swap(self, a, b):
        """ Swap a set of atoms in the geometry and return a new one

        This can be used to reorder elements of a geometry.

        Parameters
        ----------
        a : array_like
             the first list of atomic coordinates
        b : array_like
             the second list of atomic coordinates
        """
        a = self._sanitize_atom(a)
        b = self._sanitize_atom(b)
        xyz = np.copy(self.xyz)
        xyz[a, :] = self.xyz[b, :]
        xyz[b, :] = self.xyz[a, :]
        return self.__class__(xyz, atom=self.atoms.swap(a, b), sc=self.sc.copy())

    def swapaxes(self, a, b, swap='cell+xyz'):
        """ Swap the axis for the atomic coordinates and the cell vectors

        If ``swapaxes(0,1)`` it returns the 0 and 1 values
        swapped in the ``cell`` variable.

        Parameters
        ----------
        a : int
           axes 1, swaps with `b`
        b : int
           axes 2, swaps with `a`
        swap : {'cell+xyz', 'cell', 'xyz'}
           decide what to swap, if `'cell'` is in `swap` then
           the cell axis are swapped.
           if `'xyz'` is in `swap` then
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
        return self.__class__(xyz, atom=self.atoms.copy(), sc=sc)

    def center(self, atom=None, what='xyz'):
        """ Returns the center of the geometry

        By specifying `what` one can control whether it should be:

        * ``xyz|position``: Center of coordinates (default)
        * ``mm(xyz)``: Center of minimum/maximum of coordinates
        * ``mass``: Center of mass
        * ``cell``: Center of cell

        Parameters
        ----------
        atom : array_like
            list of atomic indices to find center of
        what : {'xyz', 'mm(xyz)', 'mass', 'cell'}
            determine whether center should be of 'cell', mass-centered ('mass'),
            center of minimum/maximum position of atoms or absolute center of the positions.
        """
        if 'cell' == what:
            return self.sc.center()
        if atom is None:
            g = self
        else:
            g = self.sub(atom)
        if 'mass' == what:
            mass = self.mass
            return dot(mass, g.xyz) / np.sum(mass)
        if 'mm(xyz)' == what:
            return (self.xyz.min(0) + self.xyz.max(0)) / 2
        if not ('xyz' in what or 'position' in what):
            raise ValueError(
                'Unknown what, not one of [xyz,position,mass,cell]')
        return np.mean(g.xyz, axis=0)

    def append(self, other, axis, align='none'):
        """ Appends two structures along `axis`

        This will automatically add the ``self.cell[axis,:]`` to all atomic
        coordiates in the `other` structure before appending.

        The basic algorithm is this:

        >>> oxa = other.xyz + self.cell[axis,:][None,:]
        >>> self.xyz = np.append(self.xyz,oxa)
        >>> self.cell[axis,:] += other.cell[axis,:]

        NOTE: The cell appended is only in the axis that
        is appended, which means that the other cell directions
        need not conform.

        Parameters
        ----------
        other : Geometry or SuperCell
            Other geometry class which needs to be appended
            If a `SuperCell` only the super cell will be extended
        axis : int
            Cell direction to which the `other` geometry should be
            appended.
        align : {'none', 'min'}
            By default appending two structures will simply use the coordinates,
            as is.
            With 'min', the routine will shift both the structures along the cell
            axis of `self` such that they coincide at the first atom.

        See Also
        --------
        add : add geometries
        prepend : prending geometries
        attach : attach a geometry
        insert : insert a geometry
        """
        align = align.lower()
        if isinstance(other, SuperCell):
            # Only extend the supercell.
            xyz = np.copy(self.xyz)
            atom = self.atoms.copy()
            sc = self.sc.append(other, axis)
            names = self._names.copy()

        else:
            if align == 'none':
                xyz = np.append(self.xyz, self.cell[axis, :][None, :] + other.xyz, axis=0)
            elif align == 'min':
                # We want to align at the minimum position along the `axis`
                min_f = self.fxyz[:, axis].min()
                min_other_f = dot(other.xyz, self.icell.T)[:, axis].min()
                displ = self.cell[axis, :] * (1 + min_f - min_other_f)
                xyz = np.append(self.xyz, displ[None, :] + other.xyz, axis=0)
            else:
                raise ValueError(self.__class__.__name__ + '.append requires align keyword to be one of [none, min]')
            atom = self.atoms.append(other.atom)
            sc = self.sc.append(other.sc, axis)
            names = self._names.merge(other._names, offset=len(self))

        return self.__class__(xyz, atom=atom, sc=sc, names=names)

    def prepend(self, other, axis, align='none'):
        """ Prepend two structures along `axis`

        This will automatically add the ``self.cell[axis,:]`` to all atomic
        coordiates in the `other` structure before appending.

        The basic algorithm is this:

        >>> oxa = other.xyz
        >>> self.xyz = np.append(oxa, self.xyz + other.cell[axis,:][None,:])
        >>> self.cell[axis,:] += other.cell[axis,:]

        NOTE: The cell prepended is only in the axis that
        is prependend, which means that the other cell directions
        need not conform.

        Parameters
        ----------
        other : Geometry or SuperCell
            Other geometry class which needs to be prepended
            If a `SuperCell` only the super cell will be extended
        axis : int
            Cell direction to which the `other` geometry should be
            prepended
        align : {'none', 'min'}
            By default prepending two structures will simply use the coordinates,
            as is.
            With 'min', the routine will shift both the structures along the cell
            axis of `other` such that they coincide at the first atom.

        See Also
        --------
        add : add geometries
        append : appending geometries
        attach : attach a geometry
        insert : insert a geometry
        """
        align = align.lower()
        if isinstance(other, SuperCell):
            # Only extend the supercell.
            xyz = np.copy(self.xyz)
            atom = self.atoms.copy()
            sc = self.sc.prepend(other, axis)
            names = self._names.copy()

        else:
            if align == 'none':
                xyz = np.append(other.xyz, other.cell[axis, :][None, :] + self.xyz, axis=0)
            elif align == 'min':
                # We want to align at the minimum position along the `axis`
                min_f = other.fxyz[:, axis].min()
                min_other_f = dot(self.xyz, other.icell.T)[:, axis].min()
                displ = other.cell[axis, :] * (1 + min_f - min_other_f)
                xyz = np.append(other.xyz, displ[None, :] + self.xyz, axis=0)
            else:
                raise ValueError(self.__class__.__name__ + '.prepend requires align keyword to be one of [none, min]')
            atom = self.atoms.prepend(other.atom)
            sc = self.sc.append(other.sc, axis)
            names = other._names.merge(self._names, offset=len(other))

        return self.__class__(xyz, atom=atom, sc=sc, names=names)

    def add(self, other):
        """ Merge two geometries (or a Geometry and SuperCell) by adding the two atoms together

        If `other` is a Geometry only the atoms gets added, to also add the supercell vectors
        simply do ``geom.add(other).add(other.sc)``.

        Parameters
        ----------
        other : Geometry or SuperCell
            Other geometry class which is added

        See Also
        --------
        append : appending geometries
        prepend : prending geometries
        attach : attach a geometry
        insert : insert a geometry
        """
        if isinstance(other, SuperCell):
            xyz = self.xyz.copy()
            sc = self.sc + other
            atom = self.atoms.copy()
            names = self._names.copy()
        else:
            xyz = np.append(self.xyz, other.xyz, axis=0)
            sc = self.sc.copy()
            atom = self.atoms.add(other.atom)
            names = self._names.merge(other._names, offset=len(self))
        return self.__class__(xyz, atom=atom, sc=sc, names=names)

    def insert(self, atom, geom):
        """ Inserts other atoms right before index

        We insert the `geom` `Geometry` before `atom`.
        Note that this will not change the unit cell.

        Parameters
        ----------
        atom : int
           the index at which atom the other geometry is inserted
        geom : Geometry
           the other geometry to be inserted

        See Also
        --------
        add : add geometries
        append : appending geometries
        prepend : prending geometries
        attach : attach a geometry
        """
        xyz = np.insert(self.xyz, atom, geom.xyz, axis=0)
        atoms = self.atoms.insert(atom, geom.atom)
        return self.__class__(xyz, atom=atoms, sc=self.sc.copy())

    def __add__(self, b):
        """ Merge two geometries (or geometry and supercell)

        Parameters
        ----------
        self, b : Geometry or SuperCell or tuple or list
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
        >>> (A, 1) + B == A.append(B, 1)

        See Also
        --------
        add : add geometries
        append : appending geometries
        prepend : prending geometries
        """
        if isinstance(b, (SuperCell, Geometry)):
            return self.add(b)
        return self.append(b[0], b[1])

    def __radd__(self, b):
        """ Merge two geometries (or geometry and supercell)

        Parameters
        ----------
        self, b : Geometry or SuperCell or tuple or list
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
        >>> (A, 1) + B == A.append(B, 1)

        See Also
        --------
        add : add geometries
        append : appending geometries
        prepend : prending geometries
        """
        if isinstance(b, (SuperCell, Geometry)):
            return b.add(self)
        return self + b

    def attach(self, s_idx, other, o_idx, dist='calc', axis=None):
        """ Attaches another `Geometry` at the `s_idx` index with respect to `o_idx` using different methods.

        The attached geometry will be inserted at the end of the geometry via `add`.

        Parameters
        ----------
        s_idx : int
           atomic index which is the base position of the attachment. The distance
           between `s_idx` and `o_idx` is `dist`.
        other : Geometry
           the other Geometry to attach at the given point. In this case `dist` from
           `s_idx`.
        o_idx : int
           the index of the atom in `other` that is inserted at `s_idx`.
        dist : array_like or float or str, optional
           the distance (in `Ang`) between the attached coordinates.
           If `dist` is `array_like` it should be the vector between
           the atoms;
           if `dist` is `float` the argument `axis` is required
           and the vector will be calculated along the corresponding latticevector;
           else if `dist` is `str` this will correspond to the
           `method` argument of the `Atom.radius` class of the two
           atoms. Here `axis` is also required.
        axis : int
           specify the direction of the lattice vectors used.
           Not used if `dist` is an array-like argument.
        """
        if isinstance(dist, Real):
            # We have a single rational number
            if axis is None:
                raise ValueError(self.__class__.__name__ + ".attach, `axis` has not been specified, please specify the axis when using a distance")

            # Now calculate the vector that we should have
            # between the atoms
            v = self.cell[axis, :]
            v = v / (v[0]**2 + v[1]**2 + v[2]**2) ** .5 * dist

        elif isinstance(dist, string_types):
            # We have a single rational number
            if axis is None:
                raise ValueError(self.__class__.__name__ + ".attach, `axis` has not been specified, please specify the axis when using a distance")

            # This is the empirical distance between the atoms
            d = self.atoms[s_idx].radius(dist) + other.atoms[o_idx].radius(dist)
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

        Parameters
        ----------
        atom : int or array_like, optional
             only reverse the given atomic indices, if not specified, all
             atoms will be reversed
        """
        if atom is None:
            xyz = self.xyz[::-1, :]
        else:
            atom = self._sanitize_atom(atom)
            xyz = np.copy(self.xyz)
            xyz[atom, :] = self.xyz[atom[::-1], :]
        return self.__class__(xyz, atom=self.atoms.reverse(atom), sc=self.sc.copy())

    def mirror(self, plane, atom=None):
        """ Mirrors the atomic coordinates by multiplying by -1

        This will typically move the atomic coordinates outside of the unit-cell.
        This method should be used with care.

        Parameters
        ----------
        plane : {'xy'/'ab', 'yz'/'bc', 'xz'/'ac'}
           mirror the structure across the lattice vector plane
        atom : array_like, optional
           only mirror a subset of atoms
        """
        if not atom is None:
            atom = self._sanitize_atom(atom)
        else:
            atom = slice(None)
        g = self.copy()
        lplane = ''.join(sorted(plane.lower()))
        if lplane in ['xy', 'ab']:
            g.xyz[atom, 2] *= -1
        elif lplane in ['yz', 'bc']:
            g.xyz[atom, 0] *= -1
        elif lplane in ['xz', 'ac']:
            g.xyz[atom, 1] *= -1
        return self.__class__(g.xyz, atom=g.atom, sc=self.sc.copy())

    @property
    def fxyz(self):
        """ Returns geometry coordinates in fractional coordinates """
        return dot(self.xyz, self.icell.T)

    def axyz(self, atom=None, isc=None):
        """ Return the atomic coordinates in the supercell of a given atom.

        The ``Geometry[...]`` slicing is calling this function with appropriate options.

        Parameters
        ----------
        atom : int or array_like
          atom(s) from which we should return the coordinates, the atomic indices
          may be in supercell format.
        isc : array_like, optional
            Returns the atomic coordinates shifted according to the integer
            parts of the cell. Defaults to the unit-cell

        Examples
        --------
        >>> geom = Geometry([[0, 0, 0], [0.5, 0, 0]], sc=1.)
        >>> print(geom.axyz(isc=[1,0,0])) # doctest: +NORMALIZE_WHITESPACE
        [[1.   0.   0. ]
         [1.5  0.   0. ]]

        >>> geom = Geometry([[0, 0, 0], [0.5, 0, 0]], sc=1.)
        >>> print(geom.axyz(0)) # doctest: +NORMALIZE_WHITESPACE
        [0.  0.  0.]

        """
        if atom is None and isc is None:
            return self.xyz

        if not atom is None:
            atom = self._sanitize_atom(atom)

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

        if atom.ndim == 0:
            return self.axyz(atom) + offset

        return self.axyz(atom) + offset[None, :]

    def scale(self, scale):
        """ Scale coordinates and unit-cell to get a new geometry with proper scaling

        Parameters
        ----------
        scale : float
           the scale factor for the new geometry (lattice vectors, coordinates
           and the atomic radii are scaled).
        """
        xyz = self.xyz * scale
        atom = self.atoms.scale(scale)
        sc = self.sc.scale(scale)
        return self.__class__(xyz, atom=atom, sc=sc)

    def within_sc(self, shapes, isc=None,
                  idx=None, idx_xyz=None,
                  ret_xyz=False, ret_rij=False):
        """ Indices of atoms in a given supercell within a given shape from a given coordinate

        This returns a set of atomic indices which are within a
        sphere of radius ``R``.

        If R is a tuple/list/array it will return the indices:
        in the ranges:

        >>> ( x <= R[0] , R[0] < x <= R[1], R[1] < x <= R[2] )

        Parameters
        ----------
        shapes : Shape or list of Shape
            A list of increasing shapes that define the extend of the geometric
            volume that is searched.
            It is vital that::

               shapes[0] in shapes[1] in shapes[2] ...
        isc : array_like, optional
            The super-cell which the coordinates are checked in. Defaults to ``[0, 0, 0]``
        idx : array_like, optional
            List of atoms that will be considered. This can
            be used to only take out a certain atoms.
        idx_xyz : array_like, optional
            The atomic coordinates of the equivalent `idx` variable (`idx` must also be passed)
        ret_xyz : bool, optional
            If True this method will return the coordinates
            for each of the couplings.
        ret_rij : bool, optional
            If True this method will return the distance to the center of the shapes

        Returns
        -------
        index
            indices of atoms (in supercell indices) within the shape
        xyz
            atomic coordinates of the indexed atoms (only for true `ret_xyz`)
        rij
            distance of the indexed atoms to the center of the shape (only for true `ret_rij`)
        """

        # Ensure that `shapes` is a list
        if isinstance(shapes, Shape):
            shapes = [shapes]
        nshapes = len(shapes)

        # Convert to actual array
        if idx is not None:
            if not isndarray(idx):
                idx = _a.asarrayi(idx).ravel()
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
        ix = shapes[-1].within_index(xa)
        # Reduce search space
        xa = xa[ix, :]

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
            d = sqrt(square(xa - off[None, :]).sum(1))

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
            x = s.within_index(xa)
            if i > 0:
                x = np.setdiff1d(x, cum, assume_unique=True)
            # Update elements to remove in next loop
            cum = np.append(cum, x)
            ixS.append(x)

        # Do for the first shape
        ret = [[_a.asarrayi(idx[ixS[0]]).ravel()]]
        rc = 0
        if ret_xyz:
            rc = rc + 1
            ret.append([xa[ixS[0], :]])
        if ret_rij:
            rd = rc + 1
            ret.append([d[ixS[0]]])
        for i in range(1, nshapes):
            ret[0].append(_a.asarrayi(idx[ixS[i]]).ravel())
            if ret_xyz:
                ret[rc].append(xa[ixS[i], :])
            if ret_rij:
                ret[rd].append(d[ixS[i]])

        if ret_xyz or ret_rij:
            return ret
        return ret[0]

    def close_sc(self, xyz_ia, isc=(0, 0, 0), R=None,
                 idx=None, idx_xyz=None,
                 ret_xyz=False, ret_rij=False):
        """ Indices of atoms in a given supercell within a given radius from a given coordinate

        This returns a set of atomic indices which are within a
        sphere of radius `R`.

        If `R` is a tuple/list/array it will return the indices:
        in the ranges:

        >>> ( x <= R[0] , R[0] < x <= R[1], R[1] < x <= R[2] )

        Parameters
        ----------
        xyz_ia : array_like of floats or int
            Either a point in space or an index of an atom.
            If an index is passed it is the equivalent of passing
            the atomic coordinate ``close_sc(self.xyz[xyz_ia,:])``.
        isc : array_like, optional
            The super-cell which the coordinates are checked in.
        R : float or array_like, optional
            The radii parameter to where the atomic connections are found.
            If `R` is an array it will return the indices:
            in the ranges ``( x <= R[0] , R[0] < x <= R[1], R[1] < x <= R[2] )``.
            If a single float it will return ``x <= R``.
        idx : array_like of int, optional
            List of atoms that will be considered. This can
            be used to only take out a certain atoms.
        idx_xyz : array_like of float, optional
            The atomic coordinates of the equivalent `idx` variable (`idx` must also be passed)
        ret_xyz : bool, optional
            If True this method will return the coordinates
            for each of the couplings.
        ret_rij : bool, optional
            If True this method will return the distance
            for each of the couplings.

        Returns
        -------
        index
            indices of atoms (in supercell indices) within the shells of radius `R`
        xyz
            atomic coordinates of the indexed atoms (only for true `ret_xyz`)
        rij
            distance of the indexed atoms to the center coordinate (only for true `ret_rij`)
        """
        if R is None:
            R = np.array([self.maxR()], np.float64)
        elif not isndarray(R):
            R = _a.asarrayd(R).ravel()

        # Maximum distance queried
        max_R = R[-1]

        # Convert to actual array
        if idx is not None:
            if not isndarray(idx):
                idx = _a.asarrayi(idx).ravel()
        else:
            # If idx is None, then idx_xyz cannot be used!
            idx_xyz = None

        if isinstance(xyz_ia, Integral):
            off = self.xyz[xyz_ia, :]
        elif not isndarray(xyz_ia):
            off = _a.asarrayd(xyz_ia)
        else:
            off = xyz_ia

        # Calculate the complete offset
        foff = self.sc.offset(isc)[:] - off[:]

        # Get atomic coordinate in principal cell
        if idx_xyz is None:
            dxa = self.axyz(idx) + foff.reshape(1, 3)
        else:
            # For extremely large systems re-using the
            # idx_xyz is faster than indexing
            # a very large array
            dxa = idx_xyz + foff.reshape(1, 3)

        # Immediately downscale by easy checking
        # This will reduce the computation of the vector-norm
        # which is the main culprit of the time-consumption
        # This abstraction will _only_ help very large
        # systems.
        # For smaller ones this will actually be a slower
        # method..
        if idx is None:
            idx, d = indices_in_sphere_with_dist(dxa, max_R)
            dxa = dxa[idx, :].reshape(-1, 3)
        else:
            ix, d = indices_in_sphere_with_dist(dxa, max_R)
            idx = idx[ix]
            dxa = dxa[ix, :].reshape(-1, 3)
            del ix

        if len(idx) == 0:
            # Create default return
            ret = [[_a.emptyi([0])] * len(R)]
            if ret_xyz:
                ret.append([_a.emptyd([0, 3])] * len(R))
            if ret_rij:
                ret.append([_a.emptyd([0])] * len(R))

            # Quick return if there are
            # no entries...
            if len(R) == 1:
                if ret_xyz and ret_rij:
                    return [ret[0][0], ret[1][0], ret[2][0]]
                elif ret_xyz or ret_rij:
                    return [ret[0][0], ret[1][0]]
                return ret[0][0]
            if ret_xyz or ret_rij:
                return ret
            return ret[0]

        if ret_xyz:
            xa = dxa[:, :] + off[None, :]
        del dxa  # just because this array could be very big...

        # Check whether we only have one range to check.
        # If so, we need not reduce the index space
        if len(R) == 1:
            ret = [idx]
            if ret_xyz:
                ret.append(xa)
            if ret_rij:
                ret.append(d)
            if ret_xyz or ret_rij:
                return ret
            return ret[0]

        if not is_ascending(R):
            raise ValueError(self.__class__.__name__ + '.close_sc proximity checks for several '
                             'quantities at a time requires ascending R values.')

        # The more neigbours you wish to find the faster this becomes
        # We only do "one" heavy duty search,
        # then we immediately reduce search space to this subspace
        tidx = indices_le(d, R[0])
        ret = [[idx[tidx]]]
        r_app = ret[0].append
        if ret_xyz:
            ret.append([xa[tidx]])
            r_appx = ret[1].append
        if ret_rij:
            ret.append([d[tidx]])
            r_appd = ret[-1].append

        if ret_xyz and ret_rij:
            for i in range(1, len(R)):
                # Search in the sub-space
                # Notice that this sub-space reduction will never
                # allow the same indice to be in two ranges (due to
                # numerics)
                tidx = indices_gt_le(d, R[i-1], R[i])
                r_app(idx[tidx])
                r_appx(xa[tidx])
                r_appd(d[tidx])
        elif ret_xyz:
            for i in range(1, len(R)):
                tidx = indices_gt_le(d, R[i-1], R[i])
                r_app(idx[tidx])
                r_appx(xa[tidx])
        elif ret_rij:
            for i in range(1, len(R)):
                tidx = indices_gt_le(d, R[i-1], R[i])
                r_app(idx[tidx])
                r_appd(d[tidx])
        else:
            for i in range(1, len(R)):
                tidx = indices_gt_le(d, R[i-1], R[i])
                r_app(idx[tidx])

        if ret_xyz or ret_rij:
            return ret
        return ret[0]

    def __currently_not_used_close_rec(self, xyz_ia, R=None,
                 idx=None, idx_xyz=None,
                 ret_xyz=False, ret_rij=False):
        """ Indices of atoms in a given supercell within a given radius from a given coordinate

        This returns a set of atomic indices which are within a
        sphere of radius `R`.

        If `R` is a tuple/list/array it will return the indices:
        in the ranges:

        >>> ( x <= R[0] , R[0] < x <= R[1], R[1] < x <= R[2] )

        Parameters
        ----------
        xyz_ia : array_like of floats or int
            Either a point in space or an index of an atom.
            If an index is passed it is the equivalent of passing
            the atomic coordinate ``close_rec(self.xyz[xyz_ia,:])``.
        R : float or array_like, optional
            The radii parameter to where the atomic connections are found.
            If `R` is an array it will return the indices:
            in the ranges ``( x <= R[0] , R[0] < x <= R[1], R[1] < x <= R[2] )``.
            If a single float it will return ``x <= R``.
        idx : array_like of int, optional
            List of atoms that will be considered. This can
            be used to only take out a certain atoms.
        idx_xyz : array_like of float, optional
            The atomic coordinates of the equivalent `idx` variable (`idx` must also be passed)
        ret_xyz : bool, optional
            If True this method will return the coordinates
            for each of the couplings.
        ret_rij : bool, optional
            If True this method will return the distance
            for each of the couplings.
        """
        if R is None:
            R = np.array([self.maxR()], np.float64)
        elif not isndarray(R):
            R = _a.asarrayd(R).ravel()

        # Maximum distance queried
        max_R = R[-1]

        # This way of calculating overlapping regions is heavily inspired by
        # initial work by Jose Soler from Siesta.

        # Retrieve reciprocal lattice to divide the mesh into reciprocal divisions.
        icell = self.icell

        # Calculate number of mesh-divisions
        divisions = np.maximum(2. / fnorm(icell) / max_R, 1).floor(dtype=int32)
        divisions.shape = (-1, 1)
        celld = self.cell / divisions
        idcell = divisions * icell

        # Calculate mesh indices for atoms
        xyz = self.xyz
        mesh_a = dot(xyz, imcell.T) # dmx
        mesh_i = mesh_a.floor(dtype=int32)
        subtract(mesh_a, mesh_i, out=mesh_a)
        mesh_i = mesh_i.astype(int32) # imx
        mod(mesh_i, divisions.T, out=mesh_i)

        # Calculate atomic positions in the mesh
        a_pos = dot(mesh_a, celld)

        if isinstance(xyz_ia, Integral):
            coord = self.xyz[xyz_ia, :]
        elif not isndarray(xyz_ia):
            coord = _a.asarrayd(xyz_ia)
        else:
            coord = xyz_ia

        # Transform into cell-mesh divisions
        c_a = dot(coord, rmcell.T) # dmx
        c_i = c_a.floor(dtype=int32)
        c_a = c_a - c_i
        c_i = c_i.astype(int32) # imx
        mod(c_i, divisions.ravel(), out=c_i)
        c_pos = dot(c_a, celld)

    def bond_correct(self, ia, atom, method='calc'):
        """ Corrects the bond between `ia` and the `atom`.

        Corrects the bond-length between atom `ia` and `atom` in such
        a way that the atomic radius is preserved.
        I.e. the sum of the bond-lengths minimizes the distance matrix.

        Only atom `ia` is moved.

        Parameters
        ----------
        ia : int
            The atom to be displaced according to the atomic radius
        atom : array_like or int
            The atom(s) from which the radius should be reduced.
        method : str, float, optional
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
            idx, c, d = self.close(ia, R=(0.1, 10.), idx=algo,
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
            except Exception:
                # get radius
                rad = self.atoms[idx].radius(method) \
                      + self.atoms[ia].radius(method)

            # Update the coordinate
            self.xyz[ia, :] = c + bv / d * rad

        else:
            raise NotImplementedError(
                'Changing bond-length dependent on several lacks implementation.')

    def within(self, shapes,
            idx=None, idx_xyz=None,
            ret_xyz=False, ret_rij=False):
        """ Indices of atoms in the entire supercell within a given shape from a given coordinate

        This heavily relies on the `within_sc` method.

        Note that if a connection is made in a neighbouring super-cell
        then the atomic index is shifted by the super-cell index times
        number of atoms.
        This allows one to decipher super-cell atoms from unit-cell atoms.

        Parameters
        ----------
        shapes : Shape, list of Shape
        idx : array_like, optional
            List of indices for atoms that are to be considered
        idx_xyz : array_like, optional
            The atomic coordinates of the equivalent `idx` variable (`idx` must also be passed)
        ret_xyz : bool, optional
            If true this method will return the coordinates
            for each of the couplings.
        ret_rij : bool, optional
            If true this method will return the distances from the `xyz_ia`
            for each of the couplings.

        Returns
        -------
        index
            indices of atoms (in supercell indices) within the shape
        xyz
            atomic coordinates of the indexed atoms (only for true `ret_xyz`)
        rij
            distance of the indexed atoms to the center of the shape (only for true `ret_rij`)
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

    def close(self, xyz_ia, R=None,
            idx=None, idx_xyz=None,
            ret_xyz=False, ret_rij=False):
        """ Indices of atoms in the entire supercell within a given radius from a given coordinate

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
            the atomic coordinate ``close_sc(self.xyz[xyz_ia,:])``.
        R : (None), float/tuple of float
            The radii parameter to where the atomic connections are found.
            If `R` is an array it will return the indices:
            in the ranges:

            >>> ( x <= R[0] , R[0] < x <= R[1], R[1] < x <= R[2] )

            If a single float it will return:

            >>> x <= R

        idx : array_like, optional
            List of indices for atoms that are to be considered
        idx_xyz : array_like, optional
            The atomic coordinates of the equivalent `idx` variable (`idx` must also be passed)
        ret_xyz : bool, optional
            If true this method will return the coordinates
            for each of the couplings.
        ret_rij : bool, optional
            If true this method will return the distances from the `xyz_ia`
            for each of the couplings.

        Returns
        -------
        index
            indices of atoms (in supercell indices) within the shells of radius `R`
        xyz
            atomic coordinates of the indexed atoms (only for true `ret_xyz`)
        rij
            distance of the indexed atoms to the center coordinate (only for true `ret_rij`)
        """
        if R is None:
            R = self.maxR()
        R = _a.asarrayd(R).ravel()

        # Convert inedx coordinate to point
        if isinstance(xyz_ia, Integral):
            xyz_ia = self.xyz[xyz_ia, :]
        elif not isndarray(xyz_ia):
            xyz_ia = _a.asarrayd(xyz_ia)

        # Get global calls
        # Is faster for many loops
        concat = np.concatenate

        ret = [[np.empty([0], np.int32)] * len(R)]
        i = 0
        if ret_xyz:
            c = i + 1
            i += 1
            ret.append([np.empty([0, 3], np.float64)] * len(R))
        if ret_rij:
            d = i + 1
            i += 1
            ret.append([np.empty([0], np.float64)] * len(R))

        ret_special = ret_xyz or ret_rij

        for s in range(self.n_s):

            na = self.na * s
            sret = self.close_sc(xyz_ia,
                self.sc.sc_off[s, :], R=R,
                idx=idx, idx_xyz=idx_xyz,
                ret_xyz=ret_xyz, ret_rij=ret_rij)

            if not ret_special:
                # This is to "fake" the return
                # of a list (we will do indexing!)
                sret = [sret]

            if isinstance(sret[0], list):
                # we have a list of arrays (len(R) > 1)
                for i, x in enumerate(sret[0]):
                    ret[0][i] = concat((ret[0][i], x + na), axis=0)
                    if ret_xyz:
                        ret[c][i] = concat((ret[c][i], sret[c][i]), axis=0)
                    if ret_rij:
                        ret[d][i] = concat((ret[d][i], sret[d][i]), axis=0)
            elif len(sret[0]) > 0:
                # We can add it to the list (len(R) == 1)
                # We add the atomic offset for the supercell index
                ret[0][0] = concat((ret[0][0], sret[0] + na), axis=0)
                if ret_xyz:
                    ret[c][0] = concat((ret[c][0], sret[c]), axis=0)
                if ret_rij:
                    ret[d][0] = concat((ret[d][0], sret[d]), axis=0)

        if len(R) == 1:
            if ret_xyz and ret_rij:
                return [ret[0][0], ret[1][0], ret[2][0]]
            elif ret_xyz or ret_rij:
                return [ret[0][0], ret[1][0]]
            return ret[0][0]

        if ret_special:
            return ret

        return ret[0]

    def a2transpose(self, atom1, atom2=None):
        """ Transposes connections from `atom1` to `atom2` such that supercell connections are transposed

        When handling supercell indices it is useful to get the *transposed* connection. I.e. if you have
        a connection from site ``i`` (in unit cell indices) to site ``j`` (in supercell indices) it may be
        useful to get the equivalent supercell connection such for site ``j`` (in unit cell indices) to
        site ``i`` (in supercell indices) such that they correspond to the transposed coupling.

        Note that since this transposes couplings the indices returned are always expanded to the full
        length if either of the inputs are a single index.

        Examples
        --------
        >>> gr = geom.graphene()
        >>> idx = gr.close(0, 1.5)
        >>> idx
        array([0, 1, 5, 9], dtype=int32)
        >>> gr.a2transpose(0, idx)
        (array([0, 1, 1, 1], dtype=int32), array([ 0,  0, 14, 10], dtype=int32))

        Parameters
        ----------
        atom1 : array_like
            atomic indices must have same length as `atom2` or length 1
        atom2 : array_like, optional
            atomic indices must have same length as `atom1` or length 1.
            If not present then only `atom1` will be returned in transposed indices.

        Returns
        -------
        atom2 : array_like
            transposed indices for atom2 (only returned if `atom2` is not None)
        atom1 : array_like
            transposed indices for atom1
        """
        # First check whether they have the same size, if so then do not pre-process
        atom1 = self._sanitize_atom(atom1)
        if atom2 is None:
            # we only need to transpose atom1
            offset = self.sc.sc_index(-self.a2isc(atom1)) * self.na
            return atom1 % self.na + offset

        atom2 = self._sanitize_atom(atom2)
        if atom1.size == atom2.size:
            pass
        elif atom1.size == 1: # typical case where atom1 is a single number
            atom1 = np.tile(atom1, atom2.size)
        elif atom2.size == 1:
            atom2 = np.tile(atom2, atom1.size)
        else:
            raise ValueError(self.__class__.__name__ + '.a2transpose only allows length 1 or same length arrays.')

        # Now convert atoms
        na = self.na
        sc_index = self.sc.sc_index
        isc1 = self.a2isc(atom1)
        isc2 = self.a2isc(atom2)

        atom1 = atom1 % na + sc_index(-isc2) * na
        atom2 = atom2 % na + sc_index(-isc1) * na
        return atom2, atom1

    def o2transpose(self, orb1, orb2=None):
        """ Transposes connections from `orb1` to `orb2` such that supercell connections are transposed

        When handling supercell indices it is useful to get the *transposed* connection. I.e. if you have
        a connection from site ``i`` (in unit cell indices) to site ``j`` (in supercell indices) it may be
        useful to get the equivalent supercell connection such for site ``j`` (in unit cell indices) to
        site ``i`` (in supercell indices) such that they correspond to the transposed coupling.

        Note that since this transposes couplings the indices returned are always expanded to the full
        length if either of the inputs are a single index.

        Examples
        --------
        >>> gr = geom.graphene() # one orbital per site
        >>> idx = gr.close(0, 1.5)
        >>> idx
        array([0, 1, 5, 9], dtype=int32)
        >>> gr.o2transpose(0, idx)
        (array([0, 1, 1, 1], dtype=int32), array([ 0,  0, 14, 10], dtype=int32))

        Parameters
        ----------
        orb1 : array_like
            orbital indices must have same length as `orb2` or length 1
        orb2 : array_like, optional
            orbital indices must have same length as `orb1` or length 1.
            If not present then only `orb1` will be returned in transposed indices.

        Returns
        -------
        orb2 : array_like
            transposed indices for orb2 (only returned if `orb2` is not None)
        orb1 : array_like
            transposed indices for orb1
        """
        # First check whether they have the same size, if so then do not pre-process
        orb1 = self._sanitize_orb(orb1)
        if orb2 is None:
            # we only need to transpose orb1
            offset = self.sc.sc_index(-self.o2isc(orb1)) * self.no
            return orb1 % self.no + offset

        orb2 = self._sanitize_orb(orb2)
        if orb1.size == orb2.size:
            pass
        elif orb1.size == 1: # typical case where orb1 is a single number
            orb1 = np.tile(orb1, orb2.size)
        elif orb2.size == 1:
            orb2 = np.tile(orb2, orb1.size)
        else:
            raise ValueError(self.__class__.__name__ + '.o2transpose only allows length 1 or same length arrays.')

        # Now convert orbs
        no = self.no
        sc_index = self.sc.sc_index
        isc1 = self.o2isc(orb1)
        isc2 = self.o2isc(orb2)

        orb1 = orb1 % no + sc_index(-isc2) * no
        orb2 = orb2 % no + sc_index(-isc1) * no
        return orb2, orb1

    def a2o(self, ia, all=False):
        """
        Returns an orbital index of the first orbital of said atom.
        This is particularly handy if you want to create
        TB models with more than one orbital per atom.

        Note that this will preserve the super-cell offsets.

        Parameters
        ----------
        ia : array_like
             Atomic indices
        all : bool, optional
             ``False``, return only the first orbital corresponding to the atom,
             ``True``, returns list of the full atom
        """
        ia = self._sanitize_atom(ia)
        if not all:
            return self.firsto[ia % self.na] + (ia // self.na) * self.no
        off = (ia // self.na) * self.no
        ia = ia % self.na
        ob = self.firsto[ia] + off
        oe = self.lasto[ia] + off + 1

        # Create ranges
        if isinstance(ob, Integral):
            return _a.arangei(ob, oe)

        return array_arange(ob, oe)

    def o2a(self, io, unique=False):
        """ Atomic index corresponding to the orbital indicies.

        This is a particurlaly slow algorithm due to for-loops.

        Note that this will preserve the super-cell offsets.

        Parameters
        ----------
        io : array_like
             List of indices to return the atoms for
        unique : bool, optional
             If True only return the unique atoms.
        """
        if isinstance(io, Integral):
            if unique:
                return np.unique(np.argmax(io % self.no <= self.lasto) + (io // self.no) * self.na)
            return np.argmax(io % self.no <= self.lasto) + (io // self.no) * self.na

        a = list_index_le(_a.asarrayi(io).ravel() % self.no, self.lasto)
        if unique:
            return np.unique(a + (io // self.no) * self.na)
        return a + (io // self.no) * self.na

    def uc2sc(self, atom, unique=False):
        """ Returns atom from unit-cell indices to supercell indices, possibly removing dublicates

        Parameters
        ----------
        atom : array_like or int
           the atomic unit-cell indices to be converted to supercell indices
        unique : bool, optional
           If True the returned indices are unique and sorted.
        """
        atom = self._sanitize_atom(atom) % self.na
        atom = (atom.reshape(1, -1) + _a.arangei(self.n_s).reshape(-1, 1) * self.na).ravel()
        if unique:
            return np.unique(atom)
        return atom
    auc2sc = uc2sc

    def sc2uc(self, atom, unique=False):
        """ Returns atom from supercell indices to unit-cell indices, possibly removing dublicates

        Parameters
        ----------
        atom : array_like or int
           the atomic supercell indices to be converted to unit-cell indices
        unique : bool, optional
           If True the returned indices are unique and sorted.
        """
        atom = self._sanitize_atom(atom) % self.na
        if unique:
            return np.unique(atom)
        return atom
    asc2uc = sc2uc

    def osc2uc(self, orb, unique=False):
        """ Returns orbitals from supercell indices to unit-cell indices, possibly removing dublicates

        Parameters
        ----------
        orb : array_like or int
           the orbital supercell indices to be converted to unit-cell indices
        unique : bool, optional
           If True the returned indices are unique and sorted.
        """
        orb = _a.asarrayi(orb) % self.no
        if unique:
            return np.unique(orb)
        return orb

    def ouc2sc(self, orb, unique=False):
        """ Returns orbitals from unit-cell indices to supercell indices, possibly removing dublicates

        Parameters
        ----------
        orb : array_like or int
           the orbital unit-cell indices to be converted to supercell indices
        unique : bool, optional
           If True the returned indices are unique and sorted.
        """
        orb = _a.asarrayi(orb) % self.no
        orb = (orb.reshape(1, -1) + _a.arangei(self.n_s).reshape(-1, 1) * self.no).ravel()
        if unique:
            return np.unique(orb)
        return orb

    def a2isc(self, ia):
        """ Returns super-cell index for a specific/list atom

        Returns a vector of 3 numbers with integers.
        """
        idx = self._sanitize_atom(ia) // self.na
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
        idx = _a.asarrayi(io) // self.no
        return self.sc.sc_off[idx, :]

    def o2sc(self, o):
        """
        Returns the super-cell offset for a specific orbital.
        """
        return self.sc.offset(self.o2isc(o))

    def __plot__(self, axis=None, supercell=True, axes=False,
                 atom_indices=False, *args, **kwargs):
        """ Plot the geometry in a specified ``matplotlib.Axes`` object.

        Parameters
        ----------
        axis : array_like, optional
           only plot a subset of the axis, defaults to all axis
        supercell : bool, optional
           If `True` also plot the supercell structure
        atom_indices : bool, optional
           if true, also add atomic numbering in the plot (0-based)
        axes : bool or matplotlib.Axes, optional
           the figure axes to plot in (if ``matplotlib.Axes`` object).
           If `True` it will create a new figure to plot in.
           If `False` it will try and grap the current figure and the current axes.
        """
        # Default dictionary for passing to newly created figures
        d = dict()

        colors = np.linspace(0, 1, num=self.atoms.nspecie, endpoint=False)
        colors = colors[self.atoms.specie]
        if 's' in kwargs:
            area = kwargs.pop('s')
        else:
            area = _a.arrayd(self.atoms.Z)
            area[:] *= 20 * np.pi / area.min()

        if axis is None:
            axis = [0, 1, 2]

        # Ensure we have a new 3D Axes3D
        if len(axis) == 3:
            d['projection'] = '3d'

        # The Geometry determines the axes, then we pass it to supercell.
        axes = plt.get_axes(axes, **d)

        # Start by plotting the supercell
        if supercell:
            axes = self.sc.__plot__(axis, axes=axes, *args, **kwargs)

        # Create short-hand
        xyz = self.xyz

        if axes.__class__.__name__.startswith('Axes3D'):
            # We should plot in 3D plots
            axes.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=area, c=colors, alpha=0.8)
            axes.set_zlabel('Ang')
            if atom_indices:
                for i, loc in enumerate(xyz):
                    axes.text(loc[0], loc[1], loc[2], str(i), verticalalignment='bottom')

        else:
            axes.scatter(xyz[:, axis[0]], xyz[:, axis[1]], s=area, c=colors, alpha=0.8)
            if atom_indices:
                for i, loc in enumerate(xyz):
                    axes.text(loc[axis[0]], loc[axis[1]], str(i), verticalalignment='bottom')

        axes.set_xlabel('Ang')
        axes.set_ylabel('Ang')

        return axes

    @classmethod
    def fromASE(cls, aseg):
        """ Returns geometry from an ASE object.

        Parameters
        ----------
        aseg : ASE ``Atoms`` object which contains the following routines:
            ``get_atomic_numbers``, ``get_positions``, ``get_cell``.
            From those methods a `Geometry` object will be created.
        """
        Z = aseg.get_atomic_numbers()
        xyz = aseg.get_positions()
        cell = aseg.get_cell()
        # Convert to sisl object
        return cls(xyz, atom=Z, sc=cell)

    def toASE(self):
        """ Returns the geometry as an ASE ``Atoms`` object """
        from ase import Atoms as ASE_Atoms
        return ASE_Atoms(symbols=self.atoms.Z, positions=self.xyz.tolist(),
                         cell=self.cell.tolist())

    def equal(self, other, R=True, tol=1e-4):
        """ Whether two geometries are the same (optional not check of the orbital radius)

        Parameters
        ----------
        other : Geometry
            the other Geometry to check against
        R : bool, optional
            if True also check if the orbital radii are the same (see `Atom.equal`)
        tol : float, optional
            tolerance for checking the atomic coordinates
        """
        if not isinstance(other, Geometry):
            return False
        same = self.sc.equal(other.sc, tol=tol)
        same = same and np.allclose(self.xyz, other.xyz, atol=tol)
        same = same and self.atoms.equal(other.atom, R)
        return same

    def __eq__(self, other):
        return self.equal(other)

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
        method : str, optional
           see `iter_block` for details

        Returns
        -------
        SparseAtom
           sparse matrix with all rij elements

        See Also
        --------
        iter_block : the method for looping the atoms
        distance : create a list of distances
        """
        from .sparse_geometry import SparseAtom
        rij = SparseAtom(self, nnzpr=20, dtype=dtype)

        # Get R
        R = (0.1, self.maxR())
        iR = self.iR(na_iR)

        # Do the loop
        for ias, idxs in self.iter_block(iR=iR, method=method):

            # Get all the indexed atoms...
            # This speeds up the searching for
            # coordinates...
            idxs_xyz = self[idxs, :]

            # Loop the atoms inside
            for ia in ias:
                idx, r = self.close(ia, R=R, idx=idxs, idx_xyz=idxs_xyz, ret_rij=True)
                rij[ia, ia] = 0.
                rij[ia, idx[1]] = r[1]

        return rij

    def distance(self, atom=None, R=None, tol=0.1, method='average'):
        """ Calculate the distances for all atoms in shells of radius `tol` within `max_R`

        Parameters
        ----------
        atom : int or array_like, optional
           only create list of distances from the given atoms, default to all atoms
        R : float, optional
           the maximum radius to consider, default to ``self.maxR()``.
           To retrieve all distances for atoms within the supercell structure
           you can pass `numpy.inf`.
        tol : float or array_like, optional
           the tolerance for grouping a set of atoms.
           This parameter sets the shell radius for each shell.
           I.e. the returned distances between two shells will be maximally
           ``2*tol``, but only if atoms are within two consecutive lists.
           If this is a list, the shells will be of unequal size.

           The first shell size will be ``tol * .5`` or ``tol[0] * .5`` if `tol` is a list.

        method : {'average', 'mode', '<numpy.func>', func}
           How the distance in each shell is determined.
           A list of distances within each shell is gathered and the equivalent
           method will be used to extract a single quantity from the list of
           distances in the shell.
           If `'mode'` is chosen it will use `scipy.stats.mode`.
           If a string is given it will correspond to ``getattr(numpy, method)``,
           while any callable function may be passed. The passed function
           will only be passed a list of unsorted distances that needs to be
           processed.

        Examples
        --------
        >>> geom = Geometry([0]*3, Atom(1, R=1.), sc=SuperCell(1., nsc=[5, 5, 1]))
        >>> geom.distance() # use geom.maxR() # doctest: +NORMALIZE_WHITESPACE
        array([1.])
        >>> geom.distance(tol=[0.5, 0.4, 0.3, 0.2])
        array([1.])
        >>> geom.distance(R=2, tol=[0.5, 0.4, 0.3, 0.2]) # doctest: +NORMALIZE_WHITESPACE
        array([1.        ,  1.41421356,  2.        ])
        >>> geom.distance(R=2, tol=[0.5, 0.7]) # the R = 1 and R = 2 ** .5 gets averaged # doctest: +NORMALIZE_WHITESPACE
        array([1.20710678,  2.        ])

        Returns
        -------
        numpy.ndarray
           an array of positive numbers yielding the distances from the atoms in reduced form

        See Also
        --------
        sparserij : return a sparse matrix will all distances between atoms
        """

        # Correct atom input
        if atom is None:
            atom = _a.arangei(len(self))
        else:
            atom = self._sanitize_atom(atom).ravel()

        # Figure out maximum distance
        if R is None:
            R = self.maxR()
            if R < 0:
                raise ValueError((self.__class__.__name__ +
                                  ".distance cannot determine the `R` parameter. "
                                  "The internal `maxR()` is negative and thus not set. "
                                  "Set an explicit value for `R`."))
        elif np.any(self.nsc > 1):
            maxR = fnorm(self.cell).max()
            # These loops could be leveraged if we look at angles...
            for i, j, k in product([0, self.nsc[0] // 2],
                                   [0, self.nsc[1] // 2],
                                   [0, self.nsc[2] // 2]):
                if i == 0 and j == 0 and k == 0:
                    continue
                sc = [i, j, k]
                off = self.sc.offset(sc)

                for ii, jj, kk in product([0, 1], [0, 1], [0, 1]):
                    o = self.cell[0, :] * ii + \
                        self.cell[1, :] * jj + \
                        self.cell[2, :] * kk
                    maxR = max(maxR, fnorm(off + o))

            if R > maxR:
                R = maxR

        # Convert to list
        tol = _a.asarrayd(tol).ravel()
        if len(tol) == 1:
            # Now we are in a position to determine the sizes
            dR = _a.aranged(tol[0] * .5, R + tol[0] * .55, tol[0])
        else:
            dR = tol.copy()
            dR[0] *= 0.5
            # The first tolerance, is for it-self, the second
            # has to have the first tolerance as the field
            dR = _a.cumsumd(np.insert(dR, 1, tol[0]))

            if dR[-1] < R:
                # Now finalize dR by ensuring all remaining segments are captured
                t = tol[-1]

                dR = np.concatenate((dR, _a.aranged(dR[-1] + t, R + t * .55, t)))

            # Reduce to the largest value above R
            # This ensures that R, truly is the largest considered element
            dR = dR[:(dR > R).nonzero()[0][0]+1]

        # Now we can figure out the list of atoms in each shell
        # First create the initial lists of shell atoms
        # The inner shell will never be used, because it should correspond
        # to the atom it-self.
        shells = [[] for i in range(len(dR) - 1)]

        for a in atom:
            _, r = self.close(a, R=dR, ret_rij=True)

            for i, rlist in enumerate(r[1:]):
                shells[i].extend(rlist)

        # Now parse all of the shells with the correct routine
        # First we grap the routine:
        if isinstance(method, _str):
            if method == 'median':
                def func(lst):
                    return np.median(lst, overwrite_input=True)

            elif method == 'mode':
                from scipy.stats import mode
                def func(lst):
                    return mode(lst)[0]
            else:
                try:
                    func = getattr(np, method)
                except:
                    raise ValueError(self.__class__.__name__ + ".distance `method` has wrong input value.")
        else:
            func = method

        # Reduce lists
        for i in range(len(shells)):
            lst = shells[i]
            if len(lst) == 0:
                continue

            # Reduce elements
            shells[i] = func(lst)

        # Convert to flattened numpy array and ensure shape
        d = np.hstack(shells)
        d.shape = (-1,)

        return d

    def within_inf(self, sc, periodic=None, tol=1e-5, origo=None):
        """ Find all atoms within a provided supercell

        Note this function is rather different from `close` and `within`.
        Specifically this routine is returning *all* indices for the infinite
        periodic system (where ``self.nsc > 1`` or `periodic` is true).

        Atomic coordinates lying on the boundary of the supercell will be duplicated
        on the neighbouring supercell images. Thus performing `geom.within_inf(geom.sc)`
        may result in more atoms than in the structure.

        Notes
        -----
        The name of this function may change. Currently it should only be used
        internally in sisl.

        Parameters
        ----------
        sc : SuperCell or SuperCellChild
            the supercell in which this geometry should be expanded into.
        periodic : list of bool
            explicitly define the periodic directions, by default the periodic
            directions are only where ``self.nsc > 1``.
        tol : float, optional
            length tolerance for the fractional coordinates to be on a duplicate site (in Ang).
            This allows atoms within `tol` of the cell boundaries to be taken as *inside* the
            cell.
        origo : (3, ) of float
            origo that is the basis for comparison

        Returns
        -------
        numpy.ndarray
           unit-cell atomic indices which are inside the `sc` cell
        numpy.ndarray
           atomic coordinates for the `ia` atoms (including supercell offsets)
        numpy.ndarray
           integer supercell offsets for `ia` atoms
        """
        if periodic is None:
            periodic = self.nsc > 1
        else:
            periodic = list(periodic)

        if origo is None:
            origo = _a.zerosd(3)

        # Our first task is to construct a geometry large
        # enough to fully encompass the supercell

        # 1. Number of times each lattice vector must be expanded to fit
        #    inside the "possibly" larger `sc`.
        idx = dot(sc.cell, self.icell.T)
        tile_min = floor(idx.min(0))
        tile_max = ceil(idx.max(0)).astype(dtype=int32)

        # Intrinsic offset (when atomic coordinates are outside primary unit-cell)
        idx = dot(self.xyz, self.icell.T)
        tmp = floor(idx.min(0))
        tile_min = np.where(tile_min < tmp, tile_min, tmp).astype(dtype=int32)
        tmp = ceil(idx.max(0))
        tile_max = np.where(tmp < tile_max, tile_max, tmp).astype(dtype=int32)
        del idx, tmp

        # 1a) correct for origo displacement
        idx = floor(dot(sc.origo, self.icell.T))
        tile_min = np.where(tile_min < idx, tile_min, idx).astype(dtype=int32)
        idx = floor(dot(origo, self.icell.T))
        tile_min = np.where(tile_min < idx, tile_min, idx).astype(dtype=int32)

        # 2. Reduce tiling along non-periodic directions
        tile_min = np.where(periodic, tile_min, 0)
        tile_max = np.where(periodic, tile_max, 1)

        # 3. Find the *new* origo according to the *negative* tilings.
        #    This is important for skewed cells as the placement of the new
        #    larger geometry has to be shifted to have sc inside
        big_origo = (tile_min.reshape(3, 1) * self.cell).sum(0)

        # The xyz geometry that fully encompass the (possibly) larger supercell
        tile = tile_max - tile_min
        full_geom = (self * tile).translate(big_origo - origo)

        # Now we have to figure out all atomic coordinates within
        cuboid = sc.toCuboid()

        # Now retrieve all atomic coordinates from the full geometry
        xyz = full_geom.axyz(_a.arangei(full_geom.na_s))
        idx = cuboid.within_index(xyz)
        xyz = xyz[idx, :]
        del full_geom

        # Figure out supercell connections in the smaller indices
        # Since we have shifted all coordinates into the primary unit cell we
        # are sure that these fxyz are [0:1[
        fxyz = dot(xyz, self.icell.T)

        # Since there are numerical errors for the above operation
        # we *have* to account for possible sign-errors
        # This is done by a length tolerance
        ftol = tol / fnorm(self.cell).reshape(1, 3)
        isc = floor(fxyz - ftol).astype(int32)

        # Now we can extract the indices where the two are non-matching.
        # At these indices we have some "errors" that we have to fix and
        # thus select the correct isc.
        idx_diff = (isc - floor(fxyz + ftol).astype(int32)).nonzero()

        # For these indices we can use the nearest integer as that
        # selects the closest. floor will ONLY be wrong for -0.0000, 0.99999, ...
        isc[idx_diff] = np.rint(fxyz[idx_diff]).astype(int32)

        # Convert indices to unit-cell indices and also return coordinates and
        # infinite supercell indices
        return self.sc2uc(idx), xyz, isc

    # Create pickling routines
    def __getstate__(self):
        """ Returns the state of this object """
        d = self.sc.__getstate__()
        d['xyz'] = self.xyz
        d['atom'] = self.atoms.__getstate__()
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
    @default_ArgumentParser(description="Manipulate a Geometry object in sisl.")
    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Create and return a group of argument parsers which manipulates it self `Geometry`.

        Parameters
        ----------
        parser : ArgumentParser, optional
           in case the arguments should be added to a specific parser. It defaults
           to create a new.
        limit_arguments : bool, optional
           If ``False`` additional options will be created which are similar to other options.
           For instance ``--repeat-x`` which is equivalent to ``--repeat x``.
           Default `True`.
        short : bool, optional
           Create short options for a selected range of options.
        positional_out : bool, optional
           If ``True``, adds a positional argument which acts as --out. This may be handy if only the geometry is in the argument list.
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
        if isinstance(self, Geometry):
            g = self.copy()
        else:
            g = None
        d = {
            "_geometry": g,
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

            def __call__(self, parser, ns, no_value, option_string=None):
                ns._geometry.xyz[:, :] -= np.amin(ns._geometry.xyz, axis=0)[None, :]
        p.add_argument(*opts('--origin', '-O'), action=MoveOrigin, nargs=0,
                   help='Move all atoms such that one atom will be at the origin.')

        class MoveCenterOf(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                xyz = ns._geometry.center(what='xyz')
                ns._geometry = ns._geometry.translate(ns._geometry.center(what=value) - xyz)
        p.add_argument(*opts('--center-of', '-co'), choices=['mass', 'xyz', 'position', 'cell'],
                       action=MoveCenterOf,
                       help='Move coordinates to the center of the designated choice.')

        class MoveUnitCell(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                if value in ['translate', 'tr', 't']:
                    # Simple translation
                    tmp = np.amin(ns._geometry.xyz, axis=0)
                    ns._geometry = ns._geometry.translate(-tmp)
                elif value in ['mod']:
                    g = ns._geometry
                    # Change all coordinates using the reciprocal cell and move to unit-cell (% 1.)
                    fxyz = g.fxyz % 1.
                    fxyz -= np.amin(fxyz, axis=0)
                    ns._geometry.xyz[:, :] = dot(fxyz, g.cell)
        p.add_argument(*opts('--unit-cell', '-uc'), choices=['translate', 'tr', 't', 'mod'],
                       action=MoveUnitCell,
                       help='Moves the coordinates into the unit-cell by translation or the mod-operator')

        # Rotation
        class Rotation(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                # Convert value[0] to the direction
                # The rotate function expects degree
                ang = angle(values[0], rad=False, in_rad=False)
                d = direction(values[1])
                if d == 0:
                    v = [1, 0, 0]
                elif d == 1:
                    v = [0, 1, 0]
                elif d == 2:
                    v = [0, 0, 1]
                ns._geometry = ns._geometry.rotate(ang, v)
        p.add_argument(*opts('--rotate', '-R'), nargs=2, metavar=('ANGLE', 'DIR'),
                       action=Rotation,
                       help='Rotate geometry around given axis. ANGLE defaults to be specified in degree. Prefix with "r" for input in radians.')

        if not limit_args:
            class RotationX(argparse.Action):

                def __call__(self, parser, ns, value, option_string=None):
                    # The rotate function expects degree
                    ang = angle(value, rad=False, in_rad=False)
                    ns._geometry = ns._geometry.rotate(ang, [1, 0, 0])
            p.add_argument(*opts('--rotate-x', '-Rx'), metavar='ANGLE',
                           action=RotationX,
                           help='Rotate geometry around first cell vector. ANGLE defaults to be specified in degree. Prefix with "r" for input in radians.')

            class RotationY(argparse.Action):

                def __call__(self, parser, ns, value, option_string=None):
                    # The rotate function expects degree
                    ang = angle(value, rad=False, in_rad=False)
                    ns._geometry = ns._geometry.rotate(ang, [0, 1, 0])
            p.add_argument(*opts('--rotate-y', '-Ry'), metavar='ANGLE',
                           action=RotationY,
                           help='Rotate geometry around second cell vector. ANGLE defaults to be specified in degree. Prefix with "r" for input in radians.')

            class RotationZ(argparse.Action):

                def __call__(self, parser, ns, value, option_string=None):
                    # The rotate function expects degree
                    ang = angle(value, rad=False, in_rad=False)
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
                s = int(values[0])
                d = direction(values[1])
                ns._geometry = ns._geometry.cut(s, d)
        p.add_argument(*opts('--cut', '-c'), nargs=2, metavar=('SEPS', 'DIR'),
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
                r = int(values[0])
                d = direction(values[1])
                ns._geometry = ns._geometry.repeat(r, d)
        p.add_argument(*opts('--repeat', '-r'), nargs=2, metavar=('TIMES', 'DIR'),
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
                r = int(values[0])
                d = direction(values[1])
                ns._geometry = ns._geometry.tile(r, d)
        p.add_argument(*opts('--tile'), nargs=2, metavar=('TIMES', 'DIR'),
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

            def __call__(self, parser, ns, no_value, option_string=None):
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
                    vs = getattr(ns, '_vector_scale', True)
                    if isinstance(vs, bool):
                        if vs:
                            vs = 1. / np.max(sqrt(square(v).sum(1)))
                            info('Scaling vector by: {}'.format(vs))
                        else:
                            vs = 1.

                    # Store the vectors with the scaling
                    kwargs['data'] = v * vs
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


def sgeom(geometry=None, argv=None, ret_geometry=False):
    """ Main script for sgeom.

    This routine may be called with `argv` and/or a `Sile` which is the geometry at hand.

    Parameters
    ----------
    geom : Geometry or BaseSile
       this may either be the geometry, as-is, or a `Sile` which contains
       the geometry.
    argv : list of str
       the arguments passed to sgeom
    ret_geometry : bool, optional
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

    p = argparse.ArgumentParser('Manipulates geometries.',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=description)

    # Add default sisl version stuff
    cmd.add_sisl_version_cite_arg(p)

    # First read the input "Sile"
    stdout_geom = True
    if geometry is None:
        from os.path import isfile
        argv, input_file = cmd.collect_input(argv)

        if input_file is None:
            stdout_geom = False
            geometry = Geometry([0] * 3)
        else:
            # Extract specification of the input file
            i_file, spec = str_spec(input_file)

            if isfile(i_file):
                geometry = get_sile(input_file).read_geometry()
            else:
                from .messages import info
                info("Cannot find file '{}'!".format(input_file))
                geometry = Geometry
                stdout_geom = False

    elif isinstance(geometry, Geometry):
        # Do nothing, the geometry is already created
        pass

    elif isinstance(geometry, BaseSile):
        geometry = geometry.read_geometry()
        # Store the input file...
        input_file = geometry.file

    # Do the argument parser
    p, ns = geometry.ArgumentParser(p, **geometry._ArgumentParser_args_single())

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

    if stdout_geom and not args._stored_geometry:
        # We should write out the information to the stdout
        # This is merely for testing purposes and may not be used for anything.
        print('Cell:')
        for i in (0, 1, 2):
            print('  {0:10.6f} {1:10.6f} {2:10.6f}'.format(*g.cell[i, :]))
        print('SuperCell:')
        print('  {0:d} {1:d} {2:d}'.format(*g.nsc))
        print(' {:>10s} {:>10s} {:>10s}  {:>3s}'.format('x', 'y', 'z', 'Z'))
        for ia in g:
            print(' {1:10.6f} {2:10.6f} {3:10.6f}  {0:3d}'.format(g.atoms[ia].Z,
                                                                  *g.xyz[ia, :]))

    if ret_geometry:
        return g
    return 0
