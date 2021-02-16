# To check for integers
from numbers import Integral, Real
from math import acos
from itertools import product
from collections import OrderedDict
from pathlib import Path

import numpy as np
from numpy import ndarray, int32, bool_
from numpy import dot, square, sqrt, diff
from numpy import floor, ceil, tile
from numpy import argsort, split, isin, concatenate

from ._internal import set_module, singledispatchmethod
from . import _plot as plt
from . import _array as _a
from ._math_small import is_ascending, cross3
from ._indices import indices_in_sphere_with_dist, indices_le, indices_gt_le
from ._indices import list_index_le
from .messages import info, warn, SislError, deprecate_method
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
from ._category import Category, GenericCategory
from ._dispatcher import AbstractDispatch
from ._dispatcher import ErrorDispatcher, ClassDispatcher, TypeDispatcher


__all__ = ['Geometry', 'sgeom']


# It needs to be here otherwise we can't use it in these routines
# Note how we are overwriting the module
@set_module("sisl.geom")
class AtomCategory(Category):
    __slots__ = tuple()

    @classmethod
    def is_class(cls, name, case=True):
        # Strip off `Atom`
        if case:
            return cls.__name__[4:] == name
        return cls.__name__[4:].lower() == name.lower()


@set_module("sisl")
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


    Parameters
    ----------
    xyz : array_like
        atomic coordinates
        ``xyz[i, :]`` is the atomic coordinate of the i'th atom.
    atoms : array_like or Atoms
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

    Conversion of geometries to other projects instances can be done via
    sisl's dispatch functionality

    >>> g.to.ase()
    Atoms(...)

    converts to an ASE ``Atoms`` object. See ``sisl/geometry.py`` for details
    on how to add more conversion methods.

    See Also
    --------
    Atoms : contained atoms `self.atoms`
    Atom : contained atoms are each an object of this
    """

    def __init__(self, xyz, atoms=None, sc=None, names=None):

        # Create the geometry coordinate
        # We need flatten to ensure a copy
        self.xyz = _a.asarrayd(xyz).flatten().reshape(-1, 3)

        # Default value
        if atoms is None:
            atoms = Atom('H')

        # Create the local Atoms object
        self._atoms = Atoms(atoms, na=self.na)

        # Assign a group specifier
        if isinstance(names, NamedIndex):
            self._names = names.copy()
        else:
            self._names = NamedIndex(names)

        self.__init_sc(sc)

    # Define a dispatcher for converting and requesting
    # new Geometries
    #  Geometry.new("run.fdf") will invoke Geometry.read("run.fdf")
    new = ClassDispatcher("new",
                          instance_dispatcher=TypeDispatcher,
                          obj_getattr=lambda obj, key:
                          (_ for _ in ()).throw(
                              AttributeError((f"{obj}.new does not implement '{key}' "
                                              f"dispatcher, are you using it incorrectly?"))
                          ),
    )

    # Define a dispatcher for converting Geometries
    #  Geometry().to.ase() will convert to an ase.Atoms object
    to = ClassDispatcher("to",
                         obj_getattr=lambda obj, key:
                         (_ for _ in ()).throw(
                             AttributeError((f"{obj}.to does not implement '{key}' "
                                             f"dispatcher, are you using it incorrectly?"))
                         )
    )

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
    @property
    @deprecate_method(f"Geometry.atom is deprecated, use Geometry.atoms instead")
    def atom(self):
        """ deprecated atom """
        return self._atoms

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

    def __setitem__(self, atoms, value):
        """ Specify geometry coordinates """
        if isinstance(atoms, str):
            self.names.add_name(atoms, value)
        elif isinstance(value, str):
            self.names.add_name(value, atoms)

    @singledispatchmethod
    def __getitem__(self, atoms):
        """ Geometry coordinates (allows supercell indices) """
        return self.axyz(atoms)

    @__getitem__.register(slice)
    def _(self, atoms):
        if atoms.stop is None:
            atoms = atoms.indices(self.na)
        else:
            atoms = atoms.indices(self.na_s)
        return self.axyz(_a.arangei(atoms[0], atoms[1], atoms[2]))

    @__getitem__.register(tuple)
    def _(self, atoms):
        return self[atoms[0]][..., atoms[1]]

    @singledispatchmethod
    def _sanitize_atoms(self, atoms):
        """ Converts an `atoms` to index under given inputs

        `atoms` may be one of the following:

        - boolean array -> nonzero()[0]
        - name -> self._names[name]
        - `Atom` -> (self.atoms.index(atom) == self.atoms.specie).nonzero()[0]
        - range/list/ndarray -> ndarray
        """
        if atoms is None:
            return _a.arangei(self.na)
        return np.asarray(atoms, dtype=np.int32)

    @_sanitize_atoms.register(str)
    def _(self, atoms):
        return self.names[atoms]

    @_sanitize_atoms.register(ndarray)
    def _(self, atoms):
        if atoms.dtype == bool_:
            return np.flatnonzero(atoms)
        return atoms

    @_sanitize_atoms.register(Atom)
    def _(self, atoms):
        return (self.atoms.specie == self.atoms.index(atoms)).nonzero()[0]

    @_sanitize_atoms.register(AtomCategory)
    @_sanitize_atoms.register(GenericCategory)
    def _(self, atoms):
        # First do categorization
        cat = atoms.categorize(self)
        def m(cat):
            for ia, c in enumerate(cat):
                if c == None:
                    pass
                else:
                    yield ia
        return _a.fromiteri(m(cat))

    @_sanitize_atoms.register(dict)
    def _(self, atoms):
        # First do categorization
        return self._sanitize_atoms(AtomCategory.kw(**atoms))

    def _sanitize_orbs(self, orbital):
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
            raise ValueError(f'{self.__class__.__name__}.as_primary requires the number of atoms to be divisable by the '
                             'total number of atoms.')

        n_supercells = len(self) // na_primary
        if n_supercells == 1:
            # Return a copy of self
            if ret_super:
                return self.copy(), self.nsc.copy()
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
            raise SislError(f'{self.__class__.__name__}.as_primary could not determine the optimal supercell.')

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
        s = self.__class__.__name__ + f'{{na: {self.na}, no: {self.no},\n '
        s += str(self.atoms).replace('\n', '\n ')
        if len(self.names) > 0:
            s += ',\n ' + str(self.names).replace('\n', '\n ')
        return (s + ',\n maxR: {0:.5f},\n {1}\n}}'.format(self.maxR(), str(self.sc).replace('\n', '\n '))).strip()

    def __repr__(self):
        """ A simple, short string representation. """
        return f"<{self.__module__}.{self.__class__.__name__} na={self.na}, no={self.no}, nsc={self.nsc}>"

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
        yield from range(len(self))

    __iter__ = iter

    def iter_species(self, atoms=None):
        """ Iterator over all atoms (or a subset) and species as a tuple in this geometry

        >>> for ia, a, idx_specie in self.iter_species():
        ...     isinstance(ia, int) == True
        ...     isinstance(a, Atom) == True
        ...     isinstance(idx_specie, int) == True

        with ``ia`` being the atomic index, ``a`` the `Atom` object, ``idx_specie``
        is the index of the specie

        Parameters
        ----------
        atoms : int or array_like, optional
           only loop on the given atoms, default to all atoms

        See Also
        --------
        iter : iterate over atomic indices
        iter_orbitals : iterate across atomic indices and orbital indices
        """
        if atoms is None:
            for ia in self:
                yield ia, self.atoms[ia], self.atoms.specie[ia]
        else:
            for ia in self._sanitize_atoms(atoms).ravel():
                yield ia, self.atoms[ia], self.atoms.specie[ia]

    def iter_orbitals(self, atoms=None, local=True):
        r"""
        Returns an iterator over all atoms and their associated orbitals

        >>> for ia, io in self.iter_orbitals():

        with ``ia`` being the atomic index, ``io`` the associated orbital index on atom ``ia``.
        Note that ``io`` will start from ``0``.

        Parameters
        ----------
        atoms : int or array_like, optional
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
        if atoms is None:
            if local:
                for ia, IO in enumerate(zip(self.firsto, self.lasto + 1)):
                    for io in range(IO[1] - IO[0]):
                        yield ia, io
            else:
                for ia, IO in enumerate(zip(self.firsto, self.lasto + 1)):
                    for io in range(IO[0], IO[1]):
                        yield ia, io
        else:
            atoms = self._sanitize_atoms(atoms).ravel()
            if local:
                for ia, io1, io2 in zip(atoms, self.firsto[atoms], self.lasto[atoms] + 1):
                    for io in range(io2 - io1):
                        yield ia, io
            else:
                for ia, io1, io2 in zip(atoms, self.firsto[atoms], self.lasto[atoms] + 1):
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
            raise ValueError(f"{self.__class__.__name__}.iR unable to determine a number of atoms within a sphere with negative radius, is maxR() defined?")

        # Number of atoms within 20 * R
        naiR = max(1, len(self.close(ia, R=R * iR)))

        # Convert to na atoms spherical radii
        iR = int(4 / 3 * np.pi * R ** 3 / naiR * na)

        return max(2, iR)

    def iter_block_rand(self, iR=20, R=None, atoms=None):
        """ Perform the *random* block-iteration by randomly selecting the next center of block """

        # We implement yields as we can then do nested iterators
        # create a boolean array
        na = len(self)
        not_passed = np.empty(na, dtype='b')
        if atoms is not None:
            # Reverse the values
            not_passed[:] = False
            not_passed[atoms] = True
        else:
            not_passed[:] = True

        # Figure out how many we need to loop on
        not_passed_N = np.sum(not_passed)

        if iR < 2:
            raise SislError(f'{self.__class__.__name__}.iter_block_rand too small iR!')

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
            raise SislError(f'{self.__class__.__name__}.iter_block_rand error on iterations. Not all atoms have been visited.')

    def iter_block_shape(self, shape=None, iR=20, atoms=None):
        """ Perform the *grid* block-iteration by looping a grid """

        # We implement yields as we can then do nested iterators
        # create a boolean array
        na = len(self)
        if atoms is not None:
            not_passed = np.zeros(na, dtype=bool)
            # Reverse the values
            not_passed[atoms] = True
        else:
            not_passed = np.ones(na, dtype=bool)

        # Figure out how many we need to loop on
        not_passed_N = np.sum(not_passed)

        if iR < 2:
            raise SislError(f'{self.__class__.__name__}.iter_block_shape too small iR!')

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
            raise ValueError(f'{self.__class__.__name__}.iter_block_shape, number of Shapes *must* be one or two')

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
                raise ValueError(f'{self.__class__.__name__}.iter_block_shape currently only works for '
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
            raise SislError(f'{self.__class__.__name__}.iter_block_shape error on iterations. Not all atoms have been visited.')

    def iter_block(self, iR=20, R=None, atoms=None, method='rand'):
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
        atoms : array_like, optional
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
            raise SislError(f'{self.__class__.__name__}.iter_block too small iR!')

        method = method.lower()
        if method == 'rand' or method == 'random':
            yield from self.iter_block_rand(iR, R, atoms)
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

            yield from self.iter_block_shape(dS)

    def copy(self):
        """ A copy of the object. """
        g = self.__class__(np.copy(self.xyz), atoms=self.atoms.copy(), sc=self.sc.copy())
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
        # sanitize `other`
        other = self.new(other)
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

    def sort(self, **kwargs):
        r""" Sort atoms in a nested fashion according to various criteria

        There are many ways to sort a `Geometry`.
        - by Cartesian coordinates, `axis`
        - by lattice vectors, `lattice`
        - by user defined vectors, `vector`
        - by grouping atoms, `group`
        - by a user defined function, `func`
        - by a user defined function using internal sorting algorithm, `func_sort`

        - a combination of the above in arbitrary order

        Additionally one may sort ascending or descending.

        This method allows nested sorting based on keyword arguments.

        Parameters
        ----------
        atoms : list or list of list, optional
           only perform sorting algorithm for subset of atoms. This is *NOT* a positional dependent
           argument. All sorting algorithms will _only_ be performed on these atoms.
           If a list of indices only the given atoms will be sorted, and all other atoms
           will be kept in the final structure at their initial indices.
           If a list of list of indices, each list of indices will be sorted individually (see `ret_atoms`).
           Default, all atoms will be sorted.
        ret_atoms : bool, optional
           return a list of list for the groups of atoms that have been sorted.
        axis : int or tuple of int, optional
           sort coordinates according to Cartesian coordinates, if a tuple of
           ints is passed it will be equivalent to ``sort(axis0=axis[0], axis1=axis[1])``.
           This behaves differently than `numpy.lexsort`!
        lattice : int or tuple of int, optional
           sort coordinates according to lattice vectors, if a tuple of
           ints is passed it will be equivalent to ``sort(lattice0=lattice[0], lattice1=lattice[1])``.
           Note that before sorting we multiply the fractional coordinates by the length of the
           lattice vector. This ensures that `atol` is meaningful for both `axis` and `lattice` since
           they will be on the same order of magnitude.
           This behaves differently than `numpy.lexsort`!
        vector : (3, ), optional
           sort along a user defined vector, similar to `lattice` but with a user defined
           direction. Note that `lattice` sorting and `vector` sorting are *only* equivalent
           when the lattice vector is orthogonal to the other lattice vectors.
        group : {'Z', 'symbol', 'tag', 'species'} or (str, ...), optional
           group together a set of atoms by various means.
           `group` may be one of the listed strings.
           For ``'Z'`` atoms will be grouped in atomic number
           For ``'symbol'`` atoms will be grouped by their atomic symbol.
           For ``'tag'`` atoms will be grouped by their atomic tag.
           For ``'species'`` atoms will be sorted according to their specie index.
           If a tuple/list is passed the first item is described. All subsequent items are a
           list of groups, where each group comprises elements that should be sorted on an
           equal footing. If one of the groups is None, that group will be replaced with all
           non-mentioned elements. See examples.
        func : callable or list-like of callable, optional
           pass a sorting function which should have an interface like ``func(geometry, atoms, **kwargs)``.
           The first argument is the geometry to sort. The 2nd argument is a list of indices in
           the current group of sorted atoms. And ``**kwargs`` are any optional arguments
           currently collected, i.e. `ascend`, `atol` etc.
           The function should return either a list of atoms, or a list of list of atoms (in which
           case the atomic indices have been split into several groups that will be sorted individually
           for subsequent sorting methods).
           In either case the returned indices must never hold any other indices but the ones passed
           as ``atoms``.
           If a list/tuple of functions, they will be processed in that order.
        func_sort : callable or list-like of callable, optional
           pass a function returning a 1D array corresponding to all atoms in the geometry.
           The interface should simply be: ``func(geometry)``.
           Those values will be passed down to the internal sorting algorithm.
           To be compatible with `atol` the returned values from `func_sort` should
           be on the scale of coordinates (in Ang).
        ascend, descend : bool, optional
            control ascending or descending sorting for all subsequent sorting methods.
            Default ``ascend=True``.
        atol : float, optional
            absolute tolerance when sorting numerical arrays for subsequent sorting methods.
            When a selection of sorted coordinates are grouped via `atol`, we ensure such
            a group does not alter its indices. I.e. the group is *always* ascending indices.
            Note this may have unwanted side-effects if `atol` is very large compared
            to the difference between atomic coordinates.
            Default ``1e-9``.

        Notes
        -----
        The order of arguments is also the sorting order. ``sort(axis=0, lattice=0)`` is different
        from ``sort(lattice=0, axis=0)``

        All arguments may be suffixed with integers. This allows multiple keyword arguments
        to control sorting algorithms
        in different order. It also allows changing of sorting settings between different calls.
        Note that the integers have no relevance to the order of execution!
        See examples.

        Returns
        -------
        geometry : Geometry
            sorted geometry
        index : list of list of indices
            indices that would sort the original structure to the output, only returned if `ret_atoms` is True

        Examples
        --------
        >>> geom = sisl.geom.bilayer(top_atoms=sisl.Atom[5, 7], bottom_atoms=sisl.Atom(6))
        >>> geom = geom.tile(5, 0).repeat(7, 1)

        Sort according to :math:`x` coordinate

        >>> geom.sort(axis=0)

        Sort according to :math:`z`, then :math:`x` for each group created from first sort

        >>> geom.sort(axis=(2, 0))

        Sort according to :math:`z`, then first lattice vector

        >>> geom.sort(axis=2, lattice=0)

        Sort according to :math:`z` (ascending), then first lattice vector (descending)

        >>> geom.sort(axis=2, ascend=False, lattice=0)

        Sort according to :math:`z` (descending), then first lattice vector (ascending)
        Note how integer suffixes has no importance.

        >>> geom.sort(ascend1=False, axis=2, ascend0=True, lattice=0)

        Sort only atoms ``range(1, 5)`` first by :math:`z`, then by first lattice vector

        >>> geom.sort(axis=2, lattice=0, atoms=np.arange(1, 5))

        Sort two groups of atoms ``[range(1, 5), range(5, 10)]`` (individually) by :math:`z`

        >>> geom.sort(axis=2, atoms=[np.arange(1, 5), np.arange(5, 10)])

        The returned sorting indices may be used for manual sorting. Note
        however, that this requires one to perform a sorting for all atoms.
        In such a case the following sortings are equal.

        >>> geom0, atoms0 = geom.sort(axis=2, lattice=0, ret_atoms=True)
        >>> _, atoms1 = geom.sort(axis=2, ret_atoms=True)
        >>> geom1, atoms1 = geom.sort(lattice=0, atoms=atoms1, ret_atoms=True)
        >>> geom2 = geom.sub(np.concatenate(atoms0))
        >>> geom3 = geom.sub(np.concatenate(atoms1))
        >>> assert geom0 == geom1
        >>> assert geom0 == geom2
        >>> assert geom0 == geom3

        Default sorting is equivalent to ``axis=(0, 1, 2)``

        >>> assert geom.sort() == geom.sort(axis=(0, 1, 2))

        Sort along a user defined vector ``[2.2, 1., 0.]``

        >>> geom.sort(vector=[2.2, 1., 0.])

        Integer specification has no influence on the order of operations.
        It is _always_ the keyword argument order that determines the operation.

        >>> assert geom.sort(axis2=1, axis0=0, axis1=2) == geom.sort(axis=(1, 0, 2))

        Sort by atomic numbers

        >>> geom.sort(group='Z') # 5, 6, 7

        One may group several elements together on an equal footing (``None`` means all non-mentioned elements)
        The order of the groups are important (the first two are _not_ equal, the last three _are_ equal)

        >>> geom.sort(group=('symbol', 'C', None)) # C, [B, N]
        >>> geom.sort(group=('symbol', None, 'C')) # [B, N], C
        >>> geom.sort(group=('symbol', ['N', 'B'], 'C')) # [B, N], C (B and N unaltered order)
        >>> geom.sort(group=('symbol', ['B', 'N'], 'C')) # [B, N], C (B and N unaltered order)

        A too high `atol` may have unexpected side-effects. This is because of the way
        the sorting algorithm splits the sections for nested sorting.
        So for coordinates with a continuous displacement the sorting may break and group
        a large range into 1 group. Consider the following array to be split in groups
        while sorting.

        An example would be a linear chain with a middle point with a much shorter distance.

        >>> x = np.arange(5) * 0.1
        >>> x[3:] -= 0.095
        y = z = np.zeros(5)
        geom = si.Geometry(np.stack((x, y, z), axis=1))
        >>> geom.xyz[:, 0]
        [0.    0.1   0.2   0.205 0.305]

        In this case a high tolerance (``atol>0.005``) would group atoms 2 and 3
        together

        >>> geom.sort(atol=0.01, axis=0, ret_atoms=True)[1]
        [[0], [1], [2, 3], [4]]

        However, a very low tolerance will not find these two as atoms close
        to each other.

        >>> geom.sort(atol=0.001, axis=0, ret_atoms=True)[1]
        [[0], [1], [2], [3], [4]]
        """
        # We need a way to easily handle nested lists
        # This small class handles lists and allows appending nested lists
        # while flattening them.
        class NestedList:
            __slots__ = ('_idx', )
            def __init__(self, idx=None, sort=False):
                self._idx = []
                if not idx is None:
                    self.append(idx, sort)
            def append(self, idx, sort=False):
                if isinstance(idx, (tuple, list)):
                    if isinstance(idx[0], (tuple, list, ndarray)):
                        for ix in idx:
                            self.append(ix, sort)
                        return
                elif isinstance(idx, NestedList):
                    idx = idx.tolist()
                if len(idx) > 0:
                    if sort:
                        self._idx.append(np.sort(idx))
                    else:
                        self._idx.append(np.asarray(idx))
            def __iter__(self):
                yield from self._idx
            def __len__(self):
                return len(self._idx)
            def tolist(self):
                return self._idx

        def _sort(val, atoms, **kwargs):
            """ We do not sort according to lexsort """
            if len(val) <= 1:
                # no values to sort
                return atoms

            # control ascend vs descending
            ascend = kwargs['ascend']
            atol = kwargs['atol']

            new_atoms = NestedList()
            for atom in atoms:
                if len(atom) <= 1:
                    # no need for complexity
                    new_atoms.append(atom)
                    continue

                # Sort values
                jdx = atom[argsort(val[atom])]
                if ascend:
                    d = diff(val[jdx]) > atol
                else:
                    jdx = jdx[::-1]
                    d = diff(val[jdx]) < -atol
                new_atoms.append(split(jdx, d.nonzero()[0] + 1), sort=True)
            return new_atoms

        # Functions allowed by external users
        funcs = dict()
        def _axis(axis, atoms, **kwargs):
            """ Cartesian coordinate sort """
            if isinstance(axis, int):
                axis = (axis,)
            for ax in axis:
                atoms = _sort(self.xyz[:, ax], atoms, **kwargs)
            return atoms
        funcs["axis"] = _axis

        def _lattice(lattice, atoms, **kwargs):
            """
            We scale the fractional coordinates with the lattice vector length.
            This ensures `atol` has a meaningful size for very large structures.
            """
            if isinstance(lattice, int):
                lattice = (lattice,)
            fxyz = self.fxyz
            for ax in lattice:
                atoms = _sort(fxyz[:, ax] * self.sc.length[ax], atoms, **kwargs)
            return atoms
        funcs["lattice"] = _lattice

        def _vector(vector, atoms, **kwargs):
            """
            Calculate fraction of positions along a vector and sort along it.
            We first normalize the vector to ensure that `atol` is meaningful
            for very large structures (ensures scale is on the order of Ang).

            A vector projection will only be equivalent to lattice projection
            when a lattice vector is orthogonal to the other lattice vectors.
            """
            # Ensure we are using a copied data array
            vector = _a.asarrayd(vector).copy()
            # normalize
            vector /= fnorm(vector)
            # Perform a . b^ == scalar projection
            return _sort(self.xyz.dot(vector), atoms, **kwargs)
        funcs["vector"] = _vector

        def _funcs(funcs, atoms, **kwargs):
            """
            User defined function (tuple/list of function)
            """
            def _func(func, atoms, kwargs):
                nl = NestedList()
                for atom in atoms:
                    # TODO add check that
                    #  res = func(...) in a
                    # A user *may* remove an atom from the sorting here (but
                    # that negates all sorting of that atom)
                    nl.append(func(self, atom, **kwargs))
                return nl

            if callable(funcs):
                funcs = [funcs]
            for func in funcs:
                atoms = _func(func, atoms, kwargs)
            return atoms
        funcs["func"] = _funcs

        def _func_sort(funcs, atoms, **kwargs):
            """
            User defined function, but using internal sorting
            """
            if callable(funcs):
                funcs = [funcs]
            for func in funcs:
                atoms = _sort(func(self), atoms, **kwargs)
            return atoms
        funcs["func_sort"] = _func_sort

        def _group_vals(vals, groups, atoms, **kwargs):
            """
            vals should be of size len(self) and be parsable
            by numpy
            """
            nl = NestedList()

            # Create unique list of integers
            uniq_vals = np.unique(vals)
            if len(groups) == 0:
                # fake the groups argument
                groups = [[i] for i in uniq_vals]
            else:
                # Check if one of the elements of group is None
                # In this case we replace it with the missing rest
                # of the missing unique items
                try:
                    none_idx = groups.index(None)

                    # we have a None (ensure we use a list, tuples are
                    # immutable)
                    groups = list(groups)
                    groups[none_idx] = []

                    uniq_groups = np.unique(concatenate(groups))
                    # add a new group that is in uniq_vals, but not in uniq_groups
                    rest = uniq_vals[isin(uniq_vals, uniq_groups, invert=True)]
                    groups[none_idx] = rest
                except ValueError:
                    # there is no None in the list
                    pass

            for at in atoms:
                # reduce search
                at_vals = vals[at]
                # loop group values
                for group in groups:
                    # retain original indexing
                    nl.append(at[isin(at_vals, group)])
            return nl

        def _group(method_group, atoms, **kwargs):
            """
            Group based sorting is based on a named identification.

            group: str or tuple of (str, list of lists)

            symbol: order by symbol (most cases same as Z)
            Z: order by atomic number
            tag: order by atom tag (should be the same as specie)
            specie/species: order by specie (in order of contained in the Geometry)
            """
            # Create new list
            nl = NestedList()
            if isinstance(method_group, str):
                method = method_group
                groups = []
            else:
                method, *in_groups = method_group

                # Ensure all groups are lists
                groups = []
                NoneType = type(None)
                for group in in_groups:
                    if isinstance(group, (tuple, list, ndarray, NoneType)):
                        groups.append(group)
                    else:
                        groups.append([group])

            method = method.lower()

            if method == 'z':
                return _group_vals(self.atoms.Z, groups, atoms, **kwargs)
            elif method in ['specie', 'species']:
                return _group_vals(self.atoms.specie, groups, atoms, **kwargs)
            elif method == 'tag':
                # create numpy array
                strs = np.array(list(map(lambda a: a.tag, self.atoms)))
                return _group_vals(strs, groups, atoms, **kwargs)
            elif method == 'symbol':
                strs = np.array(list(map(lambda a: a.symbol, self.atoms)))
                return _group_vals(strs, groups, atoms, **kwargs)
            else:
                raise ValueError(f"{self.__class__.__name__}.sort group only supports ['Z','species','tag','symbol'] for sorting")
        funcs["group"] = _group

        def stripint(s):
            """ Remove integers from end of string -> Allow multiple arguments """
            if s[-1] in '0123456789':
                return stripint(s[:-1])
            return s

        # Now perform cumultative sort function
        # Our point is that we would like to allow users to do consecutive sorting
        # based on different keys

        # We also allow specific keys for specific methods
        func_kw = dict()
        func_kw['ascend'] = True
        func_kw['atol'] = 1e-9

        def update_flag(kw, arg, val):
            if arg in ['ascending', 'ascend']:
                kw['ascend'] = val
                return True
            elif arg in ['descending', 'descend']:
                kw['ascend'] = not val
                return True
            elif arg == 'atol':
                kw['atol'] = val
                return True
            return False

        # Default to all atoms
        atoms = NestedList(kwargs.pop("atoms", None))
        if len(atoms) > 0:
            # Ensure the user has not supplied duplicate atomic indices
            atomsl = concatenate(atoms.tolist())
            if len(np.unique(atomsl)) != len(atomsl):
                raise ValueError(f"{self.__class__.__name__}.sort requires 'atoms' argument to not have duplicate values")
            del atomsl
        else:
            # Always ensure the first nested list has *something in it*
            atoms = NestedList(_a.arangei(len(self)))

        ret_atoms = kwargs.pop("ret_atoms", False)

        # In case the user just did geometry.sort, it will default to sort x, y, z
        if len(kwargs) == 0:
            kwargs['axis'] = (0, 1, 2)

        for key_int, method in kwargs.items():
            key = stripint(key_int)
            if update_flag(func_kw, key, method):
                continue
            if not key in funcs:
                raise ValueError(f"{self.__class__.__name__}.sort unrecognized keyword '{key}' ('{key_int}')")
            # call sorting algorithm and retrieve new grouped sorting
            atoms = funcs[key](method, atoms, **func_kw)

        atoms_flat = concatenate(atoms.tolist()).ravel()

        # Ensure that all atoms are present
        if len(atoms_flat) != len(self):
            all_atoms = _a.arangei(len(self))
            all_atoms[np.sort(atoms_flat)] = atoms_flat[:]
            atoms_flat = all_atoms

        if ret_atoms:
            return self.sub(atoms_flat), atoms.tolist()
        return self.sub(atoms_flat)

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

    def sub(self, atoms, cell=None):
        """ Create a new `Geometry` with a subset of this `Geometry`

        Indices passed *MUST* be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atoms : int or array_like
            indices/boolean of all atoms to be removed
        cell   : array_like or SuperCell, optional
            the new associated cell of the geometry (defaults to the same cell)

        See Also
        --------
        SuperCell.fit : update the supercell according to a reference supercell
        remove : the negative of this routine, i.e. remove a subset of atoms
        """
        atoms = self.sc2uc(atoms)
        if cell is None:
            return self.__class__(self.xyz[atoms, :],
                                  atoms=self.atoms.sub(atoms), sc=self.sc.copy())
        return self.__class__(self.xyz[atoms, :],
                              atoms=self.atoms.sub(atoms), sc=cell)

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
            raise ValueError(f'{self.__class__.__name__}.cut '
                             f'cannot be cut into {seps} different '
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

    def remove(self, atoms):
        """ Remove atoms from the geometry.

        Indices passed *MUST* be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atoms : int or array_like
            indices/boolean of all atoms to be removed

        See Also
        --------
        sub : the negative of this routine, i.e. retain a subset of atoms
        """
        atoms = self.sc2uc(atoms)
        atoms = np.delete(_a.arangei(self.na), atoms)
        return self.sub(atoms)

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
            raise ValueError(f'{self.__class__.__name__}.tile requires a repetition above 0')

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
        return self.__class__(xyz, atoms=self.atoms.tile(reps), sc=sc)

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
            raise ValueError(f'{self.__class__.__name__}.repeat requires a repetition above 0')

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
        return self.__class__(xyz, atoms=self.atoms.repeat(reps), sc=sc)

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
            if isinstance(m[1], str):
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

    def angle(self, atoms, dir=(1., 0, 0), ref=None, rad=False):
        r""" The angle between atom `atoms` and the direction `dir`, with possibility of a reference coordinate `ref`

        The calculated angle can be written as this

        .. math::
            \alpha = \arccos \frac{(\mathrm{atom} - \mathrm{ref})\cdot \mathrm{dir}}
            {|\mathrm{atom}-\mathrm{ref}||\mathrm{dir}|}

        and thus lies in the interval :math:`[0 ; \pi]` as one cannot distinguish orientation without
        additional vectors.

        Parameters
        ----------
        atoms : int or array_like
           indices/boolean of all atoms where angles should be calculated on
        dir : str, int or array_like, optional
           the direction from which the angle is calculated from, default to ``x``
        ref : int or array_like, optional
           the reference point from which the vectors are drawn, default to origo
        rad : bool, optional
           whether the returned value is in radians
        """
        xi = self.axyz(atoms)
        if isinstance(dir, (str, Integral)):
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

    def rotate(self, angle, v, origo=None, atoms=None, only='abc+xyz', rad=False):
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
        atoms : int or array_like, optional
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

        if not atoms is None:
            # Only rotate the unique values
            atoms = self.sc2uc(atoms, unique=True)

        # Ensure the normal vector is normalized... (flatten == copy)
        vn = _a.asarrayd(v).flatten()
        vn /= fnorm(vn)

        # Rotate by direct call
        if 'a' in only or 'b' in only or 'c' in only:
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
            xyz[atoms, :] = q.rotate(xyz[atoms, :] - origo[None, :]) + origo[None, :]

        return self.__class__(xyz, atoms=self.atoms.copy(), sc=sc)

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
        return self.rotate(a, cp, rad=True)

    def move(self, v, atoms=None, cell=False):
        """ Translates the geometry by `v`

        One can translate a subset of the atoms by supplying `atoms`.

        Returns a copy of the structure translated by `v`.

        Parameters
        ----------
        v : array_like
             the vector to displace all atomic coordinates
        atoms : int or array_like, optional
             only displace the given atomic indices, if not specified, all
             atoms will be displaced
        cell : bool, optional
             If True the supercell also gets enlarged by the vector
        """
        g = self.copy()
        if atoms is None:
            g.xyz[:, :] += np.asarray(v, g.xyz.dtype)[None, :]
        else:
            g.xyz[self._sanitize_atoms(atoms).ravel(), :] += np.asarray(v, g.xyz.dtype)[None, :]
        if cell:
            g.set_supercell(g.sc.translate(v))
        return g
    translate = move

    def swap(self, atoms_a, atoms_b):
        """ Swap a set of atoms in the geometry and return a new one

        This can be used to reorder elements of a geometry.

        Parameters
        ----------
        atoms_a : array_like
             the first list of atomic coordinates
        atoms_b : array_like
             the second list of atomic coordinates
        """
        atoms_a = self._sanitize_atoms(atoms_a)
        atoms_b = self._sanitize_atoms(atoms_b)
        xyz = np.copy(self.xyz)
        xyz[atoms_a, :] = self.xyz[atoms_b, :]
        xyz[atoms_b, :] = self.xyz[atoms_a, :]
        return self.__class__(xyz, atoms=self.atoms.swap(atoms_a, atoms_b), sc=self.sc.copy())

    def swapaxes(self, axis_a, axis_b, swap='cell+xyz'):
        """ Swap the axis for the atomic coordinates and the cell vectors

        If ``swapaxes(0,1)`` it returns the 0 and 1 values
        swapped in the ``cell`` variable.

        Parameters
        ----------
        axis_a : int
           axis 1, swaps with `b`
        axis_b : int
           axis 2, swaps with `a`
        swap : {'cell+xyz', 'cell', 'xyz'}
           decide what to swap, if `'cell'` is in `swap` then
           the cell axis are swapped.
           if `'xyz'` is in `swap` then
           the xyz (Cartesian) axis are swapped.
           Both may be in `swap`.
        """
        xyz = np.copy(self.xyz)
        if 'xyz' in swap:
            xyz[:, axis_a] = self.xyz[:, axis_b]
            xyz[:, axis_b] = self.xyz[:, axis_a]
        if 'cell' in swap:
            sc = self.sc.swapaxes(axis_a, axis_b)
        else:
            sc = self.sc.copy()
        return self.__class__(xyz, atoms=self.atoms.copy(), sc=sc)

    def center(self, atoms=None, what='xyz'):
        """ Returns the center of the geometry

        By specifying `what` one can control whether it should be:

        * ``xyz|position``: Center of coordinates (default)
        * ``mm(xyz)``: Center of minimum/maximum of coordinates
        * ``mass``: Center of mass
        * ``cell``: Center of cell

        Parameters
        ----------
        atoms : array_like
            list of atomic indices to find center of
        what : {'xyz', 'mm(xyz)', 'mass', 'cell'}
            determine whether center should be of 'cell', mass-centered ('mass'),
            center of minimum/maximum position of atoms or absolute center of the positions.
        """
        if 'cell' == what:
            return self.sc.center()
        if atoms is None:
            g = self
        else:
            g = self.sub(atoms)
        if 'mass' == what:
            mass = self.mass
            return dot(mass, g.xyz) / np.sum(mass)
        if 'mm(xyz)' == what:
            return (self.xyz.min(0) + self.xyz.max(0)) / 2
        if not ('xyz' in what or 'position' in what):
            raise ValueError(
                'Unknown what, not one of [xyz,position,mass,cell]')
        return np.mean(g.xyz, axis=0)

    def append(self, other, axis, offset='none'):
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
        offset : {'none', 'min', (3,)}
            By default appending two structures will simply use the coordinates,
            as is.
            With 'min', the routine will shift both the structures along the cell
            axis of `self` such that they coincide at the first atom, lastly one
            may use a specified offset to manually select how `other` is displaced.
            NOTE: That `self.cell[axis, :]` will be added to `offset` if `other` is
            a geometry.

        See Also
        --------
        add : add geometries
        prepend : prending geometries
        attach : attach a geometry
        insert : insert a geometry
        """
        if isinstance(other, SuperCell):
            # Only extend the supercell.
            xyz = np.copy(self.xyz)
            atoms = self.atoms.copy()
            sc = self.sc.append(other, axis)
            names = self._names.copy()
            if isinstance(offset, str):
                if offset == "none":
                    offset = [0, 0, 0]
                else:
                    raise ValueError(f"{self.__class__.__name__}.append requires offset to be (3,) for supercell input")
            xyz += _a.arrayd(offset).reshape(1, 3)

        else:
            # sanitize output
            other = self.new(other)
            if isinstance(offset, str):
                offset = offset.lower()
                if offset == 'none':
                    offset = self.cell[axis, :].reshape(1, 3)
                elif offset == 'min':
                    # We want to align at the minimum position along the `axis`
                    min_f = self.fxyz[:, axis].min()
                    min_other_f = dot(other.xyz, self.icell.T)[:, axis].min()
                    offset = self.cell[axis, :] * (1 + min_f - min_other_f)
                else:
                    raise ValueError(f'{self.__class__.__name__}.append requires align keyword to be one of [none, min, (3,)]')
            else:
                offset = (self.cell[axis, :] + _a.arrayd(offset)).reshape(1, 3)

            xyz = np.append(self.xyz, offset + other.xyz, axis=0)
            atoms = self.atoms.append(other.atoms)
            sc = self.sc.append(other.sc, axis)
            names = self._names.merge(other._names, offset=len(self))

        return self.__class__(xyz, atoms=atoms, sc=sc, names=names)

    def prepend(self, other, axis, offset='none'):
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
        offset : {'none', 'min', (3,)}
            By default appending two structures will simply use the coordinates,
            as is.
            With 'min', the routine will shift both the structures along the cell
            axis of `other` such that they coincide at the first atom, lastly one
            may use a specified offset to manually select how `self` is displaced.
            NOTE: That `other.cell[axis, :]` will be added to `offset` if `other` is
            a geometry.

        See Also
        --------
        add : add geometries
        append : appending geometries
        attach : attach a geometry
        insert : insert a geometry
        """
        if isinstance(other, SuperCell):
            # Only extend the supercell.
            xyz = np.copy(self.xyz)
            atoms = self.atoms.copy()
            sc = self.sc.prepend(other, axis)
            names = self._names.copy()
            if isinstance(offset, str):
                if offset == "none":
                    offset = [0, 0, 0]
                else:
                    raise ValueError(f"{self.__class__.__name__}.prepend requires offset to be (3,) for supercell input")
            xyz += _a.arrayd(offset).reshape(1, 3)

        else:
            # sanitize output
            other = self.new(other)
            if isinstance(offset, str):
                offset = offset.lower()
                if offset == 'none':
                    offset = other.cell[axis, :].reshape(1, 3)
                elif offset == 'min':
                    # We want to align at the minimum position along the `axis`
                    min_f = other.fxyz[:, axis].min()
                    min_other_f = dot(self.xyz, other.icell.T)[:, axis].min()
                    offset = other.cell[axis, :] * (1 + min_f - min_other_f)
                else:
                    raise ValueError(f'{self.__class__.__name__}.prepend requires align keyword to be one of [none, min, (3,)]')
            else:
                offset = (other.cell[axis, :] + _a.arrayd(offset)).reshape(1, 3)

            xyz = np.append(other.xyz, offset + self.xyz, axis=0)
            atoms = self.atoms.prepend(other.atoms)
            sc = self.sc.prepend(other.sc, axis)
            names = other._names.merge(self._names, offset=len(other))

        return self.__class__(xyz, atoms=atoms, sc=sc, names=names)

    def add(self, other, offset=(0, 0, 0)):
        """ Merge two geometries (or a Geometry and SuperCell) by adding the two atoms together

        If `other` is a Geometry only the atoms gets added, to also add the supercell vectors
        simply do ``geom.add(other).add(other.sc)``.

        Parameters
        ----------
        other : Geometry or SuperCell
            Other geometry class which is added
        offset : (3,), optional
            offset in geometry of `other` when adding the atoms. Only if `other` is
            of instance `Geometry`.

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
            atoms = self.atoms.copy()
            names = self._names.copy()
        else:
            other = self.new(other)
            xyz = np.append(self.xyz, other.xyz + _a.arrayd(offset).reshape(1, 3), axis=0)
            sc = self.sc.copy()
            atoms = self.atoms.add(other.atoms)
            names = self._names.merge(other._names, offset=len(self))
        return self.__class__(xyz, atoms=atoms, sc=sc, names=names)

    def insert(self, atom, other):
        """ Inserts other atoms right before index

        We insert the `geometry` `Geometry` before `atom`.
        Note that this will not change the unit cell.

        Parameters
        ----------
        atom : int
           the atomic index at which the other geometry is inserted
        other : Geometry
           the other geometry to be inserted

        See Also
        --------
        add : add geometries
        append : appending geometries
        prepend : prending geometries
        attach : attach a geometry
        """
        atom = self._sanitize_atoms(atom)
        if atom.size > 1:
            raise ValueError(f"{self.__class__.__name__}.insert requires only 1 atomic index for insertion.")
        other = self.new(other)
        xyz = np.insert(self.xyz, atom, other.xyz, axis=0)
        atoms = self.atoms.insert(atom, other.atoms)
        return self.__class__(xyz, atoms, sc=self.sc.copy())

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

    def attach(self, atom, other, other_atom, dist='calc', axis=None):
        """ Attaches another `Geometry` at the `atom` index with respect to `other_atom` using different methods.

        The attached geometry will be inserted at the end of the geometry via `add`.

        Parameters
        ----------
        atom : int
           atomic index which is the base position of the attachment. The distance
           between `atom` and `other_atom` is `dist`.
        other : Geometry
           the other Geometry to attach at the given point. In this case `dist` from
           `atom`.
        other_atom : int
           the index of the atom in `other` that is inserted at `atom`.
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
        other = self.new(other)
        if isinstance(dist, Real):
            # We have a single rational number
            if axis is None:
                raise ValueError(f"{self.__class__.__name__}.attach, `axis` has not been specified, please specify the axis when using a distance")

            # Now calculate the vector that we should have
            # between the atoms
            v = self.cell[axis, :]
            v = v / (v ** 2).sum() ** 0.5 * dist

        elif isinstance(dist, str):
            # We have a single rational number
            if axis is None:
                raise ValueError(f"{self.__class__.__name__}.attach, `axis` has not been specified, please specify the axis when using a distance")

            # This is the empirical distance between the atoms
            d = self.atoms[atom].radius(dist) + other.atoms[other_atom].radius(dist)
            if isinstance(axis, Integral):
                v = self.cell[axis, :]
            else:
                v = np.array(axis)

            v = v / (v ** 2).sum() ** 0.5 * d

        else:
            # The user *must* have supplied a vector
            v = np.array(dist)

        # Now create a copy of the other geometry
        # so that we move it...
        # Translate to origo, then back to position in new cell
        o = other.translate(-other.xyz[other_atom] + self.xyz[atom] + v)

        # We do not know how to handle the lattice-vectors,
        # so we will do nothing...
        return self.add(o)

    def reverse(self, atoms=None):
        """ Returns a reversed geometry

        Also enables reversing a subset of the atoms.

        Parameters
        ----------
        atoms : int or array_like, optional
             only reverse the given atomic indices, if not specified, all
             atoms will be reversed
        """
        if atoms is None:
            xyz = self.xyz[::-1, :]
        else:
            atoms = self._sanitize_atoms(atoms).ravel()
            xyz = np.copy(self.xyz)
            xyz[atoms, :] = self.xyz[atoms[::-1], :]
        return self.__class__(xyz, atoms=self.atoms.reverse(atoms), sc=self.sc.copy())

    def mirror(self, method, atoms=None):
        r""" Mirrors the atomic coordinates about the plane

        This will typically move the atomic coordinates outside of the unit-cell.
        This method should be used with care.

        Parameters
        ----------
        method : {'xy'/'z', ..., 'ab', ..., v}
           mirror the structure about a Cartesian direction (``x``, ``y``, ``z``),
           plane (``xy``, ``xz``, ``yz``) or about user defined vectors (``v``).
           A vector may also be specified by ``'ab'`` which is the vector normal
           to the plane spanned by the first and second lattice vector.
           or user defined vector (`v`)
        atoms : array_like, optional
           only mirror a subset of atoms
        """
        atoms = self._sanitize_atoms(atoms)

        g = self.copy()
        if isinstance(method, str):
            method = ''.join(sorted(method.lower()))
            if method in ['z', 'xy']:
                g.xyz[atoms, 2] *= -1
            elif method in ['x', 'yz']:
                g.xyz[atoms, 0] *= -1
            elif method in ['y', 'xz']:
                g.xyz[atoms, 1] *= -1
            elif method == 'a':
                method = self.cell[0]
            elif method == 'b':
                method = self.cell[1]
            elif method == 'c':
                method = self.cell[2]
            elif method == 'ab':
                method = cross3(self.cell[0], self.cell[1])
            elif method == 'ac':
                method = cross3(self.cell[0], self.cell[2])
            elif method == 'bc':
                method = cross3(self.cell[1], self.cell[2])
            else:
                raise ValueError(f"{self.__class__.__name__}.mirror unrecognized 'method' value")

        if not isinstance(method, str):
            # it has to be an array of length 3
            # Mirror about a user defined vector
            method = _a.asarrayd(method).copy()
            method /= fnorm(method)

            # project onto vector
            vp = self.xyz[atoms, :].dot(method) * 2

            # convert coordinates
            # first subtract the projection, then its mirror position
            self.xyz[atoms, :] -= vp.reshape(-1, 1) * method.reshape(1, 3)

        return self.__class__(g.xyz, atoms=g.atoms, sc=self.sc.copy())

    @property
    def fxyz(self):
        """ Returns geometry coordinates in fractional coordinates """
        return dot(self.xyz, self.icell.T)

    def axyz(self, atoms=None, isc=None):
        """ Return the atomic coordinates in the supercell of a given atom.

        The ``Geometry[...]`` slicing is calling this function with appropriate options.

        Parameters
        ----------
        atoms : int or array_like
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
        if atoms is None and isc is None:
            return self.xyz

        if not atoms is None:
            atoms = self._sanitize_atoms(atoms)

        # If only atoms has been specified
        if isc is None:
            # get offsets from atomic indices (note that this will be per atom)
            isc = self.a2isc(atoms)
            offset = self.sc.offset(isc)
            return self.xyz[self.sc2uc(atoms), :] + offset

        elif atoms is None:
            offset = self.sc.offset(isc)
            return self.xyz[:, :] + offset[None, :]

        # Neither of atoms, or isc are `None`, we add the offset to all coordinates
        offset = self.sc.offset(isc)

        if atoms.ndim == 0:
            return self.axyz(atoms) + offset

        return self.axyz(atoms) + offset[None, :]

    def scale(self, scale):
        """ Scale coordinates and unit-cell to get a new geometry with proper scaling

        Parameters
        ----------
        scale : float
           the scale factor for the new geometry (lattice vectors, coordinates
           and the atomic radii are scaled).
        """
        xyz = self.xyz * scale
        atoms = self.atoms.scale(scale)
        sc = self.sc.scale(scale)
        return self.__class__(xyz, atoms=atoms, sc=sc)

    def within_sc(self, shapes, isc=None,
                  atoms=None, atoms_xyz=None,
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
        atoms : array_like, optional
            List of atoms that will be considered. This can
            be used to only take out a certain atoms.
        atoms_xyz : array_like, optional
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
        if atoms is not None:
            atoms = self._sanitize_atoms(atoms)
        else:
            # If idx is None, then idx_xyz cannot be used!
            # So we force it to None
            atoms_xyz = None

        # Get shape centers
        off = shapes[-1].center[:]
        # Get the supercell offset
        soff = self.sc.offset(isc)[:]

        # Get atomic coordinate in principal cell
        if atoms_xyz is None:
            xa = self[atoms, :] + soff[None, :]
        else:
            # For extremely large systems re-using the
            # idx_xyz is faster than indexing
            # a very large array
            # However, this idx_xyz should not
            # be offset by any supercell
            xa = atoms_xyz[:, :] + soff[None, :]

        # Get indices and coordinates of the largest shape
        # The largest part of the calculation are to calculate
        # the content in the largest shape.
        ix = shapes[-1].within_index(xa)
        # Reduce search space
        xa = xa[ix, :]

        if atoms is None:
            # This is because of the pre-check of the distance checks
            atoms = ix
        else:
            atoms = atoms[ix]

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
            ret = [[atoms]]
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
        cum = np.array([], atoms.dtype)
        for i, s in enumerate(shapes):
            x = s.within_index(xa)
            if i > 0:
                x = np.setdiff1d(x, cum, assume_unique=True)
            # Update elements to remove in next loop
            cum = np.append(cum, x)
            ixS.append(x)

        # Do for the first shape
        ret = [[_a.asarrayi(atoms[ixS[0]]).ravel()]]
        rc = 0
        if ret_xyz:
            rc = rc + 1
            ret.append([xa[ixS[0], :]])
        if ret_rij:
            rd = rc + 1
            ret.append([d[ixS[0]]])
        for i in range(1, nshapes):
            ret[0].append(_a.asarrayi(atoms[ixS[i]]).ravel())
            if ret_xyz:
                ret[rc].append(xa[ixS[i], :])
            if ret_rij:
                ret[rd].append(d[ixS[i]])

        if ret_xyz or ret_rij:
            return ret
        return ret[0]

    def close_sc(self, xyz_ia, isc=(0, 0, 0), R=None,
                 atoms=None, atoms_xyz=None,
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
        isc : (3,), optional
            Integer super-cell offsets in which the coordinates are checked in.
            I.e. ``isc=[0, 0, 0]`` is the primary cell (default).
        R : float or array_like, optional
            The radii parameter to where the atomic connections are found.
            If `R` is an array it will return the indices:
            in the ranges ``( x <= R[0] , R[0] < x <= R[1], R[1] < x <= R[2] )``.
            If a single float it will return ``x <= R``.
        atoms : array_like of int, optional
            List of atoms that will be considered. This can
            be used to only take out a certain atoms.
        atoms_xyz : array_like of float, optional
            The atomic coordinates of the equivalent `atoms` variable (`atoms` must also be passed)
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
        if atoms is not None:
            atoms = self._sanitize_atoms(atoms).ravel()
        else:
            # If atoms is None, then atoms_xyz cannot be used!
            atoms_xyz = None

        if isinstance(xyz_ia, Integral):
            off = self.xyz[xyz_ia, :]
        elif not isndarray(xyz_ia):
            off = _a.asarrayd(xyz_ia)
        elif xyz_ia.ndim == 0:
            off = self.xyz[xyz_ia, :]
        else:
            off = xyz_ia

        # Calculate the complete offset
        foff = self.sc.offset(isc)[:] - off[:]

        # Get atomic coordinate in principal cell
        if atoms_xyz is None:
            dxa = self.axyz(atoms) + foff.reshape(1, 3)
        else:
            # For extremely large systems re-using the
            # atoms_xyz is faster than indexing
            # a very large array
            dxa = atoms_xyz + foff.reshape(1, 3)

        # Immediately downscale by easy checking
        # This will reduce the computation of the vector-norm
        # which is the main culprit of the time-consumption
        # This abstraction will _only_ help very large
        # systems.
        # For smaller ones this will actually be a slower
        # method..
        if atoms is None:
            atoms, d = indices_in_sphere_with_dist(dxa, max_R)
            dxa = dxa[atoms, :].reshape(-1, 3)
        else:
            ix, d = indices_in_sphere_with_dist(dxa, max_R)
            atoms = atoms[ix]
            dxa = dxa[ix, :].reshape(-1, 3)
            del ix

        if len(atoms) == 0:
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
            ret = [atoms]
            if ret_xyz:
                ret.append(xa)
            if ret_rij:
                ret.append(d)
            if ret_xyz or ret_rij:
                return ret
            return ret[0]

        if not is_ascending(R):
            raise ValueError(f'{self.__class__.__name__}.close_sc proximity checks for several '
                             'quantities at a time requires ascending R values.')

        # The more neigbours you wish to find the faster this becomes
        # We only do "one" heavy duty search,
        # then we immediately reduce search space to this subspace
        tidx = indices_le(d, R[0])
        ret = [[atoms[tidx]]]
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
                r_app(atoms[tidx])
                r_appx(xa[tidx])
                r_appd(d[tidx])
        elif ret_xyz:
            for i in range(1, len(R)):
                tidx = indices_gt_le(d, R[i-1], R[i])
                r_app(atoms[tidx])
                r_appx(xa[tidx])
        elif ret_rij:
            for i in range(1, len(R)):
                tidx = indices_gt_le(d, R[i-1], R[i])
                r_app(atoms[tidx])
                r_appd(d[tidx])
        else:
            for i in range(1, len(R)):
                tidx = indices_gt_le(d, R[i-1], R[i])
                r_app(atoms[tidx])

        if ret_xyz or ret_rij:
            return ret
        return ret[0]

    def __currently_not_used_close_rec(self, xyz_ia, R=None,
                 atoms=None, atoms_xyz=None,
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
        atoms : array_like of int, optional
            List of atoms that will be considered. This can
            be used to only take out a certain atoms.
        atoms_xyz : array_like of float, optional
            The atomic coordinates of the equivalent `atoms` variable (`atoms` must also be passed)
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

    def bond_correct(self, ia, atoms, method='calc'):
        """ Corrects the bond between `ia` and the `atoms`.

        Corrects the bond-length between atom `ia` and `atoms` in such
        a way that the atomic radius is preserved.
        I.e. the sum of the bond-lengths minimizes the distance matrix.

        Only atom `ia` is moved.

        Parameters
        ----------
        ia : int
            The atom to be displaced according to the atomic radius
        atoms : array_like or int
            The atom(s) from which the radius should be reduced.
        method : str, float, optional
            If str will use that as lookup in `Atom.radius`.
            Else it will be the new bond-length.
        """

        # Decide which algorithm to choose from
        atoms = self._sanitize_atoms(atoms).ravel()
        if len(atoms) == 1:
            algo = atoms[0]
        else:
            # signal a list of atoms
            algo = -1

        if algo >= 0:

            # We have a single atom
            # Get bond length in the closest direction
            # A bond-length HAS to be below 10
            atoms, c, d = self.close(ia, R=(0.1, 10.), atoms=algo,
                                     ret_xyz=True, ret_rij=True)
            i = np.argmin(d[1])
            # Convert to unitcell atom (and get the one atom)
            atoms = self.sc2uc(atoms[1][i])
            c = c[1][i]
            d = d[1][i]

            # Calculate the bond vector
            bv = self.xyz[ia, :] - c

            try:
                # If it is a number, we use that.
                rad = float(method)
            except Exception:
                # get radius
                rad = self.atoms[atoms].radius(method) \
                      + self.atoms[ia].radius(method)

            # Update the coordinate
            self.xyz[ia, :] = c + bv / d * rad

        else:
            raise NotImplementedError(
                'Changing bond-length dependent on several lacks implementation.')

    def within(self, shapes,
               atoms=None, atoms_xyz=None,
               ret_xyz=False, ret_rij=False, ret_isc=False):
        """ Indices of atoms in the entire supercell within a given shape from a given coordinate

        This heavily relies on the `within_sc` method.

        Note that if a connection is made in a neighbouring super-cell
        then the atomic index is shifted by the super-cell index times
        number of atoms.
        This allows one to decipher super-cell atoms from unit-cell atoms.

        Parameters
        ----------
        shapes : Shape, list of Shape
        atoms : array_like, optional
            List of indices for atoms that are to be considered
        atoms_xyz : array_like, optional
            The atomic coordinates of the equivalent `atoms` variable (`atoms` must also be passed)
        ret_xyz : bool, optional
            If true this method will return the coordinates
            for each of the couplings.
        ret_rij : bool, optional
            If true this method will return the distances from the `xyz_ia`
            for each of the couplings.
        ret_isc : bool, optional
            If true this method will return the supercell offsets for each of the couplings.

        Returns
        -------
        index
            indices of atoms (in supercell indices) within the shape
        xyz
            atomic coordinates of the indexed atoms (only for true `ret_xyz`)
        rij
            distance of the indexed atoms to the center of the shape (only for true `ret_rij`)
        isc
            supercell indices of the couplings (only for true `ret_isc`)
        """

        # Ensure that `shapes` is a list
        if isinstance(shapes, Shape):
            shapes = [shapes]
        nshapes = len(shapes)

        ret = [[np.empty([0], np.int32)] * nshapes]
        i = 0
        if ret_xyz:
            ixyz = i + 1
            i += 1
            ret.append([np.empty([0, 3], np.float64)] * nshapes)
        if ret_rij:
            irij = i + 1
            i += 1
            ret.append([np.empty([0], np.float64)] * nshapes)
        if ret_isc:
            iisc = i + 1
            i += 1
            ret.append([np.empty([0, 3], np.int32)] * nshapes)

        # number of special returns
        n_ret = i

        def isc_tile(isc, n):
            return tile(isc.reshape(1, -1), (n, 1))

        for s in range(self.n_s):

            na = self.na * s
            isc = self.sc.sc_off[s, :]
            sret = self.within_sc(shapes, self.sc.sc_off[s, :],
                                  atoms=atoms, atoms_xyz=atoms_xyz,
                                  ret_xyz=ret_xyz, ret_rij=ret_rij)
            if n_ret == 0:
                # This is to "fake" the return
                # of a list (we will do indexing!)
                sret = [sret]

            if isinstance(sret[0], list):
                # we have a list of arrays (nshapes > 1)
                for i, x in enumerate(sret[0]):
                    ret[0][i] = concatenate((ret[0][i], x + na), axis=0)
                    if ret_xyz:
                        ret[ixyz][i] = concatenate((ret[ixyz][i], sret[ixyz][i]), axis=0)
                    if ret_rij:
                        ret[irij][i] = concatenate((ret[irij][i], sret[irij][i]), axis=0)
                    if ret_isc:
                        ret[iisc][i] = concatenate((ret[iisc][i], isc_tile(isc, len(x))), axis=0)
            elif len(sret[0]) > 0:
                # We can add it to the list (nshapes == 1)
                # We add the atomic offset for the supercell index
                ret[0][0] = concatenate((ret[0][0], sret[0] + na), axis=0)
                if ret_xyz:
                    ret[ixyz][0] = concatenate((ret[ixyz][0], sret[ixyz]), axis=0)
                if ret_rij:
                    ret[irij][0] = concatenate((ret[irij][0], sret[irij]), axis=0)
                if ret_isc:
                    ret[iisc][0] = concatenate((ret[iisc][0], isc_tile(isc, len(sret[0]))), axis=0)

        if nshapes == 1:
            if n_ret == 0:
                return ret[0][0]
            return tuple(ret[i][0] for i in range(n_ret + 1))

        if n_ret == 0:
            return ret[0]
        return ret

    def close(self, xyz_ia, R=None,
              atoms=None, atoms_xyz=None,
              ret_xyz=False, ret_rij=False, ret_isc=False):
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

        atoms : array_like, optional
            List of indices for atoms that are to be considered
        atoms_xyz : array_like, optional
            The atomic coordinates of the equivalent `atoms` variable (`atoms` must also be passed)
        ret_xyz : bool, optional
            If true this method will return the coordinates
            for each of the couplings.
        ret_rij : bool, optional
            If true this method will return the distances from the `xyz_ia`
            for each of the couplings.
        ret_isc : bool, optional
            If true this method will return the lattice offset from `xyz_ia`
            for each of the couplings.

        Returns
        -------
        index
            indices of atoms (in supercell indices) within the shells of radius `R`
        xyz
            atomic coordinates of the indexed atoms (only for true `ret_xyz`)
        rij
            distance of the indexed atoms to the center coordinate (only for true `ret_rij`)
        isc
            integer lattice offsets for the couplings (related to `rij` without atomic coordinates)
        """
        if R is None:
            R = self.maxR()
        R = _a.asarrayd(R).ravel()
        nR = R.size

        # Convert inedx coordinate to point
        if isinstance(xyz_ia, Integral):
            xyz_ia = self.xyz[xyz_ia, :]
        elif not isndarray(xyz_ia):
            xyz_ia = _a.asarrayd(xyz_ia)

        ret = [[np.empty([0], np.int32)] * nR]
        i = 0
        if ret_xyz:
            ixyz = i + 1
            i += 1
            ret.append([np.empty([0, 3], np.float64)] * nR)
        if ret_rij:
            irij = i + 1
            i += 1
            ret.append([np.empty([0], np.float64)] * nR)
        if ret_isc:
            iisc = i + 1
            i += 1
            ret.append([np.empty([0, 3], np.int32)] * nR)

        # number of special returns
        n_ret = i

        def isc_tile(isc, n):
            return tile(isc.reshape(1, -1), (n, 1))

        for s in range(self.n_s):

            na = self.na * s
            isc = self.sc.sc_off[s, :]
            sret = self.close_sc(xyz_ia, isc, R=R,
                                 atoms=atoms, atoms_xyz=atoms_xyz,
                                 ret_xyz=ret_xyz, ret_rij=ret_rij)

            if n_ret == 0:
                # This is to "fake" the return
                # of a list (we will do indexing!)
                sret = [sret]

            if isinstance(sret[0], list):
                # we have a list of arrays (len(R) > 1)
                for i, x in enumerate(sret[0]):
                    ret[0][i] = concatenate((ret[0][i], x + na), axis=0)
                    if ret_xyz:
                        ret[ixyz][i] = concatenate((ret[ixyz][i], sret[ixyz][i]), axis=0)
                    if ret_rij:
                        ret[irij][i] = concatenate((ret[irij][i], sret[irij][i]), axis=0)
                    if ret_isc:
                        ret[iisc][i] = concatenate((ret[iisc][i], isc_tile(isc, len(x))), axis=0)
            elif len(sret[0]) > 0:
                # We can add it to the list (len(R) == 1)
                # We add the atomic offset for the supercell index
                ret[0][0] = concatenate((ret[0][0], sret[0] + na), axis=0)
                if ret_xyz:
                    ret[ixyz][0] = concatenate((ret[ixyz][0], sret[ixyz]), axis=0)
                if ret_rij:
                    ret[irij][0] = concatenate((ret[irij][0], sret[irij]), axis=0)
                if ret_isc:
                    ret[iisc][0] = concatenate((ret[iisc][0], isc_tile(isc, len(sret[0]))), axis=0)

        if nR == 1:
            if n_ret == 0:
                return ret[0][0]
            return tuple(ret[i][0] for i in range(n_ret + 1))

        if n_ret == 0:
            return ret[0]
        return ret

    def a2transpose(self, atoms1, atoms2=None):
        """ Transposes connections from `atoms1` to `atoms2` such that supercell connections are transposed

        When handling supercell indices it is useful to get the *transposed* connection. I.e. if you have
        a connection from site ``i`` (in unit cell indices) to site ``j`` (in supercell indices) it may be
        useful to get the equivalent supercell connection such for site ``j`` (in unit cell indices) to
        site ``i`` (in supercell indices) such that they correspond to the transposed coupling.

        Note that since this transposes couplings the indices returned are always expanded to the full
        length if either of the inputs are a single index.

        Examples
        --------
        >>> gr = geom.graphene()
        >>> atoms = gr.close(0, 1.5)
        >>> atoms
        array([0, 1, 5, 9], dtype=int32)
        >>> gr.a2transpose(0, atoms)
        (array([0, 1, 1, 1], dtype=int32), array([ 0,  0, 14, 10], dtype=int32))

        Parameters
        ----------
        atoms1 : array_like
            atomic indices must have same length as `atoms2` or length 1
        atoms2 : array_like, optional
            atomic indices must have same length as `atoms1` or length 1.
            If not present then only `atoms1` will be returned in transposed indices.

        Returns
        -------
        atoms2 : array_like
            transposed indices for atoms2 (only returned if `atoms2` is not None)
        atoms1 : array_like
            transposed indices for atoms1
        """
        # First check whether they have the same size, if so then do not pre-process
        atoms1 = self._sanitize_atoms(atoms1)
        if atoms2 is None:
            # we only need to transpose atoms1
            offset = self.sc.sc_index(-self.a2isc(atoms1)) * self.na
            return atoms1 % self.na + offset

        atoms2 = self._sanitize_atoms(atoms2)
        if atoms1.size == atoms2.size:
            pass
        elif atoms1.size == 1: # typical case where atoms1 is a single number
            atoms1 = np.tile(atoms1, atoms2.size)
        elif atoms2.size == 1:
            atoms2 = np.tile(atoms2, atoms1.size)
        else:
            raise ValueError(f'{self.__class__.__name__}.a2transpose only allows length 1 or same length arrays.')

        # Now convert atoms
        na = self.na
        sc_index = self.sc.sc_index
        isc1 = self.a2isc(atoms1)
        isc2 = self.a2isc(atoms2)

        atoms1 = atoms1 % na + sc_index(-isc2) * na
        atoms2 = atoms2 % na + sc_index(-isc1) * na
        return atoms2, atoms1

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
        >>> atoms = gr.close(0, 1.5)
        >>> atoms
        array([0, 1, 5, 9], dtype=int32)
        >>> gr.o2transpose(0, atoms)
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
        orb1 = self._sanitize_orbs(orb1)
        if orb2 is None:
            # we only need to transpose orb1
            offset = self.sc.sc_index(-self.o2isc(orb1)) * self.no
            return orb1 % self.no + offset

        orb2 = self._sanitize_orbs(orb2)
        if orb1.size == orb2.size:
            pass
        elif orb1.size == 1: # typical case where orb1 is a single number
            orb1 = np.tile(orb1, orb2.size)
        elif orb2.size == 1:
            orb2 = np.tile(orb2, orb1.size)
        else:
            raise ValueError(f'{self.__class__.__name__}.o2transpose only allows length 1 or same length arrays.')

        # Now convert orbs
        no = self.no
        sc_index = self.sc.sc_index
        isc1 = self.o2isc(orb1)
        isc2 = self.o2isc(orb2)

        orb1 = orb1 % no + sc_index(-isc2) * no
        orb2 = orb2 % no + sc_index(-isc1) * no
        return orb2, orb1

    def a2o(self, atoms, all=False):
        """
        Returns an orbital index of the first orbital of said atom.
        This is particularly handy if you want to create
        TB models with more than one orbital per atom.

        Note that this will preserve the super-cell offsets.

        Parameters
        ----------
        atoms : array_like
             Atomic indices
        all : bool, optional
             ``False``, return only the first orbital corresponding to the atom,
             ``True``, returns list of the full atom
        """
        # we must not alter `atoms` as it may come from outside
        off, atoms = np.divmod(self._sanitize_atoms(atoms), self.na)
        off *= self.no
        if not all:
            return self.firsto[atoms] + off
        ob = self.firsto[atoms] + off
        oe = self.lasto[atoms] + off + 1

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

        isc, io = np.divmod(_a.asarrayi(io).ravel(), self.no)
        a = list_index_le(io, self.lasto)
        if unique:
            return np.unique(a + isc * self.na)
        return a + isc * self.na

    def uc2sc(self, atoms, unique=False):
        """ Returns atom from unit-cell indices to supercell indices, possibly removing dublicates

        Parameters
        ----------
        atoms : array_like or int
           the atomic unit-cell indices to be converted to supercell indices
        unique : bool, optional
           If True the returned indices are unique and sorted.
        """
        atoms = self._sanitize_atoms(atoms) % self.na
        atoms = (atoms.reshape(1, -1) + _a.arangei(self.n_s).reshape(-1, 1) * self.na).ravel()
        if unique:
            return np.unique(atoms)
        return atoms
    auc2sc = uc2sc

    def sc2uc(self, atoms, unique=False):
        """ Returns atoms from supercell indices to unit-cell indices, possibly removing dublicates

        Parameters
        ----------
        atoms : array_like or int
           the atomic supercell indices to be converted to unit-cell indices
        unique : bool, optional
           If True the returned indices are unique and sorted.
        """
        atoms = self._sanitize_atoms(atoms) % self.na
        if unique:
            return np.unique(atoms)
        return atoms
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

    def ouc2sc(self, orbitals, unique=False):
        """ Returns orbitals from unit-cell indices to supercell indices, possibly removing dublicates

        Parameters
        ----------
        orbitals : array_like or int
           the orbital unit-cell indices to be converted to supercell indices
        unique : bool, optional
           If True the returned indices are unique and sorted.
        """
        orbitals = _a.asarrayi(orbitals) % self.no
        orbitals = (orbitals.reshape(1, -1) +
                    _a.arangei(self.n_s).reshape(-1, 1) * self.no).ravel()
        if unique:
            return np.unique(orbitals)
        return orbitals

    def a2isc(self, atoms):
        """ Returns super-cell index for a specific/list atom

        Returns a vector of 3 numbers with integers.
        """
        atoms = self._sanitize_atoms(atoms) // self.na
        return self.sc.sc_off[atoms, :]

    # This function is a bit weird, it returns a real array,
    # however, there should be no ambiguity as it corresponds to th
    # offset and "what else" is there to query?
    def a2sc(self, atoms):
        """
        Returns the super-cell offset for a specific atom
        """
        return self.sc.offset(self.a2isc(atoms))

    def o2isc(self, orbitals):
        """
        Returns the super-cell index for a specific orbital.

        Returns a vector of 3 numbers with integers.
        """
        orbitals = _a.asarrayi(orbitals) // self.no
        return self.sc.sc_off[orbitals, :]

    def o2sc(self, orbitals):
        """
        Returns the super-cell offset for a specific orbital.
        """
        return self.sc.offset(self.o2isc(orbitals))

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
    @deprecate_method("Geometry.fromASE is deprecated in favor of Geometry.new(ase_obj)")
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
        return cls(xyz, atoms=Z, sc=cell)

    @deprecate_method("Geometry.toASE is deprecated in favor of Geometry.to.ase()")
    def toASE(self):
        """ Returns the geometry as an ASE ``Atoms`` object """
        from ase import Atoms as ASE_Atoms
        return ASE_Atoms(symbols=self.atoms.Z, positions=self.xyz.tolist(),
                         cell=self.cell.tolist(), pbc=self.nsc > 1)

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
        other = self.new(other)
        if not isinstance(other, Geometry):
            return False
        same = self.sc.equal(other.sc, tol=tol)
        same = same and np.allclose(self.xyz, other.xyz, atol=tol)
        same = same and self.atoms.equal(other.atoms, R)
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
        for ias, atoms in self.iter_block(iR=iR, method=method):

            # Get all the indexed atoms...
            # This speeds up the searching for
            # coordinates...
            atoms_xyz = self[atoms, :]

            # Loop the atoms inside
            for ia in ias:
                idx, r = self.close(ia, R=R, atoms=atoms, atoms_xyz=atoms_xyz, ret_rij=True)
                rij[ia, ia] = 0.
                rij[ia, idx[1]] = r[1]

        return rij

    def distance(self, atoms=None, R=None, tol=0.1, method='average'):
        """ Calculate the distances for all atoms in shells of radius `tol` within `max_R`

        Parameters
        ----------
        atoms : int or array_like, optional
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
        atoms = self._sanitize_atoms(atoms).ravel()

        # Figure out maximum distance
        if R is None:
            R = self.maxR()
            if R < 0:
                raise ValueError(self.__class__.__name__ +
                                  ".distance cannot determine the `R` parameter. "
                                  "The internal `maxR()` is negative and thus not set. "
                                  "Set an explicit value for `R`.")
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

                dR = concatenate((dR, _a.aranged(dR[-1] + t, R + t * .55, t)))

            # Reduce to the largest value above R
            # This ensures that R, truly is the largest considered element
            dR = dR[:(dR > R).nonzero()[0][0]+1]

        # Now we can figure out the list of atoms in each shell
        # First create the initial lists of shell atoms
        # The inner shell will never be used, because it should correspond
        # to the atom it-self.
        shells = [[] for i in range(len(dR) - 1)]

        for a in atoms:
            _, r = self.close(a, R=dR, ret_rij=True)

            for i, rlist in enumerate(r[1:]):
                shells[i].extend(rlist)

        # Now parse all of the shells with the correct routine
        # First we grap the routine:
        if isinstance(method, str):
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
                    raise ValueError(f"{self.__class__.__name__}.distance `method` has wrong input value.")
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
        d['atoms'] = self.atoms.__getstate__()
        return d

    def __setstate__(self, d):
        """ Re-create the state of this object """
        sc = SuperCell([1, 1, 1])
        sc.__setstate__(d)
        atoms = Atoms()
        atoms.__setstate__(d['atoms'])
        self.__init__(d['xyz'], atoms=atoms, sc=sc)

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
                   help='Move all atoms such that the smallest value along each Cartesian direction will be at the origin.')

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
                g = Geometry([float(x) for x in values[0].split(',')], atoms=Atom(values[1]))
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

        # Sort
        class Sort(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                # call geometry.sort(...) using appropriate keywords (and ordered dict)
                kwargs = OrderedDict()
                opts = values[0].split(';')
                for i, opt in enumerate(opts):
                    # Split for equal
                    opt = opt.split('=', 1)
                    if len(opt) > 1:
                        opt, val = opt
                    else:
                        opt = opt[0]
                        val = "True"
                    if val.lower() in ['t', 'true']:
                        val = True
                    elif val.lower() in ['f', 'false']:
                        val = False
                    elif opt in ['atol']:
                        # float values
                        val = float(val)
                    elif opt == 'group':
                        pass
                    else:
                        # it must be a range/tuple
                        val = lstranges(strmap(int, val))

                    # we always add integers to allow users to use the same keywords on commandline
                    kwargs[opt.strip() + str(i)] = val
                ns._geometry = ns._geometry.sort(**kwargs)
        p.add_argument(*opts('--sort'), nargs=1, metavar='SORT',
                       action=Sort,
                       help='Semi-colon separated options for sort, please always encapsulate in quotation ["axis=0;descend;lattice=(1, 2);group=Z"].')

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
                    vs = getattr(ns, '_vector_scale')
                    if isinstance(vs, bool):
                        if vs:
                            vs = 1. / np.max(sqrt(square(v).sum(1)))
                            info(f"Scaling vector by: {vs}")
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


# Define base-class for this
class GeometryNewDispatcher(AbstractDispatch):
    """ Base dispatcher from class passing arguments to Geometry class

    This forwards all `__call__` calls to `dispatch`
    """

    def __call__(self, *args, **kwargs):
        return self.dispatch(*args, **kwargs)


class GeometryNewAseDispatcher(GeometryNewDispatcher):
    def dispatch(self, aseg, **kwargs):
        """ Convert an ``ase`` object into a `Geometry` """
        Z = aseg.get_atomic_numbers()
        xyz = aseg.get_positions()
        cell = aseg.get_cell()
        nsc = [3 if pbc else 1 for pbc in aseg.pbc]
        sc = SuperCell(cell, nsc=nsc)
        return self._obj(xyz, atoms=Z, sc=sc, **kwargs)
Geometry.new.register("ase", GeometryNewAseDispatcher)

# currently we can't ensure the ase Atoms type
# to get it by type(). That requires ase to be importable.
try:
    from ase import Atoms as ase_Atoms
    Geometry.new.register(ase_Atoms, GeometryNewAseDispatcher)
    # ensure we don't pollute name-space
    del ase_Atoms
except:
    pass


# Bypass regular Geometry to be returned as is
class GeometryNewGeometryDispatcher(GeometryNewDispatcher):
    def dispatch(self, geom):
        """ Return geometry as-is (no copy), for sanitization purposes """
        return geom
Geometry.new.register(Geometry, GeometryNewGeometryDispatcher)


class GeometryNewFileDispatcher(GeometryNewDispatcher):
    def dispatch(self, *args, **kwargs):
        """ Defer the `Geometry.read` method by passing down arguments """
        return self._obj.read(*args, **kwargs)
Geometry.new.register(str, GeometryNewFileDispatcher)
Geometry.new.register(Path, GeometryNewFileDispatcher)
# see sisl/__init__.py for Geometry.new.register(BaseSile, GeometryNewFileDispatcher)


class GeometryToDispatcher(AbstractDispatch):
    """ Base dispatcher from class passing from Geometry class """
    @staticmethod
    def _ensure_object(obj):
        if isinstance(obj, type):
            raise ValueError(f"Dispatcher on {obj} must not be called on the class.")


class GeometryToAseDispatcher(GeometryToDispatcher):
    def dispatch(self, *args, **kwargs):
        from ase import Atoms as ase_Atoms
        geom = self._obj
        self._ensure_object(geom)
        if len(args) > 0:
            raise ValueError(f"{geom}.to.ase only accepts keyword arguments")
        return ase_Atoms(symbols=geom.atoms.Z, positions=geom.xyz.tolist(),
                         cell=geom.cell.tolist(), pbc=geom.nsc > 1, **kwargs)

Geometry.to.register("ase", GeometryToAseDispatcher)


class GeometryToSileDispatcher(GeometryToDispatcher):
    def dispatch(self, *args, **kwargs):
        geom = self._obj
        self._ensure_object(geom)
        return geom.write(*args, **kwargs)
Geometry.to.register("str", GeometryToSileDispatcher)
Geometry.to.register("path", GeometryToSileDispatcher)
# to do geom.to[Path](path)
Geometry.to.register(str, GeometryToSileDispatcher)
Geometry.to.register(Path, GeometryToSileDispatcher)


class GeometryToDataframeDispatcher(GeometryToDispatcher):
    def dispatch(self, *args, **kwargs):
        import pandas as pd
        geom = self._obj
        self._ensure_object(geom)

        # Now create data-frame
        # Currently we will populate it with
        # - xyz
        # - symbol
        # - Z
        # - tag
        # - R
        # - mass
        # - valence q
        # - norbs
        data = {}
        x, y, z = geom.xyz.T.tolist()
        data["x"] = x
        data["y"] = y
        data["z"] = z

        atoms = geom.atoms
        data["Z"] = atoms.Z
        data["mass"] = atoms.mass
        data["R"] = atoms.maxR(all=True)
        data["q0"] = atoms.q0
        data["norbitals"] = atoms.orbitals

        return pd.DataFrame(data)
Geometry.to.register("df", GeometryToDataframeDispatcher)


@set_module("sisl")
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
    import argparse
    from pathlib import Path

    from sisl.io import get_sile, BaseSile

    # The geometry-file *MUST* be the first argument
    # (except --help|-h)
    exe = Path(sys.argv[0]).name

    # We cannot create a separate ArgumentParser to retrieve a positional arguments
    # as that will grab the first argument for an option!

    # Start creating the command-line utilities that are the actual ones.
    description = f"""
This manipulation utility is highly advanced and one should note that the ORDER of
options is determining the final structure. For instance:

   {exe} geom.xyz --repeat x 2 --repeat y 2

is NOT equivalent to:

   {exe} geom.xyz --repeat y 2 --repeat x 2

This may be unexpected but enables one to do advanced manipulations.

Additionally, in between arguments, one may store the current state of the geometry
by writing to a standard file.

   {exe} geom.xyz --repeat y 2 geom_repy.xyz --repeat x 2 geom_repy_repx.xyz

will create two files:
   geom_repy.xyz
will only be repeated 2 times along the second lattice vector, while:
   geom_repy_repx.xyz
will be repeated 2 times along the second lattice vector, and then the first
lattice vector.
    """

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

    p = argparse.ArgumentParser(exe,
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
                info(f"Cannot find file '{input_file}'!")
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
            print('  {:10.6f} {:10.6f} {:10.6f}'.format(*g.cell[i, :]))
        print('SuperCell:')
        print('  {:d} {:d} {:d}'.format(*g.nsc))
        print(' {:>10s} {:>10s} {:>10s}  {:>3s}'.format('x', 'y', 'z', 'Z'))
        for ia in g:
            print(' {1:10.6f} {2:10.6f} {3:10.6f}  {0:3d}'.format(g.atoms[ia].Z,
                                                                  *g.xyz[ia, :]))

    if ret_geometry:
        return g
    return 0
