"""
Geometry class to retain the atomic structure.
"""
from __future__ import print_function, division

# To check for integers
from numbers import Integral
from math import acos, pi
import sys
import warnings

import numpy as np

from .utils import *
from .quaternion import Quaternion
from .supercell import SuperCell, SuperCellChild
from .atom import Atom
from ._help import array_fill_repeat, ensure_array

__all__ = ['Geometry']



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
    xyz : array_like
        atomic coordinates
        ``xyz[i,:]`` is the atomic coordinate of the i'th atom.
    atom : array_like
        atomic species retrieved from the `PeriodicTable`
    sc : `SuperCell`
        the unit-cell describing the atoms in a periodic
        super-cell
    """

    def __init__(self, xyz, atom=None, sc=None):

        # Create the geometry coordinate
        self.xyz = np.copy(np.asarray(xyz, dtype=np.float64))
        self.xyz.shape = (-1, 3)
        self.na = len(self.xyz)

        # Default value
        if atom is None:
            atom = Atom('H')

        # Correct the atoms input to Atom
        if isinstance(atom, list):
            if isinstance(atom[0], (str, Integral)):
                A = np.array([Atom(a) for a in atom])
            elif isinstance(atom[0], Atom):
                A = np.array(atom)
            else:
                raise ValueError('atom keyword was wrong input')
        elif isinstance(atom, str):
            A = np.array([Atom(atom)])
        else:
            A = np.array([atom]).flatten()

        # Create atom objects
        self.atom = array_fill_repeat(A, self.na, cls=Atom)

        # Get total number of orbitals
        orbs = np.array([a.orbs for a in self.atom], np.int32)

        # Get total number of orbitals
        self.no = np.sum(orbs)

        # Create local lasto
        lasto = np.append(np.array(0, np.int32), orbs)
        self.lasto = np.cumsum(lasto)

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
    def dR(self):
        """ Returns the maximum orbital range of the atoms """
        return np.amax([a.dR for a in self.atom])

    @property
    def atoms(self):
        """ Returns the atoms (mainly for backwards compatibility) """
        return self.atom

    @property
    def no_s(self):
        """ Number of supercell orbitals """
        return self.no * self.n_s

    def __len__(self):
        """ Return number of atoms in this geometry """
        return self.na

    def __getitem__(self, key):
        """ Returns geometry coordinates """
        return self.xyz[key]

    @staticmethod
    def read(sile):
        """ Reads geometry from the `Sile` using `Sile.read_geom`

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
            return sile.read_geom()
        else:
            return get_sile(sile).read_geom()

    def write(self, sile, *args, **kwargs):
        """ Writes geometry to the `Sile` using `sile.write_geom`

        Parameters
        ----------
        sile : Sile, str
            a `Sile` object which will be used to write the geometry
            if it is a string it will create a new sile using `get_sile`
        *args, **kwargs: Any other args will be passed directly to the
                         underlying routine
        """

        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            sile.write_geom(self, *args, **kwargs)
        else:
            get_sile(sile, 'w').write_geom(self, *args, **kwargs)

    def __repr__(self):
        """ Representation of the object """
        spec = self._species_order()
        s = '{{na: {0}, no: {1}, species:\n {{ n: {2},'.format(
            self.na, self.no, len(spec))
        for z in spec:
            s += '\n   [{0}], '.format(str(spec[z][1]))
        return s[
            :-2] + '\n }},\n nsc: [{1}, {2}, {3}], dR: {0}\n}}'.format(self.dR, *self.nsc)

    def iter_species(self):
        """
        Returns an iterator over all atoms and species as a tuple in this geometry

         >>> for ia,a,idx_specie in self.iter_species():

        with ``ia`` being the atomic index, ``a`` the `Atom` object, `idx_specie`
        is the index of the species
        """
        # Count for the species
        spec = []
        for ia, a in enumerate(self.atom):
            if a.tag not in spec:
                spec.append(a.tag)
                yield ia, a, len(spec) - 1
            else:
                # It must already exist in the species list
                yield ia, a, spec.index(a.tag)

    def iter_linear(self):
        """
        Returns an iterator for simple linear ranges.

        This iterator is the same as:

          >>> for ia in range(len(self)):
          >>>    <do something>
        or equivalently
          >>> for ia in self:
          >>>    <do something>
        """
        for ia in range(len(self)):
            yield ia

    # Default iteration module to loop over atoms
    __iter__ = iter_linear

    def iter_block(self, iR=10, dR=None):
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
        iR  : (10) integer
            the number of ``dR`` ranges taken into account when doing the iterator
        dR  : (self.dR), float
            enables overwriting the local dR quantity.

        Returns two lists with [0] being a list of atoms to be looped and [1] being the atoms that
        need searched.
        """

        # We implement yields as we can then do nested iterators
        # create a boolean array
        na = len(self)
        not_passed = np.empty(na, dtype='b')
        not_passed[:] = True
        not_passed_N = na

        if dR is None:
            selfdR = self.dR
            # The boundaries (ensure complete overlap)
            dr = (selfdR * (iR - 1), selfdR * (iR + .1))
        else:
            dr = (dR * (iR - 1), dR * (iR + .1))

        # loop until all passed are true
        while not_passed_N > 0:

            # Take a random non-passed element
            all_true = np.where(not_passed)[0]

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
            all_idx = self.close(idx, dR=dr)

            # Get unit-cell atoms
            all_idx[0] = self.sc2uc(all_idx[0], uniq=True)
            # First extend the search-space (before reducing)
            all_idx[1] = self.sc2uc(
                np.append(
                    all_idx[1],
                    all_idx[0]),
                uniq=True)

            # Only select those who have not been runned yet
            all_idx[0] = all_idx[0][np.where(not_passed[all_idx[0]])[0]]
            if len(all_idx[0]) == 0:
                raise ValueError(
                    'Internal error, please report to the developers')

            # Tell the next loop to skip those passed
            not_passed[all_idx[0]] = False
            # Update looped variables
            not_passed_N -= len(all_idx[0])

            # Now we want to yield the stuff revealed
            # all_idx[0] contains the elements that should be looped
            # all_idx[1] contains the indices that can be searched
            yield all_idx[0], all_idx[1]

        if np.any(not_passed):
            raise ValueError(
                'Error on iterations. Not all atoms has been visited.')

    def sub(self, atom, cell=None):
        """
        Returns a subset of atoms from the geometry.

        Indices passed *MUST* be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atom  : array_like
            indices of all atoms to be removed.
        cell   : (``self.cell``), array_like, optional
            the new associated cell of the geometry
        """
        atms = np.asarray([atom], np.int32).flatten() % self.na
        if cell is None:
            return self.__class__(
                self.xyz[
                    atms, :], atom=[
                    self.atom[i] for i in atms], sc=self.sc.copy())
        return self.__class__(self.xyz[atms, :],
                              atom=[self.atom[i] for i in atms], sc=cell)

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

        Doing ``geom.cut(2,1).tile(seps=2,axis=1)``, could for symmetric setups,
        be equivalent to a no-op operation. A ``UserWarning`` will be issued
        if this is not the case.

        Parameters
        ----------
        seps  : int
            number of times the structure will be cut.
        axis  : int
            the axis that will be cut
        seg : int, optional (0)
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
        if not np.allclose(
                new.tile(
                    seps,
                    axis).xyz,
                self.xyz,
                rtol=rtol,
                atol=atol):
            st = 'The cut structure cannot be re-created by tiling'
            st += '\nThe difference between the coordinates can be altered using rtol, atol'
            warnings.warn(st, UserWarning)
        return new

    def _species_order(self):
        """ Returns dictionary with species indices for the atoms.
        They will be populated in order of appearence"""

        # Count for the species
        spec = {}
        ispec = 0
        for a in self.atom:
            if not a.tag is None:
                if a.tag not in spec:
                    ispec += 1
                    spec[a.tag] = (ispec, a)
            elif not a.Z in spec:
                ispec += 1
                spec[a.Z] = (ispec, a)
        return spec

    def copy(self):
        """
        Returns a copy of the object.
        """
        return self.__class__(np.copy(self.xyz),
                              atom=self.atom, sc=self.sc.copy())

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
        atms = np.asarray([atom], np.int32).flatten() % self.na
        idx = np.setdiff1d(np.arange(self.na), atms, assume_unique=True)
        return self.sub(idx)

    def tile(self, reps, axis):
        """
        Returns a geometry tiled, i.e. copied.

        The atomic indices are retained for the base structure.

        Parameters
        ----------
        reps  : number of tiles (repetitions)
        axis  : direction of tiling
                  0, 1, 2 according to the cell-direction

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
        return self.__class__(xyz, atom=self.atom, sc=sc)

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
        tight-binding parameter sets for TBtrans.

        For geometries with a single atom this routine returns the same as
        `tile`.

        It is adviced to only use this for electrode Bloch's theorem
        purposes as `tile` is faster.

        Parameters
        ----------
        reps  : number of repetitions
        axis  : direction of repetition
                  0, 1, 2 according to the cell-direction

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
        atom = [None for i in range(na)]
        dx = np.dot(np.arange(reps)[:, None], self.cell[axis, :][None, :])
        # Start the repetition
        ja = 0
        for ia in range(self.na):
            # Single atom displacements
            # First add the basic atomic coordinate,
            # then add displacement for each repetition.
            xyz[ja:ja + reps, :] = self.xyz[ia, :][None, :] + dx[:, :]
            for i in range(reps):
                atom[ja + i] = self.atom[ia]
            ja += reps
        # Create the geometry and return it
        return self.__class__(xyz, atom=atom, sc=sc)

    def rotatea(self, angle, only='abc+xyz', radians=False):
        return self.rotate(angle, self.cell[0, :], only=only, radians=radians)

    def rotateb(self, angle, only='abc+xyz', radians=False):
        return self.rotate(angle, self.cell[1, :], only=only, radians=radians)

    def rotatec(self, angle, only='abc+xyz', radians=False):
        return self.rotate(angle, self.cell[2, :], only=only, radians=radians)

    def rotate(self, angle, v, only='abc+xyz', radians=False):
        """
        Rotates the geometry, in-place by the angle around the vector

        Per default will the entire geometry be rotated, such that everything
        is aligned as before rotation.

        However, by supplying ``only='abc|xyz'`` one can designate which
        part of the geometry that will be rotated.

        Parameters
        ----------
        angle : float
             the angle in radians of which the geometry should be rotated
        v     : array_like [3]
             the vector around the rotation is going to happen
             v = [1,0,0] will rotate in the ``yz`` plane
        only  : ('abc+xyz'), str, optional
             which coordinate subject should be rotated,
             if ``abc`` is in this string the cell will be rotated
             if ``xyz`` is in this string the coordinates will be rotated
        """
        vn = np.copy(np.asarray(v, dtype=np.float64)[:])
        vn /= np.sum(vn ** 2) ** .5
        q = Quaternion(angle, vn, radians=radians)
        q /= q.norm()  # normalize the quaternion

        # Rotate by direct call
        sc = self.sc.rotate(angle, vn, radians=radians, only=only)

        if 'xyz' in only:
            xyz = q.rotate(self.xyz)
        else:
            xyz = np.copy(self.xyz)

        return self.__class__(xyz, atom=self.atom, sc=sc)

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

    def translate(self, v, atom=None, cell=False):
        """ Translates the geometry by ``v``

        One can translate a subset of the atoms by supplying ``atom``.

        Returns a copy of the structure translated by ``v``.
        """
        g = self.copy()
        if atom is None:
            g.xyz[:, :] += np.asarray(v, g.xyz.dtype)[None, :]
        else:
            g.xyz[atom, :] += np.asarray(v, g.xyz.dtype)[None, :]
        if cell:
            g.set_supercell(g.sc.translate(v))
        return g

    def swap(self, a, b):
        """ Returns a geometry with swapped atoms

        This can be used to reorder elements of a geometry.
        """
        xyz = np.copy(self.xyz)
        xyz[a, :] = self.xyz[b, :]
        xyz[b, :] = self.xyz[a, :]
        atom = np.copy(self.atom)
        atom[a] = self.atom[b]
        atom[b] = self.atom[a]
        return self.__class__(xyz, atom=atom, sc=self.sc.copy())

    def swapaxes(self, a, b, swap='cell+xyz'):
        """ Returns geometry with swapped axis

        If ``swapaxes(0,1)`` it returns the 0 and 1 values
        swapped in the ``cell`` variable.

        Parameters
        ----------
        a : int
           axes 1, swaps with ``b``
        b : int
           axes 2, swaps with ``a``
        swap : str, "cell+xyz"
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
        return self.__class__(xyz, atom=np.copy(self.atom), sc=sc)

    def center(self, atom=None, which='xyz'):
        """ Returns the center of the geometry

        By specifying ``which`` one can control whether it should be:

        * ``xyz|position``: Center of coordinates (default)
        * ``mass``: Center of mass
        * ``cell``: Center of cell

        Parameters
        ----------
        atom : list, ndarray
            list of atomic indices to find center of
        which : str
            determine whether center should be of 'cell', mass-centered ('mass'),
            or absolute center of the positions.
        """
        if 'cell' in which:
            return self.sc.center()
        if atom is None:
            g = self
        else:
            g = self.sub(atom)
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
         >>> self.lasto = np.append(self.lasto,other.lasto)

        NOTE: The cell appended is only in the axis that
        is appended, which means that the other cell directions
        need not conform.

        Parameters
        ----------
        other : `Geometry`/`SuperCell`
            Other geometry class which needs to be appended
            If a `SuperCell` only the super cell will be extended
        axis  : int
            Cell direction to which the ``other`` geometry should be
            appended.
        """
        if isinstance(other, SuperCell):
            # Only extend the supercell.
            xyz = np.copy(self.xyz)
            atom = np.copy(self.atom)
            sc = self.sc.append(other, axis)
        else:
            xyz = np.append(self.xyz,
                            self.cell[axis, :][None, :] + other.xyz,
                            axis=0)
            atom = np.append(self.atom, other.atom)
            sc = self.sc.append(other.sc, axis)
        return self.__class__(xyz, atom=atom, sc=sc)


    def add(self, other):
        """
        Adds atoms (as is) from the ``other`` geometry.
        This will not alter the residing cell vectors.

        Parameters
        ----------
        other : `Geometry`
            Other geometry class which is added
        """
        xyz = np.append(self.xyz,
                        other.xyz,
                        axis=0)
        atom = np.append(self.atom, other.atom)
        sc = self.sc.copy()
        return self.__class__(xyz, atom=atom, sc=sc)
    

    def reverse(self, atom=None):
        """ Returns a reversed geometry

        Also enables reversing a subset
        """
        if atom is None:
            xyz = self.xyz[::-1, :]
            atms = self.atom[::-1]
        else:
            xyz = np.copy(self.xyz)
            xyz[atom, :] = self.xyz[atom[::-1], :]
            atms = np.copy(self.atom)
            atms[atom] = atms[atom][::-1]
        return self.__class__(xyz, atom=atms, sc=self.sc.copy())

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
        atom : int
           the index at which atom the other geometry is inserted
        geom : `Geometry`
           the other geometry to be inserted
        """
        xyz = np.insert(self.xyz, atom, geom.xyz, axis=0)
        atoms = np.insert(self.atom, atom, geom.atom)
        return self.__class__(xyz, atom=atoms, sc=self.sc.copy())

    def coords(self, isc=None, idx=None):
        """
        Returns the coordinates of a given super-cell index

        Parameters
        ----------
        isc   : array_like, ([0,0,0])
            Returns the atomic coordinates shifted according to the integer
            parts of the cell.
        idx   : int/array_like
            Only return the coordinates of these indices

        Examples
        --------

        >>> geom = Geometry(cell=[[1.,0,0],[0,1.,0.],[0,0,1.]],xyz=[[0,0,0],[0.5,0,0]])
        >>> print(geom.coords(isc=[1,0,0])
        [[ 1.   0.   0. ]
         [ 1.5  0.   0. ]]

        """
        offset = self.sc.offset(isc)
        if idx is None:
            return self.xyz + offset[None, :]
        else:
            return self.xyz[idx, :] + offset[None, :]

    def axyzsc(self, ia):
        return self.coords(self.a2isc(ia), self.sc2uc(ia))


    def close_sc(self, xyz_ia,
                 isc=None,
                 dR=None,
                 idx=None,
                 ret_coord=False,
                 ret_dist=False):
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
        xyz_ia    : coordinate/index
            Either a point in space or an index of an atom.
            If an index is passed it is the equivalent of passing
            the atomic coordinate ``close_sc(self.xyz[xyz_ia,:])``.
        isc       : ([0,0,0]), array_like, optional
            The super-cell which the coordinates are checked in.
        dR        : (None), float/tuple of float
            The radii parameter to where the atomic connections are found.
            If ``dR`` is an array it will return the indices:
            in the ranges:
               ``( x <= dR[0] , dR[0] < x <= dR[1], dR[1] < x <= dR[2] )``
            If a single float it will return:
               ``x <= dR``
        idx       : (None), array_like
            List of atoms that will be considered. This can
            be used to only take out a certain atoms.
        ret_coord : (False), boolean
            If true this method will return the coordinates
            for each of the couplings.
        ret_dist : (False), boolean
            If true this method will return the distance
            for each of the couplings.
        """

        # Common numpy used functions (reduces function look-ups)
        where = np.where
        log_and = np.logical_and

        if dR is None:
            ddR = np.array([self.dR], np.float64)
        else:
            ddR = np.array([dR], np.float64).flatten()
        # Maximum distance queried
        max_dR = ddR[-1]

        # Convert to actual array
        if idx is not None:
            idx = ensure_array(idx)

        if isinstance(xyz_ia, Integral):
            off = self.xyz[xyz_ia, :]
        else:
            off = xyz_ia
        # Get atomic coordinate in principal cell
        dxa = self.coords(isc=isc, idx=idx) - off[None, :]

        # Immediately downscale by easy checking
        # This will reduce the computation of the vector-norm
        # which is the main culprit of the time-consumption
        # This abstraction will _only_ help very large
        # systems.
        # For smaller ones this will actually be a slower
        # method...
        # TODO should we abstract the methods dependent on size?
        ix = log_and(log_and(dxa[:, 0] <= max_dR,
                             dxa[:, 1] <= max_dR),
                     dxa[:, 2] <= max_dR)
        if idx is None:
            # This is because of the pre-check of the
            # distance checks
            idx = where(ix)[0]
        else:
            idx = idx[ix]
        dxa = dxa[ix, :]

        # Create default return
        ret = [[]]*len(ddR)
        i = 0
        if ret_coord:
            i += 1
            rc = i
            ret.append([[]]*len(ddR))
        if ret_dist:
            i += 1
            rc = i
            ret.append([[]]*len(ddR))
        
        if len(dxa) == 0:
            # Quick return if there are
            # no entries...
            if ret_coord or ret_dist:
                return ret
            return ret[0]


        # Retrieve all atomic indices which are closer
        # than our delta-R
        # The linear algebra norm function could be used, but it
        # has a lot of checks, hence we do it manually
        #xaR = np.linalg.norm(dxa,axis=-1)
        xaR = (dxa[:, 0]**2 + dxa[:, 1]**2 + dxa[:, 2]**2) ** .5
        ix = where(xaR <= max_dR)[0]
        if ret_coord:
            xa = dxa[ix, :] + off[None, :]
        if ret_dist:
            d = xaR[ix]
        del dxa  # just because these arrays could be very big...

        # Check whether we only have one range to check.
        # If so, we need not reduce the index space
        if len(ddR) == 1:
            ret = [idx[ix]]
            if ret_coord:
                ret.append(xa)
            if ret_dist:
                ret.append(d)
            if ret_coord or ret_dist:
                return ret
            return ret[0]

        if np.any(np.diff(ddR) < 0.):
            raise ValueError('Proximity checks for several quantities ' +
                             'at a time requires ascending dR values.')

        # Reduce search space!
        # The more neigbours you wish to find the faster this becomes
        # We only do "one" heavy duty search,
        # then we immediately reduce search space to this subspace
        xaR = xaR[ix]
        tidx = where(xaR <= ddR[0])[0]
        ret = [[ensure_array(idx[ix[tidx]])]]
        i = 0
        if ret_coord:
            rc = i + 1
            i += 1
            ret.append([xa[tidx]])
        if ret_dist:
            rd = i + 1
            i += 1
            ret.append([d[tidx]])
        for i in range(1, len(ddR)):
            # Search in the sub-space
            # Notice that this sub-space reduction will never
            # allow the same indice to be in two ranges (due to
            # numerics)
            tidx = where(log_and(ddR[i - 1] < xaR, xaR <= ddR[i]))[0]
            ret[0].append(ensure_array(idx[ix[tidx]]))
            if ret_coord:
                ret[rc].append(xa[tidx])
            if ret_dist:
                ret[rd].append(d[tidx])
        if ret_coord or ret_dist:
            return ret
        return ret[0]

    def bond_correct(self, ia, atom, radii='calc'):
        """ Corrects the bond between `ia` and the `atom`.

        Corrects the bond-length between atom `ia` and `atom` in such
        a way that the atomic radii is preserved.
        I.e. the sum of the bond-lengths minimizes the distance matrix.

        Only atom `ia` is moved.

        Parameters
        ----------
        ia : int
            The atom to be displaced according to the atomic radii
        atom : int, array_like
            The atom(s) from which the radii should be reduced.
        radii : str/float
            If str will use that as lookup in `Atom.radii`.
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
                                   ret_coord=True, ret_dist=True)
            i = np.argmin(d[1])
            # Convert to unitcell atoms
            idx = self.sc2uc(idx[1][i])
            c = c[1][i]
            d = d[1][i]

            # Calculate the bond vector
            bv = self.xyz[ia, :] - c

            try:
                # If it is a number, we use that.
                rad = float(radii)
            except:
                # get radii
                rad = (self.atom[idx].radii(radii=radii) +
                       self.atom[ia].radii(radii=radii))

            # Update the coordinate
            self.xyz[ia, :] = c + bv / d * rad

        else:
            raise NotImplementedError(
                'Changing bond-length dependent on several lacks implementation.')

    def close(self, xyz_ia,
            dR=None,
            idx=None,
            ret_coord=False,
            ret_dist=False):
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
        ret_coord : (False), boolean
            If true this method will return the coordinates
            for each of the couplings.
        ret_dist : (False), boolean
            If true this method will return the distances from the ``xyz_ia``
            for each of the couplings.
        """

        # Get global calls
        # Is faster for many loops
        append = np.append
        vstack = np.vstack
        hstack = np.hstack

        # Convert to actual array
        if isinstance(idx, Integral):
            idx = np.array([idx], np.int32)

        ret = [None]
        i = 0
        if ret_coord:
            c = i + 1
            i += 1
            ret.append(None)
        if ret_dist:
            d = i + 1
            i += 1
            ret.append(None)
        ret_special = ret_coord or ret_dist
        for s in range(self.n_s):
            na = self.na * s
            sret = self.close_sc(
                xyz_ia,
                self.sc.sc_off[s, :],
                dR=dR,
                idx=idx,
                ret_coord=ret_coord,
                ret_dist=ret_dist)
            if not ret_special:
                sret = [sret]
            if isinstance(sret[0], list):
                # we have a list of arrays
                if ret[0] is None:
                    ret[0] = (np.array(sret[0]) + na).tolist()
                    if ret_coord:
                        ret[c] = sret[c]
                    if ret_dist:
                        ret[d] = sret[d]
                else:
                    for i, x in enumerate(sret[0]):
                        ret[0][i] = append(ret[0][i], x + na)
                        if ret_coord:
                            ret[c][i] = vstack((ret[c][i], sret[c][i]))
                        if ret_dist:
                            ret[d][i] = hstack((ret[d][i], sret[d][i]))
            elif len(sret[0]) > 0:
                # We can add it to the list
                # We add the atomic offset for the supercell index
                if ret[0] is None:
                    ret[0] = sret[0] + na
                    if ret_coord:
                        ret[c] = sret[c]
                    if ret_dist:
                        ret[d] = sret[d]
                else:
                    ret[0] = append(ret[0], sret[0] + na)
                    if ret_coord:
                        ret[c] = vstack((ret[c], sret[c]))
                    if ret_dist:
                        ret[d] = hstack((ret[d], sret[d]))
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
        ia : `list` of `int`
             Atomic indices
        all : `bool = False`
             `False`, return only the first orbital corresponding to the atom,
             `True`, returns list of the full atom
        """
        if not all:
            return self.lasto[ia % self.na] + (ia // self.na) * self.no
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

    def o2a(self, io):
        """
        Returns an atomic index corresponding to the orbital indicies.

        This is a particurlaly slow algorithm due to for-loops.

        Note that this will preserve the super-cell offsets.

        Parameters
        ----------
        io: `list` of `int`
             List of indices to return the atoms for
        """
        rlasto = self.lasto[::-1]
        iio = np.asarray([io % self.no]).flatten()
        a = [self.na - np.argmax(rlasto <= i) for i in iio]
        return np.asarray(a) + (io // self.no) * self.na

    def sc2uc(self, atom, uniq=False):
        """ Returns atom from super-cell indices to unit-cell indices, possibly removing dublicates """
        if uniq:
            return np.unique(atom % self.na)
        return atom % self.na
    asc2uc = sc2uc

    def osc2uc(self, orbs, uniq=False):
        """ Returns orbitals from super-cell indices to unit-cell indices, possibly removing dublicates """
        if uniq:
            return np.unique(orbs % self.no)
        return orbs % self.no

    def a2isc(self, a):
        """
        Returns the super-cell index for a specific atom

        Hence one can easily figure out the supercell
        """
        idx = np.where(a < self.na * np.arange(1, self.n_s + 1))[0][0]
        return self.sc.sc_off[idx, :]

    def o2isc(self, o):
        """
        Returns the super-cell index for a specific orbital.

        Hence one can easily figure out the supercell
        """
        idx = np.where(o < self.no * np.arange(1, self.n_s + 1))[0][0]
        return self.sc.sc_off[idx, :]

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
        return np.array([a.mass for a in self.atom], np.float64)


    def __eq__(self, other):
        if not isinstance(other, Geometry):
            return False
        same = self.sc == other.sc
        same = same and np.allclose(self.xyz, other.xyz)
        same = same and np.all(self.atom == other.atom)
        return same

    def __ne__(self, other):
        return not (self == other)


    # Create pickling routines
    def __getstate__(self):
        """ Returns the state of this object """
        d = self.sc.__getstate__()
        d['xyz'] = self.xyz
        d['atom'] = self.atom
        return d

    def __setstate__(self, d):
        """ Re-create the state of this object """
        sc = SuperCell([1, 1, 1])
        sc.__setstate__(d)
        self.__init__(d['xyz'], d['atom'], sc=sc)


    @classmethod
    def _ArgumentParser_args_single(cls):
        """ Returns the options for `Geometry.ArgumentParser` in case they are the only options """
        return {'limit_arguments' : False,
                'short'           : True,
                'positional_out'  : True,
            }

    # Hook into the Geometry class to create
    # an automatic ArgumentParser which makes actions
    # as the options are read.
    @dec_default_AP("Manipulate a Geometry object in sisl.")
    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Create and return a group of argument parsers which manipulates it self `Geometry`. 

        Parameters
        ----------
        parser: ArgumentParser, None
           in case the arguments should be added to a specific parser. It defaults
           to create a new.
        limit_arguments: bool, True
           If `False` additional options will be created which are similar to other options.
           For instance `--repeat-x` which is equivalent to `--repeat x`.
        short: bool, False
           Create short options for a selected range of options
        positional_out: bool, False
           If `True`, adds a positional argument which acts as --out. This may be handy if only the geometry is in the argument list.
        """
        limit_args = kwargs.get('limit_arguments', True)
        short = kwargs.get('short', False)

        def opts(*args):
            if short:
                return args
            return [args[0]]
        
        # We limit the import to occur here
        import argparse as arg

        # The first thing we do is adding the geometry to the NameSpace of the
        # parser.
        # This will enable custom actions to interact with the geometry in a
        # straight forward manner.
        d = {
            "_geometry"        : self.copy(),
            "_stored_geometry" : False,
        }
        namespace = default_namespace(**d)

        # Create actions
        class MoveOrigin(arg.Action):
            def __call__(self, parser, ns, value, option_string=None):
                ns._geometry.xyz[:,:] -= np.amin(ns._geometry.xyz, axis=0)[None,:]
        p.add_argument(*opts('--origin','-O'), action=MoveOrigin, nargs=0,
                   help='Move all atoms such that one atom will be at the origin.')
        
        class MoveCenterOf(arg.Action):
            def __call__(self, parser, ns, value, option_string=None):
                xyz = ns._geometry.center(which='xyz')
                ns._geometry = ns._geometry.translate(ns._geometry.center(which=value) - xyz)
        p.add_argument(*opts('--center-of', '-co'), choices=['mass','xyz','position','cell'], 
                       action=MoveCenterOf,
                       help='Move coordinates to the center of the designated choice.')

        class MoveUnitCell(arg.Action):
            def __call__(self, parser, ns, value, option_string=None):
                if value in ['translate','tr','t']:
                    # Simple translation
                    tmp = np.amin(ns._geometry.xyz, axis=0)
                    # Find the smallest distance from the first atom
                    _, d = ns._geometry.close(0, dR=(0.1,20.), ret_dist=True)
                    d = np.amin(d[1]) / 2
                    ns._geometry = ns._geometry.translate(-tmp + np.array([d,d,d]))
                elif args.unit_cell in ['mod']:
                    # Change all coordinates using the reciprocal cell
                    rcell = ns._geometry.rcell / ( 2. * np.pi )
                    idx = np.abs(np.array(np.dot(ns._geometry.xyz, rcell),np.int32))
                    # change supercell
                    nsc = np.amax(idx * 2 + 1,axis=0)
                    ns._geometry.set_nsc(nsc)
                    # Change the coordinates
                    for ia in ns._geometry:
                        ns._geometry.xyz[ia,:] = ns._geometry.coords(isc=idx[ia,:], idx=ia)
        p.add_argument(*opts('--unit-cell', '-uc'), choices=['translate','tr','t','mod'],
                       action=MoveUnitCell,
                       help='Moves the coordinates into the unit-cell by translation or the mod-operator')

        # Rotation
        class Rotation(arg.Action):
            def __call__(self, parser, ns, values, option_string=None):
                # Convert value[0] to the direction
                d = direction(values[0])
                # The rotate function expects radians
                ang = angle(values[1] + 'r', in_radians=False)
                if d == 0:
                    v = [1,0,0]
                elif d == 1:
                    v = [0,1,0]
                elif d == 2:
                    v = [0,0,1]
                ns._geometry = ns._geometry.rotate(ang, v)
        p.add_argument(*opts('--rotate', '-R'), nargs=2, metavar=('DIR','ANGLE'),
                       action=Rotation,
                       help='Rotate geometry around given axis. ANGLE defaults to be specified in degree. Prefix with "r" for input in radians.')
        
        if not limit_args:
            class RotationX(arg.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    # The rotate function expects radians
                    ang = angle(value + 'r', in_radians=False)
                    ns._geometry = ns._geometry.rotate(ang, [1,0,0])
            p.add_argument(*opts('--rotate-x', '-Rx'), nargs=1, metavar='ANGLE',
                           action=RotationX,
                           help='Rotate geometry around first cell vector. ANGLE defaults to be specified in degree. Prefix with "r" for input in radians.')
            
            class RotationY(arg.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    # The rotate function expects radians
                    ang = angle(value + 'r', in_radians=False)
                    ns._geometry = ns._geometry.rotate(ang, [0,1,0])
            p.add_argument(*opts('--rotate-y', '-Ry'), nargs=1, metavar='ANGLE',
                           action=RotationY,
                           help='Rotate geometry around second cell vector. ANGLE defaults to be specified in degree. Prefix with "r" for input in radians.')
            
            class RotationZ(arg.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    # The rotate function expects radians
                    ang = angle(value + 'r', in_radians=False)
                    ns._geometry = ns._geometry.rotate(ang, [0,0,1])
            p.add_argument(*opts('--rotate-z', '-Rz'), nargs=1, metavar='ANGLE',
                           action=RotationZ,
                           help='Rotate geometry around third cell vector. ANGLE defaults to be specified in degree. Prefix with "r" for input in radians.')
            

        # Reduce size of geometry
        class ReduceSub(arg.Action):
            def __call__(self, parser, ns, value, option_string=None):
                # Get atomic indices
                rng = lstranges(strmap(int, value, sep='-'))
                ns._geometry = ns._geometry.sub(rng)
        p.add_argument(*opts('--sub','-s'),metavar='RNG',
                       action=ReduceSub,
                       help='Removes specified atoms, can be complex ranges.')

        class ReduceCut(arg.Action):
            def __call__(self, parser, ns, values, option_string=None):
                d = direction(values[0])
                s = int(values[1])
                ns._geometry = ns._geometry.cut(s, d)
        p.add_argument(*opts('--cut','-c'), nargs=2, metavar=('DIR', 'SEPS'),
                       action=ReduceCut,
                       help='Cuts the geometry into `seps` parts along the unit-cell direction `dir`.')

        # Add an atom
        class AtomAdd(arg.Action):
            def __call__(self, parser, ns, values, option_string=None):
                # Create an atom from the input
                g = Geometry([float(x) for x in values[0].split(',')], atom=Atom(values[1]))
                ns._geometry = ns._geometry.add(g)
        p.add_argument(*opts('--add'), nargs=2, metavar=('COORD','Z'),
                       action=AtomAdd,
                       help='Adds an atom, coordinate is comma separated (in Ang). Z is the atomic number.')


        # Periodicly increase the structure
        class PeriodRepeat(arg.Action):
            def __call__(self, parser, ns, values, option_string=None):
                d = direction(values[0])
                r = int(values[1])
                ns._geometry = ns._geometry.repeat(r, d)
        p.add_argument(*opts('--repeat','-r'), nargs=2, metavar=('DIR', 'TIMES'),
                       action=PeriodRepeat,
                       help='Repeats the geometry in the specified direction.')
        
        if not limit_args:
            class PeriodRepeatX(arg.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.repeat(int(value), 0)
            p.add_argument(*opts('--repeat-x','-rx'),nargs=1, metavar='TIMES',
                           action=PeriodRepeatX,
                           help='Repeats the geometry along the first cell vector.')
            
            class PeriodRepeatY(arg.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.repeat(int(value), 1)
            p.add_argument(*opts('--repeat-y','-ry'),nargs=1, metavar='TIMES',
                           action=PeriodRepeatY,
                           help='Repeats the geometry along the second cell vector.')

            class PeriodRepeatZ(arg.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.repeat(int(value), 2)
            p.add_argument(*opts('--repeat-z','-rz'),nargs=1, metavar='TIMES',
                           action=PeriodRepeatZ,
                           help='Repeats the geometry along the third cell vector.')


        class PeriodTile(arg.Action):
            def __call__(self, parser, ns, values, option_string=None):
                d = direction(values[0])
                r = int(values[1])
                ns._geometry = ns._geometry.tile(r, d)
        p.add_argument(*opts('--tile','-t'), nargs=2, metavar=('DIR', 'TIMES'),
                       action=PeriodTile,
                       help='Tiles the geometry in the specified direction.')

        if not limit_args:
            class PeriodTileX(arg.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.tile(int(value), 0)
            p.add_argument(*opts('--tile-x','-tx'), nargs=1, metavar='TIMES',
                           action=PeriodTileX,
                           help='Tiles the geometry along the first cell vector.')

            class PeriodTileY(arg.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.tile(int(value), 1)
            p.add_argument(*opts('--tile-y','-ty'), nargs=1, metavar='TIMES',
                           action=PeriodTileY,
                           help='Tiles the geometry along the second cell vector.')

            class PeriodTileZ(arg.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.tile(int(value), 2)
            p.add_argument(*opts('--tile-z','-tz'), nargs=1, metavar='TIMES',
                           action=PeriodTileZ,
                           help='Tiles the geometry along the third cell vector.')

        # We will add the vector data
        class Vectors(arg.Action):
            def __call__(self, parser, ns, values, option_string=None):
                if len(values) == 1:
                    # the vectors should be read from the input stuff...
                    input_file = getattr(ns, '_input_file', None)
                else:
                    input_file = values[1]
                # Quick return if there is no input-file...
                if input_file is None:
                    return
                # Try and read the vector
                from sisl.io import get_sile
                vector = getattr(get_sile(input_file), 'read_{}'.format(values[0]))()
                if vector is None:
                    raise ValueError('{} could not be read from file: {}.'.format(values[0].title(), input_file))

                if len(vector) != len(ns._geometry):
                    raise ValueError('{} could read from file: {}, does not conform to read geometry.'.format(values[0].title(), input_file))
                setattr(ns, '_vector', vector)
        p.add_argument(*opts('--vector','-v'),metavar='DATA',nargs='+',
                       action=Vectors,
                       help='''Adds vector arrows for each atom, first argument is type (force, moment, ...).
                       If the current input file contains the vectors no second argument is necessary, else the file containing the data is required as a second input.
                       ''')

        # Print some common information about the
        # geometry (to stdout)
        class PrintInfo(arg.Action):
            def __call__(self, parser, ns, values, option_string=None):
                print(ns._geometry)
        p.add_argument(*opts('--info'), nargs=0,
                       action=PrintInfo,
                       help='Print, to stdout, some regular information about the geometry.')

            
        class Out(arg.Action):
            def __call__(self, parser, ns, value, option_string=None):
                if value is None:
                    return
                if len(value) == 0:
                    return
                # If the vector, exists, we should write it
                if hasattr(ns, '_vector'):
                    ns._geometry.write(value[0], data=getattr(ns, '_vector', None))
                else:
                    ns._geometry.write(value[0])
                # Issue to the namespace that the geometry has been written, at least once.
                ns._stored_geometry = True
        p.add_argument(*opts('--out','-o'), nargs=1, action=Out,
                       help='Store the geometry (at its current invocation) to the out file.')

        # If the user requests positional out arguments, we also add that.
        if kwargs.get('positional_out', False):
            p.add_argument('out', nargs='*',default=None,
                           action=Out,
                           help='Store the geometry (at its current invocation) to the out file.')
            
        # We have now created all arguments
        return p, namespace
