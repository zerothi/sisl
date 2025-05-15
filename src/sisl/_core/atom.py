# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from numbers import Integral, Real
from typing import Literal

import numpy as np

import sisl._array as _a
from sisl._dispatch_class import _Dispatchs
from sisl._dispatcher import AbstractDispatch, ClassDispatcher
from sisl._help import array_fill_repeat
from sisl._indices import list_index_le
from sisl._internal import set_module
from sisl.messages import deprecate_argument, deprecation, info
from sisl.shape import Sphere

from .orbital import Orbital
from .periodictable import PeriodicTable

__all__ = ["Atom", "AtomUnknown", "AtomGhost", "Atoms"]


# Create a local instance of the periodic table to
# faster look up
_ptbl = PeriodicTable()


class AtomMeta(type):
    """Meta class for key-lookup on the class."""

    def __getitem__(cls, key):
        """Create a new atom object"""
        if isinstance(key, Atom):
            # if the key already is an atomic object
            # return it
            return key
        elif isinstance(key, dict):
            # The key is a dictionary, hence
            # we can return the atom directly
            return cls(**key)
        elif isinstance(key, (list, tuple)):
            # The key is a list,
            # we need to create a list of atoms
            return [cls[k] for k in key]  # pylint: disable=E1136
        # Index Z based
        return cls(key)


# Note the with_metaclass which is required for python3 support.
# The designation of metaclass in python3 is actually:
#   class ...(..., metaclass=MetaClass)
# This below construct handles both python2 and python3 cases
@set_module("sisl")
class Atom(
    _Dispatchs,
    dispatchs=[ClassDispatcher("to", type_dispatcher=None)],
    when_subclassing="keep",
    metaclass=AtomMeta,
):
    """Atomic information for a single atomic species

    An atomic object retaining information about a single atomic species.
    It describes the atomic number (integer), the mass of the atom, and
    holds a list of atomic centered orbitals. It also allows one
    to tag the atom to distinguish it from other atoms of the same species.

    Parameters
    ----------
    Z :
        determine species for the atomic species.
    orbitals : list of Orbital or float, optional
        orbitals associated with this atom. See `Orbital` for details on
        how to define orbitals.
        Defaults to one orbital.
    mass :
        the atomic mass, defaults to the mass found in `PeriodicTable`.
    tag :
        arbitrary designation for user handling similar atoms with
        different settings (defaults to the label of the atom)


    Examples
    --------
    >>> Carbon = Atom(6)
    >>> Carbon = Atom("C")
    >>> Carbon = Atom("Carbon")

    Add a tag to be able to distinguish it from other atoms
    >>> tagged_Carbon = Atom("Carbon", tag="siteA")

    Create deuterium
    >>> D = Atom("H", mass=2.014)

    Define an atom with 3 orbitals, each with a range of 2 Angstroem
    >>> C3 = Atom("C", orbitals=[2, 2, 2])

    Define an atom outside of the periodic table (negative will yield an
    AtomGhost object)
    >>> ghost_C = Atom(-6)

    Define an unknown atom (basically anything can do)
    >>> unknown_atom = Atom(1000)

    Notes
    -----
    One can define atoms outside of the periodic table. They will generally
    be handled in this order:

    * negative numbers will be converted into positive ones, and the returned
      object will be an `AtomGhost`
    * any other number (or name) not found in the periodic table will be returned
      in an AtomUnknown object

    The mass for atoms outside the periodic table will default to 1e40 amu.

    See Also
    --------
    Orbital : define an orbital
    Atoms : an efficient collection of Atom objects
    """

    def __new__(cls, *args, **kwargs):
        """Figure out which class to actually use"""
        # Handle the case where no arguments are passed (e.g. for serializing stuff)
        if len(args) == 0 and "Z" not in kwargs:
            return super().__new__(cls)

        # direct call
        if len(args) > 0:
            Z = args[0]
            if "Z" in kwargs:
                raise ValueError(
                    f"{cls.__name__} got both Z as argument and keyword argument. Please only use one."
                )
        else:
            Z = None
        Z = kwargs.get("Z", Z)
        if isinstance(Z, Atom):
            return super().__new__(Z.__class__)

        try:
            # Try and convert to an integer
            Z = int(Z)
        except Exception:
            pass

        if isinstance(Z, Integral) and not issubclass(cls, AtomGhost) and Z < 0:
            cls = AtomGhost
        elif Z not in _ptbl._Z_int and not issubclass(cls, AtomUnknown):
            cls = AtomUnknown

        return super().__new__(cls)

    def __init__(
        self,
        Z: Union[str, int],
        orbitals=None,
        mass: Optional[float] = None,
        tag: Optional[str] = None,
        **kwargs,
    ):
        # try and cast to integer, it might be cast differently later
        # but this is to try and see if we can easily get it
        mass_Z = Z
        try:
            Z = int(Z)
        except Exception:
            pass

        if isinstance(Z, Atom):
            self._Z = Z.Z
            mass_Z = self.Z
        elif isinstance(Z, Integral):
            self._Z = Z
            mass_Z = self.Z
        else:
            self._Z = _ptbl.Z_int(Z)
        if not isinstance(self._Z, Integral):
            raise ValueError(
                f"{self.__class__.__name__} got an unparseable Z argument, needs to be an integer, got='{Z}'."
            )

        self._orbitals = None
        if isinstance(orbitals, (tuple, list, np.ndarray)):
            if len(orbitals) == 0:
                # This may be the same as only regarding `R` argument
                pass
            elif isinstance(orbitals[0], Orbital):
                # all is good
                self._orbitals = orbitals
            elif isinstance(orbitals[0], Real):
                # radius has been given
                self._orbitals = [Orbital(R) for R in orbitals]
            elif isinstance(orbitals[0], str):
                # radius has been given
                self._orbitals = [Orbital(-1, tag=tag) for tag in orbitals]
            elif isinstance(orbitals[0], tuple):
                # likely a radiuse + tag
                self._orbitals = [Orbital(R, tag=tag) for R, tag in orbitals]
            elif all(orb is None for orb in orbitals):
                orbitals = None

        elif isinstance(orbitals, Orbital):
            self._orbitals = [orbitals]

        elif isinstance(orbitals, Real):
            self._orbitals = [Orbital(orbitals)]

        if self._orbitals is None:
            if orbitals is not None:
                raise ValueError(
                    f"{self.__class__.__name__}.__init__ got unparseable 'orbitals' argument: {orbitals}"
                )
            if "R" in kwargs:
                # backwards compatibility (possibly remove this in the future)
                R = _a.asarrayd(kwargs["R"]).ravel()
                self._orbitals = [Orbital(r) for r in R]
            else:
                self._orbitals = [Orbital(-1.0)]

        if mass is None:
            self._mass = _ptbl.atomic_mass(mass_Z)
        else:
            self._mass = mass

        # self.tag will return self.symbol if not set
        self._tag = tag

    def __hash__(self):
        return hash((self.tag, self._mass, self._Z, *self._orbitals))

    @property
    def Z(self) -> int:
        """Atomic number"""
        return self._Z

    @property
    def orbitals(self):
        """List of orbitals"""
        return self._orbitals

    @property
    def mass(self) -> float:
        """Atomic mass"""
        return self._mass

    @property
    def tag(self) -> str:
        """Tag for atom"""
        if self._tag is None:
            return self.symbol
        return self._tag

    @property
    def no(self) -> int:
        """Number of orbitals on this atom"""
        return len(self.orbitals)

    def __len__(self):
        """Return number of orbitals in this atom"""
        return self.no

    @property
    def row(self) -> int:
        """The row of the atom in the periodic table.

        May return `NotImplemented` if the element isn't found
        in the periodic table.

        Only covers up to Z=118.

        See Also
        --------
        column : for getting the periodic table column
        PeriodicTable.Z_row : used to extract the periodic table row of an atomic number
        PeriodicTable.Z_column : used to extract the periodic table column of an atomic number
        """
        return PeriodicTable.Z_row(self.Z)

    @property
    def column(self) -> int:
        """The column of the atom in the periodic table.

        May return `NotImplemented` if the element isn't found
        in the periodic table.

        Only covers up to Z=118.

        See Also
        --------
        row : for getting the periodic table row
        PeriodicTable.Z_row : used to extract the periodic table row of an atomic number
        PeriodicTable.Z_column : used to extract the periodic table column of an atomic number
        """
        return PeriodicTable.Z_column(self.Z)

    def index(self, orbital):
        """Return the index of the orbital in the atom object"""
        if not isinstance(orbital, Orbital):
            orbital = self[orbital]
        for i, o in enumerate(self.orbitals):
            if o == orbital:
                return i
        raise KeyError("Could not find `orbital` in the list of orbitals.")

    def radius(self, method: Literal["calc", "empirical", "vdw"] = "calc"):
        """Return the atomic radius of the atom (in Ang)

        See `PeriodicTable.radius` for details on the argument.
        """
        return _ptbl.radius(self.Z, method)

    @property
    def symbol(self):
        """Return short atomic name (Au==79)."""
        return _ptbl.Z_short(self.Z)

    def __getitem__(self, key):
        """The orbital corresponding to index `key`"""
        if isinstance(key, slice):
            ol = key.indices(len(self))
            return [self.orbitals[o] for o in range(*ol)]
        elif isinstance(key, Integral):
            return self.orbitals[key]
        elif isinstance(key, str):
            orbs = [orb for orb in self.orbitals if key in orb.name()]
            # In case none are found, None will be returned
            if not orbs:
                return None
            return orbs if len(orbs) != 1 else orbs[0]
        return [self.orbitals[o] for o in key]

    def maxR(self):
        """Return the maximum range of orbitals."""
        mR = -1e10
        for o in self.orbitals:
            mR = max(mR, o.R)
        return mR

    def __iter__(self):
        """Loop on all orbitals in this atom"""
        yield from self.orbitals

    def iter(self, group: bool = False):
        """Loop on all orbitals in this atom

        Parameters
        ----------
        group : bool, optional
           if two orbitals share the same radius
           one may be able to group two orbitals together

        Returns
        -------
        Orbital
            current orbital, if `group` is ``True`` this is a list of orbitals,
            otherwise a single orbital is returned
        """
        if group:
            i = 0
            no = self.no - 1
            while i <= no:
                # Figure out how many share the same radial part
                j = i + 1
                while j <= no:
                    if np.allclose(self.orbitals[i].R, self.orbitals[j].R):
                        j += 1
                    else:
                        break
                yield self.orbitals[i:j]
                i = j
            return
        yield from self.orbitals

    def __str__(self):
        # Create orbitals output
        orbs = ",\n ".join([str(o) for o in self.orbitals])
        return (
            self.__class__.__name__
            + "{{{0}, Z: {1:d}, mass(au): {2:.5f}, maxR: {3:.5f},\n {4}\n}}".format(
                self.tag, self.Z, self.mass, self.maxR(), orbs
            )
        )

    def __repr__(self):
        return f"<{self.__module__}.{self.__class__.__name__} {self.tag}, Z={self.Z}, M={self.mass}, maxR={self.maxR()}, no={len(self.orbitals)}>"

    def __getattr__(self, attr):
        """Pass attribute calls to the orbital classes and return lists/array

        Parameters
        ----------
        attr : str
        """

        # First we create a list of values that the orbitals have
        # Some may have it, others may not
        vals = [None] * len(self.orbitals)
        found = False
        is_Integral = is_Real = is_callable = True
        for io, orb in enumerate(self.orbitals):
            try:
                vals[io] = getattr(orb, attr)
                found = True
                is_callable &= callable(vals[io])
                is_Integral &= isinstance(vals[io], Integral)
                is_Real &= isinstance(vals[io], Real)
            except AttributeError:
                pass

        if found == 0:
            # we never got any values, reraise the AttributeError
            raise AttributeError(
                f"'{self.__class__.__name__}.orbitals' objects has no attribute '{attr}'"
            )

        # Now parse the data, currently we'll only allow Integral, Real, Complex
        if is_Integral:
            for io in range(len(vals)):
                if vals[io] is None:
                    vals[io] = 0
            return _a.arrayi(vals)
        elif is_Real:
            for io in range(len(vals)):
                if vals[io] is None:
                    vals[io] = 0.0
            return _a.arrayd(vals)
        elif is_callable:

            def _ret_none(*args, **kwargs):
                return None

            for io in range(len(vals)):
                if vals[io] is None:
                    vals[io] = _ret_none

            # Now subclass the content and return values per method
            class ArrayCall:
                def __init__(self, methods):
                    self.methods = methods

                def __call__(self, *args, **kwargs):
                    return [m(*args, **kwargs) for m in self.methods]

            return ArrayCall(vals)

        # We don't know how to handle this, simply return...
        return vals

    @deprecation(
        "toSphere is deprecated, use shape.to.Sphere(...) instead.", "0.15", "0.17"
    )
    def toSphere(self, center=None):
        """Return a sphere with the maximum orbital radius equal

        Returns
        -------
        ~sisl.shape.Sphere
             a sphere with radius equal to the maximum radius of the orbitals
        """
        return self.to.Sphere(center=center)

    def equal(self, other, R: bool = True, psi: bool = False):
        """True if `other` is the same as this atomic species

        Parameters
        ----------
        other : Atom
           the other object to check againts
        R : bool, optional
           if True the equality check also checks the orbital radius, else they are not compared
        psi : bool, optional
           if True, also check the wave-function component of the orbitals, see `Orbital.psi`
        """
        if not isinstance(other, Atom):
            return False
        same = self.Z == other.Z
        same &= self.no == other.no
        if same and R:
            same &= all(
                [
                    self.orbitals[i].equal(other.orbitals[i], psi=psi)
                    for i in range(self.no)
                ]
            )
        same &= np.isclose(self.mass, other.mass)
        same &= self.tag == other.tag
        return same

    # Check whether they are equal
    def __eq__(self, b):
        """Return true if the saved quantities are the same"""
        return self.equal(b)

    def __ne__(self, b):
        return not (self == b)

    # Create pickling routines
    def __getstate__(self):
        """Return the state of this object"""
        return {
            "Z": self.Z,
            "orbitals": self.orbitals,
            "mass": self.mass,
            "tag": self.tag,
        }

    def __setstate__(self, d):
        """Re-create the state of this object"""
        self.__init__(d["Z"], d["orbitals"], d["mass"], d["tag"])


@set_module("sisl")
class AtomUnknown(Atom):
    def __init__(self, Z, *args, **kwargs):
        """Instantiate with overridden tag"""
        if len(args) < 3 and "tag" not in kwargs:
            kwargs["tag"] = "unknown"
        if len(args) < 2 and "mass" not in kwargs:
            kwargs["mass"] = 1e40
        super().__init__(Z, *args, **kwargs)


@set_module("sisl")
class AtomGhost(AtomUnknown):
    def __init__(self, Z, *args, **kwargs):
        """Instantiate with overridden tag and taking the absolute value of Z"""
        try:
            # here we also need to do the conversion as we want
            # to remove the negative sign
            Z = abs(int(Z))
        except Execption:
            pass

        if len(args) < 3 and "tag" not in kwargs:
            kwargs["tag"] = "ghost"
        super().__init__(Z, *args, **kwargs)


class AtomToDispatch(AbstractDispatch):
    """Base dispatcher from class passing from an Atom class"""


to_dispatch = Atom.to


class ToSphereDispatch(AtomToDispatch):
    def dispatch(self, *args, center=None, **kwargs):
        return Sphere(self._get_object().maxR(), center)


to_dispatch.register("Sphere", ToSphereDispatch)


@set_module("sisl")
class Atoms:
    """Efficient collection of `Atom` objects

    A container object for `Atom` objects in a specific order.
    No two `Atom` objects will be duplicated and indices will be used
    to determine which `Atom` any indexable atom corresponds to.
    This is convenient when having geometries with millions of atoms
    because it will not duplicate the `Atom` object, only a list index.

    Parameters
    ----------
    atoms :
       atoms to be contained in this list of atoms
       If a str, or a single `Atom` it will be the only atom in the resulting
       class repeated `na` times.
       If a list, it will create all unique atoms and retain these, each item in
       the list may a single argument passed to the `Atom` or a dictionary
       that is passed to `Atom`, see examples.
    na :
       total number of atoms, if ``len(atoms)`` is smaller than `na` it will
       be repeated to match `na`.

    Examples
    --------

    Creating an atoms object consisting of 5 atoms, all the same.

    >>> atoms = Atoms("H", na=5)

    Creating a set of 4 atoms, 2 Hydrogen, 2 Helium, in an alternate
    ordere

    >>> Z = [1, 2, 1, 2]
    >>> atoms = Atoms(Z)
    >>> atoms = Atoms([1, 2], na=4) # equivalent

    Creating individual atoms using dictionary entries, two
    Hydrogen atoms, one with a tag H_ghost.

    >>> Atoms([dict(Z=1, tag="H_ghost"), 1])
    """

    # Using the slots should make this class slightly faster.
    __slots__ = ("_atom", "_species", "_firsto")

    def __init__(self, atoms: AtomsLike = "H", na: Optional[int] = None):
        # Default value of the atom object
        if atoms is None:
            atoms = Atom("H")

        # Correct the atoms input to Atom
        if isinstance(atoms, Atom):
            uatoms = [atoms]
            species = [0]

        elif isinstance(atoms, Atoms):
            # Ensure we make a copy to not operate
            # on the same data.
            catoms = atoms.copy()
            uatoms = catoms.atom[:]
            species = catoms.species[:]

        elif isinstance(atoms, (str, Integral)):
            uatoms = [Atom(atoms)]
            species = [0]

        elif isinstance(atoms, dict):
            uatoms = [Atom(**atoms)]
            species = [0]

        elif isinstance(atoms, Iterable):
            # TODO this is very inefficient for large MD files
            uatoms = []
            species = []
            for a in atoms:
                if isinstance(a, dict):
                    a = Atom(**a)
                elif not isinstance(a, Atom):
                    a = Atom(a)
                try:
                    s = uatoms.index(a)
                except Exception:
                    s = len(uatoms)
                    uatoms.append(a)
                species.append(s)

        else:
            raise ValueError(f"atoms keyword type is not acceptable {type(atoms)}")

        # Default for number of atoms
        if na is None:
            na = len(species)

        # Create atom and species objects
        self._atom = list(uatoms)

        self._species = array_fill_repeat(species, na, cls=np.int16)

        self._update_orbitals()

    def _update_orbitals(self):
        """Internal routine for updating the `firsto` attribute"""
        # Get number of orbitals per specie
        uorbs = _a.arrayi([a.no for a in self.atom])
        self._firsto = np.insert(_a.cumsumi(uorbs[self.species]), 0, 0)

    @property
    def atom(self):
        """List of unique atoms in this group of atoms"""
        return self._atom

    @property
    @deprecation("nspecie is deprecated, use nspecies instead.", "0.15", "0.17")
    def nspecie(self):
        """Number of different species"""
        return len(self._atom)

    @property
    def nspecies(self):
        """Number of different species"""
        return len(self._atom)

    @property
    def species(self):
        """List of atomic species"""
        return self._species

    @property
    @deprecation("specie is deprecated, use species instead.", "0.15", "0.17")
    def specie(self):
        """List of atomic species"""
        return self._species

    @property
    def no(self) -> int:
        """Total number of orbitals in this list of atoms"""
        uorbs = _a.arrayi([a.no for a in self.atom])
        return int(uorbs[self.species].sum())

    @property
    def orbitals(self):
        """Array of orbitals of the contained objects"""
        return np.diff(self.firsto)

    @property
    def firsto(self):
        """First orbital of the corresponding atom in the consecutive list of orbitals"""
        return self._firsto

    @property
    def lasto(self):
        """Last orbital of the corresponding atom in the consecutive list of orbitals"""
        return self._firsto[1:] - 1

    @property
    def q0(self):
        """Initial charge per atom"""
        q0 = _a.arrayd([a.q0.sum() for a in self.atom])
        return q0[self.species]

    def orbital(self, io):
        """Return an array of orbital of the contained objects"""
        io = _a.asarrayi(io)
        ndim = io.ndim
        io = io.ravel() % self.no
        a = list_index_le(io, self.lasto)
        io = io - self.firsto[a]
        a = self.species[a]
        # Now extract the list of orbitals
        if ndim == 0:
            return self.atom[a[0]].orbitals[io[0]]
        return [self.atom[ia].orbitals[o] for ia, o in zip(a, io)]

    def maxR(self, all: bool = False):
        """The maximum radius of the atoms

        Parameters
        ----------
        all : bool
            determine the returned maximum radii.
            If `True` is passed an array of all atoms maximum radii is returned (array).
            Else, if `False` the maximum of all atoms maximum radii is returned (scalar).
        """
        if all:
            maxR = _a.arrayd([a.maxR() for a in self.atom])
            return maxR[self.species]
        return np.amax([a.maxR() for a in self.atom])

    @property
    def mass(self):
        """Array of masses of the contained objects"""
        umass = _a.arrayd([a.mass for a in self.atom])
        return umass[self.species]

    @property
    def Z(self):
        """Array of atomic numbers"""
        uZ = _a.arrayi([a.Z for a in self.atom])
        return uZ[self.species]

    def index(self, atom):
        """Return the indices of the atom object"""
        return (self._species == self.species_index(atom)).nonzero()[0]

    def species_index(self, atom):
        """Return the species index of the atom object"""
        if not isinstance(atom, Atom):
            atom = self[atom]
        for s, a in enumerate(self.atom):
            if a == atom:
                return s
        raise KeyError("Could not find `atom` in the list of atoms.")

    specie_index = deprecation(
        "specie_index is deprecated, use species_index instead.", "0.15", "0.17"
    )(species_index)

    def group_atom_data(self, data, axis=0):
        r"""Group data for each atom based on number of orbitals

        This is useful for grouping data that is orbitally resolved.
        This will return a list of length ``len(self)`` and with each item
        having the sub-slice of the data corresponding to the orbitals on the given
        atom.

        Examples
        --------
        >>> atoms = Atoms([Atom(4, [0.1, 0.2]), Atom(6, [0.2, 0.3, 0.5])])
        >>> orb_data = np.arange(10).reshape(2, 5)
        >>> atoms.group_data(orb_data, axis=1)
        [
         [[0, 1], [2, 3]],
         [[4, 5, 6], [7, 8, 9]]
        ]

        Parameters
        ----------
        data : numpy.ndarray
           data to be grouped
        axis : int, optional
           along which axis one should split the data
        """
        return np.split(data, self.lasto[:-1] + 1, axis=axis)

    @deprecate_argument(
        "in_place",
        "inplace",
        "argument in_place has been deprecated in favor of inplace, please update your code.",
        "0.15",
        "0.17",
    )
    def reorder(self, inplace: bool = False):
        """Reorders the atoms and species index so that they are ascending (starting with a species that exists)

        Parameters
        ----------
        inplace :
            whether the re-order is done *in-place*
        """

        # Contains the minimum atomic index for a given specie
        smin = _a.emptyi(len(self.atom))
        smin.fill(len(self))
        for a in range(len(self.atom)):
            lst = (self.species == a).nonzero()[0]
            if len(lst) > 0:
                smin[a] = lst.min()

        if inplace:
            atoms = self
        else:
            atoms = self.copy()

        # Now swap indices into correct place
        # This will give the indices of the species
        # in the ascending order
        isort = np.argsort(smin)
        if np.allclose(np.diff(isort), 0):
            return atoms

        atoms._atom[:] = [atoms._atom[i] for i in isort]
        atoms._species[:] = isort[atoms._species]

        atoms._update_orbitals()
        return atoms

    def formula(self, system="Hill"):
        """Return the chemical formula for the species in this object

        Parameters
        ----------
        system : {"Hill"}, optional
           which notation system to use
           Is not case-sensitive
        """
        # loop different species
        c = Counter()
        for atom, indices in self.iter(species=True):
            if len(indices) > 0:
                c[atom.symbol] += len(indices)

        # now we have all elements, and the counts of them
        systeml = system.lower()
        if systeml == "hill":
            # sort lexographically
            symbols = sorted(c.keys())

            def parse(symbol_c):
                symbol, c = symbol_c
                if c == 1:
                    return symbol
                return f"{symbol}{c}"

            return "".join(map(parse, sorted(c.items())))

        raise ValueError(
            f"{self.__class__.__name__}.formula got unrecognized argument 'system' {system}"
        )

    @deprecate_argument(
        "in_place",
        "inplace",
        "argument in_place has been deprecated in favor of inplace, please update your code.",
        "0.15",
        "0.17",
    )
    def reduce(self, inplace: bool = False):
        """Returns a new `Atoms` object by removing non-used atoms"""
        if inplace:
            atoms = self
        else:
            atoms = self.copy()
        atom = atoms._atom
        species = atoms._species

        rem = []
        for i in range(len(self.atom)):
            if np.all(species != i):
                rem.append(i)

        # Remove the atoms
        for i in rem[::-1]:
            atom.pop(i)
            species = np.where(species > i, species - 1, species)

        atoms._atom = atom
        atoms._species = species
        atoms._update_orbitals()

        return atoms

    def swap_atom(self, a, b):
        """Swap species index positions"""
        speciesa = self.species_index(a)
        speciesb = self.species_index(b)

        idx_a = (self._species == speciesa).nonzero()[0]
        idx_b = (self._species == speciesb).nonzero()[0]

        atoms = self.copy()
        atoms._atom[speciesa], atoms._atom[speciesb] = (
            atoms._atom[speciesb],
            atoms._atom[speciesa],
        )
        atoms._species[idx_a] = speciesb
        atoms._species[idx_b] = speciesa
        atoms._update_orbitals()
        return atoms

    def reverse(self, atoms=None):
        """Returns a reversed geometry

        Also enables reversing a subset of the atoms.
        """
        copy = self.copy()
        if atoms is None:
            copy._species = self._species[::-1]
        else:
            copy._species[atoms] = self._species[atoms[::-1]]
        copy._update_orbitals()
        return copy

    def __str__(self):
        """Return the `Atoms` in str"""
        s = f"{self.__class__.__name__}{{species: {len(self._atom)},\n"
        for a, idx in self.iter(True):
            s += " {1}: {0},\n".format(len(idx), str(a).replace("\n", "\n "))
        return f"{s}}}"

    def __repr__(self):
        return f"<{self.__module__}.{self.__class__.__name__} nspecies={len(self._atom)}, na={len(self)}, no={self.no}>"

    def __len__(self):
        """Return number of atoms in the object"""
        return len(self._species)

    def iter(self, species=False):
        """Loop on all atoms

        This iterator may be used in two contexts:

        1. `species` is ``False``, this is the slowest method and will yield the
           `Atom` per contained atom.
        2. `species` is ``True``, which yields a tuple of `(Atom, list)` where
           ``list`` contains all indices of atoms that has the `Atom` species.
           This is much faster than the first option.

        Parameters
        ----------
        species : bool, optional
           If ``True`` loops only on different species and yields a tuple of (Atom, list)
           Else yields the atom for the equivalent index.
        """
        if species:
            for s, atom in enumerate(self._atom):
                yield atom, (self.species == s).nonzero()[0]
        else:
            for s in self.species:
                yield self._atom[s]

    def __iter__(self):
        """Loop on all atoms with the same species in order of atoms"""
        yield from self.iter()

    def __contains__(self, key):
        """Determine whether the `key` is in the unique atoms list"""
        return key in self.atom

    def __getitem__(self, key):
        """Return an `Atom` object corresponding to the key(s)"""
        if isinstance(key, slice):
            sl = key.indices(len(self))
            return [self.atom[self._species[s]] for s in range(sl[0], sl[1], sl[2])]
        elif isinstance(key, Integral):
            return self.atom[self._species[key]]
        elif isinstance(key, str):
            for at in self.atom:
                if at.tag == key:
                    return at
            return None
        key = np.asarray(key)
        if key.ndim == 0:
            return self.atom[self._species[key]]
        return [self.atom[i] for i in self._species[key]]

    def __setitem__(self, key, value):
        """Overwrite an `Atom` object corresponding to the key(s)"""
        # If key is a string, we replace the atom that matches 'key'
        if isinstance(key, str):
            self.replace_atom(self[key], value)
            return

        # Convert to array
        if isinstance(key, slice):
            sl = key.indices(len(self))
            key = _a.arangei(sl[0], sl[1], sl[2])
        else:
            key = _a.asarrayi(key).ravel()

        if len(key) == 0:
            if value not in self:
                self._atom.append(value)
            return

        # Create new atoms object to iterate
        other = Atoms(value, na=len(key))

        # Append the new Atom objects
        for atom, s_i in other.iter(True):
            if atom not in self:
                self._atom.append(atom)
            self._species[key[s_i]] = self.species_index(atom)
        self._update_orbitals()

    def replace(self, index, atom):
        """Replace all atomic indices `index` with the atom `atom` (in-place)

        This is the preferred way of replacing atoms in geometries.

        Parameters
        ----------
        index : list of int or Atom
           the indices of the atoms that should be replaced by the new atom.
           If an `Atom` is passed, this routine defers its call to `replace_atom`.
        atom : Atom
           the replacement atom.
        """
        if isinstance(index, Atom):
            self.replace_atom(index, atom)
            return
        if not isinstance(atom, Atom):
            raise TypeError(
                f"{self.__class__.__name__}.replace requires input arguments to "
                "be of the type Atom"
            )
        index = _a.asarrayi(index).ravel()

        # Be sure to add the atom
        if atom not in self.atom:
            self._atom.append(atom)

        # Get species index of the atom
        species = self.species_index(atom)

        # Loop unique species and check that we have the correct number of orbitals
        for ius in np.unique(self._species[index]):
            a = self._atom[ius]
            if a.no != atom.no:
                a1 = "  " + str(a).replace("\n", "\n  ")
                a2 = "  " + str(atom).replace("\n", "\n  ")
                info(
                    f"Substituting atom\n{a1}\n->\n{a2}\nwith a different number of orbitals!"
                )
        self._species[index] = species
        # Update orbital counts...
        self._update_orbitals()

    def replace_atom(self, atom_from: Atom, atom_to: Atom):
        """Replace all atoms equivalent to `atom_from` with `atom_to` (in-place)

        I.e. this is the preferred way of adapting all atoms of a specific type
        with another one.

        If the two atoms does not have the same number of orbitals a warning will
        be raised.

        Parameters
        ----------
        atom_from : Atom
           the atom that should be replaced, if not found in the current list
           of atoms, nothing will happen.
        atom_to : Atom
           the replacement atom.

        Raises
        ------
        KeyError
           if `atom_from` does not exist in the list of atoms
        UserWarning
           if the atoms does not have the same number of orbitals.
        """
        if not isinstance(atom_from, Atom):
            raise TypeError(
                f"{self.__class__.__name__}.replace_atom requires input arguments to "
                "be of the class Atom"
            )
        if not isinstance(atom_to, Atom):
            raise TypeError(
                f"{self.__class__.__name__}.replace_atom requires input arguments to "
                "be of the class Atom"
            )

        # Get index of `atom_from`
        idx_from = self.species_index(atom_from)
        try:
            idx_to = self.species_index(atom_to)
            if idx_from == idx_to:
                raise KeyError("")

            # Decrement indices of the atoms that are
            # changed to one already there
            self._species[self.species == idx_from] = idx_to
            self._species[self.species > idx_from] -= 1
            # Now delete the old index, we replace, so we *have* to remove it
            self._atom.pop(idx_from)
        except KeyError:
            # The atom_to is not in the list
            # Simply change
            self._atom[idx_from] = atom_to

        if atom_from.no != atom_to.no:
            a1 = "  " + str(atom_from).replace("\n", "\n  ")
            a2 = "  " + str(atom_to).replace("\n", "\n  ")
            info(
                f"Replacing atom\n{a1}\n->\n{a2}\nwith a different number of orbitals!"
            )

            # Update orbital counts...
            self._update_orbitals()

    def hassame(self, other, R=True):
        """True if the contained atoms are the same in the two lists

        Notes
        -----
        This does not necessarily mean that the order, nor the number of atoms
        are the same.

        Parameters
        ----------
        other : Atoms
           the list of atoms to check against
        R : bool, optional
           if True also checks that the orbital radius are the same

        See Also
        --------
        equal : explicit check of the indices *and* the contained atoms
        """
        if len(self.atom) != len(other.atom):
            return False
        for A in self.atom:
            is_in = False
            for B in other.atom:
                if A.equal(B, R):
                    is_in = True
                    break
            if not is_in:
                return False
        return True

    def equal(self, other, R=True):
        """True if the contained atoms are the same in the two lists (also checks indices)

        Parameters
        ----------
        other : Atoms
           the list of atoms to check against
        R : bool, optional
           if True also checks that the orbital radius are the same

        See Also
        --------
        hassame : only check whether the two atoms are contained in both
        """
        if len(self.atom) > len(other.atom):
            for iA, A in enumerate(self.atom):
                is_in = -1
                for iB, B in enumerate(other.atom):
                    if A.equal(B, R):
                        is_in = iB
                        break
                if is_in == -1:
                    return False
                # We should check that they also have the same indices
                if not np.all(
                    np.nonzero(self.species == iA)[0]
                    == np.nonzero(other.species == is_in)[0]
                ):
                    return False
        else:
            for iB, B in enumerate(other.atom):
                is_in = -1
                for iA, A in enumerate(self.atom):
                    if B.equal(A, R):
                        is_in = iA
                        break
                if is_in == -1:
                    return False
                # We should check that they also have the same indices
                if not np.all(
                    np.nonzero(other.species == iB)[0]
                    == np.nonzero(self.species == is_in)[0]
                ):
                    return False
        return True

    def __eq__(self, b):
        """Returns true if the contained atoms are the same"""
        return self.equal(b)

    # Create pickling routines
    def __getstate__(self):
        """Return the state of this object"""
        return {"atom": self.atom, "species": self.species}

    def __setstate__(self, d):
        """Re-create the state of this object"""
        self.__init__()
        self._atom = d["atom"]
        self._species = d["species"]
