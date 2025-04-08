# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import operator
from functools import wraps
from numbers import Integral

import numpy as np

import sisl._array as _a
from sisl._category import CategoryMeta
from sisl._core import Geometry, Lattice, LatticeChild
from sisl._core._lattice import cell_invert
from sisl._internal import set_module
from sisl.messages import deprecate_argument
from sisl.shape import Shape
from sisl.typing import AtomsIndex
from sisl.utils.misc import direction

from .base import AtomCategory, NullCategory

__all__ = ["AtomFracSite", "AtomXYZ"]


@set_module("sisl.geom")
class AtomFracSite(AtomCategory):
    r"""Classify atoms based on fractional sites for a given supercell

    Match atomic coordinates based on the fractional positions.

    Parameters
    ----------
    lattice : Lattice, LatticeChild or argument to Lattice
       an object that defines the lattice vectors (will be passed through to `Lattice`
       if not an object instance of `Lattice` or `LatticeChild`
    atol : float, optional
       the absolute tolerance (in Ang) to check whether the site is an integer
       site.
    offset : array_like, optional
       an offset made to the geometry coordinates before calculating the fractional
       coordinates according to `lattice`
    foffset : array_like, optional
       fractional offset of the fractional coordinates, this allows to select sub-regions
       in the `lattice` lattice vectors.

    Examples
    --------
    >>> gr = graphene() * (4, 5, 1)
    >>> A_site = AtomFracSite(graphene())
    >>> B_site = AtomFracSite(graphene(), foffset=(-1/3, -1/3, 0))
    >>> cat = (A_site | B_site).categorize(gr)
    >>> for ia, c in enumerate(cat):
    ...    if ia % 2 == 0:
    ...        assert c == A_site
    ...    else:
    ...        assert c == B_site
    """

    __slots__ = ("_cell", "_icell", "_length", "_atol", "_offset", "_foffset")

    @deprecate_argument("sc", "lattice", "use lattice= instead of sc=", "0.15", "0.17")
    def __init__(
        self, lattice, atol=1.0e-5, offset=(0.0, 0.0, 0.0), foffset=(0.0, 0.0, 0.0)
    ):
        if isinstance(lattice, LatticeChild):
            lattice = lattice.lattice
        elif not isinstance(lattice, Lattice):
            lattice = Lattice(lattice)

        # Unit-cell to fractionalize
        self._cell = lattice.cell.copy()
        # lengths of lattice vectors
        self._length = lattice.length.copy().reshape(1, 3)
        # inverse cell (for fractional coordinate calculations)
        self._icell = cell_invert(self._cell)
        # absolute tolerance [Ang]
        self._atol = atol
        # offset of coordinates before calculating the fractional coordinates
        self._offset = _a.arrayd(offset).reshape(1, 3)
        # fractional offset before comparing to the integer part of the fractional coordinate
        self._foffset = _a.arrayd(foffset).reshape(1, 3)

        super().__init__(
            f"fracsite(atol={self._atol}, offset={self._offset}, foffset={self._foffset})"
        )

    def categorize(self, geometry: Geometry, atoms: AtomsIndex = None):
        # _sanitize_loop will ensure that atoms will always be an integer
        if atoms is None:
            fxyz = np.dot(geometry.xyz + self._offset, self._icell.T) + self._foffset
        else:
            atoms = geometry._sanitize_atoms(atoms)
            fxyz = (
                np.dot(geometry.xyz[atoms].reshape(-1, 3) + self._offset, self._icell.T)
                + self._foffset
            )
        # Find fractional indices that match to an integer of the passed cell
        # We multiply with the length of the cell to get an error in Ang
        ret = np.where(
            np.fabs((fxyz - np.rint(fxyz)) * self._length).max(1) <= self._atol,
            self,
            NullCategory(),
        ).tolist()
        if isinstance(atoms, Integral):
            ret = ret[0]
        return ret

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            for s, o in map(
                lambda a: (getattr(self, f"_{a}"), getattr(other, f"_{a}")),
                ("cell", "icell", "atol", "offset", "foffset"),
            ):
                if not np.allclose(s, o):
                    return False
            return True
        return False


@set_module("sisl.geom")
class AtomXYZ(AtomCategory):
    r"""Classify atoms based on coordinates

    Parameters
    ----------
    *args : Shape
       any shape that implements `Shape.within`
    **kwargs:
       keys are operator specifications and values are
       used in those specifications.
       The keys are split into 3 sections
       ``<options>_<direction>_<operator>``
       - ``options`` are made of combinations of ``['a', 'f']``
         i.e. ``"af"``, ``"f"`` or ``"a"`` are all valid.
         An ``a`` takes the absolute value, ``f`` means a fractional
         coordinate. This part is optional.
       - ``direction`` is anything that gets parsed in `sisl.utils.misc.direction`
         either one of ``{0, "X", "x", "a", 1, "Y", "y", "b", 2, "Z", "z", "c"}``.
       - ``operator`` is a name for an operator defined in the `operator` module.

       For instance ``a_z_lt=3.`` will be equivalent to the
       boolean operation ``np.fabs(geometry.xyz[:, 2]) < 3.``.

       Optionally one need not specify the operator in which case one should
       provide an argument of two values.

       For instance ``c=(3., 6.)`` will be equivalent to the
       boolean operation ``3. <= geometry.xyz[:, 2]) <= 6.``.

    Attributes
    -----------
    x, y, z, fx, fy, fz, ax, ay, az: AtomCategory
        Shortcuts for creating an `AtomXYZ` category.

        .. code::

            AtomXYZ.x(-2, 3) == AtomXYZ(x=(-2, 3))
            AtomXYZ.fx < 3 == AtomXYZ.fx(None, 3) == AtomXYZ(f_x=(None, 3)) == AtomXYZ(f_x_lt=3)

    """

    __slots__ = ("_coord_check",)

    def __init__(self, *args, **kwargs):
        def create1(is_frac, is_abs, op, d):
            if is_abs:

                @wraps(op)
                def func(a, b):
                    return op(np.fabs(a))

            else:

                @wraps(op)
                def func(a, b):
                    return op(a)

            return is_frac, func, d, None

        def create2(is_frac, is_abs, op, d, val):
            if is_abs:

                @wraps(op)
                def func(a, b):
                    return op(np.fabs(a), b)

            else:

                @wraps(op)
                def func(a, b):
                    return op(a, b)

            return is_frac, func, d, val

        coord_ops = []

        # For each *args we expect this to be a shape
        for arg in args:
            if not isinstance(arg, Shape):
                raise ValueError(
                    f"{self.__class__.__name__} requires non-keyword arguments "
                    f"to be of type Shape {type(arg)}."
                )

            coord_ops.append(create1(False, False, arg.within, (0, 1, 2)))

        for key, value in kwargs.items():
            # will allow us to do value.size
            value = np.array(value)

            # Parse key for to get values
            # The key must have this specification:
            # [fa]_"dir"_"op"
            spec = ""
            if key.count("_") == 2:
                spec, sdir, op = key.split("_")
            elif key.count("_") == 1:
                if value.size == 2:
                    spec, sdir = key.split("_")
                else:
                    sdir, op = key.split("_")
            elif value.size == 2:
                sdir = key
            else:
                raise ValueError(
                    f"{self.__class__.__name__} could not determine the operations for {key}={value}.\n"
                    f"{key} must be on the form [fa]_<dir>_<operator>"
                )

            # parse options
            is_abs = "a" in spec
            is_frac = "f" in spec
            # Convert to integer axis
            sdir = direction(sdir)

            # Now we are ready to build our scheme
            if value.size == 2:
                # do it twice
                if not value[0] is None:
                    coord_ops.append(
                        create2(is_frac, is_abs, operator.ge, sdir, value[0])
                    )
                if not value[1] is None:
                    coord_ops.append(
                        create2(is_frac, is_abs, operator.le, sdir, value[1])
                    )
            else:
                coord_ops.append(
                    create2(is_frac, is_abs, getattr(operator, op), sdir, value)
                )

        self._coord_check = coord_ops
        super().__init__("coord")

    def categorize(self, geometry: Geometry, atoms: AtomsIndex = None):
        if atoms is None:
            xyz = geometry.xyz
            fxyz = geometry.fxyz
        else:
            atoms = geometry._sanitize_atoms(atoms)
            xyz = geometry.xyz[atoms]
            fxyz = geometry.fxyz[atoms]

        def call(frac, func, d, val):
            if frac:
                return func(fxyz[..., d], val)
            return func(xyz[..., d], val)

        and_reduce = np.logical_and.reduce
        ret = np.where(
            and_reduce([call(*four) for four in self._coord_check]),
            self,
            NullCategory(),
        ).tolist()
        if isinstance(atoms, Integral):
            ret = ret[0]
        return ret

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            if len(self._coord_check) == len(other._coord_check):
                for s4 in self._coord_check:
                    if s4 not in other._coord_check:
                        return False
                return True
        return False


# The following code is just to make the use of coordinate categories more convenient
# We will create one individual category for each direction. In this way, one can access
# them directly in the keyword category builder. E.g.: `AtomCategory(x=(1,2))` creates `AtomXYZ(x=(1,2))`
# without needing to nest "x", as in `AtomCategory(xyz={"x": (1,2)})`.
class AtomXYZMeta(CategoryMeta):
    """
    Metaclass to give extra functionality to individual coordinates categories.

    The classes generated by this metaclass can use `<` and `>` to create categories.
    See examples.

    Examples
    ----------

    If `AtomX` has been built using this metaclass:
    >>> AtomX < 5 == AtomX((None, 5)) == AtomXYZ(x=(None, 5))
    >>> AtomX > 5 == AtomX(gt=5) == AtomXYZ(x=(5, None))
    """

    def __lt__(self, other):
        return self(lt=other)

    def __le__(self, other):
        return self(le=other)

    def __gt__(self, other):
        return self(gt=other)

    def __ge__(self, other):
        return self(ge=other)


def _new_factory(key):
    def _new(cls, *interval, **kwargs):
        """
        Will go into the __new__ method of the coordinate classes
        """

        def _apply_key(k, v):
            return f"{key}_{k}", v

        if len(kwargs) > 0:
            new_kwargs = dict(map(_apply_key, *zip(*kwargs.items())))
        else:
            new_kwargs = {}

        # Convert interval to correct interpretation
        if len(interval) == 1:
            # we want to do it explicitly to let AtomXYZ raise an
            # error for multiple entries
            return AtomXYZ(**{key: interval[0]}, **new_kwargs)
        elif len(interval) == 2:
            # we want to do it explicitly to let AtomXYZ raise an
            # error for multiple entries
            return AtomXYZ(**{key: interval}, **new_kwargs)
        elif len(interval) != 0:
            raise ValueError(
                f"{cls.__name__} non-keyword argumest must be 1 tuple, or 2 values"
            )
        return AtomXYZ(**new_kwargs)

    return _new


# Iterate over all directions that deserve an individual class
for key in ("x", "y", "z", "f_x", "f_y", "f_z", "a_x", "a_y", "a_z"):
    # The underscores are just kept for the key that is passed to AtomXYZ,
    # but the name of the category will have no underscore
    name = key.replace("_", "")

    # Create the class for this direction
    # note this is lower case since AtomZ should not interfere with Atomz
    new_cls = AtomXYZMeta(
        f"Atom{name}", (AtomCategory,), {"__new__": _new_factory(key), "__slots__": ()}
    )

    new_cls = set_module("sisl.geom")(new_cls)

    # Set it as an attribute of `AtomXYZ`, so that you can access them from it.
    # Note that the classes are never exposed, so you can not import them, and the way
    # to use them is either through the kw builder in `AtomCategory` or from `AtomXYZ` attributes.
    # This enables AtomXYZ.fz < 0.5, for example.
    setattr(AtomXYZ, name, new_cls)
