from functools import partial, wraps
import operator
from numbers import Integral

import numpy as np
from numpy import dot, fabs, where

from sisl._internal import set_module, singledispatchmethod
from sisl._help import isiterable
from sisl.utils.misc import direction
from sisl.shape import *
from sisl.supercell import SuperCell, SuperCellChild
from sisl._supercell import cell_invert
import sisl._array as _a
from .base import AtomCategory, NullCategory, _sanitize_loop


__all__ = ["AtomFracSite", "AtomXYZ"]


@set_module("sisl.geom")
class AtomFracSite(AtomCategory):
    r""" Classify atoms based on fractional sites for a given supercell

    Match atomic coordinates based on the fractional positions.

    Parameters
    ----------
    sc : SuperCell, SuperCellChild or argument to SuperCell
       an object that defines the lattice vectors (will be passed through to `SuperCell`
       if not an object instance of `SuperCell` or `SuperCellChild`
    atol : float, optional
       the absolute tolerance (in Ang) to check whether the site is an integer
       site.
    offset : array_like, optional
       an offset made to the geometry coordinates before calculating the fractional
       coordinates according to `sc`
    foffset : array_like, optional
       fractional offset of the fractional coordinates, this allows to select sub-regions
       in the `sc` lattice vectors.

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
    __slots__ = (f"_{a}" for a in ("cell", "icell", "length", "atol", "offset", "foffset"))

    def __init__(self, sc, atol=1.e-5, offset=(0., 0., 0.), foffset=(0., 0., 0.)):
        if isinstance(sc, SuperCellChild):
            sc = sc.sc
        elif not isinstance(sc, SuperCell):
            sc = SuperCell(sc)

        # Unit-cell to fractionalize
        self._cell = sc.cell.copy()
        # lengths of lattice vectors
        self._length = sc.length.copy().reshape(1, 3)
        # inverse cell (for fractional coordinate calculations)
        self._icell = cell_invert(self._cell)
        # absolute tolerance [Ang]
        self._atol = atol
        # offset of coordinates before calculating the fractional coordinates
        self._offset = _a.arrayd(offset).reshape(1, 3)
        # fractional offset before comparing to the integer part of the fractional coordinate
        self._foffset = _a.arrayd(foffset).reshape(1, 3)

        super().__init__(f"fracsite(atol={self._atol}, offset={self._offset}, foffset={self._foffset})")

    def categorize(self, geometry, atoms=None):
        # _sanitize_loop will ensure that atoms will always be an integer
        if atoms is None:
            fxyz = dot(geometry.xyz + self._offset, self._icell.T) + self._foffset
        else:
            fxyz = dot(geometry.xyz[atoms].reshape(-1, 3) + self._offset,
                       self._icell.T) + self._foffset
        # Find fractional indices that match to an integer of the passed cell
        # We multiply with the length of the cell to get an error in Ang
        ret = where(np.fabs((fxyz - np.rint(fxyz))*self._length).max(1) <= self._atol,
                    self, NullCategory()).tolist()
        if isinstance(atoms, Integral):
            ret = ret[0]
        return ret

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            for s, o in map(lambda a: (getattr(self, f"_{a}"), getattr(other, f"_{a}")),
                            ("cell", "icell", "atol", "offset", "foffset")):
                if not np.allclose(s, o):
                    return False
            return True
        return False


@set_module("sisl.geom")
class AtomXYZ(AtomCategory):
    r""" Classify atoms based on coordinates

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

       For instance `a_z_lt=3.` will be equivalent to the
       boolean operation ``np.fabs(geometry.xyz[:, 2]) < 3.``.

       Optionally one need not specify the operator in which case one should
       provide an argument of two values.

       For instance `c=(3., 6.)` will be equivalent to the
       boolean operation ``3. <= geometry.xyz[:, 2]) <= 6.``.
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
                raise ValueError(f"{self.__class__.__name__} requires non-keyword arguments "
                                 f"to be of type Shape {type(arg)}.")

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
                raise ValueError(f"{self.__class__.__name__} could not determine the operations for {key}={value}.\n"
                                 f"{key} must be on the form [fa]_<dir>_<operator>")

            # parse options
            is_abs = "a" in spec
            is_frac = "f" in spec
            # Convert to integer axis
            sdir = direction(sdir)

            # Now we are ready to build our scheme
            if value.size == 2:
                # do it twice
                if not value[0] is None:
                    coord_ops.append(create2(is_frac, is_abs, operator.ge, sdir, value[0]))
                if not value[1] is None:
                    coord_ops.append(create2(is_frac, is_abs, operator.le, sdir, value[1]))
            else:
                coord_ops.append(create2(is_frac, is_abs, getattr(operator, op), sdir, value))

        self._coord_check = coord_ops
        super().__init__("coord")

    def categorize(self, geometry, atoms=None):
        if atoms is None:
            xyz = geometry.xyz
            fxyz = geometry.fxyz
        else:
            xyz = geometry.xyz[atoms]
            fxyz = geometry.fxyz[atoms]
        def call(frac, func, d, val):
            if frac:
                return func(fxyz[..., d], val)
            return func(xyz[..., d], val)

        and_reduce = np.logical_and.reduce
        ret = where(and_reduce([call(*four) for four in self._coord_check]),
                    self, NullCategory()).tolist()
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
