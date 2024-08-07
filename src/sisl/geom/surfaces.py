# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections import namedtuple
from collections.abc import Sequence
from itertools import groupby
from numbers import Integral
from typing import Union

import numpy as np

from sisl import Atom, Geometry, Lattice
from sisl._internal import set_module
from sisl.typing import AtomsLike

from ._common import geometry2uc, geometry_define_nsc

__all__ = ["fcc_slab", "bcc_slab", "rocksalt_slab"]


def _layer2int(layer, periodicity):
    """Convert layer specification to integer"""
    if layer is None:
        return None
    if isinstance(layer, str):
        layer = "ABCDEF".index(layer.upper())
    return layer % periodicity


def _calc_info(start, end, layers, periodicity):
    """Determine offset index from start or end specification"""
    if start is not None and end is not None:
        raise ValueError("Only one of 'start' or 'end' may be supplied")

    Info = namedtuple("Info", ["layers", "nlayers", "offset", "periodicity"])

    # First check valid input, start or end should conform or die
    stacking = "ABCDEF"[:periodicity]

    # convert to integers in range range(periodicity)
    # However, if they are None, they will still be none
    start = _layer2int(start, periodicity)
    end = _layer2int(end, periodicity)

    # First convert `layers` to integer, and possibly determine start/end
    if layers is None:
        # default to a single stacking
        layers = periodicity

    if isinstance(layers, Integral):
        # convert to proper layers
        nlayers = layers

        # + 2 to allow rotating
        layers = stacking * (nlayers // periodicity + 2)

        if start is None and end is None:
            # the following will figure it out
            layers = layers[:nlayers]
        elif start is None:
            # end is not none
            layers = layers[end + 1 :] + layers[: end + 1]
            layers = layers[-nlayers:]
        elif end is None:
            # start is not none
            layers = layers[start:] + layers[:start]
            layers = layers[:nlayers]

    elif isinstance(layers, str):
        nlayers = len(layers)

        try:
            # + 2 to allow rotating
            (stacking * (nlayers // periodicity + 2)).index(layers)
        except ValueError:
            raise NotImplementedError(
                f"Stacking faults are not implemented, requested {layers} with stacking {stacking}"
            )

        if start is None and end is None:
            # easy case, we just calculate one of them
            start = _layer2int(layers[0], periodicity)

        elif start is not None:
            if _layer2int(layers[0], periodicity) != start:
                raise ValueError(
                    f"Passing both 'layers' and 'start' requires them to be conforming; found layers={layers} "
                    f"and start={'ABCDEF'[start]}"
                )
        elif end is not None:
            if _layer2int(layers[-1], periodicity) != end:
                raise ValueError(
                    f"Passing both 'layers' and 'end' requires them to be conforming; found layers={layers} "
                    f"and end={'ABCDEF'[end]}"
                )

    # a sanity check for the algorithm, should always hold!
    if start is not None:
        assert _layer2int(layers[0], periodicity) == start
    if end is not None:
        assert _layer2int(layers[-1], periodicity) == end

    # Convert layers variable to the list of layers in integer space
    layers = [_layer2int(l, periodicity) for l in layers]
    return Info(layers, nlayers, -_layer2int(layers[0], periodicity), periodicity)


def _finish_slab(g, vacuum):
    """Move slab to the unit-cell and move it very slightly to
    stick to the lower side of the unit-cell borders.
    """
    g = geometry2uc(g).sort(lattice=[2, 1, 0])
    if vacuum is not None:
        geometry_define_nsc(g, [True, True, False])
        g.cell[2, 2] = g.xyz[:, 2].max() + vacuum
    else:
        geometry_define_nsc(g, [True, True, True])
    return g


def _convert_miller(miller):
    """Convert miller specification to 3-tuple"""
    if isinstance(miller, int):
        miller = str(miller)
    if isinstance(miller, str):
        miller = [int(i) for i in miller]
    if isinstance(miller, list):
        miller = tuple(miller)
    if len(miller) != 3:
        raise ValueError(f"Invalid Miller indices, must have length 3")
    return miller


def _slab_with_vacuum(func, *args, **kwargs):
    """Function to wrap `func` with vacuum in between"""
    layers = kwargs.pop("layers")
    if layers is None or isinstance(layers, Integral):
        return None

    def is_vacuum(layer):
        """A vacuum is defined by one of these variables:

        - None
        - ' '
        - 0
        """
        if layer is None:
            return True
        if isinstance(layer, str):
            return layer == " "
        if isinstance(layer, Integral):
            return layer == 0
        return False

    # we are dealing either with a list of ints or str
    if isinstance(layers, str):
        nvacuums = layers.count(" ")
        if nvacuums == 0:
            return None

        if layers.count("  ") > 0:
            raise ValueError(
                "Denoting several vacuum layers next to each other is not supported. "
                "Please pass 'vacuum' as an array instead."
            )

        # determine number of slabs
        nslabs = len(layers.strip().split())

    else:
        # this must be a list of ints, fill in none between ints
        def are_layers(a, b):
            a_layer = not is_vacuum(a)
            b_layer = not is_vacuum(b)
            return a_layer and b_layer

        # convert list correctly
        layers = [
            [p, None] if are_layers(p, n) else [p]
            for p, n in zip(layers[:-1], layers[1:])
        ] + [[layers[-1]]]
        layers = [l for ls in layers for l in ls]
        nvacuums = sum([1 if is_vacuum(l) else 0 for l in layers])
        nslabs = sum([0 if is_vacuum(l) else 1 for l in layers])

    # Now we need to ensure that `start` and `end` are the same
    # length as nslabs
    def ensure_length(var, nslabs, name):
        if var is None:
            return [None] * nslabs
        if isinstance(var, (Integral, str)):
            return [var] * nslabs

        if len(var) > nslabs:
            raise ValueError(
                f"Specification of {name} has too many elements compared to the "
                f"number of slabs {nslabs}, please reduce length from {len(var)}."
            )

        # it must be an array of some sorts
        out = [None] * nslabs
        out[: len(var)] = var[:]
        return out

    start = ensure_length(kwargs.pop("start"), nslabs, "start")
    end = ensure_length(kwargs.pop("end"), nslabs, "end")

    vacuum = np.asarray(kwargs.pop("vacuum"))
    vacuums = np.full(nvacuums, 0.0)
    if vacuum.ndim == 0:
        vacuums[:] = vacuum
    else:
        vacuums[: len(vacuum)] = vacuum
        vacuums[len(vacuum) :] = vacuum[-1]
    vacuums = vacuums.tolist()

    # We are now sure that there is a vacuum!
    def iter_func(key, layer):
        if key == 0:
            return None

        # layer is an iterator, convert to list
        layer = list(layer)
        if isinstance(layer[0], str):
            layer = "".join(layer)
        elif len(layer) > 1:
            raise ValueError(f"Grouper returned long list {layer}")
        else:
            layer = layer[0]
        if is_vacuum(layer):
            return None
        return layer

    # group stuff
    layers = [
        iter_func(key, group)
        for key, group in groupby(
            layers,
            # group by vacuum positions and not vacuum positions
            lambda l: 0 if is_vacuum(l) else 1,
        )
    ]

    # Now we need to loop and create the things
    reduce_nsc_c = layers[0] is None or layers[-1] is None
    ivacuum = 0
    islab = 0
    if layers[0] is None:
        layers.pop(0)  # vacuum specification
        out = func(
            *args,
            layers=layers.pop(0),
            start=start.pop(0),
            end=end.pop(0),
            vacuum=None,
            **kwargs,
        )
        # add vacuum
        vacuum = vacuums.pop(0)
        out = out.add_vacuum(vacuum, 2, offset=(0, 0, vacuum))
        ivacuum += 1
        islab += 1

    else:
        out = func(
            *args,
            layers=layers.pop(0),
            start=start.pop(0),
            end=end.pop(0),
            vacuum=None,
            **kwargs,
        )
        islab += 1

    while len(layers) > 0:
        layer = layers.pop(0)
        if layer is None:
            dx = out.cell[2, 2] - out.xyz[:, 2].max()
            # this ensures the vacuum is exactly vacuums[iv]
            vacuum = vacuums.pop(0) - dx
            ivacuum += 1
            out = out.add_vacuum(vacuum, 2)
        else:
            geom = func(
                *args,
                layers=layer,
                start=start.pop(0),
                end=end.pop(0),
                vacuum=None,
                **kwargs,
            )
            out = out.append(geom, 2)
            islab += 1

    assert islab == nslabs, "Error in determining correct slab counts"
    assert ivacuum == nvacuums, "Error in determining correct vacuum counts"

    if reduce_nsc_c:
        out.set_nsc(c=1)

    return out


@set_module("sisl.geom")
def fcc_slab(
    alat: float,
    atoms: AtomsLike,
    miller: Union[int, str, tuple[int, int, int]],
    layers=None,
    vacuum: Union[float, Sequence[float]] = 20.0,
    *,
    orthogonal: bool = False,
    start=None,
    end=None,
) -> Geometry:
    r"""Surface slab forming a face-centered cubic (FCC) crystal

    The slab layers are stacked along the :math:`z`-axis. The default stacking is the first
    layer as an A-layer, defined as the plane containing an atom at :math:`(x,y)=(0,0)`.

    Several vacuum separated segments can be created by specifying specific positions through
    either `layers` being a list, or by having spaces in its `str` form, see Examples.

    Parameters
    ----------
    alat :
        lattice constant of the fcc crystal
    atoms :
        the atom that the crystal consists of
    miller :
        Miller indices of the surface facet
    layers : int or str or array_like of ints, optional
        Number of layers in the slab or explicit layer specification.
        For array like arguments vacuum will be placed between each index of the layers.
        Each element can either be an int or a str to specify number of layers or an explicit
        order of layers.
        If a `str` it can contain spaces to specify vacuum positions (then equivalent to ``layers.split()``).
        If there are no vacuum positions specified a vacuum will be placed *after* the layers.
        See examples for details.
    vacuum :
        size of vacuum at locations specified in `layers`. The vacuum will always
        be placed along the :math:`z`-axis (3rd lattice vector).
        Each segment in `layers` will be appended the vacuum as found by ``zip_longest(layers, vacuum)``.
        None means that no vacuum is inserted (keeps the crystal structure intact).
    orthogonal :
        if True returns an orthogonal lattice
    start : int or str or array_like, optional
        sets the first layer in the slab. Only one of `start` or `end` must be specified.
        Discouraged to pass if `layers` is a str since a `ValueError` will be raised if they do
        not match.
    end : int or str or array_like, optional
        sets the last layer in the slab. Only one of `start` or `end` must be specified.
        Discouraged to pass if `layers` is a str since a `ValueError` will be raised if they do
        not match.

    Examples
    --------
    111 surface, starting with the A layer

    >>> fcc_slab(alat, atoms, "111", start=0)

    111 surface, starting with the B layer

    >>> fcc_slab(alat, atoms, "111", start=1)

    111 surface, ending with the B layer

    >>> fcc_slab(alat, atoms, "111", end='B')

    fcc crystal with 6 layers and the 111 orientation (periodic also on the z-direction, e.g., for an electrode geometry)

    >>> fcc_slab(alat, atoms, "111", layers=6, vacuum=None)

    111 surface, with explicit layers in a given order

    >>> fcc_slab(alat, atoms, "111", layers='BCABCA')

    111 surface, with (1 Ang vacuum)BCA(2 Ang vacuum)ABC(3 Ang vacuum)

    >>> fcc_slab(alat, atoms, "111", layers=' BCA ABC ', vacuum=(1, 2, 3))

    111 surface, with (20 Ang vacuum)BCA

    >>> fcc_slab(alat, atoms, "111", layers=' BCA', vacuum=20)

    111 surface, with (2 Ang vacuum)BCA(1 Ang vacuum)ABC(1 Ang vacuum)
    The last item in `vacuum` gets repeated.

    >>> fcc_slab(alat, atoms, "111", layers=' BCA ABC ', vacuum=(2, 1))

    111 periodic structure with ABC(20 Ang vacuum)BC
    The unit cell parameters will be periodic in this case, and it will not be
    a slab.

    >>> fcc_slab(alat, atoms, "111", layers='ABC BC', vacuum=20.)

    111 surface in an orthogonal (4x5) cell, maintaining the atom ordering
    according to `lattice=[2, 1, 0]`:

    >>> fcc_slab(alat, atoms, "111", orthogonal=True).repeat(5, axis=1).repeat(4, axis=0)

    111 surface with number specifications of layers together with start
    Between each number an implicit vacuum is inserted, only the first and last
    are required if vacuum surrounding the slab is needed. The following two calls
    are equivalent.
    Structure: (10 Ang vacuum)(ABC)(1 Ang vacuum)(BCABC)(2 Ang vacuum)(CAB)

    >>> fcc_slab(alat, atoms, "111", layers=(' ', 3, 5, 3), start=(0, 1, 2), vacuum=(10, 1, 2))
    >>> fcc_slab(alat, atoms, "111", layers=' ABC BCABC CAB', vacuum=(10, 1, 2))

    Raises
    ------
    NotImplementedError
        In case the Miller index has not been implemented or a stacking fault is
        introduced in `layers`.

    ValueError
        For wrongly specified `layers` and `vacuum` arguments.

    See Also
    --------
    fcc : Fully periodic equivalent of this slab structure
    bcc_slab : Slab in BCC structure
    rocksalt_slab : Slab in rocksalt/halite structure
    """
    geom = _slab_with_vacuum(
        fcc_slab,
        alat,
        atoms,
        miller,
        vacuum=vacuum,
        orthogonal=orthogonal,
        layers=layers,
        start=start,
        end=end,
    )
    if geom is not None:
        return geom

    miller = _convert_miller(miller)

    if miller == (1, 0, 0):
        info = _calc_info(start, end, layers, 2)

        lattice = Lattice(np.array([0.5**0.5, 0.5**0.5, 0.5]) * alat)
        g = Geometry([0, 0, 0], atoms=atoms, lattice=lattice)
        g = g.tile(info.nlayers, 2)

        # slide AB layers relative to each other
        B = (info.offset + 1) % 2
        g.xyz[B::2] += (lattice.cell[0] + lattice.cell[1]) / 2

    elif miller == (1, 1, 0):
        info = _calc_info(start, end, layers, 2)

        lattice = Lattice(np.array([1.0, 0.5, 0.125]) ** 0.5 * alat)
        g = Geometry([0, 0, 0], atoms=atoms, lattice=lattice)
        g = g.tile(info.nlayers, 2)

        # slide AB layers relative to each other
        B = (info.offset + 1) % 2
        g.xyz[B::2] += (lattice.cell[0] + lattice.cell[1]) / 2

    elif miller == (1, 1, 1):
        info = _calc_info(start, end, layers, 3)

        if orthogonal:
            lattice = Lattice(np.array([0.5, 4 * 0.375, 1 / 3]) ** 0.5 * alat)
            g = Geometry(
                np.array([[0, 0, 0], [0.125, 0.375, 0]]) ** 0.5 * alat,
                atoms=atoms,
                lattice=lattice,
            )
            g = g.tile(info.nlayers, 2)

            # slide ABC layers relative to each other
            B = 2 * (info.offset + 1) % 6
            C = 2 * (info.offset + 2) % 6
            vec = (3 * lattice.cell[0] + lattice.cell[1]) / 6
            g.xyz[B::6] += vec
            g.xyz[B + 1 :: 6] += vec
            g.xyz[C::6] += 2 * vec
            g.xyz[C + 1 :: 6] += 2 * vec

        else:
            lattice = Lattice(
                np.array([[0.5, 0, 0], [0.125, 0.375, 0], [0, 0, 1 / 3]]) ** 0.5 * alat
            )
            g = Geometry([0, 0, 0], atoms=atoms, lattice=lattice)
            g = g.tile(info.nlayers, 2)

            # slide ABC layers relative to each other
            B = (info.offset + 1) % 3
            C = (info.offset + 2) % 3
            vec = (lattice.cell[0] + lattice.cell[1]) / 3
            g.xyz[B::3] += vec
            g.xyz[C::3] += 2 * vec

    else:
        raise NotImplementedError(f"fcc_slab: miller={miller} is not implemented")

    g = _finish_slab(g, vacuum)
    return g


@set_module("sisl.geom")
def bcc_slab(
    alat: float,
    atoms: AtomsLike,
    miller: Union[int, str, tuple[int, int, int]],
    layers=None,
    vacuum: Union[float, Sequence[float]] = 20.0,
    *,
    orthogonal: bool = False,
    start=None,
    end=None,
) -> Geometry:
    r"""Construction of a surface slab from a body-centered cubic (BCC) crystal

    The slab layers are stacked along the :math:`z`-axis. The default stacking is the first
    layer as an A-layer, defined as the plane containing an atom at :math:`(x,y)=(0,0)`.

    Several vacuum separated segments can be created by specifying specific positions through
    either `layers` being a list, or by having spaces in its `str` form, see Examples.

    Parameters
    ----------
    alat :
        lattice constant of the fcc crystal
    atoms :
        the atom that the crystal consists of
    miller :
        Miller indices of the surface facet
    layers : int or str or array_like of ints, optional
        Number of layers in the slab or explicit layer specification.
        For array like arguments vacuum will be placed between each index of the layers.
        Each element can either be an int or a str to specify number of layers or an explicit
        order of layers.
        If a `str` it can contain spaces to specify vacuum positions (then equivalent to ``layers.split()``).
        If there are no vacuum positions specified a vacuum will be placed *after* the layers.
        See examples for details.
    vacuum :
        size of vacuum at locations specified in `layers`. The vacuum will always
        be placed along the :math:`z`-axis (3rd lattice vector).
        Each segment in `layers` will be appended the vacuum as found by ``zip_longest(layers, vacuum)``.
        None means that no vacuum is inserted (keeps the crystal structure intact).
    orthogonal :
        if True returns an orthogonal lattice
    start : int or str or array_like, optional
        sets the first layer in the slab. Only one of `start` or `end` must be specified.
        Discouraged to pass if `layers` is a str.
    end : int or str or array_like, optional
        sets the last layer in the slab. Only one of `start` or `end` must be specified.
        Discouraged to pass if `layers` is a str.

    Examples
    --------

    Please see `fcc_slab` for examples, they are equivalent to this method.

    Raises
    ------
    NotImplementedError
        In case the Miller index has not been implemented or a stacking fault is
        introduced in `layers`.

    See Also
    --------
    bcc : Fully periodic equivalent of this slab structure
    fcc_slab : Slab in FCC structure
    rocksalt_slab : Slab in rocksalt/halite structure
    """
    geom = _slab_with_vacuum(
        bcc_slab,
        alat,
        atoms,
        miller,
        vacuum=vacuum,
        orthogonal=orthogonal,
        layers=layers,
        start=start,
        end=end,
    )
    if geom is not None:
        return geom

    miller = _convert_miller(miller)

    if miller == (1, 0, 0):
        info = _calc_info(start, end, layers, 2)

        lattice = Lattice(np.array([1, 1, 0.5]) * alat)
        g = Geometry([0, 0, 0], atoms=atoms, lattice=lattice)
        g = g.tile(info.nlayers, 2)

        # slide AB layers relative to each other
        B = (info.offset + 1) % 2
        g.xyz[B::2] += (lattice.cell[0] + lattice.cell[1]) / 2

    elif miller == (1, 1, 0):
        info = _calc_info(start, end, layers, 2)

        if orthogonal:
            lattice = Lattice(np.array([1, 2, 0.5]) ** 0.5 * alat)
            g = Geometry(
                np.array([[0, 0, 0], [0.5, 0.5**0.5, 0]]) * alat,
                atoms=atoms,
                lattice=lattice,
            )
            g = g.tile(info.nlayers, 2)

            # slide ABC layers relative to each other
            B = 2 * (info.offset + 1) % 4
            vec = lattice.cell[1] / 2
            g.xyz[B::4] += vec
            g.xyz[B + 1 :: 4] += vec

        else:
            lattice = Lattice(
                np.array([[1, 0, 0], [0.5, 0.5**0.5, 0], [0, 0, 0.5**0.5]]) * alat
            )
            g = Geometry([0, 0, 0], atoms=atoms, lattice=lattice)
            g = g.tile(info.nlayers, 2)

            # slide AB layers relative to each other
            B = (info.offset + 1) % 2
            g.xyz[B::2] += lattice.cell[0] / 2

    elif miller == (1, 1, 1):
        info = _calc_info(start, end, layers, 3)

        if orthogonal:
            lattice = Lattice(np.array([2, 4 * 1.5, 1 / 12]) ** 0.5 * alat)
            g = Geometry(
                np.array([[0, 0, 0], [0.5, 1.5, 0]]) ** 0.5 * alat,
                atoms=atoms,
                lattice=lattice,
            )
            g = g.tile(info.nlayers, 2)

            # slide ABC layers relative to each other
            B = 2 * (info.offset + 1) % 6
            C = 2 * (info.offset + 2) % 6
            vec = (lattice.cell[0] + lattice.cell[1]) / 3
            for i in range(2):
                g.xyz[B + i :: 6] += vec
                g.xyz[C + i :: 6] += 2 * vec

        else:
            lattice = Lattice(
                np.array([[2, 0, 0], [0.5, 1.5, 0], [0, 0, 1 / 12]]) ** 0.5 * alat
            )
            g = Geometry([0, 0, 0], atoms=atoms, lattice=lattice)
            g = g.tile(info.nlayers, 2)

            # slide ABC layers relative to each other
            B = (info.offset + 1) % 3
            C = (info.offset + 2) % 3
            vec = (lattice.cell[0] + lattice.cell[1]) / 3
            g.xyz[B::3] += vec
            g.xyz[C::3] += 2 * vec

    else:
        raise NotImplementedError(f"bcc_slab: miller={miller} is not implemented")

    g = _finish_slab(g, vacuum)
    return g


@set_module("sisl.geom")
def rocksalt_slab(
    alat: float,
    atoms: AtomsLike,
    miller: Union[int, str, tuple[int, int, int]],
    layers=None,
    vacuum: Union[float, Sequence[float]] = 20.0,
    *,
    orthogonal: bool = False,
    start=None,
    end=None,
) -> Geometry:
    r"""Surface slab forming a rock-salt crystal (halite)

    This structure is formed by two interlocked fcc crystals for each of the two elements.

    The slab layers are stacked along the :math:`z`-axis. The default stacking is the first
    layer as an A-layer, defined as the plane containing the first atom in the atoms list
    at :math:`(x,y)=(0,0)`.

    Several vacuum separated segments can be created by specifying specific positions through
    either `layers` being a list, or by having spaces in its `str` form, see Examples.

    This is equivalent to the NaCl crystal structure (halite).

    Parameters
    ----------
    alat :
        lattice constant of the rock-salt crystal
    atoms :
        a list of two atoms that the crystal consist of
    miller :
        Miller indices of the surface facet
    layers : int or str or array_like of ints, optional
        Number of layers in the slab or explicit layer specification.
        For array like arguments vacuum will be placed between each index of the layers.
        Each element can either be an int or a str to specify number of layers or an explicit
        order of layers.
        If a `str` it can contain spaces to specify vacuum positions (then equivalent to ``layers.split()``).
        If there are no vacuum positions specified a vacuum will be placed *after* the layers.
        See examples for details.
    vacuum :
        size of vacuum at locations specified in `layers`. The vacuum will always
        be placed along the :math:`z`-axis (3rd lattice vector).
        Each segment in `layers` will be appended the vacuum as found by ``zip_longest(layers, vacuum)``.
        None means that no vacuum is inserted (keeps the crystal structure intact).
    orthogonal :
        if True returns an orthogonal lattice
    start : int or str or array_like, optional
        sets the first layer in the slab. Only one of `start` or `end` must be specified.
        Discouraged to pass if `layers` is a str.
    end : int or str or array_like, optional
        sets the last layer in the slab. Only one of `start` or `end` must be specified.
        Discouraged to pass if `layers` is a str.

    Examples
    --------
    NaCl(100) slab, starting with A-layer

    >>> rocksalt_slab(5.64, ['Na', 'Cl'], 100)

    6-layer NaCl(100) slab, ending with A-layer

    >>> rocksalt_slab(5.64, ['Na', 'Cl'], 100, layers=6, end='A')

    6-layer NaCl(100) slab, starting with Cl A layer and with a vacuum
    gap of 20 Å on both sides of the slab

    >>> rocksalt_slab(5.64, ['Cl', 'Na'], 100, layers=' ABAB ')

    For more examples see `fcc_slab`, the vacuum displacements are directly
    translateable to this function.

    Raises
    ------
    NotImplementedError
        In case the Miller index has not been implemented or a stacking fault is
        introduced in `layers`.

    See Also
    --------
    rocksalt : Basic structure of this one
    fcc_slab : Slab in FCC structure (this slab is a combination of fcc slab structures)
    bcc_slab : Slab in BCC structure
    """
    geom = _slab_with_vacuum(
        rocksalt_slab,
        alat,
        atoms,
        miller,
        vacuum=vacuum,
        orthogonal=orthogonal,
        layers=layers,
        start=start,
        end=end,
    )
    if geom is not None:
        return geom

    if isinstance(atoms, (str, Integral, Atom)):
        atoms = [atoms, atoms]
    if len(atoms) != 2:
        raise ValueError(f"Invalid list of atoms, must have length 2")

    miller = _convert_miller(miller)

    g1 = fcc_slab(
        alat,
        atoms[0],
        miller,
        layers=layers,
        vacuum=None,
        orthogonal=orthogonal,
        start=start,
        end=end,
    )
    g2 = fcc_slab(
        alat,
        atoms[1],
        miller,
        layers=layers,
        vacuum=None,
        orthogonal=orthogonal,
        start=start,
        end=end,
    )

    if miller == (1, 0, 0):
        g2 = g2.move(np.array([0.5, 0.5, 0]) ** 0.5 * alat / 2)

    elif miller == (1, 1, 0):
        g2 = g2.move(np.array([1, 0, 0]) * alat / 2)

    elif miller == (1, 1, 1):
        g2 = g2.move(np.array([0, 2 / 3, 1 / 3]) ** 0.5 * alat / 2)

    else:
        raise NotImplementedError(f"rocksalt_slab: miller={miller} is not implemented")

    g = g1.add(g2)

    g = _finish_slab(g, vacuum)
    return g
