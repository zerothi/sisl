# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Optional, Union

import numpy as np

from sisl import Atom, Geometry
from sisl._internal import set_module
from sisl.typing import AtomsLike

from ._common import geometry_define_nsc
from ._composite import CompositeGeometrySection, composite_geometry
from .flat import honeycomb

__all__ = [
    "nanoribbon",
    "graphene_nanoribbon",
    "agnr",
    "zgnr",
    "cgnr",
    "heteroribbon",
    "graphene_heteroribbon",
]

FloatOrFloat2 = Union[float, tuple[float, float]]


@set_module("sisl.geom")
def nanoribbon(
    width: int,
    bond: float,
    atoms: AtomsLike,
    kind: Literal["armchair", "zigzag", "chiral"] = "armchair",
    vacuum: FloatOrFloat2 = 20.0,
    chirality: tuple[int, int] = (3, 1),
) -> Geometry:
    r"""Construction of a nanoribbon unit cell of type armchair, zigzag or (n,m)-chiral.

    The geometry is oriented along the :math:`x` axis.

    Parameters
    ----------
    width :
       number of atoms in the transverse direction
    bond :
       bond length between atoms in the honeycomb lattice
    atoms :
       atom (or atoms) in the honeycomb lattice
    kind :
       type of ribbon
    vacuum :
       separation in transverse and perpendicular direction
    chirality :
       index (n, m), only used if ``kind=chiral``

    See Also
    --------
    honeycomb : honeycomb lattices
    graphene : graphene geometry
    graphene_nanoribbon : graphene nanoribbon
    agnr : armchair graphene nanoribbon
    cgnr : chiral graphene nanoribbon
    zgnr : zigzag graphene nanoribbon
    """
    if not isinstance(width, Integral):
        raise ValueError(f"nanoribbon: width needs to be a postive integer ({width})!")

    try:
        vacuum_trans, vacuum_perp = vacuum
    except Exception:
        vacuum_trans = vacuum_perp = vacuum

    # Width characterization
    width = max(width, 1)
    n, m = width // 2, width % 2

    ribbon = honeycomb(bond, atoms, orthogonal=True, vacuum=vacuum_perp)
    angle = 0

    kind = kind.lower()
    if kind == "armchair":
        # Construct armchair GNR
        if m == 1:
            ribbon = ribbon.repeat(n + 1, 1)
            ribbon = ribbon.remove(3 * (n + 1)).remove(0)
        else:
            ribbon = ribbon.repeat(n, 1)

        # fix vacuum
        xyz = ribbon.xyz
        ribbon.cell[1, 1] = xyz[:, 1].max() - xyz[:, 1].min() + vacuum_trans

    elif kind in ("zigzag", "chiral"):
        # Construct zigzag GNR
        ribbon = ribbon.rotate(90, [0, 0, -1], what="abc+xyz")
        if m == 1:
            ribbon = ribbon.tile(n + 1, 0)
            ribbon = ribbon.remove(-1).remove(-1)
        else:
            ribbon = ribbon.tile(n, 0)
        # Invert y-coordinates
        ribbon.xyz[:, 1] *= -1

        # Set lattice vectors strictly orthogonal
        ribbon.set_lattice([ribbon.cell[1, 0], -ribbon.cell[0, 1], ribbon.cell[2, 2]])

        # Sort along x, then y
        ribbon = ribbon.sort(axes=(0, 1))

        if kind == "chiral":
            # continue with the zigzag ribbon as building block
            n, m = chirality
            ribbon = ribbon.tile(n + 1, 0)
            r = ribbon.xyz[1] - ribbon.xyz[width]
            ribbon.cell[0] += r + (m - 1) * (ribbon.xyz[2] - ribbon.xyz[0])
            ribbon = ribbon.remove(range(width))
            # determine rotation angle
            x = ribbon.cell[0]
            angle = np.arccos(
                x.dot([1, 0, 0]) / x.dot(x) ** 0.5
            )  # angle of vectors, x and b=[1, 0, 0]
            ribbon = ribbon.rotate(
                angle, [0, 0, -1], origin=ribbon.xyz[0], rad=True, what="abc+xyz"
            )
            # first lattice vector strictly along x
            ribbon.cell[0, 1] = 0

            # This will make the lattice vector exactly the vacuum length longer
            y = (ribbon.fxyz * ribbon.lattice.length)[:, 1]
            ribbon.cell[1] = ribbon.lattice.cell2length(
                y.max() - y.min() + vacuum_trans, axes=1
            )

        else:
            # fix vacuum
            xyz = ribbon.xyz
            ribbon.cell[1, 1] = xyz[:, 1].max() - xyz[:, 1].min() + vacuum_trans

    else:
        raise ValueError(f"nanoribbon: kind must be armchair or zigzag ({kind})")

    geometry_define_nsc(ribbon, [True, False, False])

    # move geometry into middle of the cell
    ribbon = ribbon.move(ribbon.center(what="cell") - ribbon.center())

    # first atom to zero along the first lattice vector
    x = ribbon.xyz[0]
    ribbon = ribbon.move([x[1] * np.tan(angle) - x[0], 0, 0])

    return ribbon


@set_module("sisl.geom")
def graphene_nanoribbon(
    width: int,
    bond: float = 1.42,
    atoms: Optional[AtomsLike] = None,
    kind: Literal["armchair", "zigzag", "chiral"] = "armchair",
    vacuum: FloatOrFloat2 = 20.0,
    chirality: tuple[int, int] = (3, 1),
) -> Geometry:
    r"""Construction of a graphene nanoribbon

    Parameters
    ----------
    width :
       number of atoms in the transverse direction
    bond :
       C-C bond length
    atoms :
       atom (or atoms) in the honeycomb lattice. Defaults to ``Atom(6)``
    kind :
       type of ribbon
    vacuum :
       separation in transverse direction
    chirality :
       index (n, m), only used if ``kind=chiral``

    See Also
    --------
    honeycomb : honeycomb lattices
    graphene : graphene geometry
    nanoribbon : honeycomb nanoribbon (used for this method)
    agnr : armchair graphene nanoribbon
    zgnr : zigzag graphene nanoribbon
    cgnr : chiral graphene nanoribbon
    """
    if atoms is None:
        atoms = Atom(Z=6, R=bond * 1.01)
    return nanoribbon(width, bond, atoms, kind=kind, vacuum=vacuum, chirality=chirality)


@set_module("sisl.geom")
def agnr(
    width: int,
    bond: float = 1.42,
    atoms: Optional[AtomsLike] = None,
    vacuum: FloatOrFloat2 = 20.0,
) -> Geometry:
    r"""Construction of an armchair graphene nanoribbon

    Parameters
    ----------
    width :
       number of atoms in the transverse direction
    bond :
       C-C bond length
    atoms :
       atom (or atoms) in the honeycomb lattice. Defaults to ``Atom(6)``
    vacuum :
       separation in transverse and perpendicular direction

    See Also
    --------
    honeycomb : honeycomb lattices
    graphene : graphene geometry
    nanoribbon : honeycomb nanoribbon
    graphene_nanoribbon : graphene nanoribbon
    zgnr : zigzag graphene nanoribbon
    cgnr : chiral graphene nanoribbon
    """
    return graphene_nanoribbon(width, bond, atoms, kind="armchair", vacuum=vacuum)


@set_module("sisl.geom")
def zgnr(
    width: int,
    bond: float = 1.42,
    atoms: Optional[AtomsLike] = None,
    vacuum: FloatOrFloat2 = 20.0,
) -> Geometry:
    r"""Construction of a zigzag graphene nanoribbon

    Parameters
    ----------
    width :
       number of atoms in the transverse direction
    bond :
       C-C bond length
    atoms :
       atom (or atoms) in the honeycomb lattice. Defaults to ``Atom(6)``
    vacuum :
       separation in transverse and perpendicular direction

    See Also
    --------
    honeycomb : honeycomb lattices
    graphene : graphene geometry
    nanoribbon : honeycomb nanoribbon
    graphene_nanoribbon : graphene nanoribbon
    agnr : armchair graphene nanoribbon
    cgnr : chiral graphene nanoribbon
    """
    return graphene_nanoribbon(width, bond, atoms, kind="zigzag", vacuum=vacuum)


@set_module("sisl.geom")
def cgnr(
    width: int,
    chirality: tuple[int, int],
    bond: float = 1.42,
    atoms: Optional[AtomsLike] = None,
    vacuum: FloatOrFloat2 = 20.0,
) -> Geometry:
    r"""Construction of an (n, m, w)-chiral graphene nanoribbon

    Parameters
    ----------
    width :
       number of atoms in the transverse direction
    chirality :
       index (n, m) corresponding to an edge with n zigzag segments followed by m armchair segments
    bond :
       C-C bond length
    atoms :
       atom (or atoms) in the honeycomb lattice. Defaults to ``Atom(6)``
    vacuum :
       separation in transverse and perpendicular direction

    See Also
    --------
    honeycomb : honeycomb lattices
    graphene : graphene geometry
    nanoribbon : honeycomb nanoribbon
    graphene_nanoribbon : graphene nanoribbon
    agnr : armchair graphene nanoribbon
    zgnr : zigzag graphene nanoribbon
    """
    return graphene_nanoribbon(
        width, bond, atoms, kind="chiral", vacuum=vacuum, chirality=chirality
    )


@set_module("sisl.geom")
@dataclass
class heteroribbon_section(CompositeGeometrySection):
    """
    Parameters
    ----------
    W: int
        The width of the section.
    L: int, optional
        The number of units of the section. Note that a "unit" is
        not a unit cell, but half of it. I.e. a transversal string of
        atoms.
    shift: int, optional
        The shift of this section with respect to the previous one.
        It can be both positive (upwards shift) or negative (downwards shift).
    align: {"bottom"/"b", "top"/"t", "center"/"c", "auto"/"a"}
        Indicates how the section should be aligned with respect to the
        previous one.

        If automatic alignment is requested, sections are aligned:
         - If both sections are odd: On their center.
         - If previous section is even: On its open edge (top or bottom)
         - If previous section is odd and incoming section is even: On the bottom.
    atoms: Atom
        Value to pass to the `atoms` argument of `nanoribbon`. If not provided,
        it defaults to the `atoms` argument passed to this function.
    bond: float
        The bond length of the ribbon.
    kind: {'armchair', 'zigzag'}
        The kind of ribbon that this section should be.
    vacuum :
        minimum separation in transverse direction
    shift_quantum: bool, optional
        Whether the implementation will assist avoiding lone atoms (< 2 neighbors).

        If ``False``, sections are just shifted (`shift`) number of atoms.

        If ``True``, shifts are quantized in the sense that shifts that produce
        lone atoms (< 2 neighbors) are ignored. Then:
            - ``shift = 0`` means aligned.
            - ``shift = -1`` means first possible downwards shift (if available).
            - ``shift = 1`` means first possible upwards shift (if available).
        If this is set to `True`, `on_lone_atom` is overwritten to `"raise"`.
    on_lone_atom: {'ignore', 'warn', 'raise'}
        What to do when a junction between sections produces lone atoms (< 2 neighbors)

        Messages contain hopefully useful explanations to understand what
        to do to fix it.
    invert_first: bool, optional
        Whether, if this is the first section, it should be inverted with respect
        to the one provided by `sisl.geom.nanoribbon`.
    """

    W: int
    L: int = 1
    shift: int = 0
    align: str = "bottom"
    atoms: AtomsLike = None
    bond: float = None
    kind: str = "armchair"
    vacuum: float = 20.0
    shift_quantum: bool = False
    on_lone_atom: str = field(default="ignore", repr=False)
    invert_first: bool = field(default=False, repr=False)

    def __post_init__(self):
        if self.kind in ("armchair", "zigzag"):
            self.long_ax, self.trans_ax = 0, 1
        else:
            raise ValueError("Unknown kind={kind}, must be one of zigzag or armchair")

        if self.shift_quantum:
            self.on_lone_atom = "raise"

        self._open_borders = [False, False]

    def _shift_unit_cell(self, geometry):
        """Changes the border used for a ribbon.

        It does so by shifting half unit cell. This must be done before any
        tiling of the geometry.
        """
        move = np.array([0.0, 0.0, 0.0])
        move[self.long_ax] = -geometry.xyz[self.W, self.long_ax]
        geometry = geometry.move(move)
        geometry.xyz = (geometry.fxyz % 1).dot(geometry.cell)
        return geometry

    def _align_offset(self, prev, new_xyz):
        """Helper function to align the sections.

        It returns the offset to apply to the incoming section in order to
        align it to the previous one.
        """
        align = self.align.lower()
        if prev is None:
            return align, 0

        W = self.W
        W_diff = W - prev.W
        if align in ("a", "auto"):
            if W % 2 == 1 and W_diff % 2 == 0:
                # Both ribbons are odd, so we align on the center
                align = "c"
            elif prev.W % 2 == 0:
                # The previous section is even, so it dictates how to align.
                # We should align on its open edge.
                align = {True: "t", False: "b"}[prev._open_borders[1]]
            else:
                # We have an incoming even section and we can align it however we wish.
                # We will align them on the bottom.
                align = "b"

        if align in ("c", "center"):
            if W_diff % 2 == 1:
                self._junction_error(
                    prev,
                    "Different parity sections can not be aligned by their centers",
                    "raise",
                )
            return (
                align,
                prev.xyz[:, self.trans_ax].mean() - new_xyz[:, self.trans_ax].mean(),
            )
        elif align in ("t", "top"):
            return (
                align,
                prev.xyz[:, self.trans_ax].max() - new_xyz[:, self.trans_ax].max(),
            )
        elif align in ("b", "bottom"):
            return (
                align,
                prev.xyz[:, self.trans_ax].min() - new_xyz[:, self.trans_ax].min(),
            )
        else:
            raise ValueError(
                f"Invalid value for 'align': {align}. Must be one of"
                " {'c', 'center', 't', 'top', 'b', 'bottom', 'a', 'auto'}"
            )

    def _offset_from_center(self, align, prev):
        align = align.lower()[0]
        W_diff = self.W - prev.W

        if align in ("t", "b"):
            # Number of atoms that hang out if we align on the center
            edge_offset = W_diff // 2

            # Now shift the limits.
            if align == "t":
                offset_sign = -1
            elif align == "b":
                offset_sign = 1
            return edge_offset * offset_sign

        return 0

    def build_section(self, prev):
        new_section = nanoribbon(
            bond=self.bond,
            atoms=self.atoms,
            width=self.W,
            kind=self.kind,
            vacuum=self.vacuum,
        )

        align, offset = self._align_offset(prev, new_section)

        if prev is not None:
            if not isinstance(prev, heteroribbon_section):
                self._junction_error(
                    prev,
                    f"{self.__class__.__name__} can not be appended to {type(prev).__name__}",
                    "raise",
                )
            if self.kind != prev.kind:
                self._junction_error(prev, f"Ribbons must be of same type.", "raise")
            if self.bond != prev.bond:
                self._junction_error(
                    prev, f"Ribbons must have same bond length.", "raise"
                )

            shift = self._parse_shift(self.shift, prev, align)

            # Get the distance of an atom shift. (sin(60) = 3**.5 / 2)
            atom_shift = self.bond * 3**0.5 / 2

            # if (last_W % 2 == 1 and W < last_W) and last_open:
            # _junction_error(i, "DANGLING BONDS: Previous odd section, which has an open end,"
            #     " is wider than the incoming one. A wider odd section must always"
            #     " have a closed end. You can solve this by making the previous section"
            #     " one unit smaller or larger (L = L +- 1).", on_lone_atom
            # )
            W_diff = self.W - prev.W

            # Check whether aligned sections naturally match
            if align in ("c", "center"):
                # When sections are aligned by the center or the top, it is very easy to check if
                # they match.
                aligned_match = (not prev._open_borders[1]) == (W_diff % 4 == 0)
            elif align in ("t", "top"):
                aligned_match = not prev._open_borders[1]
            elif align in ("b", "bottom"):
                # However, when sections are aligned by the bottom, it is a bit more complex.
                # This is because a "closed" even section means that its bottom edge is open.
                last_bot_edge_open = (prev.W % 2 == 1) == prev._open_borders[1]
                this_bot_edge_open = self.W % 2 == 0
                aligned_match = last_bot_edge_open == this_bot_edge_open

            # Shift the incoming section if the vertical shift makes them not match.
            if aligned_match == (shift % 2 == 1):
                new_section = self._shift_unit_cell(new_section)
                self._open_borders[0] = not self._open_borders[0]

            # Apply the offset that we have calculated.
            move = np.zeros(3)
            move[self.trans_ax] = offset + shift * atom_shift
            new_section = new_section.move(move)
        else:
            if self.invert_first:
                new_section = self._shift_unit_cell(new_section)
                self._open_borders[0] = not self._open_borders[0]

        # Check how many times we have to tile the unit cell (tiles) and whether
        # we have to cut the last string of atoms (cut_last)
        tiles, res = divmod(self.L + 1, 2)
        cut_last = res == 0

        # Tile the current section unit cell
        new_section = new_section.tile(tiles, self.long_ax)
        # Cut the last string of atoms.
        if cut_last:
            new_section.cell[0, 0] *= self.L / (self.L + 1)
            new_section = new_section.remove(
                {
                    "xy"[self.long_ax]: (
                        new_section.cell[self.long_ax, self.long_ax] - 0.01,
                        None,
                    )
                }
            )

        self._open_borders[1] = self._open_borders[0] != cut_last

        self.xyz = new_section.xyz
        return new_section

    def add_section(self, geom, new_section):
        # Avoid going out of the cell in the transversal direction
        new_min = new_section[:, self.trans_ax].min()
        new_max = new_section[:, self.trans_ax].max()
        if new_min < 0:
            cell_offset = -new_min + self.vacuum
            geom = geom.add_vacuum(cell_offset, self.trans_ax)
            move = np.zeros(3)
            move[self.trans_ax] = cell_offset
            geom = geom.move(move)
            new_section = new_section.move(move)
        if new_max > geom.cell[1, 1]:
            geom = geom.add_vacuum(
                new_max - geom.cell[self.trans_ax, self.trans_ax] + self.vacuum,
                self.trans_ax,
            )

        self.xyz = new_section.xyz
        # Finally, we can safely append the geometry.
        return geom.append(new_section, self.long_ax)

    def _parse_shift(self, shift, prev, align):
        if self.on_lone_atom == "ignore":
            return shift

        W = self.W

        # Check that we are not joining an open odd ribbon with
        # a smaller ribbon, since no matter what the shift is there will
        # always be dangling bonds.
        if (prev.W % 2 == 1 and W < prev.W) and prev._open_borders[1]:
            self._junction_error(
                prev,
                "LONE ATOMS: Previous odd section, which has an open end,"
                " is wider than the incoming one. A wider odd section must always"
                " have a closed end. You can solve this by making the previous section"
                " one unit smaller or larger (L = L +- 1).",
                self.on_lone_atom,
            )

        # Get the difference in width between the previous and this ribbon section
        W_diff = W - prev.W
        # And also the mod(4) because there are certain differences if the width differs
        # on 1, 2, 3 or 4 atoms. After that, the cycle just repeats (e.g. 5 == 1, etc).
        diff_mod = W_diff % 4

        # Calculate the shifts that are valid (don't leave atoms with less than 2 bonds)
        # This depends on several factors.
        if diff_mod % 2 == 0 and W % 2 == 1:
            # Both sections are odd

            if W < prev.W:
                # The incoming section is thinner than the last one. Note that at this point
                # we are sure that the last section has a closed border, otherwise we
                # would have raised an error. At this point, centers can differ by any
                # integer number of atoms without leaving dangling bonds.

                # Shift limits are a bit complicated and are different for even and odd shifts.
                # This is because a closed incoming section can shift until there is no connection
                # between ribbons, while an open one needs to stop before its edge goes outside
                # the previous section.
                shift_lims = {
                    "closed": prev.W // 2 + W // 2 - 2,
                    "open": prev.W // 2 - W // 2 - 1,
                }

                shift_pars = {lim % 2: lim for k, lim in shift_lims.items()}

                # Build an array with all the valid shifts.
                valid_shifts = np.sort(
                    [
                        *np.arange(0, shift_pars[0] + 1, 2),
                        *np.arange(1, shift_pars[1] + 1, 2),
                    ]
                )
                valid_shifts = np.array([*(np.flip(-valid_shifts)[:-1]), *valid_shifts])

                # Update the valid shift limits if the sections are aligned on any of the edges.
                shift_offset = self._offset_from_center(align, prev)
                valid_shifts -= shift_offset
            elif prev.W == W:
                valid_shifts = np.array([0])
            else:
                # At this point, we already know that the incoming section is wider and
                # therefore it MUST have a closed start, otherwise there will be dangling bonds.
                if (
                    diff_mod == 2
                    and prev._open_borders[1]
                    or diff_mod == 0
                    and not prev._open_borders[1]
                ):
                    # In these cases, centers must match or differ by an even number of atoms.
                    # And this is the limit for the shift from the center.
                    shift_lim = ((W_diff // 2) // 2) * 2
                else:
                    # Otherwise, centers must differ by an odd number of atoms.
                    # And these are the limits for the shift from the center
                    if prev._open_borders[1]:
                        # To avoid the current open section leaving dangling bonds.
                        shift_lim = (W_diff // 2) - 1
                    else:
                        # To avoid sections being disconnected.
                        shift_lim = W_diff // 2 + ((prev.W // 2) - 1) * 2

                # Update the shift limits if the sections are aligned on any of the edges.
                shift_offset = self._offset_from_center(align, prev)

                # Apply the offsets and calculate the maximum and minimum shifts.
                min_shift, max_shift = (
                    -shift_lim - shift_offset,
                    shift_lim - shift_offset,
                )

                valid_shifts = np.arange(min_shift, max_shift + 1, 2)
        else:
            # There is at least one even section.

            # We have to make sure that the open edge of the even ribbons (however
            # many there are) is always shifted towards the center. Shifting in the
            # opposite direction would result in dangling bonds.

            # We will calculate all the valid shifts from a bottom alignment perspective.
            # Then convert if needed.
            special_shifts = []

            if diff_mod % 2 == 0:
                # Both ribbons are even
                if prev._open_borders[1]:
                    special_shifts = [prev.W - W]
                    min_shift = prev.W - W + 1
                    max_shift = prev.W - 1
                else:
                    special_shifts = [0]
                    min_shift = -W + 1
                    max_shift = -1

            elif W % 2 == 1:
                # Last section was even, incoming section is odd.
                if W < prev.W:
                    if prev._open_borders[1]:
                        special_shifts = [prev.W - W]
                        min_shift = prev.W - W
                        max_shift = prev.W - W + 1 + ((W - 2) // 2) * 2
                    else:
                        special_shifts = [0]
                        min_shift = -1 - ((W - 2) // 2) * 2
                        max_shift = -1
                else:
                    if prev._open_borders[1]:
                        min_shift = 0
                        max_shift = prev.W - 2
                    else:
                        max_shift = -1
                        min_shift = -(W - 2)
            else:
                # Last section was odd, incoming section is even.
                if prev._open_borders[1]:
                    special_shifts = [0, prev.W - W]
                    min_shift = None
                else:
                    min_shift = [1, -(W - 2)]
                    max_shift = [prev.W - 2, prev.W - W - 1]

            # We have gone over all possible situations, now just build the
            # array of valid shifts.
            valid_shifts = [*special_shifts]
            if isinstance(min_shift, int):
                valid_shifts.extend(np.arange(min_shift, max_shift + 1, 2))
            elif isinstance(min_shift, list):
                for m, mx in zip(min_shift, max_shift):
                    valid_shifts.extend(np.arange(m, mx + 1, 2))

            # Apply offset on shifts based on the actual alignment requested
            # for the sections.
            shift_offset = 0
            if align[0] == "t":
                shift_offset = W_diff
            elif align[0] == "c":
                shift_offset = -self._offset_from_center("b", prev)
            valid_shifts = np.array(valid_shifts) + shift_offset

        # Finally, check if the provided shift value is valid or not.
        valid_shifts = np.sort(valid_shifts)
        if self.shift_quantum:
            n_valid_shifts = len(valid_shifts)
            # Find out if it is possible to perfectly align.
            if np.any(valid_shifts == 0):
                aligned_shift = np.where(valid_shifts == 0)[0][0]
            else:
                # If not, we have to find the smallest shift.
                # What flip does is prioritize upwards shifts.
                # That is, if both "-1" and "1" shifts are valid,
                # "1" will be picked as the reference.
                aligned_shift = (
                    n_valid_shifts - 1 - np.argmin(np.abs(np.flip(valid_shifts)))
                )

            # Calculate which index we really need to retrieve
            corrected_shift = aligned_shift + shift

            if corrected_shift < 0 or corrected_shift >= n_valid_shifts:
                self._junction_error(
                    prev,
                    f"LONE ATOMS: Shift must be between {-aligned_shift}"
                    f" and {n_valid_shifts - aligned_shift - 1}, but {shift} was provided.",
                    self.on_lone_atom,
                )

            # And finally get the shift value
            shift = valid_shifts[corrected_shift]
        else:
            if shift not in valid_shifts:
                self._junction_error(
                    prev,
                    f"LONE ATOMS: Shift must be one of {valid_shifts}"
                    f" but {shift} was provided.",
                    self.on_lone_atom,
                )

        return shift


@set_module("sisl.geom")
def heteroribbon(sections, section_cls=heteroribbon_section, **kwargs) -> Geometry:
    """Build a nanoribbon consisting of several nanoribbons of different widths.

    This function uses `composite_geometry`, but defaulting to the usage
    of `heteroribbon.section` as the section class.

    See `heteroribbon.section` and `composite_geometry` for arguments.

    Returns
    -------
    Geometry:
        The final structure of the heteroribbon.

    Notes
    -----
    It only works for armchair ribbons for now.

    Examples
    --------
    >>> # A simple 7-11-AGNR with the sections aligned on the center
    >>> heteroribbon([(7, 2), (11, 2)], bond=1.42, atoms="C")
    >>> # The same AGNR but shifted up
    >>> heteroribbon([(7, 2), (11, 2, 1)], bond=1.42, atoms="C")
    >>> # And down
    >>> heteroribbon([(7, 2), (11, 2, -1)], bond=1.42, atoms="C")
    >>> # The same AGNR but with a bigger 11 section and a 9-atom bridge
    >>> heteroribbon([(7, 1), (9,1), (11, 4), (9,1), (7,1)], bond=1.42, atoms="C")
    >>> # One that you have probably never seen before
    >>> heteroribbon([(7, 1j), (10, 2), (9, 1), (8, 2j, -1)], bond=1.42, atoms="C")

    See also
    --------
    composite_geometry: Underlying method used to build the heteroribbon.
    heteroribbon_section: The class that describes each section.
    nanoribbon : method used to create the segments
    graphene_heteroribbon: same as this function, but with defaults for graphene GNR's
    """
    return composite_geometry(sections, section_cls=section_cls, **kwargs)


heteroribbon.section = heteroribbon_section


@set_module("sisl.geom")
def graphene_heteroribbon(
    sections,
    section_cls=heteroribbon_section,
    bond: float = 1.42,
    atoms: Optional[AtomsLike] = None,
    **kwargs,
) -> Geometry:
    """Build a graphene nanoribbon consisting of several nanoribbons of different widths

    Please see `heteroribbon` for arguments, the only difference is that the `bond` and `atoms`
    arguments default to ``bond=1.42`` and ``Atoms(Z=6, R=bond*1.01)``, respectively.

    See Also
    --------
    heteroribbon : for argument details and how it behaves
    """
    if atoms is None:
        atoms = Atom(Z=6, R=bond * 1.01)
    return composite_geometry(
        sections, section_cls=section_cls, bond=bond, atoms=atoms, **kwargs
    )


graphene_heteroribbon.section = heteroribbon_section
