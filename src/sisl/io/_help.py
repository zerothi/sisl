# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Sequence
from re import compile as re_compile
from typing import Optional, Union

import numpy as np

from sisl import Atoms, AtomUnknown, Geometry
from sisl.messages import warn

__all__ = ["starts_with_list", "header_to_dict", "grid_reduce_indices", "parse_order"]
__all__ += ["_fill_basis_empty", "_replace_basis"]


def starts_with_list(l, comments):
    for comment in comments:
        if l.strip().startswith(comment):
            return True
    return False


def header_to_dict(header):
    """Convert a header line with 'key=val key1=val1' sequences to a single dictionary"""
    e = re_compile(r"(\S+)=")

    # 1. Remove *any* entry with 0 length
    # 2. Ensure it is a list
    # 3. Reverse the list order (for popping)
    kv = list(filter(lambda x: len(x.strip()) > 0, e.split(header)))[::-1]

    # Now create the dictionary
    d = {}
    while len(kv) >= 2:
        # We have reversed the list
        key = kv.pop().strip(" =")  # remove white-space *and* =
        val = kv.pop().strip()  # remove outer whitespace
        d[key] = val

    return d


def grid_reduce_indices(grids, factors, axis=0, out=None):
    """Reduce `grids` into a single `grid` value along `axis` by summing the `factors`

    If `out` is defined the data will be stored there.
    """
    if len(factors) > grids.shape[axis]:
        raise ValueError(
            f"Trying to reduce a grid with too many factors: {len(factors)} > {grids.shape[axis]}"
        )

    if out is not None:
        grid = out
        np.take(grids, 0, axis=axis, out=grid)
        grid *= factors[0]
    else:
        grid = np.take(grids, 0, axis=axis) * factors[0]

    for idx, factor in enumerate(factors[1:], start=1):
        grid += np.take(grids, idx, axis=axis) * factor

    return grid


def _listify_str(arg):
    if isinstance(arg, str):
        return [arg]
    return arg


def parse_order(
    order: Optional[Union[str, Sequence[str]]],
    choice_dict="",
    choice=None,
    case: bool = False,
):
    """Converts `order` in to a proper order list, depending on the `output` value

    Can sort through `order` by removing those from ``choice_dict[choice]`` if prefixed with ``^``.

    If `order` is not present, it will return ``choice_dict[choice]``.

    If any elements in `order` is ^name, then all `name` will be removed from the order list.
    This enables one to remove some from the default order elements.

    For instance:

    >>> read_geometry(order="^fdf", output=True)

    where internally the `order` is parsed as:

    >>> order = parse_order(order, {True: ["fdf", "TSHS", "nc"], False: []}, choice)

    then the `order` on return will retain only ``[TSHS, nc]``

    Parameters
    ----------
    order : str or list of str
        some kind of specifier that is used in the `Sile` to determine how to parse data
    choice_dict :
        dictionary of values that will be chosen via `choice`
    choice :
        a hashable variable used to extract from `choice_dict`
    case:
        if `True`, do not lower case
    """
    if choice is None:
        choice = 1
        choice_dict = {1: choice_dict}
    if order is None:
        return _listify_str(choice_dict[choice])

    # now handle the cases where the users wants to not use something
    order = _listify_str(order)

    rem = []
    for el in order:
        if el.startswith("^"):
            if case:
                rem.append(el)
                rem.append(el[1:])
            else:
                rem.append(el.lower())
                rem.append(el[1:].lower())

    if rem:
        order.extend(_listify_str(choice_dict[choice]))
        if case:
            order = [el for el in order if el not in rem]
        else:
            order = [el for el in order if el.lower() not in rem]

    return order


def _fill_basis_empty(sp: np.ndarray, basis: list[Atom], start_Z=1000) -> Atoms:
    """Adds atoms in `basis` with ``AtomUnknown(start_Z + species_idx)``

    This is useful when one does not have the required atom in the basis,
    but one knows that the species are existing.

    Parameters
    ----------
    basis:
        the basis to insert *empty* atoms into
    sp:
        the array of species indices
    start_Z:
        the offset used in AtomUnknown
    """
    n_species = sp.max() + 1

    only_basis = False
    if isinstance(basis, Atoms):
        # for atoms objects that holds the reference
        # basis, it *must* be complete.
        # Hence we extract the species list, and specifies
        # it as the *only* basis option.
        only_basis = True
        n_species = max(n_species, len(basis.atom))
        basis = basis.atom
    else:
        # it *can* be the correct basis, in case there are more basis
        # atoms than actually used, but fewer than the actual number
        # of atoms
        only_basis = n_species <= len(basis) and len(basis) < len(sp)

    if only_basis:
        n_species = max(n_species, len(basis))

    # Retrieve the first atom object
    # This is important to initialize the Atoms object with the
    # correct species index
    def get_atom(basis, isp, idx=None):
        nonlocal sp
        if idx is None:
            idx = (sp == isp).nonzero()[0]

        if only_basis:
            return basis[isp]
        elif len(idx) > 0:
            return basis[idx[0]]
        return AtomUnknown(start_Z + isp)

    atoms = Atoms(get_atom(basis, 0), na=len(sp))

    for isp in range(n_species):
        idx = (sp == isp).nonzero()[0]
        atoms[idx] = get_atom(basis, isp, idx)

    return atoms


def _replace_basis(basis: Atoms, ref_basis: Union[Atoms, Geometry]) -> None:
    """Replace the `basis` with the atoms in `ref_basis`"""
    if isinstance(ref_basis, Geometry):
        ref_basis = ref_basis.atoms

    only_basis = len(basis) == len(ref_basis.atom)
    if not only_basis:
        # it *can* be the correct basis, in case there are more basis
        # atoms than actually used, but fewer than the actual number
        # of atoms
        only_basis = len(ref_basis.atom) < len(basis) and len(basis) < len(ref_basis)

    # Try and replace stuff
    # We *must* assume that the species are aligned
    for (atom_in, in_idx), (atom_ref, ref_idx) in zip(
        basis.iter(True), ref_basis.iter(True)
    ):
        # Check if the indices are the same
        if len(in_idx) > 0 and (np.allclose(in_idx, ref_idx) or only_basis):
            if atom_in.Z == atom_ref.Z:
                # replace
                basis.replace_atom(atom_in, atom_ref)
        elif len(in_idx) == 0 and (len(ref_idx) == 0 or only_basis):
            # replace because it is a missing
            basis.replace_atom(atom_in, atom_ref)
        elif not only_basis:
            warn(
                f"Trying to replace atom {atom_in!r} with {atom_ref!r}, but they don't share the same atoms in the geometry."
            )
