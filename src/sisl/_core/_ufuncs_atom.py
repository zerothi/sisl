# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from sisl import _array as _a
from sisl._ufuncs import register_sisl_dispatch
from sisl.typing import SimpleIndex

from .atom import Atom, Atoms

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(Atoms, module="sisl")
def sub(atom: Atoms, atoms: SimpleIndex) -> Atoms:
    """Return a subset of the list"""
    atoms = _a.asarray(atoms).ravel()
    new_atoms = Atoms()
    new_atoms._atom = atom._atom[:]
    new_atoms._species = atom._species[atoms]
    new_atoms._update_orbitals()
    return new_atoms


@register_sisl_dispatch(Atoms, module="sisl")
def remove(atom: Atoms, atoms: SimpleIndex) -> Atoms:
    """Remove a set of atoms"""
    atoms = _a.asarray(atoms).ravel()
    idx = np.setdiff1d(np.arange(len(atom)), atoms, assume_unique=True)
    return atom.sub(idx)


@register_sisl_dispatch(Atoms, module="sisl")
def tile(atom: Atoms, reps: int) -> Atoms:
    """Tile this atom object"""
    atoms = atom.copy()
    atoms._species = np.tile(atoms.species, reps)
    atoms._update_orbitals()
    return atoms


@register_sisl_dispatch(Atoms, module="sisl")
def repeat(atom: Atoms, reps: int) -> Atoms:
    """Repeat this atom object"""
    atoms = atom.copy()
    atoms._species = np.repeat(atoms.species, reps)
    atoms._update_orbitals()
    return atoms


@register_sisl_dispatch(Atoms, module="sisl")
def swap(atom: Atoms, atoms1: SimpleIndex, atoms2: SimpleIndex) -> Atoms:
    """Swaps atoms by index"""
    a = _a.asarray(atoms1)
    b = _a.asarray(atoms2)
    atoms = atom.copy()
    spec = np.copy(atoms.species)
    atoms._species[a] = spec[b]
    atoms._species[b] = spec[a]
    atoms._update_orbitals()
    return atoms


@register_sisl_dispatch(Atoms, module="sisl")
def append(atom: Atoms, other: Union[Atom, Atoms]) -> Atoms:
    """Append `other` to this list of atoms and return the appended version

    Parameters
    ----------
    other :
       new atoms to be added

    Returns
    -------
    Atoms
        merging of this objects atoms and the `other` objects atoms.
    """
    if not isinstance(other, Atoms):
        other = Atoms(other)

    atoms = atom.copy()
    spec = np.copy(other.species)
    for i, atom in enumerate(other.atom):
        try:
            s = atoms.species_index(atom)
        except KeyError:
            s = len(atoms.atom)
            atoms._atom.append(atom)
        spec = np.where(other.species == i, s, spec)
    atoms._species = np.concatenate((atoms.species, spec))
    atoms._update_orbitals()
    return atoms


add = append
add.__name__ = "add"
register_sisl_dispatch(Atoms, module="sisl")(add)


@register_sisl_dispatch(Atoms, module="sisl")
def prepend(atom: Atoms, other: Union[Atom, Atoms]) -> Atoms:
    if not isinstance(other, Atoms):
        other = Atoms(other)
    return other.append(atom)


@register_sisl_dispatch(Atoms, module="sisl")
def insert(atom: Atoms, index: SimpleIndex, other: Union[Atom, Atoms]) -> Atoms:
    """Insert other atoms into the list of atoms at index"""
    if isinstance(other, Atom):
        other = Atoms(other)
    else:
        other = other.copy()

    # Create a copy for insertion
    atoms = atom.copy()

    spec = other.species
    for i, atom in enumerate(other.atom):
        if atom not in atoms:
            s = len(atoms.atom)
            atoms._atom.append(atom)
        else:
            s = atoms.species_index(atom)
        spec = np.where(spec == i, s, spec)
    atoms._species = np.insert(atoms.species, index, spec)
    atoms._update_orbitals()
    return atoms


@register_sisl_dispatch(Atoms, module="sisl")
def scale(atoms: Atoms, scale: float) -> Atoms:
    """Scale the atomic radii and return an equivalent atom.

    Parameters
    ----------
    scale :
       the scale factor for the atomic radii
    """
    out = Atoms()
    out._atom = [a.scale(scale) for a in atoms.atom]
    out._species = np.copy(atoms.species)
    return out


@register_sisl_dispatch(Atoms, module="sisl")
def copy(atoms: Atoms) -> Atoms:
    """Return a copy of this atom"""
    out = Atoms()
    out._atom = [a.copy() for a in atoms._atom]
    out._species = np.copy(atoms.species)
    out._update_orbitals()
    return out


@register_sisl_dispatch(Atom, module="sisl")
def sub(atom: Atom, orbitals: SimpleIndex) -> Atom:
    """Return the same atom with only a subset of the orbitals present

    Parameters
    ----------
    orbitals :
       indices of the orbitals to retain

    Returns
    -------
    Atom
        with only the subset of orbitals

    Raises
    ------
    ValueError
       if the number of orbitals removed is too large or some indices are outside the allowed range
    """
    orbitals = _a.asarray(orbitals).ravel()
    if len(orbitals) > atom.no:
        raise ValueError(
            f"{atom.__class__.__name__}.sub tries to remove more than the number of orbitals on an atom."
        )
    if np.any(orbitals >= atom.no):
        raise ValueError(
            f"{atom.__class__.__name__}.sub tries to remove a non-existing orbital io > no."
        )

    orbs = [atom.orbitals[o].copy() for o in orbitals]
    return atom.copy(orbitals=orbs)


@register_sisl_dispatch(Atom, module="sisl")
def remove(atom: Atom, orbitals: SimpleIndex) -> Atom:
    """Return the same atom without a specific set of orbitals

    Parameters
    ----------
    orbitals :
       indices of the orbitals to remove

    Returns
    -------
    Atom
        without the specified orbitals

    See Also
    --------
    Atom.sub : retain a selected set of orbitals
    """
    orbs = np.delete(_a.arangei(atom.no), orbitals)
    return atom.sub(orbs)


@register_sisl_dispatch(Atom, module="sisl")
def scale(atom: Atom, scale: float) -> Atom:
    """Scale the atomic radii and return an equivalent atom.

    Parameters
    ----------
    scale :
       the scale factor for the atomic radii
    """
    new = atom.copy()
    new._orbitals = [o.scale(scale) for o in atom.orbitals]
    return new


@register_sisl_dispatch(Atom, module="sisl")
def copy(
    atom: Atom,
    Z: Optional[Union[int, str]] = None,
    orbitals=None,
    mass: Optional[float] = None,
    tag: Optional[str] = None,
) -> Atom:
    """Return copy of this object"""
    if orbitals is None:
        orbitals = [orb.copy() for orb in atom]
    return atom.__class__(
        atom.Z if Z is None else Z,
        orbitals,
        atom.mass if mass is None else mass,
        atom.tag if tag is None else tag,
    )
