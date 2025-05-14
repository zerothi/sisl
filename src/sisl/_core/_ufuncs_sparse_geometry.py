# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from functools import partial

import numpy as np

from sisl import _array as _a
from sisl._ufuncs import register_sisl_dispatch
from sisl.typing import AtomsIndex

from .sparse import SparseCSR, _ncol_to_indptr, _to_cd
from .sparse_geometry import SparseAtom, SparseOrbital, _SparseGeometry

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(_SparseGeometry, module="sisl")
def copy(S: _SparseGeometry, dtype=None) -> _SparseGeometry:
    """A copy of this object

    Parameters
    ----------
    dtype : numpy.dtype, optional
       it is possible to convert the data to a different data-type
       If not specified, it will use ``self.dtype``
    """
    if dtype is None:
        dtype = S.dtype
    new = S.__class__(S.geometry.copy(), S.dim, dtype, 1, **S._cls_kwargs())
    # Be sure to copy the content of the SparseCSR object
    new._csr = S._csr.copy(dtype=dtype)
    return new


@register_sisl_dispatch(_SparseGeometry, module="sisl")
def swap(
    S: _SparseGeometry, atoms_a: AtomsIndex, atoms_b: AtomsIndex
) -> _SparseGeometry:
    """Swaps atoms in the sparse geometry to obtain a new order of atoms

    This can be used to reorder elements of a geometry.

    Parameters
    ----------
    atoms_a :
         the first list of atomic coordinates
    atoms_b :
         the second list of atomic coordinates
    """
    atoms_a = S.geometry._sanitize_atoms(atoms_a)
    atoms_b = S.geometry._sanitize_atoms(atoms_b)
    # Create full index list
    full = _a.arange(len(S.geometry))
    # Regardless of whether swapping or new indices are requested
    # this should work.
    full[atoms_a] = atoms_b
    full[atoms_b] = atoms_a
    return S.sub(full)


@register_sisl_dispatch(SparseAtom, module="sisl")
def tile(SA: SparseAtom, reps: int, axis: int) -> SparseAtom:
    """Create a tiled sparse atom object, equivalent to `Geometry.tile`

    The already existing sparse elements are extrapolated
    to the new supercell by repeating them in blocks like the coordinates.

    Notes
    -----
    Calling this routine will automatically `finalize` the `SparseAtom`. This
    is required to greatly increase performance.

    Parameters
    ----------
    reps :
        number of repetitions along cell-vector `axis`
    axis:
        0, 1, 2 according to the cell-direction

    See Also
    --------
    SparseAtom.repeat: a different ordering of the final geometry
    SparseAtom.untile : opposite of this method
    Geometry.tile: the same ordering as the final geometry
    Geometry.repeat: a different ordering of the final geometry
    """
    # Create the new sparse object
    g = SA.geometry.tile(reps, axis)
    S = SA.__class__(g, SA.dim, SA.dtype, 1, **SA._cls_kwargs())

    # Now begin to populate it accordingly
    # Retrieve local pointers to the information
    # regarding the current Hamiltonian sparse matrix
    geom = SA.geometry
    na = np.int32(SA.na)
    csr = SA._csr
    col, D = _to_cd(csr)

    # Information for the new Hamiltonian sparse matrix
    na_n = np.int32(S.na)
    geom_n = S.geometry

    # First loop on axis tiling and local
    # atoms in the geometry
    sc_index = geom_n.sc_index

    # Create new indptr, indices and D
    indptr = _ncol_to_indptr(np.tile(csr.ncol, reps))
    indices = _a.emptyi([indptr[-1]])
    indices.shape = (reps, -1)

    # Now we should fill the data
    isc = geom.a2isc(col)
    # resulting atom in the new geometry (without wrapping
    # for correct supercell, that will happen below)
    JA = col % na + na * isc[:, axis]

    # Create repetitions
    for rep in range(reps):
        # Correct the supercell information
        isc[:, axis], mod = np.divmod(JA, na_n)

        indices[rep, :] = mod + sc_index(isc) * na_n

        # Step atoms
        JA += na

    # Clean-up
    del isc, JA

    S._csr = SparseCSR(
        (np.tile(D, (reps, 1)), indices.ravel(), indptr),
        shape=(geom_n.na, geom_n.na_s),
    )

    return S


@register_sisl_dispatch(SparseAtom, module="sisl")
def repeat(SA: SparseAtom, reps: int, axis: int) -> SparseAtom:
    """Create a repeated sparse atom object, equivalent to `Geometry.repeat`

    The already existing sparse elements are extrapolated
    to the new supercell by repeating them in blocks like the coordinates.

    Parameters
    ----------
    reps :
        number of repetitions along cell-vector `axis`
    axis :
        0, 1, 2 according to the cell-direction

    See Also
    --------
    Geometry.repeat: the same ordering as the final geometry
    Geometry.tile: a different ordering of the final geometry
    SparseAtom.tile: a different ordering of the final geometry
    """
    # Create the new sparse object
    g = SA.geometry.repeat(reps, axis)
    S = SA.__class__(g, SA.dim, SA.dtype, 1, **SA._cls_kwargs())

    # Now begin to populate it accordingly
    # Retrieve local pointers to the information
    # regarding the current Hamiltonian sparse matrix
    geom = SA.geometry
    na = np.int32(SA.na)
    csr = SA._csr
    col, D = _to_cd(csr)
    ncol = csr.ncol

    # Information for the new Hamiltonian sparse matrix
    na_n = np.int32(S.na)
    geom_n = S.geometry

    # First loop on axis tiling and local
    # atoms in the geometry
    sc_index = geom_n.sc_index

    # Create new indptr, indices and D
    # Now indptr is complete
    indptr = _ncol_to_indptr(np.repeat(ncol, reps))
    indices = _a.emptyi([indptr[-1]])

    # Now we should fill the data
    isc = geom.a2isc(col)
    # resulting atom in the new geometry (without wrapping
    # for correct supercell, that will happen below)
    JA = (col % na) * reps
    # Get the offset atoms
    A = isc[:, axis] - 1

    for rep in range(reps):
        # Update the offset
        A += 1
        # Correct supercell information
        isc[:, axis], mod = np.divmod(A, reps)

        # Create the indices for the repetition
        idx = _a.array_arange(indptr[rep:-1:reps], n=ncol)
        indices[idx] = JA + mod + sc_index(isc) * na_n

    # Clean-up
    del isc, JA, A, idx

    # In the repeat we have to tile individual atomic couplings
    # So we should split the arrays and tile them individually
    # Now D is made up of D values, per atom
    if geom.na == 1:
        D = np.tile(D, (reps, 1))
    else:
        ntile = partial(np.tile, reps=(reps, 1))
        D = np.vstack(tuple(map(ntile, np.split(D, _a.cumsumi(ncol[:-1]), axis=0))))

    S._csr = SparseCSR((D, indices, indptr), shape=(geom_n.na, geom_n.na_s))

    return S


@register_sisl_dispatch(SparseOrbital, module="sisl")
def tile(SO: SparseOrbital, reps: int, axis: int) -> SparseOrbital:
    """Create a tiled sparse orbital object, equivalent to `Geometry.tile`

    The already existing sparse elements are extrapolated
    to the new supercell by repeating them in blocks like the coordinates.

    Parameters
    ----------
    reps :
        number of repetitions along cell-vector `axis`
    axis :
        0, 1, 2 according to the cell-direction

    See Also
    --------
    SparseOrbital.repeat: a different ordering of the final geometry
    SparseOrbital.untile : opposite of this method
    Geometry.tile: the same ordering as the final geometry
    Geometry.repeat: a different ordering of the final geometry
    """
    # Create the new sparse object
    g = SO.geometry.tile(reps, axis)
    S = SO.__class__(g, SO.dim, SO.dtype, 1, **SO._cls_kwargs())

    # Now begin to populate it accordingly
    # Retrieve local pointers to the information
    # regarding the current Hamiltonian sparse matrix
    geom = SO.geometry
    no = np.int32(SO.no)
    csr = SO._csr
    col, D = _to_cd(csr)
    ncol = csr.ncol

    # Information for the new Hamiltonian sparse matrix
    no_n = np.int32(S.no)
    geom_n = S.geometry

    # First loop on axis tiling and local
    # atoms in the geometry
    sc_index = geom_n.sc_index

    # Create new indptr, indices and D
    indptr = _ncol_to_indptr(np.tile(ncol, reps))
    indices = _a.emptyi([indptr[-1]])
    indices.shape = (reps, -1)

    # Now we should fill the data
    isc = geom.o2isc(col)
    # resulting atom in the new geometry (without wrapping
    # for correct supercell, that will happen below)
    JO = col % no + no * isc[:, axis]

    # Create repetitions
    for rep in range(reps):
        # Correct the supercell information
        isc[:, axis], mod = np.divmod(JO, no_n)

        indices[rep, :] = mod + sc_index(isc) * no_n

        # Step orbitals
        JO += no

    # Clean-up
    del isc, JO

    S._csr = SparseCSR(
        (np.tile(D, (reps, 1)), indices.ravel(), indptr),
        shape=(geom_n.no, geom_n.no_s),
    )

    return S


@register_sisl_dispatch(SparseOrbital, module="sisl")
def repeat(SO: SparseOrbital, reps: int, axis: int) -> SparseOrbital:
    """Create a repeated sparse orbital object, equivalent to `Geometry.repeat`

    The already existing sparse elements are extrapolated
    to the new supercell by repeating them in blocks like the coordinates.

    Parameters
    ----------
    reps :
        number of repetitions along cell-vector `axis`
    axis :
        0, 1, 2 according to the cell-direction

    See Also
    --------
    Geometry.repeat: the same ordering as the final geometry
    Geometry.tile: a different ordering of the final geometry
    SparseOrbital.tile: a different ordering of the final geometry
    """
    # Create the new sparse object
    g = SO.geometry.repeat(reps, axis)
    S = SO.__class__(g, SO.dim, SO.dtype, 1, **SO._cls_kwargs())

    # Now begin to populate it accordingly
    # Retrieve local pointers to the information
    # regarding the current Hamiltonian sparse matrix
    geom = SO.geometry
    no = np.int32(SO.no)
    csr = SO._csr
    col, D = _to_cd(csr)

    # Information for the new Hamiltonian sparse matrix
    no_n = np.int32(S.no)
    geom_n = S.geometry

    # First loop on axis tiling and local
    # orbitals in the geometry
    sc_index = geom_n.sc_index

    # Create new indptr, indices and D
    idx = np.repeat(geom.firsto, reps)
    idx = _a.array_arange(idx[:-reps], idx[reps:])
    # This will repeat the ncol elements in correct order
    ncol = csr.ncol[idx]
    # Now indptr is complete
    indptr = _ncol_to_indptr(ncol)

    # Note that D above is already reduced to a *finalized* state
    # So we have to re-create the reduced index pointer
    # Then we take repeat the data by smart indexing
    D = D[_a.array_arange(_ncol_to_indptr(csr.ncol)[idx], n=ncol), :]
    del ncol, idx
    indices = _a.emptyi([indptr[-1]])

    # Now we should fill the data
    isc = geom.o2isc(col)
    # resulting orbital in the new geometry (without wrapping
    # for correct supercell, that will happen below)
    JO = col % no
    # Get number of orbitals per atom (lasto - firsto + 1)
    # This is faster than the direct call

    ja = geom.o2a(JO)
    oJ = geom.firsto[ja]
    oA = geom.lasto[ja] + 1 - oJ
    # Shift the orbitals corresponding to the
    # repetitions of all previous atoms
    JO += oJ * (reps - 1)
    # Get the offset orbitals
    O = isc[:, axis] - 1
    # We need to create and indexable atomic array
    # This is required for multi-orbital cases where
    # we should tile atomic orbitals, and repeat the atoms (only).
    # 'A' is now the first (non-repeated) atom in the new structure
    A = _a.arangei(geom.na) * reps
    AO = geom_n.lasto[A] - geom_n.firsto[A] + 1
    # subtract AO for first iteration in repetition loop
    OA = geom_n.firsto[A] - AO

    # Clean
    del ja, oJ, A

    # Get view of ncol
    ncol = SO._csr.ncol

    # Create repetitions
    for _ in range(reps):
        # Update atomic offset
        OA += AO
        # Update the offset
        O += 1
        # Correct supercell information
        isc[:, axis], mod = np.divmod(O, reps)

        # Create the indices for the repetition
        idx = _a.array_arange(indptr[_a.array_arange(OA, n=AO)], n=ncol)
        indices[idx] = JO + oA * mod + sc_index(isc) * no_n

    # Clean-up
    del isc, JO, O, OA, AO, idx

    # In the repeat we have to tile individual atomic couplings
    # So we should split the arrays and tile them individually
    S._csr = SparseCSR((D, indices, indptr), shape=(geom_n.no, geom_n.no_s))

    return S


@register_sisl_dispatch(SparseAtom, module="sisl")
def sub(SA: SparseAtom, atoms: AtomsIndex) -> SparseAtom:
    """Create a subset of this sparse matrix by only retaining the elements corresponding to the `atoms`

    Indices passed *MUST* be unique.

    Negative indices are wrapped and thus works.

    Parameters
    ----------
    atoms :
        indices of retained atoms

    See Also
    --------
    Geometry.remove : the negative of `Geometry.sub`
    Geometry.sub : equivalent to the resulting `Geometry` from this routine
    SparseAtom.remove : the negative of `sub`, i.e. remove a subset of atoms
    """
    atoms = SA.asc2uc(atoms)
    geom = SA.geometry.sub(atoms)

    idx = np.tile(atoms, SA.n_s)
    # Use broadcasting rules
    idx.shape = (SA.n_s, -1)
    idx += (_a.arangei(SA.n_s) * SA.na).reshape(-1, 1)
    idx.shape = (-1,)

    # Now create the new sparse orbital class
    S = SA.__class__(geom, SA.dim, SA.dtype, 1, **SA._cls_kwargs())
    S._csr = SA._csr.sub(idx)

    return S


@register_sisl_dispatch(SparseOrbital, module="sisl")
def sub(SO: SparseOrbital, atoms: AtomsIndex) -> SparseOrbital:
    """Create a subset of this sparse matrix by only retaining the atoms corresponding to `atoms`

    Negative indices are wrapped and thus works, supercell atoms are also wrapped to the unit-cell.

    Parameters
    ----------
    atoms :
        indices of retained atoms or `Atom` for retaining only *that* atom

    Examples
    --------

    >>> obj = SparseOrbital(...)
    >>> obj.sub(1) # only retain the second atom in the SparseGeometry
    >>> obj.sub(obj.atoms.atom[0]) # retain all atoms which is equivalent to
    >>>                            # the first atomic specie

    See Also
    --------
    Geometry.remove : the negative of `Geometry.sub`
    Geometry.sub : equivalent to the resulting `Geometry` from this routine
    SparseOrbital.remove : the negative of `sub`, i.e. remove a subset of atoms
    """
    atoms = SO.asc2uc(atoms)
    orbs = SO.a2o(atoms, all=True)
    geom = SO.geometry.sub(atoms)

    idx = np.tile(orbs, SO.n_s)
    # Use broadcasting rules
    idx.shape = (SO.n_s, -1)
    idx += (_a.arangei(SO.n_s) * SO.no).reshape(-1, 1)
    idx.shape = (-1,)

    # Now create the new sparse orbital class
    S = SO.__class__(geom, SO.dim, SO.dtype, 1, **SO._cls_kwargs())
    S._csr = SO._csr.sub(idx)

    return S


@register_sisl_dispatch(SparseAtom, module="sisl")
@register_sisl_dispatch(SparseOrbital, module="sisl")
def remove(S: _SparseGeometry, atoms: AtomsIndex) -> _SparseGeometry:
    """Create a subset of this sparse matrix by removing the atoms corresponding to `atoms`

    Negative indices are wrapped and thus works.

    Parameters
    ----------
    atoms :
        indices of removed atoms

    See Also
    --------
    Geometry.remove : equivalent to the resulting `Geometry` from this routine
    Geometry.sub : the negative of `Geometry.remove`
    sub : the opposite of `remove`, i.e. retain a subset of atoms
    """
    atoms = S.asc2uc(atoms)
    atoms = np.delete(_a.arangei(S.na), atoms)
    return S.sub(atoms)
