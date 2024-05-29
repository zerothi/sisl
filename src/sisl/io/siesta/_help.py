# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import warnings

import numpy as np

import sisl._array as _a
from sisl import SislError
from sisl.messages import warn

try:
    from . import _siesta

    has_fortran_module = True
except ImportError:
    has_fortran_module = False

__all__ = ["_csr_from_siesta", "_csr_from_sc_off"]
__all__ += ["_csr_to_siesta", "_csr_to_sc_off"]
__all__ += ["_mat_spin_convert", "_fc_correct"]


def _ensure_diagonal(csr):
    """Ensures that the sparsity pattern has diagonal entries

    This will set the wrong values in non-orthogonal basis-sets
    since missing items will be set to 0 which should be 1 in
    non-orthogonal basis sets.
    """
    old_nnz = csr.nnz
    csr += csr.diags(0, dim=1)
    n_added = csr.nnz - old_nnz
    if n_added > 0:
        warn(f"added {n_added} diagonal elements. This changes the sparsity pattern.")


def _csr_from_siesta(geom, csr):
    """Internal routine to convert *any* SparseCSR matrix from sisl nsc to siesta nsc"""
    if not has_fortran_module:
        raise SislError(
            "sisl cannot convert the sparse matrix from a Siesta conforming sparsity pattern! Please install with fortran support!"
        )

    _csr_from_sc_off(geom, _siesta.siesta_sc_off(*geom.nsc).T, csr)


def _csr_to_siesta(geom, csr, diag=True):
    """Internal routine to convert *any* SparseCSR matrix from sisl nsc to siesta nsc

    Parameters
    ----------
    ...
    diag: bool, optional
       whether the csr matrix will be ensured diagonal as well
    """
    if not has_fortran_module:
        raise SislError(
            "sisl cannot convert the sparse matrix into a Siesta conforming sparsity pattern! Please install with fortran support!"
        )

    if diag:
        _ensure_diagonal(csr)
    _csr_to_sc_off(geom, _siesta.siesta_sc_off(*geom.nsc).T, csr)


def _csr_from_sc_off(geom, sc_off, csr):
    """Internal routine to convert *any* SparseCSR matrix from sisl nsc to siesta nsc"""
    nsc = geom.lattice.nsc.astype(np.int32)
    sc = geom.lattice.__class__([1], nsc=nsc)
    sc.sc_off = sc_off
    from_sc_off = sc.sc_index(geom.sc_off)
    # this transfers the local siesta csr matrix ordering to the geometry ordering
    col_from = (
        from_sc_off.reshape(-1, 1) * geom.no + _a.arangei(geom.no).reshape(1, -1)
    ).ravel()
    _csr_from(col_from, csr)


def _csr_to_sc_off(geom, sc_off, csr):
    """Internal routine to convert *any* SparseCSR matrix from sisl nsc to siesta nsc"""
    # Find the equivalent indices in the geometry supercell
    to_sc_off = geom.sc_index(sc_off)
    # this transfers the local csr matrix ordering to the geometry ordering
    col_from = (
        to_sc_off.reshape(-1, 1) * geom.no + _a.arangei(geom.no).reshape(1, -1)
    ).ravel()
    _csr_from(col_from, csr)


def _csr_from(col_from, csr):
    """Internal routine to convert columns in a SparseCSR matrix"""
    # local csr matrix ordering
    col_to = _a.arangei(csr.shape[1])
    csr.translate_columns(col_from, col_to)


def _mat_spin_convert(M, spin=None):
    """Conversion of Siesta spin matrices to sisl spin matrices

    The matrices from Siesta are given in a format adheering to the following
    concept:

    A non-colinear calculation has the following entries (in C-index) for
    the sparse matrix:

    H[:, [0, 1, 2, 3]]
    H11 == H[:, 0]
    H22 == H[:, 1]
    H12 == H[:, 2] - 1j H[:, 3] # spin-box Hermitian
    H21 == H[:, 2] + 1j H[:, 3]

    Although it really does not make sense to change anything, we
    do change it to adhere to the spin-orbit case (see below).
    I.e. what Siesta *saves* is the -Im[H12], which we now store
    as Im[H12].


    A spin-orbit calculation has the following entries (in C-index) for
    the sparse matrix:

    H[:, [0, 1, 2, 3, 4, 5, 6, 7]]
    H11 == H[:, 0] + 1j H[:, 4]
    H22 == H[:, 1] + 1j H[:, 5]
    H12 == H[:, 2] + 1j H[:, 3] # spin-box Hermitian
    H21 == H[:, 6] + 1j H[:, 7]
    """
    if spin is None:
        if M.spin.is_noncolinear:
            M._csr._D[:, 3] = -M._csr._D[:, 3]
        elif M.spin.is_spinorbit:
            M._csr._D[:, 3] = -M._csr._D[:, 3]
    elif spin.is_noncolinear:
        M._D[:, 3] = -M._D[:, 3]
    elif spin.is_spinorbit:
        M._D[:, 3] = -M._D[:, 3]


def _geom2hsx(geometry):
    """Convert the geometry into the correct lists of species and lists"""
    atoms = geometry.atoms
    nspecies = atoms.nspecies
    isa = atoms.species
    label, Z, no = [], [], []
    n, l, zeta = [], [], []
    for atom in atoms.atom:
        label.append(atom.tag)
        Z.append(atom.Z)
        no.append(len(atom))
        try:
            n.append([orb.n for orb in atom])
            l.append([orb.l for orb in atom])
            zeta.append([orb.zeta for orb in atom])
        except Exception:
            n.append([-1 for orb in atom])
            l.append([-1 for orb in atom])
            zeta.append([1 for orb in atom])
    return (label, Z, no), (n, l, zeta)


def _fc_correct(fc, trans_inv=True, sum0=True, hermitian=True):
    r"""Correct a force-constant matrix to retain translational invariance and sum of forces == 0 on atoms

    Parameters
    ----------
    fc : (*, 3, *nsc, *, 3)
       force constant, dimensions like this:
       1. atoms displaced
       2. displacement directions (x, y, z)
       3. number of supercells along x
       4. number of supercells along y
       5. number of supercells along z
       6. number of atoms in unit-cell
       7. Cartesian directions
    trans_inv : bool, optional
       if true, the total sum of forces will be forced to zero (translational invariance)
    sum0 : bool, optional
       if true, the sum of forces on atoms for each displacement will be forced to 0.
    hermitian: bool, optional
       whether hermiticity will be forced
    """
    if not (trans_inv or sum0 or hermitian):
        return fc

    # to not disturb the outside fc
    fc = fc.copy()
    shape = fc.shape

    is_subset = shape[0] != shape[5]
    if is_subset:
        raise ValueError(
            f"fc_correct cannot figure out the displaced atoms in the unit-cell, please limit atoms to the displaced atoms."
        )

    # NOTE:
    # This is not exactly the same as Vibra does it.
    # In Vibra this is done:
    # fc *= 0.5
    # fc += np.transpose(fc, (5, 6, 2, 3, 4, 0, 1))[:, :, ::-1, ::-1, ::-1, :, :]
    # zero = fc.sum((2, 3, 4, 5)) / np.prod(shape[2:6])
    # zeroo = zero.sum(0) / shape[0]
    # zeroo = 0.5 * (zeroo + zeroo.T).reshape(1, 3, 1, 1, 1, 1, 3)
    # zeroT = np.transpose(zero, (2, 0, 1)).reshape(1, 3, 1, 1, 1, -1, 3)
    # fc += zeroo - zero.reshape(-1, 3, 1, 1, 1, 1, 3) - zeroT

    if hermitian:
        fc *= 0.5
        fc += np.transpose(fc, (5, 6, 2, 3, 4, 0, 1))[:, :, ::-1, ::-1, ::-1, :, :]

    if trans_inv:
        fc_total = fc.sum((0, 2, 3, 4, 5)) / (shape[0] * np.prod(shape[2:6]))
        if hermitian:
            fc_total = (fc_total + fc_total.T) * 0.5
        # It is unclear to me what happens for cases where
        # the displacements are done on a sub-set of atoms
        # but not in a supercell arangement.
        # Say a molecule sandwiched between two electrodes.
        # For now we will take N as total number of atoms
        # in the cell.
        fc -= fc_total.reshape(1, 3, 1, 1, 1, 1, 3)

    if sum0:
        fc_atom = (fc.sum((2, 3, 4, 5)) / np.prod(shape[2:6])).reshape(-1, 3, 1, 3)
        if hermitian:
            # this will fail if is_subset
            # TODO add case for subset
            fc_atom = 0.5 * (fc_atom + np.transpose(fc_atom, (2, 3, 0, 1)))
            fc -= fc_atom.reshape(shape[0], 3, 1, 1, 1, shape[5], 3)
        else:
            fc -= fc_atom.reshape(shape[0], 3, 1, 1, 1, 1, 3)

    return fc
