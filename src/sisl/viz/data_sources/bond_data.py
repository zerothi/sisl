# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Union

import numpy as np

import sisl
from sisl.utils.mathematics import fnorm

from .data_source import DataSource


class BondData(DataSource):
    ndim: int

    @staticmethod
    def function(geometry, bonds):
        raise NotImplementedError("")

    pass


def bond_lengths(geometry: sisl.Geometry, bonds: np.ndarray):
    # Get an array with the coordinates defining the start and end of each bond.
    # The array will be of shape (nbonds, 2, 3)
    coords = geometry[np.ravel(bonds)].reshape(-1, 2, 3)
    # Take the diff between the end and start -> shape (nbonds, 1 , 3)
    # And then the norm of each vector -> shape (nbonds, 1, 1)
    # Finally, we just ravel it to an array of shape (nbonds, )
    return fnorm(np.diff(coords, axis=1), axis=-1).ravel()


def bond_strains(
    ref_geometry: sisl.Geometry, geometry: sisl.Geometry, bonds: np.ndarray
):
    assert ref_geometry.na == geometry.na, (
        f"Geometry provided (na={geometry.na}) does not have the"
        f" same number of atoms as the reference geometry (na={ref_geometry.na})"
    )

    ref_bond_lengths = bond_lengths(ref_geometry, bonds)
    this_bond_lengths = bond_lengths(geometry, bonds)

    return (this_bond_lengths - ref_bond_lengths) / ref_bond_lengths


def bond_data_from_atom(
    atom_data: np.ndarray,
    geometry: sisl.Geometry,
    bonds: np.ndarray,
    fold_to_uc: bool = False,
):
    if fold_to_uc:
        bonds = geometry.asc2uc(bonds)

    return atom_data[bonds[:, 0]]


def bond_data_from_matrix(
    matrix, geometry: sisl.Geometry, bonds: np.ndarray, fold_to_uc: bool = False
):
    if fold_to_uc:
        bonds = geometry.asc2uc(bonds)

    return matrix[bonds[:, 0], bonds[:, 1]]


def bond_random(
    geometry: sisl.Geometry, bonds: np.ndarray, seed: Union[int, None] = None
):
    if seed is not None:
        np.random.seed(seed)

    return np.random.random(len(bonds))


BondLength = BondData.from_func(bond_lengths)
BondStrain = BondData.from_func(bond_strains)
BondDataFromAtom = BondData.from_func(bond_data_from_atom)
BondDataFromMatrix = BondData.from_func(bond_data_from_matrix)
BondRandom = BondData.from_func(bond_random)
