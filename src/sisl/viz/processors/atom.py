# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import numpy as np
from xarray import DataArray, Dataset

from sisl._core import Geometry
from sisl.messages import SislError

from .xarray import Group, group_reduce


class AtomsGroup(Group, total=False):
    name: str
    atoms: Any
    reduce_func: Optional[Callable]


def reduce_atom_data(
    atom_data: Union[DataArray, Dataset],
    groups: Sequence[AtomsGroup],
    geometry: Optional[Geometry] = None,
    reduce_func: Callable = np.mean,
    atom_dim: str = "atom",
    groups_dim: str = "group",
    sanitize_group: Callable = lambda x: x,
    group_vars: Optional[Sequence[str]] = None,
    drop_empty: bool = False,
    fill_empty: Any = 0.0,
) -> Union[DataArray, Dataset]:
    """Groups contributions of atoms into a new dimension.

    Given an xarray object containing atom information and the specification of groups of atoms, this function
    computes the total contribution for each group of atoms. It therefore removes the atoms dimension and
    creates a new one to account for the groups.

    Parameters
    ----------
    atom_data : DataArray or Dataset
        The xarray object to reduce.
    groups : Sequence[AtomsGroup]
        A sequence containing the specifications for each group of atoms. See ``AtomsGroup``.
    geometry : Geometry, optional
        The geometry object that will be used to parse atom specifications into actual atom indices. Knowing the
        geometry therefore allows you to specify more complex selections.
        If not provided, it will be searched in the ``geometry`` attribute of the ``atom_data`` object.
    reduce_func : Callable, optional
        The function that will compute the reduction along the atoms dimension once the selection is done.
        This could be for example `numpy.mean` or `numpy.sum`.
        Notice that this will only be used in case the group specification doesn't specify a particular function
        in its "reduce_func" field, which will take preference.
    spin_reduce: Callable, optional
        The function that will compute the reduction along the spin dimension once the selection is done.
    orb_dim: str, optional
        Name of the dimension that contains the atom indices in ``atom_data``.
    groups_dim: str, optional
        Name of the new dimension that will be created for the groups.
    sanitize_group: Callable, optional
        A function that will be used to sanitize the group specification before it is used.
    group_vars: Sequence[str], optional
        If set, this argument specifies extra variables that depend on the group and the user would like to
        introduce in the new xarray object. These variables will be searched as fields for each group specification.
        A data variable will be created for each group_var and they will be added to the final xarray object.
        Note that this forces the returned object to be a Dataset, even if the input data is a DataArray.
    drop_empty: bool, optional
        If set to `True`, group specifications that do not correspond to any atom will not appear in the final
        returned object.
    fill_empty: Any, optional
        If ``drop_empty`` is set to ``False``, this argument specifies the value to use for group specifications
        that do not correspond to any atom.
    """
    # If no geometry was provided, then get it from the attrs of the xarray object.
    if geometry is None:
        geometry = atom_data.attrs.get("geometry")

    if geometry is None:

        def _sanitize_group(group):
            group = group.copy()
            group = sanitize_group(group)
            atoms = group["atoms"]
            try:
                group["atoms"] = np.array(atoms, dtype=int)
                assert atoms.ndim == 1
            except:
                raise SislError(
                    "A geometry was neither provided nor found in the xarray object. Therefore we can't"
                    f" convert the provided atom selection ({atoms}) to an array of integers."
                )

            group["selector"] = group["atoms"]

            return group

    else:

        def _sanitize_group(group):
            group = group.copy()
            group = sanitize_group(group)
            group["atoms"] = geometry._sanitize_atoms(group["atoms"])
            group["selector"] = group["atoms"]
            return group

    return group_reduce(
        data=atom_data,
        groups=groups,
        reduce_dim=atom_dim,
        reduce_func=reduce_func,
        groups_dim=groups_dim,
        sanitize_group=_sanitize_group,
        group_vars=group_vars,
        drop_empty=drop_empty,
        fill_empty=fill_empty,
    )
