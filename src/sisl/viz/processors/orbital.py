# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# TODO when forward refs work with annotations
# from __future__ import annotations

from collections import ChainMap, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict, Union

import numpy as np
import xarray
from xarray import DataArray, Dataset

import sisl
from sisl import Geometry, Spin
from sisl.messages import SislError
from sisl.typing import AtomsIndex
from sisl.viz.types import OrbitalStyleQuery

from .._single_dispatch import singledispatchmethod
from ..data import Data
from ..processors.xarray import group_reduce
from .spin import get_spin_options


class OrbitalGroup(TypedDict):
    name: str
    orbitals: Any
    spin: Any
    reduce_func: Optional[Callable]
    spin_reduce: Optional[Callable]


class OrbitalQueriesManager:
    """
    This class implements an input field that allows you to select orbitals by atom, species, etc...
    """

    _item_input_type = OrbitalStyleQuery

    _keys_to_cols = {
        "atoms": "atom",
        "orbitals": "orbital_name",
    }

    geometry: Geometry
    spin: Spin

    key_gens: dict[str, Callable] = {}

    @singledispatchmethod
    @classmethod
    def new(
        cls,
        geometry: Geometry,
        spin: Union[str, Spin] = "",
        key_gens: dict[str, Callable] = {},
    ):
        return cls(geometry=geometry, spin=spin or "", key_gens=key_gens)

    @new.register
    @classmethod
    def from_geometry(
        cls,
        geometry: Geometry,
        spin: Union[str, Spin] = "",
        key_gens: dict[str, Callable] = {},
    ):
        return cls(geometry=geometry, spin=spin or "", key_gens=key_gens)

    @new.register
    @classmethod
    def from_string(
        cls,
        string: str,
        spin: Union[str, Spin] = "",
        key_gens: dict[str, Callable] = {},
    ):
        """Initializes an OrbitalQueriesManager from a string, assuming it is a path."""
        return cls.new(Path(string), spin=spin, key_gens=key_gens)

    @new.register
    @classmethod
    def from_path(
        cls, path: Path, spin: Union[str, Spin] = "", key_gens: dict[str, Callable] = {}
    ):
        """Initializes an OrbitalQueriesManager from a path, converting it to a sile."""
        return cls.new(sisl.get_sile(path), spin=spin, key_gens=key_gens)

    @new.register
    @classmethod
    def from_sile(
        cls,
        sile: sisl.io.BaseSile,
        spin: Union[str, Spin] = "",
        key_gens: dict[str, Callable] = {},
    ):
        """Initializes an OrbitalQueriesManager from a sile."""
        return cls.new(sile.read_geometry(), spin=spin, key_gens=key_gens)

    @new.register
    @classmethod
    def from_xarray(
        cls,
        array: xarray.core.common.AttrAccessMixin,
        spin: Optional[Union[str, Spin]] = None,
        key_gens: dict[str, Callable] = {},
    ):
        """Initializes an OrbitalQueriesManager from an xarray object."""
        if spin is None:
            spin = array.attrs.get("spin", "")

        return cls.new(array.attrs.get("geometry"), spin=spin, key_gens=key_gens)

    @new.register
    @classmethod
    def from_data(
        cls,
        data: Data,
        spin: Optional[Union[str, Spin]] = None,
        key_gens: dict[str, Callable] = {},
    ):
        """Initializes an OrbitalQueriesManager from a sisl Data object."""
        return cls.new(data._data, spin=spin, key_gens=key_gens)

    def __init__(
        self,
        geometry: Optional[Geometry] = None,
        spin: Union[str, Spin] = "",
        key_gens: dict[str, Callable] = {},
    ):
        self.geometry = geometry
        self.spin = Spin(spin)

        self.key_gens = key_gens

        self._build_orb_filtering_df(geometry)

    def complete_query(self, query={}, **kwargs):
        """
        Completes a partially build query with the default values

        Parameters
        -----------
        query: dict
            the query to be completed.
        **kwargs:
            other keys that need to be added to the query IN CASE THEY DON'T ALREADY EXIST
        """
        kwargs.update(query)

        # If it's a non-colinear or spin orbit spin class, the default spin will be total,
        # since averaging/summing over "x","y","z" does not make sense.
        if "spin" not in kwargs and not self.spin.is_diagonal:
            kwargs["spin"] = ["total"]

        return self._item_input_type(**kwargs)

    def filter_df(self, df, query, key_to_cols, raise_not_active=False):
        """
        Filters a dataframe according to a query

        Parameters
        -----------
        df: pd.DataFrame
            the dataframe to filter.
        query: dict
            the query to be used as a filter. Can be incomplete, it will be completed using
            `self.complete_query()`
        keys_to_cols: array-like of tuples
            An array of tuples that look like (key, col)
            where key is the key of the parameter in the query and col the corresponding
            column in the dataframe.
        """
        query = asdict(self.complete_query(query))

        if raise_not_active:
            if not query["active"]:
                raise ValueError(
                    f"Query {query} is not active and you are trying to use it"
                )

        query_str = []
        for key, val in query.items():
            if (
                key == "orbitals"
                and val is not None
                and len(val) > 0
                and isinstance(val[0], int)
            ):
                df = df.iloc[val]
                continue

            key = key_to_cols.get(key, key)
            if key in df and val is not None:
                if isinstance(val, (np.ndarray, tuple)):
                    val = np.ravel(val).tolist()
                query_str.append(f"{key}=={repr(val)}")

        if len(query_str) == 0:
            return df
        else:
            return df.query(" & ".join(query_str))

    def _build_orb_filtering_df(self, geom):
        import pandas as pd

        orb_props = defaultdict(list)
        del_key = set()
        # Loop over all orbitals of the basis
        for at, iorb in geom.iter_orbitals():
            atom = geom.atoms[at]
            orb = atom[iorb]

            orb_props["atom"].append(at)
            orb_props["Z"].append(atom.Z)
            orb_props["species"].append(atom.symbol)
            orb_props["orbital_name"].append(orb.name())

            for key in ("n", "l", "m", "zeta"):
                val = getattr(orb, key, None)
                if val is None:
                    del_key.add(key)
                orb_props[key].append(val)

        for key in del_key:
            del orb_props[key]

        self.orb_filtering_df = pd.DataFrame(orb_props)

    def get_options(self, key, **kwargs):
        """
        Gets the options for a given key or combination of keys.

        Parameters
        ------------
        key: str, {"species", "atoms", "Z", "orbitals", "n", "l", "m", "zeta", "spin"}
            the parameter that you want the options for.

            Note that you can combine them with a "+" to get all the possible combinations.
            You can get the same effect also by passing a list.
            See examples.
        **kwargs:
            keyword arguments that add additional conditions to the query. The values of this
            keyword arguments can be lists, in which case it indicates that you want a value
            that is in the list. See examples.

        Returns
        ----------
        np.ndarray of shape (n_options, [n_keys])
            all the possible options.

            If only one key was provided, it is a one dimensional array.

        Examples
        -----------

        >>> orb_manager = OrbitalQueriesManager(geometry)
        >>> orb_manager.get_options("l", species="Au")
        >>> orb_manager.get_options("n+l", atoms=[0,1])
        """
        # Get the tadatframe
        df = self.orb_filtering_df

        # Filter the dataframe according to the constraints imposed by the kwargs,
        # if there are any.
        if kwargs:
            if "atoms" in kwargs:
                kwargs["atoms"] = self.geometry._sanitize_atoms(kwargs["atoms"])

            def _repr(v):
                if isinstance(v, np.ndarray):
                    v = v.ravel().tolist()
                if isinstance(v, dict):
                    raise Exception(str(v))
                return repr(v)

            query = " & ".join(
                [
                    f"{self._keys_to_cols.get(k, k)}=={_repr(v)}"
                    for k, v in kwargs.items()
                    if self._keys_to_cols.get(k, k) in df
                ]
            )
            if query:
                df = df.query(query)

        # If + is in key, it is a composite key. In that case we are going to
        # split it into all the keys that are present and get the options for all
        # of them. At the end we are going to return a list of tuples that will be all
        # the possible combinations of the keys.
        keys = [self._keys_to_cols.get(k, k) for k in key.split("+")]

        # Spin values are not stored in the orbital filtering dataframe. If the options
        # for spin are requested, we need to pop the key out and get the current options
        # for spin from the input field
        spin_in_keys = "spin" in keys
        if spin_in_keys:
            spin_key_i = keys.index("spin")
            keys.remove("spin")
            spin_options = get_spin_options(self.spin)

            # We might have some constraints on what the spin value can be
            if "spin" in kwargs:
                spin_options = set(spin_options).intersection(kwargs["spin"])

        # Now get the unique options from the dataframe
        if keys:
            options = df.drop_duplicates(subset=keys)[keys].values.astype(object)
        else:
            # It might be the only key was "spin", then we are going to fake it
            # to get an options array that can be treated in the same way.
            options = np.array([[]], dtype=object)

        # If "spin" was one of the keys, we are going to incorporate the spin options, taking into
        # account the position (column index) where they are expected to be returned.
        if spin_in_keys and len(spin_options) > 0:
            options = np.concatenate(
                [np.insert(options, spin_key_i, spin, axis=1) for spin in spin_options]
            )

        # Squeeze the options array, just in case there is only one key
        # There's a special case: if there is only one option for that key,
        # squeeze converts it to a number, so we need to make sure there is at least 1d
        if options.shape[1] == 1:
            options = options.squeeze()
            options = np.atleast_1d(options)

        return options

    def get_orbitals(self, query):
        if "atoms" in query:
            query["atoms"] = self.geometry._sanitize_atoms(query["atoms"])

        filtered_df = self.filter_df(self.orb_filtering_df, query, self._keys_to_cols)

        return filtered_df.index.values

    def get_atoms(self, query):
        if "atoms" in query:
            query["atoms"] = self.geometry._sanitize_atoms(query["atoms"])

        filtered_df = self.filter_df(self.orb_filtering_df, query, self._keys_to_cols)

        return np.unique(filtered_df["atom"].values)

    def _split_query(
        self,
        query,
        on,
        only=None,
        exclude=None,
        query_gen=None,
        ignore_constraints=False,
        **kwargs,
    ):
        """
        Splits a query into multiple queries based on one of its parameters.

        Parameters
        --------
        query: dict
            the query that we want to split
        on: str, {"species", "atoms", "Z", "orbitals", "n", "l", "m", "zeta", "spin"}, or list of str
            the parameter to split along.
            Note that you can combine parameters with a "+" to split along multiple parameters
            at the same time. You can get the same effect also by passing a list.
        only: array-like, optional
            if desired, the only values that should be plotted out of
            all of the values that come from the splitting.
        exclude: array-like, optional
            values of the splitting that should not be plotted.
        query_gen: function, optional
            the request generator. It is a function that takes all the parameters for each
            request that this method has come up with and gets a chance to do some modifications.

            This may be useful, for example, to give each request a color, or a custom name.
        ignore_constraints: boolean or array-like, optional
            determines whether constraints (imposed by the query that you want to split)
            on the parameters that we want to split along should be taken into consideration.

            If `False`: all constraints considered.
            If `True`: no constraints considered.
            If array-like: parameters contained in the list ignore their constraints.
        **kwargs:
            keyword arguments that go directly to each new request.

            This is useful to add extra filters. For example:

            `self._split_query(request, on="orbitals", spin=[0])`
            will split the request on the different orbitals but will take
            only the contributions from spin up.
        """
        if exclude is None:
            exclude = []

        # Divide the splitting request into all the parameters
        if isinstance(on, str):
            on = on.split("+")

        # Get the current values of the parameters that we want to split the request on
        # because these will be our constraints. If a parameter is set to None or not
        # provided, we have no constraints for that parameter.
        if ignore_constraints is True:
            constraints = {}
        else:
            constraints = ChainMap(kwargs, query)

            if ignore_constraints is False:
                ignore_constraints = ()

            constraints = {
                key: val
                for key, val in constraints.items()
                if key not in ignore_constraints and val is not None
            }

        # Knowing what are our constraints (which may be none), get the available options
        values = self.get_options("+".join(on), **constraints)

        # We are going to make sure that, even if there was only one parameter to split on,
        # the values are two dimensional. In this way, we can take the same actions for the
        # case when there is only one parameter and the case when there are multiple.
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        # If no function to modify queries was provided we are going to use the default one
        # associated to this class.
        if query_gen is None:
            query_gen = self.complete_query

        # We ensure that on is a list even if there is only one parameter, for the same
        # reason we ensured values was 2 dimensional
        if isinstance(on, str):
            on = on.split("+")

        # Define the name that we will give to the new queries, using templating
        # If a splitting parameter is not used by the name, we are going to
        # append it, in order to make names unique and self-explanatory.
        base_name = kwargs.pop("name", query.get("name", "")) or ""
        first_added = True
        for key in on:
            kwargs.pop(key, None)

            if f"${key}" not in base_name:
                base_name += f"{' | ' if first_added else ', '}{key}=${key}"
                first_added = False

        # Now build all the queries
        queries = []
        for i, value in enumerate(values):
            if value not in exclude and (only is None or value in only):
                # Use the name template to generate the name for this query
                name = base_name
                for key, val in zip(on, value):
                    name = name.replace(f"${key}", str(val))

                # Build the query
                query = query_gen(
                    **{
                        **query,
                        **{key: [val] for key, val in zip(on, value)},
                        "name": name,
                        **kwargs,
                    }
                )

                # Make sure it is a dict
                if is_dataclass(query):
                    query = asdict(query)

                # And append the new query to the queries
                queries.append(query)

        return queries

    def generate_queries(
        self,
        split: str,
        only: Optional[Sequence] = None,
        exclude: Optional[Sequence] = None,
        query_gen: Optional[Callable[[dict], dict]] = None,
        **kwargs,
    ):
        """
        Automatically generates queries based on the current options.

        Parameters
        --------
        split: str, {"species", "atoms", "Z", "orbitals", "n", "l", "m", "zeta", "spin"} or list of str
            the parameter to split on.
            Note that you can combine parameters with a "+" to split along multiple parameters
            at the same time. You can get the same effect also by passing a list.
        only: array-like, optional
            if desired, the only values that should be plotted out of
            all of the values that come from the splitting.
        exclude: array-like, optional
            values that should not be plotted
        query_gen: function, optional
            the request generator. It is a function that takes all the parameters for each
            request that this method has come up with and gets a chance to do some modifications.

            This may be useful, for example, to give each request a color, or a custom name.
        **kwargs:
            keyword arguments that go directly to each request.

            This is useful to add extra filters. For example:
            `generate_queries(split="orbitals", species=["C"])`
            will split the PDOS on the different orbitals but will take
            only those that belong to carbon atoms.
        """
        return self._split_query(
            {}, on=split, only=only, exclude=exclude, query_gen=query_gen, **kwargs
        )

    def sanitize_query(self, query):
        # Get the complete request and make sure it is a dict.
        query = self.complete_query(query)
        if is_dataclass(query):
            query = asdict(query)

        # Determine the reduce function from the "reduce" passed and the scale factor.
        def _reduce_func(arr, **kwargs):
            reduce_ = query["reduce"]
            if isinstance(reduce_, str):
                reduce_ = getattr(np, reduce_)

            if kwargs["axis"] == ():
                return arr
            return reduce_(arr, **kwargs) * query.get("scale", 1)

        # Finally, return the sanitized request, converting the request (contains "species", "n", "l", etc...)
        # into a list of orbitals.
        return {
            **query,
            "orbitals": self.get_orbitals(query),
            "reduce_func": _reduce_func,
            **{k: gen(query) for k, gen in self.key_gens.items()},
        }


def generate_orbital_queries(
    orb_manager: OrbitalQueriesManager,
    split: str,
    only: Optional[Sequence] = None,
    exclude: Optional[Sequence] = None,
    query_gen: Optional[Callable[[dict], dict]] = None,
):
    return orb_manager.generate_queries(
        split, only=only, exclude=exclude, query_gen=query_gen
    )


def reduce_orbital_data(
    orbital_data: Union[DataArray, Dataset],
    groups: Sequence[OrbitalGroup],
    geometry: Optional[Geometry] = None,
    reduce_func: Callable = np.mean,
    spin_reduce: Union[None, Callable, Literal[False]] = None,
    orb_dim: str = "orb",
    spin_dim: str = "spin",
    groups_dim: str = "group",
    sanitize_group: Union[Callable, OrbitalQueriesManager, None] = None,
    group_vars: Optional[Sequence[str]] = None,
    drop_empty: bool = False,
    fill_empty: Any = 0.0,
) -> Union[DataArray, Dataset]:
    """Groups contributions of orbitals into a new dimension.

    Given an xarray object containing orbital information and the specification of groups of orbitals, this function
    computes the total contribution for each group of orbitals. It therefore removes the orbitals dimension and
    creates a new one to account for the groups.

    It can also reduce spin in the same go if requested. In that case, groups can also specify particular spin components.

    Parameters
    ----------
    orbital_data : DataArray or Dataset
        The xarray object to reduce.
    groups : Sequence[OrbitalGroup]
        A sequence containing the specifications for each group of orbitals. See `OrbitalGroup`.
    geometry : Geometry, optional
        The geometry object that will be used to parse orbital specifications into actual orbital indices. Knowing the
        geometry therefore allows you to specify more complex selections.
        If not provided, it will be searched in the ``geometry`` attribute of the `orbital_data` object and
        afterwards in the ``parent`` attribute, under ``parent.geometry``.
    reduce_func : Callable, optional
        The function that will compute the reduction along the orbitals dimension once the selection is done.
        This could be for example `numpy.mean` or `numpy.sum`.
        Notice that this will only be used in case the group specification doesn't specify a particular function
        in its "reduce_func" field, which will take preference.
    spin_reduce: Callable, optional
        The function that will compute the reduction along the spin dimension once the selection is done.

        If False, the spin dimension will not be reduced.
    orb_dim: str, optional
        Name of the dimension that contains the orbital indices in `orbital_data`.
    spin_dim: str, optional
        Name of the dimension that contains the spin components in `orbital_data`.
    groups_dim: str, optional
        Name of the new dimension that will be created for the groups.
    sanitize_group: Union[Callable, OrbitalQueriesManager], optional
        A function that will be used to sanitize the group specification before it is used.
        If a ``OrbitalQueriesManager`` is passed, its `sanitize_query` method will be used.
        If not provided and a geometry is found in the attributes of the `orbital_data` object,
        an `OrbitalQueriesManager` will be automatically created from it.
    group_vars: Sequence[str], optional
        If set, this argument specifies extra variables that depend on the group and the user would like to
        introduce in the new xarray object. These variables will be searched as fields for each group specification.
        A data variable will be created for each group_var and they will be added to the final xarray object.
        Note that this forces the returned object to be a Dataset, even if the input data is a DataArray.
    drop_empty: bool, optional
        If set to `True`, group specifications that do not correspond to any orbital will not appear in the final
        returned object.
    fill_empty: Any, optional
        If ``drop_empty`` is set to ``False``, this argument specifies the value to use for group specifications
        that do not correspond to any orbital.
    """
    # If no geometry was provided, then get it from the attrs of the xarray object.
    if geometry is None:
        geometry = orbital_data.attrs.get("geometry")
        if geometry is None:
            parent = orbital_data.attrs.get("parent")
            if parent is not None:
                getattr(parent, "geometry")

    if sanitize_group is None:
        if geometry is not None:
            sanitize_group = OrbitalQueriesManager(
                geometry=geometry, spin=orbital_data.attrs.get("spin", "")
            )
        else:
            sanitize_group = lambda x: x
    if isinstance(sanitize_group, OrbitalQueriesManager):
        sanitize_group = sanitize_group.sanitize_query

    data_spin = orbital_data.attrs.get("spin", Spin(""))

    # Determine whether the spin dimension should be reduced.
    should_reduce_spin = (
        (spin_reduce is not None or data_spin.is_polarized)
        and (spin_reduce is not False)
        and (spin_dim in orbital_data.dims)
    )

    original_spin_coord = None
    if (
        data_spin.is_polarized
        and spin_dim in orbital_data.coords
        and spin_reduce is not False
    ):
        if not isinstance(orbital_data, (DataArray, Dataset)):
            orbital_data = orbital_data._data

        original_spin_coord = orbital_data.coords[spin_dim].values

        if "total" in orbital_data.coords["spin"]:
            spin_up = (
                (orbital_data.sel(spin="total") - orbital_data.sel(spin="z")) / 2
            ).assign_coords(spin=0)
            spin_down = (
                (orbital_data.sel(spin="total") + orbital_data.sel(spin="z")) / 2
            ).assign_coords(spin=1)

            orbital_data = xarray.concat([orbital_data, spin_up, spin_down], "spin")
        else:
            total = orbital_data.sum(spin_dim).assign_coords(spin="total")
            z = (orbital_data.sel(spin=0) - orbital_data.sel(spin=1)).assign_coords(
                spin="z"
            )

            orbital_data = xarray.concat([total, z, orbital_data], "spin")

    # If a reduction for spin was requested, then pass the two different functions to reduce
    # each coordinate.
    reduce_funcs = reduce_func
    reduce_dims = orb_dim

    if should_reduce_spin:
        reduce_funcs = (reduce_func, spin_reduce)
        reduce_dims = (orb_dim, spin_dim)

    def _sanitize_group(group):
        group = group.copy()
        group = sanitize_group(group)

        if geometry is None:
            orbitals = group.get("orbitals")
            try:
                group["orbitals"] = np.array(orbitals, dtype=int)
                assert orbitals.ndim == 1
            except:
                raise SislError(
                    "A geometry was neither provided nor found in the xarray object. Therefore we can't"
                    f" convert the provided atom selection ({orbitals}) to an array of integers."
                )
        else:
            group["orbitals"] = geometry._sanitize_orbs(group["orbitals"])

        group["selector"] = group["orbitals"]

        req_spin = group.get("spin")
        if (
            req_spin is None
            and data_spin.is_polarized
            and spin_dim in orbital_data.coords
        ):
            if spin_reduce is None:
                group["spin"] = original_spin_coord
            else:
                group["spin"] = [0, 1]

        if (
            spin_reduce is not None or group.get("spin") is not None
        ) and spin_dim in orbital_data.dims:
            group["selector"] = (group["selector"], group.get("spin"))
            group["reduce_func"] = (group.get("reduce_func", reduce_func), spin_reduce)

        return group

    return group_reduce(
        data=orbital_data,
        groups=groups,
        reduce_dim=reduce_dims,
        reduce_func=reduce_funcs,
        groups_dim=groups_dim,
        sanitize_group=_sanitize_group,
        group_vars=group_vars,
        drop_empty=drop_empty,
        fill_empty=fill_empty,
    )


def get_orbital_queries_manager(
    obj, spin: Optional[str] = None, key_gens: dict[str, Callable] = {}
) -> OrbitalQueriesManager:
    return OrbitalQueriesManager.new(obj, spin=spin, key_gens=key_gens)


def split_orbitals(
    orbital_data,
    on="species",
    only=None,
    exclude=None,
    geometry: Optional[Geometry] = None,
    reduce_func: Callable = np.mean,
    spin_reduce: Optional[Callable] = None,
    orb_dim: str = "orb",
    spin_dim: str = "spin",
    groups_dim: str = "group",
    group_vars: Optional[Sequence[str]] = None,
    drop_empty: bool = False,
    fill_empty: Any = 0.0,
    **kwargs,
):
    if geometry is not None:
        orbital_data = orbital_data.copy()
        orbital_data.attrs["geometry"] = geometry

    orbital_data = orbital_data.copy()

    orb_manager = get_orbital_queries_manager(
        orbital_data, key_gens=kwargs.pop("key_gens", {})
    )

    groups = orb_manager.generate_queries(
        split=on, only=only, exclude=exclude, **kwargs
    )

    return reduce_orbital_data(
        orbital_data,
        groups=groups,
        sanitize_group=orb_manager,
        reduce_func=reduce_func,
        spin_reduce=spin_reduce,
        orb_dim=orb_dim,
        spin_dim=spin_dim,
        groups_dim=groups_dim,
        group_vars=group_vars,
        drop_empty=drop_empty,
        fill_empty=fill_empty,
    )


def atom_data_from_orbital_data(
    orbital_data,
    atoms: AtomsIndex = None,
    request_kwargs: dict = {},
    geometry: Optional[Geometry] = None,
    reduce_func: Callable = np.mean,
    spin_reduce: Optional[Callable] = None,
    orb_dim: str = "orb",
    spin_dim: str = "spin",
    groups_dim: str = "atom",
    group_vars: Optional[Sequence[str]] = None,
    drop_empty: bool = False,
    fill_empty: Any = 0.0,
):
    request_kwargs["name"] = "$atoms"

    atom_data = split_orbitals(
        orbital_data,
        on="atoms",
        only=atoms,
        reduce_func=reduce_func,
        spin_reduce=spin_reduce,
        orb_dim=orb_dim,
        spin_dim=spin_dim,
        groups_dim=groups_dim,
        group_vars=group_vars,
        drop_empty=drop_empty,
        fill_empty=fill_empty,
        **request_kwargs,
    )

    atom_data = atom_data.assign_coords(atom=atom_data.atom.astype(int))

    return atom_data
