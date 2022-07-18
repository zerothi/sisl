from typing import Sequence, Callable, Optional, Union, Any, TypedDict

from collections import defaultdict
import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

from sisl import Geometry
from sisl.messages import SislError
from sisl.viz.nodes.processors.xarray import group_reduce

class OrbitalGroup(TypedDict):
    name: str
    orbitals: Any
    spin: Any
    reduce_func: Optional[Callable]
    spin_reduce: Optional[Callable]

def reduce_orbital_data(orbital_data: Union[DataArray, Dataset], groups: Sequence[OrbitalGroup], geometry: Optional[Geometry] = None, 
    reduce_func: Callable = np.mean, spin_reduce: Optional[Callable] = None, orb_dim: str = "orb", spin_dim="spin", 
    groups_dim: str = "group", sanitize_group: Callable = lambda x: x, group_vars: Optional[Sequence[str]] = None,
    drop_empty: bool = False, fill_empty: Any = 0.
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
        A sequence containing the specifications for each group of orbitals. See ``OrbitalGroup``.
    geometry : Geometry, optional
        The geometry object that will be used to parse orbital specifications into actual orbital indices. Knowing the
        geometry therefore allows you to specify more complex selections.
        If not provided, it will be searched in the ``geometry`` attribute of the ``orbital_data`` object.
    reduce_func : Callable, optional
        The function that will compute the reduction along the orbitals dimension once the selection is done.
        This could be for example ``numpy.mean`` or ``numpy.sum``. 
        Notice that this will only be used in case the group specification doesn't specify a particular function
        in its "reduce_func" field, which will take preference.
    spin_reduce: Callable, optional
        The function that will compute the reduction along the spin dimension once the selection is done.
    orb_dim: str, optional
        Name of the dimension that contains the orbital indices in ``orbital_data``.
    spin_dim: str, optional
        Name of the dimension that contains the spin components in ``orbital_data``.
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
        def _sanitize_group(group):
            group = group.copy()
            group = sanitize_group(group)
            orbitals = group['orbitals']
            try:
                group['orbitals'] = np.array(orbitals, dtype=int)
                assert orbitals.ndim == 1
            except:
                raise SislError("A geometry was neither provided nor found in the xarray object. Therefore we can't"
                    f" convert the provided atom selection ({orbitals}) to an array of integers.")

            group['selector'] = group['orbitals']
            if spin_reduce is not None and spin_dim in orbital_data.dims:
                group['selector'] = (group['selector'], group.get('spin'))
                group['reduce_func'] = (group.get('reduce_func', reduce_func), spin_reduce)

            return group
    else:
        def _sanitize_group(group):
            group = group.copy()
            group = sanitize_group(group)
            group["orbitals"] = geometry._sanitize_orbs(group["orbitals"])
            group['selector'] = group['orbitals']
            if spin_reduce is not None and spin_dim in orbital_data.dims:
                group['selector'] = (group['selector'], group.get('spin'))
                group['reduce_func'] = (group.get('reduce_func', reduce_func), spin_reduce)
            
            return group
    
    # If a reduction for spin was requested, then pass the two different functions to reduce
    # each coordinate.
    reduce_funcs = reduce_func
    reduce_dims = orb_dim
    if spin_reduce is not None and spin_dim in orbital_data.dims:
        reduce_funcs = (reduce_func, spin_reduce)
        reduce_dims = (orb_dim, spin_dim)

    return group_reduce(
        data=orbital_data, groups=groups, reduce_dim=reduce_dims, reduce_func=reduce_funcs,
        groups_dim=groups_dim, sanitize_group=_sanitize_group, group_vars=group_vars,
        drop_empty=drop_empty, fill_empty=fill_empty
    )

    # If no geometry was provided, then get it from the attrs of the xarray object.
    if geometry is None:
        geometry = orbital_data.attrs.get("geometry")

    group_vars_dict = defaultdict(list)
    groups_vals = []
    for i_group, group in enumerate(groups):
        group = sanitize_group(group)
        # Get the orbitals of the group
        orbitals = group['orbitals']
        # And sanitize them if we have a geometry available.
        if geometry is not None:
            orbitals = geometry._sanitize_orbs(orbitals)
        else:
            try:
                orbitals = np.array(orbitals, dtype=int)
                assert orbitals.ndim == 1
            except:
                SislError("A geometry was neither provided nor found in the xarray object. Therefore we can't"
                    f" convert the provided orbital selection ({orbitals}) to an array of integers.")

        if len(orbitals) == 0:
            # Handle the case where sanitizing led to no orbitals.
            if drop_empty:
                continue
            else:
                group_vals = orbital_data.sel(orb=0, drop=True).copy()
                group_vals[:] = fill_empty
        else:
            # Select the orbitals and remove the orbital dimension.
            group_vals = orbital_data.sel(**{orb_dim: orbitals}).reduce(group.get("reduce_func", reduce_func), dim=orb_dim)
        
        # Select and reduce spin if it was asked and the spin dimension is present
        if spin_reduce is not None and spin_dim in orbital_data.dims:
            # Get the spin specification
            spin = group.get('spin')
            # If there was a spin specification and it was not None, select the values,
            # otherwise select all spin contributions.
            if spin is not None:
                group_vals = group_vals.sel(**{spin_dim: spin})
            
            # Reduce the spin dimension
            group_vals = group_vals.reduce(spin_reduce, dim=spin_dim)

        # Assign the name to this group and add it to the list of groups.
        group_vals.name = group.get('name') or i_group
        groups_vals.append(group_vals)

        # Add the extra variables to the group.
        if group_vars is not None:
            for var in group_vars:
                group_vars_dict[var].append(group.get(var))
    
    # Get the values of the new coordinate that we are going to create for groups
    names = [group_vals.name for group_vals in groups_vals]
    # Concatenate all the groups into a single xarray object creating a new coordinate.
    new_obj = xr.concat(groups_vals, dim=groups_dim).assign_coords({groups_dim: names})
    new_obj.name = orbital_data.name
    # Set the attributes of the passed array to the new one.
    new_obj.attrs = {**orbital_data.attrs, **new_obj.attrs}

    # If there were extra group variables, then create a Dataset with them
    if group_vars is not None:

        if isinstance(new_obj, DataArray):
            new_obj = new_obj.to_dataset()
        
        new_obj = new_obj.assign({
            k: DataArray(v, dims=[groups_dim], name=k) for k,v in group_vars_dict.items()
        })

    return new_obj