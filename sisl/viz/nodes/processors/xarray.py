from typing import Sequence, Callable, Optional, Tuple, Union, Any, TypedDict

from collections import defaultdict
import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

from sisl import Geometry
from sisl.messages import SislError

class Group(TypedDict, total=False):
    name: str
    selector: Any
    reduce_func: Optional[Callable]
    ...

def group_reduce(data: Union[DataArray, Dataset], groups: Sequence[Group], 
    reduce_dim: Union[str, Tuple[str, ...]], reduce_func: Union[Callable, Tuple[Callable, ...]] = np.mean, groups_dim: str = "group", 
    sanitize_group: Callable = lambda x: x, group_vars: Optional[Sequence[str]] = None,
    drop_empty: bool = False, fill_empty: Any = 0.
) -> Union[DataArray, Dataset]:
    """Groups contributions of orbitals into a new dimension.

    Given an xarray object containing orbital information and the specification of groups of orbitals, this function
    computes the total contribution for each group of orbitals. It therefore removes the orbitals dimension and 
    creates a new one to account for the groups.

    It can also reduce spin in the same go if requested. In that case, groups can also specify particular spin components.

    Parameters
    ----------
    data : DataArray or Dataset
        The xarray object to reduce.
    groups : Sequence[Group]
        A sequence containing the specifications for each group of orbitals. See ``Group``.
    reduce_func : Callable or tuple of Callable, optional
        The function that will compute the reduction along the reduced dimension once the selection is done.
        This could be for example ``numpy.mean`` or ``numpy.sum``. 
        Notice that this will only be used in case the group specification doesn't specify a particular function
        in its "reduce_func" field, which will take preference.
        If ``reduce_dim`` is a tuple, this can also be a tuple to indicate different reducing methods for each
        dimension.
    reduce_dim: str or tuple of str, optional
        Name of the dimension that should be reduced. If a tuple is provided, multiple dimensions will be reduced.
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
        If set to `True`, group specifications that do not result in any matches will not appear in the final
        returned object.
    fill_empty: Any, optional
        If ``drop_empty`` is set to ``False``, this argument specifies the value to use for group specifications
        that do not result in any matches.

    Returns
    ----------
    DataArray or Dataset
        The new xarray object with the grouped and reduced dataarray object.
    """
    input_is_dataarray = isinstance(data, DataArray)

    if not isinstance(reduce_dim, tuple):
        reduce_dim = (reduce_dim,)

    group_vars_dict = defaultdict(list)
    groups_vals = []
    names = []
    for i_group, group in enumerate(groups):
        group = sanitize_group(group)
        # Get the orbitals of the group
        selector = group['selector']
        if not isinstance(selector, tuple):
            selector = (selector,)

        # Select the data we are interested in
        group_vals = data.sel(**{dim: sel for dim, sel in zip(reduce_dim, selector)})

        empty = False
        for dim in reduce_dim:
            selected = getattr(group_vals, dim, [])
            empty = len(selected) == 0
            if empty:
                break

        if empty:
            # Handle the case where the selection found no matches.
            if drop_empty:
                continue
            else:
                group_vals = data.isel({dim: 0 for dim in reduce_dim}, drop=True).copy(deep=True)
                if input_is_dataarray:
                    group_vals[...] = fill_empty
                else:
                    for da in group_vals.values():
                        da[...] = fill_empty

        else:
            # If it did find matches, reduce the data.
            reduce_funcs = group.get("reduce_func", reduce_func)
            if not isinstance(reduce_funcs, tuple):
                reduce_funcs = tuple([reduce_funcs] * len(reduce_dim))
            for dim, func in zip(reduce_dim, reduce_funcs):
                group_vals = group_vals.reduce(func, dim=dim)

        
        # Assign the name to this group and add it to the list of groups.
        name = group.get('name') or i_group
        names.append(name)
        if input_is_dataarray:
            group_vals.name = name
        groups_vals.append(group_vals)

        # Add the extra variables to the group.
        if group_vars is not None:
            for var in group_vars:
                group_vars_dict[var].append(group.get(var))
    
    # Concatenate all the groups into a single xarray object creating a new coordinate.
    new_obj = xr.concat(groups_vals, dim=groups_dim).assign_coords({groups_dim: names})
    if input_is_dataarray:
        new_obj.name = data.name
    # Set the attributes of the passed array to the new one.
    new_obj.attrs = {**data.attrs, **new_obj.attrs}

    # If there were extra group variables, then create a Dataset with them
    if group_vars is not None:

        if isinstance(new_obj, DataArray):
            new_obj = new_obj.to_dataset()
        
        new_obj = new_obj.assign({
            k: DataArray(v, dims=[groups_dim], name=k) for k,v in group_vars_dict.items()
        })

    return new_obj