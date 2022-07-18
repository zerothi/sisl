# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from collections import defaultdict
from copy import deepcopy
from xarray import DataArray, Dataset

from .bands import BandsData, BandsDataH, BandsDataWFSX, BandsProcessor
from ...plotutils import random_color
from ...input_fields import OrbitalQueries, TextInput, BoolInput, ColorInput, FloatInput

def _weights_from_eigenstate(eigenstate, spin, spin_index):
    """Function that calculates the weights from an eigenstate"""
    weights = eigenstate.norm2(sum=False)

    if not spin.is_diagonal:
        # If it is a non-colinear or spin orbit calculation, we have two weights for each
        # orbital (one for each spin component of the state), so we just pair them together
        # and sum their contributions to get the weight of the orbital.
        weights = weights.reshape(len(weights), -1, 2).sum(2)

    return weights.real

class FatBandsData(BandsData):
    
    _fatbands_extra_vars = [{"coords": ("band", "orb"), "name": "weight", "getter": _weights_from_eigenstate}]

class FatBandsDataH(BandsDataH, FatBandsData):

    def _get(self, extra_vars=(), **kwargs):
        return super()._get(extra_vars=(*extra_vars, *self._fatbands_extra_vars), **kwargs)

class FatBandsDataWFSX(BandsDataWFSX, FatBandsData):

    def _get(self, extra_vars=(), **kwargs):
        return super()._get(extra_vars=(*extra_vars, *self._fatbands_extra_vars), **kwargs)


_groups_param = OrbitalQueries(
    key="groups", name="Fatbands groups",
    default=None,
    help="""The different groups that are displayed in the fatbands""",
    queryForm=[

        TextInput(
            key="name", name="Name",
            default="Group",
            params={
                "placeholder": "Name of the line..."
            },
        ),

        'species', 'atoms', 'orbitals', 'spin',

        BoolInput(
            key="normalize", name="Normalize",
            default=True,
            params={
                "offLabel": "No",
                "onLabel": "Yes"
            }
        ),

        ColorInput(
            key="color", name="Color",
            default=None,
        ),

        FloatInput(
            key="scale", name="Scale factor",
            default=1,
        ),
    ]
)

@BandsProcessor.from_func
def style_fatbands(bands_data, groups, scale=1):
    """Returns a dictionary with information about all the weights that have been requested
    The return of this function is expected to be passed to the drawers.
    """
    # Get the weights of the bands array.
    plot_weights = bands_data.weight

    if groups is None:
        groups = ()
    
    def _get_group_weights(group, weights, groups_param, storage):
        """Extracts the weight values that correspond to a specific fatbands request.
        Parameters
        --------------
        group: dict
            the request to process.
        weights: DataArray
            the part of the weights dataarray that falls in the energy range that we want to draw.
        groups_param: InputField
            a parameter containing the options and the filtering abilities.
        values_storage: dict, optional
            a dictionary where the weights values will be stored using the request's name as the key.
        metadata_storage: dict, optional
            a dictionary where metadata for the request will be stored using the request's name as the key.
        Returns
        ----------
        xarray.DataArray
            The weights resulting from the request. They are indexed by spin, band and k value.
        """
        group = groups_param.complete_query(group)

        orb = groups_param.get_orbitals(group)

        # Get the weights for the requested orbitals
        weights = weights.sel(orb=orb)

        if group["normalize"]:
            weights = weights.mean("orb")
        else:
            weights = weights.sum("orb")

        if group["color"] is None:
            group["color"] = random_color()

        group_name = group["name"]
        values = weights * group["scale"]

        storage['names'].append(group_name)
        storage['values'].append(values)

        for key in ('color', ):
            storage[key].append(group[key])

        return values

    groups_param = deepcopy(_groups_param)
    geometry = bands_data.attrs.get('geometry') or bands_data.attrs['parent'].geometry
    groups_param.update_options(geometry, bands_data.attrs['spin'])

    storage = defaultdict(list)
    # Here we get the values of the weights for each group of orbitals.
    for i, group in enumerate(groups):
        group = {**group}

        # Use only the active requests
        if not group.get("active", True):
            continue

        # Give a name to the request in case it didn't have one.
        if group.get("name") is None:
            group["name"] = f"Group {i}"

        # Multiply the groups' scale by the global scale
        group["scale"] = group.get("scale", 1) * scale

        # Get the weight values for the request and store them to send to the drawer
        _get_group_weights(group, plot_weights, groups_param, storage)

    area_width = DataArray(
        storage['values'],
        coords={
            'group': storage['names']
        },
        dims=("group", *bands_data.E.dims),
        name="Orbital weight",
    )

    area_color = DataArray(storage['color'], dims=["group"], name='Area color')

    old_attrs = {}
    for coord in bands_data.coords:
        old_attrs[coord] = bands_data[coord].attrs

    bands_data = bands_data.assign(area_width=area_width, area_color=area_color)

    for k, v in old_attrs.items():
        bands_data.coords[k].attrs.update(v)

    return bands_data
