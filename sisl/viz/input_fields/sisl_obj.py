# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
""" This input field is prepared to receive sisl objects that are plotables """
from pathlib import Path
from sisl.viz.input_fields.dropdown import DropdownInput

import sisl
from sisl.physics import distribution

from .._input_field import InputField
from .queries import QueriesInput
from .number import FloatInput, IntegerInput
from .text import FilePathInput, TextInput

from sisl import BaseSile

if not hasattr(BaseSile, "to_json"):
    # Little patch so that Siles can be sent to the GUI
    def sile_to_json(self):
        return str(self.file)

    BaseSile.to_json = sile_to_json


forced_keys = {
    sisl.Geometry: 'geometry',
    sisl.Hamiltonian: 'H',
    sisl.BandStructure: 'band_structure',
    sisl.BrillouinZone: 'brillouin_zone',
    sisl.Grid: 'grid',
    sisl.EigenstateElectron: 'eigenstate',
}


class SislObjectInput(InputField):

    _type = "sisl_object"

    def __init__(self, key, *args, **kwargs):

        super().__init__(key, *args, **kwargs)

        if self.dtype is None:
            raise ValueError(f'Please provide a dtype for {key}')

        valid_key = forced_keys.get(self.dtype, None)

        if valid_key is not None and not key.endswith(valid_key):

            raise ValueError(
                f'Invalid key ("{key}") for an input that accepts {kwargs["dtype"]}, please use {valid_key}'
                'to help keeping consistency across sisl and therefore make the world a better place.'
                f'If there are multiple settings that accept {kwargs["dtype"]}, please use *_{valid_key}'
            )


class GeometryInput(SislObjectInput):

    dtype = (sisl.Geometry, "sile (or path to file) that contains a geometry")
    _dtype = (str, sisl.Geometry, *sisl.get_siles(attrs=['read_geometry']))

    def parse(self, val):

        if isinstance(val, (str, Path)):
            val = sisl.get_sile(val)
        if isinstance(val, sisl.io.BaseSile):
            val = val.read_geometry()

        return val


class HamiltonianInput(SislObjectInput):
    pass


class BandStructureInput(QueriesInput, SislObjectInput):

    dtype = sisl.BandStructure

    def __init__(self, *args, **kwargs):
        kwargs["help"] = """A band structure. it can either be provided as a sisl.BandStructure object or
        as a list of points, which will be parsed into a band structure object.
        """

        # Let's define the queryform. Each query will be a point of the path.
        kwargs["queryForm"] = [

            FloatInput(
                key="x", name="X",
                width = "s50% m20% l10%",
                default=0,
                params={
                    "step": 0.01
                }
            ),

            FloatInput(
                key="y", name="Y",
                width = "s50% m20% l10%",
                default=0,
                params={
                    "step": 0.01
                }
            ),

            FloatInput(
                key="z", name="Z",
                width="s50% m20% l10%",
                default=0,
                params={
                    "step": 0.01
                }
            ),

            IntegerInput(
                key="divisions", name="Divisions",
                width="s50% m20% l10%",
                default=50,
                params={
                    "min": 0,
                    "step": 10
                }
            ),

            TextInput(
                key="name", name="Name",
                width = "s50% m20% l10%",
                default=None,
                params = {
                    "placeholder": "Name..."
                },
                help = "Tick that should be displayed at this corner of the path."
            )
        ]

        super().__init__(*args, **kwargs)

    def parse(self, val):
        if not isinstance(val, sisl.BandStructure) and val is not None:
            # Then let's parse the list of points into a band structure object.
            # Use only those points that are active.
            val = [point for point in val if point.get("active", True)]

            val = sisl.BandStructure(
                None,
                point=[[
                        point.get("x", None) or 0, point.get("y", None) or 0, point.get("z", None) or 0
                    ] for point in val],
                division=[int(point["divisions"]) for point in val[1:]],
                name=[point.get("name", '') for point in val]
            )

        return val


class BrillouinZoneInput(SislObjectInput):
    pass


class GridInput(SislObjectInput):
    pass


class EigenstateElectronInput(SislObjectInput):
    pass


class PlotableInput(SislObjectInput):

    _type = "plotable"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DistributionInput(QueriesInput, SislObjectInput):

    def __init__(self, *args, **kwargs):
        # Let's define the queryform (although we only want one point for now we use QueriesInput for convenience)
        kwargs["queryForm"] = [

            DropdownInput(
                key="method", name="Method",
                default="gaussian",
                params={
                    "options": [{"label": dist, "value": dist} for dist in distribution.__all__ if dist != "get_distribution"],
                    "isMulti": False,
                    "isClearable": False,
                }
            ),

            FloatInput(
                key="smearing", name="Smearing",
                default=0.1,
                params={
                    "step": 0.01
                }
            ),

            FloatInput(
                key="x0", name="Center",
                default=0.0,
                params={
                    "step": 0.01
                }
            ),
        ]

        super().__init__(*args, **kwargs)

    def parse(self, val):
        if val and not callable(val):
            if isinstance(val, str):
                val = distribution.get_distribution(val)
            else:
                # QueriesInput returns a list of dicts, we only want one.
                if not isinstance(val, dict):
                    val = val[0]

                val = distribution.get_distribution(**val)

        return val


class SileInput(FilePathInput, SislObjectInput):

    def __init__(self, *args, required_attrs=None, **kwargs):

        if required_attrs:
            self._required_attrs = required_attrs
            kwargs["dtype"] = None

        super().__init__(*args, **kwargs)

    def _get_dtype(self):
        """
        This is a temporal fix because for some reason some sile classes can not be pickled
        """
        if hasattr(self, "_required_attrs"):
            return tuple(sisl.get_siles(attrs=self._required_attrs))
        else:
            return self.__dict__["dtype"]

    def _set_dtype(self, val):
        self.__dict__["dtype"] = val

    dtype = property(fget=_get_dtype, fset=_set_dtype, )
