""" This input field is prepared to receive sisl objects that are plotables """

import sisl

from .._input_field import InputField
from .text import FilePathInput

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

        if 'dtype' not in kwargs:
            raise ValueError(f'Please provide a dtype for {key}')

        valid_key = forced_keys.get(kwargs['dtype'], None)

        if valid_key is not None and not key.endswith(valid_key):

            raise ValueError(
                f'Invalid key ("{key}") for an input that accepts {kwargs["dtype"]}, please use {valid_key}'
                'to help keeping consistency across sisl and therefore make the world a better place.'
                f'If there are multiple settings that accept {kwargs["dtype"]}, please use *_{valid_key}'
            )

        super().__init__(key, *args, **kwargs)


class GeometryInput(SislObjectInput):

    _dtype = (str, sisl.Geometry, *sisl.get_siles(attrs=['read_geometry']))

    def parse(self, val):

        if isinstance(val, str):
            val = sisl.get_sile(val)
        if isinstance(val, sisl.io.Sile):
            val = val.read_geometry()

        return val


class HamiltonianInput(SislObjectInput):
    pass


class BandStructureInput(SislObjectInput):
    pass


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
