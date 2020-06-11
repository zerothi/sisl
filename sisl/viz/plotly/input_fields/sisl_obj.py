'''
This input field is prepared to receive sisl objects that are plotables
'''

import sisl

from .._input_field import InputField


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

        if not key.endswith(valid_key):

            raise ValueError(
                f'Invalid key ("{key}") for an input that accepts {kwargs["dtype"]}, please use {valid_key}'
                'to help keeping consistency across sisl and therefore make the world a better place.'
                f'If there are multiple settings that accept {kwargs["dtype"]}, please use *_{valid_key}'
            )

        super().__init__(key, *args, **kwargs)

class GeometryInput(SislObjectInput):
    pass

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
