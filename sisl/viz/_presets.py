import itertools
import os

from ._user_customs import get_user_presets
from .plotutils import get_file_vars

PRESETS = {
    "Dark theme": {
        "paper_bgcolor": "black",
        "plot_bgcolor": "black",
        **{f"{ax}_{key}": "white" for ax, key in itertools.product(
            ("xaxis", "yaxis"), ("color", "linecolor", "zerolinecolor", "gridcolor", "tickcolor")
        )},
        "bands_color": "#ccc",
        "bands_width": 2
    },

    "Barbie theme": {  
        "paper_bgcolor": "#f29ad8",
        "plot_bgcolor": "#f29ad8",
        **{f"{ax}_{key}": "#e305ad" for ax, key in itertools.product(
            ("xaxis", "yaxis"), ("color", "linecolor", "zerolinecolor", "gridcolor", "tickcolor")
        )},
        "bands_color": "gold",
        "bands_width": 2
    }
}

PRESETS = { **PRESETS, **get_user_presets()}

def get_preset(name):
    '''
    Gets the asked preset.

    Parameters
    -----------
    name: str
        the name of the preset that you are looking for
    '''

    return PRESETS.get(name, None)
