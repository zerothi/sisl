import itertools
import os

from .plotutils import get_file_vars

PRESETS = {
    "siesta": {
        
    },

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

def add_presets(**presets):
    '''
    Registers new presets

    Parameters
    ----------
    **presets:
        as many as you want. Each preset is a dict.
    '''

    global PRESETS

    PRESETS = {**PRESETS, **presets}

def get_preset(name):
    '''
    Gets the asked preset.

    Parameters
    -----------
    name: str
        the name of the preset that you are looking for
    '''

    return PRESETS.get(name, None)
