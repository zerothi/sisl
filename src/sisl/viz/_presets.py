# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
__all__ = ["add_presets", "get_preset"]

PRESETS = {

    "dark": {
        "layout": {"template": "sisl_dark"},
        "bands_color": "#ccc",
        "bands_width": 2
    },

}


def add_presets(**presets):
    """
    Registers new presets

    Parameters
    ----------
    **presets:
        as many as you want. Each preset is a dict.
    """
    PRESETS.update(presets)


def get_preset(name):
    """
    Gets the asked preset.

    Parameters
    -----------
    name: str
        the name of the preset that you are looking for
    """
    return PRESETS.get(name, None)
