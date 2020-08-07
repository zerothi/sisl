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
