from typing import List, Literal, Union

from sisl import Spin

_options = {
    Spin.UNPOLARIZED: [],
    Spin.POLARIZED: [{"label": "↑", "value": 0}, {"label": "↓", "value": 1}, 
        {"label": "Total", "value": "total"}, {"label": "Net z", "value": "z"}],
    Spin.NONCOLINEAR: [{"label": val, "value": val} for val in ("total", "x", "y", "z")],
    Spin.SPINORBIT: [{"label": val, "value": val} for val in ("total", "x", "y", "z")]
}

def get_spin_options(spin: Union[Spin, str], only_if_polarized: bool = False) -> List[Literal[0, 1, "total", "x", "y", "z"]]:
    """Returns the options for a given spin class.
    
    Parameters
    ----------
    spin: sisl.Spin or str
        The spin class to get the options for.
    only_if_polarized: bool, optional
        If set to `True`, non colinear spins will not have multiple options. 
    """
    spin = Spin(spin)

    if only_if_polarized and not spin.is_polarized:
        options_spin = Spin("")
    else:
        options_spin = spin

    return [option['value'] for option in _options[options_spin.kind]]