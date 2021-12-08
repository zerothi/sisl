# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import importlib

__all__ = ["load_backend", "load_backends"]


def load_backend(backend):
    """ Load backend from this module level

    Parameters
    ----------
    backend : str
       name of backend to load

    Raises
    ------
    ModuleNotFoundError
    """
    importlib.import_module(f".{backend}", __name__)


def load_backends():
    """ Loads all available backends from this module level

    Will *not* raise any errors.
    """
    for backend in ("templates", "plotly", "matplotlib", "blender"):
        try:
            load_backend(backend)
        except ModuleNotFoundError:
            pass
