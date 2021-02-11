import os
from pathlib import Path
import sys
import importlib

from sisl.messages import warn
from sisl._environ import get_environ_variable

__all__ = ["import_user_presets", "import_user_plots",
           "import_user_sessions", "import_user_plugins"]

USER_CUSTOM_FOLDER = get_environ_variable("SISL_CONFIGDIR") / "viz" / "plotly"

# Here we let python know that there are importable files
# in USER_CUSTOM_FOLDER
sys.path.append(str(USER_CUSTOM_FOLDER.resolve()))


def import_user_extension(extension_file):
    """
    Basis for importing users extensions.

    Parameters
    ------------
    extension_file: str
        the name of the file that you want to import (NOT THE FULL PATH).
    """
    try:
        return importlib.import_module(str(extension_file).replace(".py", ""))
    except ModuleNotFoundError:
        return None

#--------------------------------------
#              Presets
#--------------------------------------
# File where the user's presets will be searched
PRESETS_FILE_NAME = "presets.py"
PRESETS_FILE = USER_CUSTOM_FOLDER / PRESETS_FILE_NAME
# We will look for presets under this variable
PRESETS_VARIABLE = "presets"


def import_user_presets():
    """
    Imports the users presets.

    All the presets that the user wants to import into sisl
    should be in the 'presets' variable as a dict in the 'user_presets.py'
    file. Then, this method will add them to the global dictionary of presets.
    """
    from ._presets import add_presets

    module = import_user_extension(PRESETS_FILE_NAME)

    # Add these presets
    if module is not None:
        if PRESETS_VARIABLE in vars(module):
            add_presets(**vars(module)[PRESETS_VARIABLE])
        else:
            warn(f"We found the custom presets file ({PRESETS_FILE}) but no '{PRESETS_VARIABLE}' variable was found.\n Please put your presets as a dict under this variable.")

    return module

#--------------------------------------
#               Plots
#--------------------------------------
# File where the user's plots will be searched
PLOTS_FILE_NAME = "plots.py"
PLOTS_FILE = USER_CUSTOM_FOLDER / PLOTS_FILE_NAME


def import_user_plots():
    """
    Imports the user's plots.

    We don't need to do anything here because all plots available
    are tracked by checking the subclasses of `Plot`.
    Therefore, the user only needs to make sure that their plot classes
    are defined.
    """
    return import_user_extension(PLOTS_FILE_NAME)

#--------------------------------------
#              Sessions
#--------------------------------------
# File where the user's sessions will be searched
SESSION_FILE_NAME = "sessions.py"
SESSION_FILE = USER_CUSTOM_FOLDER / SESSION_FILE_NAME


def import_user_sessions():
    """
    Imports the user's sessions.

    We don't need to do anything here because all sessions available
    are tracked by checking the subclasses of `Session`.
    Therefore, the user only needs to make sure that their session classes
    are defined.
    """
    return import_user_extension(SESSION_FILE_NAME)


#----------------------------------------
#              Plugins
#---------------------------------------
# This is a general file that the user can have for convenience so that everytime
# that sisl is imported, it can automatically import all their utilities that they
# developed to work with sisl
PLUGINS_FILE_NAME = "plugins.py"


def import_user_plugins():
    """
    This imports an extra file where the user can do really anything
    that they want to finish customizing the package.
    """
    return import_user_extension(PLUGINS_FILE_NAME)
