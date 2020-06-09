import os
import sys
import importlib

from .._env_vars import register_env_var

__all__ = ['import_user_presets', 'import_user_plots',
    'import_user_sessions', 'import_user_plugins']

# Define the folder where the user will store their stuff
DEFAULT_USER_CUSTOM_FOLDER = os.path.join(os.path.expanduser('~'), ".sisl")
USER_CUSTOM_FOLDER = register_env_var(
    'USER_DIR', DEFAULT_USER_CUSTOM_FOLDER,
    "Path to the directory where the user stores their custom scripts"
    " to extend sisl"
)

sys.path.append(os.path.abspath(USER_CUSTOM_FOLDER))

def import_user_extension(extension_file):

    try:
        return importlib.import_module(extension_file.replace(".py", ""))
    except ModuleNotFoundError:
        return None

#--------------------------------------
#              Presets
#--------------------------------------
# File where the user's presets will be searched
PRESETS_FILE_NAME = "user_presets.py"
PRESETS_FILE = os.path.join(USER_CUSTOM_FOLDER, PRESETS_FILE_NAME)
# We will look for presets under this variable
PRESETS_VARIABLE = "presets"

def import_user_presets():
    from ._presets import add_presets

    module = import_user_extension(PRESETS_FILE_NAME)

    # Add this presets 
    if module is not None:
        if PRESETS_VARIABLE in vars(module):
            add_presets(**vars(module)[PRESETS_VARIABLE])
        else:
            print(
                f"We found the custom presets file ({PRESETS_FILE}) but no '{PRESETS_VARIABLE}' variable was found.\n Please put your presets as a dict under this variable.")

    return module

#--------------------------------------
#               Plots
#--------------------------------------
# File where the user's plots will be searched
PLOTS_FILE_NAME = "user_plots.py"
PLOTS_FILE = os.path.join(USER_CUSTOM_FOLDER, PLOTS_FILE_NAME)
# We will look for plots under this variable
PLOTS_VARIABLE = "plots"

def import_user_plots():
    return import_user_extension(PLOTS_FILE_NAME)

#--------------------------------------
#              Sessions
#--------------------------------------
# File where the user's sessions will be searched
SESSION_FILE_NAME = "user_sessions.py"
SESSION_FILE = os.path.join(USER_CUSTOM_FOLDER, SESSION_FILE_NAME)
# We will look for sessions under this variable
SESSION_VARIABLE = "sessions"

def import_user_sessions():
    return import_user_extension(SESSION_FILE_NAME)


#----------------------------------------
#              Plugins
#---------------------------------------
# This is a general file that the user can have for convenience so that everytime
# that sisl is imported, it can automatically import all their utilities that they
# developed to work with sisl
PLUGINS_FILE_NAME = "user_plugins.py"

def import_user_plugins():
    return import_user_extension(PLUGINS_FILE_NAME)
