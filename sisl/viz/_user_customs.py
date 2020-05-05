import os

from .plotutils import get_file_vars

USER_CUSTOM_FOLDER = os.path.join(os.path.expanduser('~'), ".sisl")

#--------------------------------------
#              Presets
#--------------------------------------
# File where the user's presets will be searched
PRESETS_FILE = os.path.join(USER_CUSTOM_FOLDER, "presets.py")
# We will look for presets under this variable
PRESETS_VARIABLE = "presets"

def get_user_presets():
    return get_file_vars(PRESETS_FILE).get(PRESETS_VARIABLE, {})

#--------------------------------------
#               Plots
#--------------------------------------
# File where the user's plots will be searched
PLOTS_FILE = os.path.join(USER_CUSTOM_FOLDER, "plots.py")
# We will look for plots under this variable
PLOTS_VARIABLE = "plots"

def get_user_plots():
    return get_file_vars(PLOTS_FILE).get(PLOTS_VARIABLE, {})

#--------------------------------------
#              Sessions
#--------------------------------------
# File where the user's sessions will be searched
SESSION_FILE = os.path.join(USER_CUSTOM_FOLDER, "sessions.py")
# We will look for sessions under this variable
SESSION_VARIABLE = "sessions"

def get_user_sessions():
    return get_file_vars(SESSION_FILE).get(SESSION_VARIABLE, {})