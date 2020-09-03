# Import everything from the plotly submodule
from .plotly import *
from .plotly import plotutils

# And then import user customs (we need to do it here
# to allow the user importing sisl.viz.Plot, for example)
user_plots = import_user_plots()
user_presets = import_user_presets()
user_sessions = import_user_sessions()
user_plugins = import_user_plugins()
