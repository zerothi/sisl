"""
This file makes documentation of plots and sessions easy based on their parameters.

Basically, you just need to have your class defined and have a flag in the docs
specifying where the settings documentation should go. This script will fill it
for you.

Example:

class FakePlot(Plot):
    '''
    This plot does really nothing useful

    Parameters
    -----------
    %%configurable_settings%%
    '''

%%configurable_settings%% is the key to let the script now where to put the documentation.

IF YOU HAVE MORE THAN ONE PLOT CLASS IN A FILE, YOU SHOULD SPECIFY %%FakePlot_configurable_settings%%

Then, just run this script and it will update all the classes documentation.
Or you can use fill_class_docs to only update a certain class.
"""

from sisl.viz.plotutils import get_configurable_docstring, get_plot_classes, get_session_classes
from sisl.viz import Plot, MultiplePlot, Animation, SubPlots, Session
import inspect


def get_parameters_docstrings(cls):
    """
    Returns the documentation for the configurable's parameters.

    Parameters
    -----------
    cls:
        the class you want the docstring for

    Returns
    -----------
    str:
        the docs with the settings added.
    """
    import re

    if isinstance(cls, type):
        params = cls._get_class_params()[0]
        doc = cls.__doc__
        if doc is None:
            doc = ""
    else:
        # It's really an instance, not the class
        params = cls.params
        doc = ""

    configurable_settings = "\n".join(
        [param._get_docstring() for param in params])

    html_cleaner = re.compile('<.*?>')
    configurable_settings = re.sub(html_cleaner, '', configurable_settings)

    return configurable_settings


def fill_class_docs(Cls):
    """
    Fills the documentation for a class that inherits from Configurable.

    You just need to use the placeholder %%configurable_settings%% or 
    %%ClassName_configurable_settings%% for more specificity, where ClassName is the name
    of the class that you want to document. Then, this function replaces that placeholder
    with the documentation for all the settings.

    Parameters
    -----------
    cls:
        the class you want to document.

    """

    filename = inspect.getfile(Cls)

    parameters_docs = get_parameters_docstrings(Cls)

    parameters_docs = "\n\t".join(
        parameters_docs.split("\n")).replace("\t", "    ")

    with open(filename, 'r') as fi:
        lines = fi.read()
        new_lines = lines.replace("%%configurable_settings%%", parameters_docs)
        new_lines = new_lines.replace(f"%%{Cls.__name__}_configurable_settings%%", parameters_docs)

    with open(filename, 'w') as fo:
        fo.write(new_lines)

if __name__ == "__main__":
    for Cls in [*get_plot_classes(), Plot, MultiplePlot, Animation, SubPlots, Session, *get_session_classes().values()]:
        fill_class_docs(Cls)
