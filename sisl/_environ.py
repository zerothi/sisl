import os
from pathlib import Path

__all__ = ["register_environ_variable", "get_environ_variable"]

# Local variable for retaining the variables, may be used for
# extroversion
SISL_ENVIRON = {}


def register_environ_variable(name, default,
                              description=None,
                              process=None):
    """Register a new global sisl environment variable.

    Parameters
    -----------
    name: str or list-like of str
        the name of the environment variable. Needs to
        be correctly prefixed with "SISL_".
    default: any, optional
        the default value for this environment variable
    description: str, optional
        a description of what this variable does.
    process : callable, optional
        a callable which will be used to post-process the value when retrieving
        it

    Raises
    ------
    ValueError: if `name` does not start with "SISL_"
    """
    if not name.startswith("SISL_"):
        raise ValueError("register_environ_variable: name should start with 'SISL_'")
    if process is None:
        def process(arg):
            return arg

    global SISL_ENVIRON

    if name in SISL_ENVIRON:
        raise NameError(f"register_environ_variable: name {name} already registered")

    SISL_ENVIRON[name] = {
        "default": default,
        "description": description,
        "process": process,
    }


def get_environ_variable(name):
    """ Gets the value of a registered environment variable.

    Parameters
    -----------
    name: str
        the name of the environment variable.
    """
    variable = SISL_ENVIRON[name]
    value = variable["process"](os.environ.get(name, variable["default"]))
    variable["value"] = value
    return value


# We register a few variables that may be used several places
register_environ_variable("SISL_NPROCS", os.cpu_count(),
                          "Maximum number of CPU's used for parallel computing",
                          process=int)

register_environ_variable("SISL_TMP", ".sisl_tmp",
                          "Path where temporary files should be stored",
                          process=Path)

register_environ_variable("SISL_CONFIGDIR", Path.home() / ".config" / "sisl",
                          "Directory where configuration files for sisl should be stored",
                          process=Path)

register_environ_variable("SISL_FILES_TESTS", "_THIS_DIRECTORY_DOES_NOT_EXIST_",
                          """Full path of the sisl/files folder.
                          Generally this is only used for tests and for documentations.""",
                          process=Path)

register_environ_variable("SISL_VIZ_AUTOLOAD", "false",
                          """Determines whether the visualization module is automatically loaded.
                          It may be good to leave auto load off if you are doing performance critical
                          calculations to avoid the overhead of loading the visualization module.""",
                          process=lambda val: val and val.lower().strip() in ["1", "t", "true"])
