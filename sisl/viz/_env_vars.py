import os

__all__ = ['ENV_VARS']

ENV_VARS = {}

_PREFIX = 'SISL_VIZ'

def register_env_var(name, default, description=None):
    '''
    Registers a new environment variable.

    Parameters
    -----------
    name: str
        the name of the environment variable. If it doesn't
        start with SISL_VIZ, this prefix will be added.
    default: any
        the default value for this environment variable
    description: str
        a description of what this variable does.
    '''

    if not name.startswith(_PREFIX):
        name = f'{_PREFIX}_{name}'
    name = name.upper()

    ENV_VARS[name] = {'default': default, 'description': description}

    return get_env_var(name)

def get_env_var(name):
    '''
    Gets the value of a registered environment variable.

    Parameters
    -----------
    name: str
        the name of the environment variable. If it doesn't
        start with SISL_VIZ, this prefix will be added.
    '''

    if not name.startswith(_PREFIX):
        name = f'{_PREFIX}_{name}'
    name = name.upper()

    val = os.environ.get(name, ENV_VARS[name]['default'])

    ENV_VARS[name].update({'value': val})

    return val

    