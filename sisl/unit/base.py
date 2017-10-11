from __future__ import print_function, division

__all__ = ['unit_group', 'unit_convert', 'unit_default', 'unit_table_base']


# We do not import anything as it depends on the package.
# Here we only add the conversions according to the
# standard. Other programs may use their units as they
# please with non-standard conversion factors.

unit_table_base = {
    'mass': {
        'DEFAULT': 'amu',
        'kg': 1.,
        'g': 1.e-3,
        'amu': 1.66054e-27,
        },
    'length': {
        'DEFAULT': 'Ang',
        'm': 1.,
        'cm': 0.01,
        'nm': 1.e-9,
        'pm': 1.e-12,
        'fm': 1.e-15,
        'Ang': 1.e-10,
        'Bohr': 5.29177249e-11,
        },
    'time': {
        'DEFAULT': 'fs',
        's': 1.,
        'ns': 1.e-9,
        'ps': 1.e-12,
        'fs': 1.e-15,
        'min': 60.,
        'hour': 3600.,
        'day': 86400.,
        },
    'energy': {
        'DEFAULT': 'eV',
        'J': 1.,
        'erg': 1.e-7,
        'meV': 1.60217733e-22,
        'eV': 1.60217733e-19,
        'mRy': 2.1798741e-21,
        'Ry': 2.1798741e-18,
        'mHa': 4.3597482e-21,
        'Ha': 4.3597482e-18,
        'Hartree': 4.3597482e-18,
        'K': 1.380648780669e-23,
        },
    'force': {
        'DEFAULT': 'eV/Ang',
        'N': 1.,
        'eV/Ang': 1.60217733e-9,
        }
    }


def unit_group(unit, tbl=None):
    """ The group of units that `unit` belong to

    Parameters
    ----------
    unit : str
      unit, e.g. kg, Ang, eV etc. returns the type of unit it is.
    tbl : dict, optional
        dictionary of units (default to the global table)

    Examples
    --------
    >>> unit_group('kg')
    'mass'
    >>> unit_group('eV')
    'energy'
    """
    if tbl is None:
        global unit_table_base
        tbl = unit_table_base

    for k in tbl:
        if unit in tbl[k]:
            return k
    raise ValueError('The unit "'+str(unit)+'" could not be located in the table.')


def unit_default(group, tbl=None):
    """ The default unit of the unit group `group`.

    Parameters
    ----------
    group : str
       look-up in the table for the default unit.
    tbl : dict, optional
        dictionary of units (default to the global table)

    Examples
    --------
    >>> unit_default('energy')
    'eV'
    """
    if tbl is None:
        global unit_table_base
        tbl = unit_table_base

    for k in tbl:
        if group == k:
            return tbl[k]['DEFAULT']

    raise ValueError('The unit-group does not exist!')


def unit_convert(fr, to, opts=None, tbl=None):
    """ Factor that takes 'fr' to the units of 'to'.

    Parameters
    ----------
    fr : str
        starting unit
    to : str
        ending unit
    opts : dict, optional
        controls whether the unit conversion is in powers or fractional units
    tbl : dict, optional
        dictionary of units (default to the global table)

    Examples
    --------
    >>> unit_convert('kg','g')
    1000.0
    >>> unit_convert('eV','J')
    1.60217733e-19
    """
    if tbl is None:
        global unit_table_base
        tbl = unit_table_base
    if opts is None:
        opts = dict()

    # In the case that the conversion to is None, we should do nothing.
    frU = 'FromNotFound'
    frV = None
    toU = 'ToNotFound'
    toV = None

    # Check that the unit types live in the same
    # space
    # TODO this currently does not handle if powers are taken into
    # consideration.

    for k in tbl:
        if fr in tbl[k]:
            frU = k
            frV = tbl[k][fr]
        if to in tbl[k]:
            toU = k
            toV = tbl[k][to]
    if frU != toU:
        raise ValueError('The unit conversion is not from the same group: '+frU+' to '+toU)

    # Calculate conversion factor
    val = frV / toV
    for opt in ['^', 'power', 'p']:
        if opt in opts:
            val = val ** opts[opt]
    for opt in ['*', 'factor', 'fac']:
        if opt in opts:
            val = val * opts[opt]
    for opt in ['/', 'divide', 'div']:
        if opt in opts:
            val = val / opts[opt]

    return val
