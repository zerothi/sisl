""" Atomic information in different object containers.

Atomic information can be created and handled using the
`PeriodicTable` object or the `Atom` object, or lastly the combined
object `Atoms`.

* The `PeriodicTable` enables a *lookup* table for generic information
  about the atomic species in the periodic table of elements.
* The `Atom` enables creating atoms with associated information, such as

  * Mass
  * Species
  * Number of associated orbitals
  * Radii of each associated orbital
  * Custom tag (useful for denoting pseudo potentials)

* The `Atoms` object is a sorted, unique list of `Atom` such that one
  can contain a list of atoms. Instead of storing all `Atom` objects
  which may have *many* dublicates, the `Atoms` object has a unique
  list of `Atom` objects and an index list (`Atoms.specie`).
  This object enables a fast lookup of atoms without having
  to duplicate too much memory.

"""
from __future__ import print_function, division

# We need this for python3 support PY3
from six import with_metaclass

from numbers import Integral

import numpy as np

from ._help import array_fill_repeat, ensure_array, _str

__all__ = ['PeriodicTable', 'Atom', 'Atoms']


class PeriodicTable(object):
    """ Periodic table for creating an `Atom` via atomic numbers

    Enables *lookup* of atomic numbers/names/labels to get
    the atomic number.

    >>> 79 == PeriodicTable().Z('Au')
    >>> 79 == PeriodicTable().Z_int('Au')
    >>> 'Au' == PeriodicTable().Z_short(79)
    >>> 'Au' == PeriodicTable().Z_label(79)
    >>> 'Au' == PeriodicTable().Z_label('Gold')

    Several quantities available to the atomic species are available
    from
      <https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)>

    The following values are accesible:

    * atomic mass (in atomic units)
    * empirical atomic radii (in Ang)
    * calculated atomic radii (in Ang)
    * van der Waals atomic radii (in Ang)

    For certain species the above quantities are not available
    and a negative number is returned.

    Examples
    --------

    >>> 12.0107 == PeriodicTable().atomic_mass('C')
    True
    >>> 12.0107 == PeriodicTable().atomic_mass(6)
    True
    >>> 12.0107 == PeriodicTable().atomic_mass('Carbon')
    True
    >>> .67 == PeriodicTable().radii('Carbon')
    True
    >>> .67 == PeriodicTable().radii(6,'calc')
    True
    >>> .7  == PeriodicTable().radii(6,'empirical')
    True
    >>> 1.7 == PeriodicTable().radii(6,'vdw')
    True

    """
    _Z_int = {
        'Actinium': 89, 'Ac': 89, '89': 89, 89: 89,
        'Aluminum': 13, 'Al': 13, '13': 13, 13: 13,
        'Americium': 95, 'Am': 95, '95': 95, 95: 95,
        'Antimony': 51, 'Sb': 51, '51': 51, 51: 51,
        'Argon': 18, 'Ar': 18, '18': 18, 18: 18,
        'Arsenic': 33, 'As': 33, '33': 33, 33: 33,
        'Astatine': 85, 'At': 85, '85': 85, 85: 85,
        'Barium': 56, 'Ba': 56, '56': 56, 56: 56,
        'Berkelium': 97, 'Bk': 97, '97': 97, 97: 97,
        'Beryllium': 4, 'Be': 4, '4': 4, 4: 4,
        'Bismuth': 83, 'Bi': 83, '83': 83, 83: 83,
        'Bohrium': 107, 'Bh': 107, '107': 107, 107: 107,
        'Boron': 5, 'B': 5, '5': 5, 5: 5,
        'Bromine': 35, 'Br': 35, '35': 35, 35: 35,
        'Cadmium': 48, 'Cd': 48, '48': 48, 48: 48,
        'Calcium': 20, 'Ca': 20, '20': 20, 20: 20,
        'Californium': 98, 'Cf': 98, '98': 98, 98: 98,
        'Carbon': 6, 'C': 6, '6': 6, 6: 6,
        'Cerium': 58, 'Ce': 58, '58': 58, 58: 58,
        'Cesium': 55, 'Cs': 55, '55': 55, 55: 55,
        'Chlorine': 17, 'Cl': 17, '17': 17, 17: 17,
        'Chromium': 24, 'Cr': 24, '24': 24, 24: 24,
        'Cobalt': 27, 'Co': 27, '27': 27, 27: 27,
        'Copper': 29, 'Cu': 29, '29': 29, 29: 29,
        'Curium': 96, 'Cm': 96, '96': 96, 96: 96,
        'Darmstadtium': 110, 'Ds': 110, '110': 110, 110: 110,
        'Dubnium': 105, 'Db': 105, '105': 105, 105: 105,
        'Dysprosium': 66, 'Dy': 66, '66': 66, 66: 66,
        'Einsteinium': 99, 'Es': 99, '99': 99, 99: 99,
        'Erbium': 68, 'Er': 68, '68': 68, 68: 68,
        'Europium': 63, 'Eu': 63, '63': 63, 63: 63,
        'Fermium': 100, 'Fm': 100, '100': 100, 100: 100,
        'Fluorine': 9, 'F': 9, '9': 9, 9: 9,
        'Francium': 87, 'Fr': 87, '87': 87, 87: 87,
        'Gadolinium': 64, 'Gd': 64, '64': 64, 64: 64,
        'Gallium': 31, 'Ga': 31, '31': 31, 31: 31,
        'Germanium': 32, 'Ge': 32, '32': 32, 32: 32,
        'Gold': 79, 'Au': 79, '79': 79, 79: 79,
        'Hafnium': 72, 'Hf': 72, '72': 72, 72: 72,
        'Hassium': 108, 'Hs': 108, '108': 108, 108: 108,
        'Helium': 2, 'He': 2, '2': 2, 2: 2,
        'Holmium': 67, 'Ho': 67, '67': 67, 67: 67,
        'Hydrogen': 1, 'H': 1, '1': 1, 1: 1,
        'Indium': 49, 'In': 49, '49': 49, 49: 49,
        'Iodine': 53, 'I': 53, '53': 53, 53: 53,
        'Iridium': 77, 'Ir': 77, '77': 77, 77: 77,
        'Iron': 26, 'Fe': 26, '26': 26, 26: 26,
        'Krypton': 36, 'Kr': 36, '36': 36, 36: 36,
        'Lanthanum': 57, 'La': 57, '57': 57, 57: 57,
        'Lawrencium': 103, 'Lr': 103, '103': 103, 103: 103,
        'Lead': 82, 'Pb': 82, '82': 82, 82: 82,
        'Lithium': 3, 'Li': 3, '3': 3, 3: 3,
        'Lutetium': 71, 'Lu': 71, '71': 71, 71: 71,
        'Magnesium': 12, 'Mg': 12, '12': 12, 12: 12,
        'Manganese': 25, 'Mn': 25, '25': 25, 25: 25,
        'Meitnerium': 109, 'Mt': 109, '109': 109, 109: 109,
        'Mendelevium': 101, 'Md': 101, '101': 101, 101: 101,
        'Mercury': 80, 'Hg': 80, '80': 80, 80: 80,
        'Molybdenum': 42, 'Mo': 42, '42': 42, 42: 42,
        'Neodymium': 60, 'Nd': 60, '60': 60, 60: 60,
        'Neon': 10, 'Ne': 10, '10': 10, 10: 10,
        'Neptunium': 93, 'Np': 93, '93': 93, 93: 93,
        'Nickel': 28, 'Ni': 28, '28': 28, 28: 28,
        'Niobium': 41, 'Nb': 41, '41': 41, 41: 41,
        'Nitrogen': 7, 'N': 7, '7': 7, 7: 7,
        'Nobelium': 102, 'No': 102, '102': 102, 102: 102,
        'Osmium': 76, 'Os': 76, '76': 76, 76: 76,
        'Oxygen': 8, 'O': 8, '8': 8, 8: 8,
        'Palladium': 46, 'Pd': 46, '46': 46, 46: 46,
        'Phosphorus': 15, 'P': 15, '15': 15, 15: 15,
        'Platinum': 78, 'Pt': 78, '78': 78, 78: 78,
        'Plutonium': 94, 'Pu': 94, '94': 94, 94: 94,
        'Polonium': 84, 'Po': 84, '84': 84, 84: 84,
        'Potassium': 19, 'K': 19, '19': 19, 19: 19,
        'Praseodymium': 59, 'Pr': 59, '59': 59, 59: 59,
        'Promethium': 61, 'Pm': 61, '61': 61, 61: 61,
        'Protactinium': 91, 'Pa': 91, '91': 91, 91: 91,
        'Radium': 88, 'Ra': 88, '88': 88, 88: 88,
        'Radon': 86, 'Rn': 86, '86': 86, 86: 86,
        'Rhenium': 75, 'Re': 75, '75': 75, 75: 75,
        'Rhodium': 45, 'Rh': 45, '45': 45, 45: 45,
        'Rubidium': 37, 'Rb': 37, '37': 37, 37: 37,
        'Ruthenium': 44, 'Ru': 44, '44': 44, 44: 44,
        'Rutherfordium': 104, 'Rf': 104, '104': 104, 104: 104,
        'Samarium': 62, 'Sm': 62, '62': 62, 62: 62,
        'Scandium': 21, 'Sc': 21, '21': 21, 21: 21,
        'Seaborgium': 106, 'Sg': 106, '106': 106, 106: 106,
        'Selenium': 34, 'Se': 34, '34': 34, 34: 34,
        'Silicon': 14, 'Si': 14, '14': 14, 14: 14,
        'Silver': 47, 'Ag': 47, '47': 47, 47: 47,
        'Sodium': 11, 'Na': 11, '11': 11, 11: 11,
        'Strontium': 38, 'Sr': 38, '38': 38, 38: 38,
        'Sulfur': 16, 'S': 16, '16': 16, 16: 16,
        'Tantalum': 73, 'Ta': 73, '73': 73, 73: 73,
        'Technetium': 43, 'Tc': 43, '43': 43, 43: 43,
        'Tellurium': 52, 'Te': 52, '52': 52, 52: 52,
        'Terbium': 65, 'Tb': 65, '65': 65, 65: 65,
        'Thallium': 81, 'Tl': 81, '81': 81, 81: 81,
        'Thorium': 90, 'Th': 90, '90': 90, 90: 90,
        'Thulium': 69, 'Tm': 69, '69': 69, 69: 69,
        'Tin': 50, 'Sn': 50, '50': 50, 50: 50,
        'Titanium': 22, 'Ti': 22, '22': 22, 22: 22,
        'Tungsten': 74, 'W': 74, '74': 74, 74: 74,
        'Ununbium': 112, 'Uub': 112, '112': 112, 112: 112,
        'Ununhexium': 116, 'Uuh': 116, '116': 116, 116: 116,
        'Ununoctium': 118, 'Uuo': 118, '118': 118, 118: 118,
        'Ununpentium': 115, 'Uup': 115, '115': 115, 115: 115,
        'Ununquadium': 114, 'Uuq': 114, '114': 114, 114: 114,
        'Ununseptium': 117, 'Uus': 117, '117': 117, 117: 117,
        'Ununtrium': 113, 'Uut': 113, '113': 113, 113: 113,
        'Ununium': 111, 'Uuu': 111, '111': 111, 111: 111,
        'Uranium': 92, 'U': 92, '92': 92, 92: 92,
        'Vanadium': 23, 'V': 23, '23': 23, 23: 23,
        'Xenon': 54, 'Xe': 54, '54': 54, 54: 54,
        'Ytterbium': 70, 'Yb': 70, '70': 70, 70: 70,
        'Yttrium': 39, 'Y': 39, '39': 39, 39: 39,
        'Zinc': 30, 'Zn': 30, '30': 30, 30: 30,
        'Zirconium': 40, 'Zr': 40, '40': 40, 40: 40,
    }

    _Z_short = {
        'Actinium': 'Ac', 'Ac': 'Ac', '89': 'Ac', 89: 'Ac',
        'Aluminum': 'Al', 'Al': 'Al', '13': 'Al', 13: 'Al',
        'Americium': 'Am', 'Am': 'Am', '95': 'Am', 95: 'Am',
        'Antimony': 'Sb', 'Sb': 'Sb', '51': 'Sb', 51: 'Sb',
        'Argon': 'Ar', 'Ar': 'Ar', '18': 'Ar', 18: 'Ar',
        'Arsenic': 'As', 'As': 'As', '33': 'As', 33: 'As',
        'Astatine': 'At', 'At': 'At', '85': 'At', 85: 'At',
        'Barium': 'Ba', 'Ba': 'Ba', '56': 'Ba', 56: 'Ba',
        'Berkelium': 'Bk', 'Bk': 'Bk', '97': 'Bk', 97: 'Bk',
        'Beryllium': 'Be', 'Be': 'Be', '4': 'Be', 4: 'Be',
        'Bismuth': 'Bi', 'Bi': 'Bi', '83': 'Bi', 83: 'Bi',
        'Bohrium': 'Bh', 'Bh': 'Bh', '107': 'Bh', 107: 'Bh',
        'Boron': 'B', 'B': 'B', '5': 'B', 5: 'B',
        'Bromine': 'Br', 'Br': 'Br', '35': 'Br', 35: 'Br',
        'Cadmium': 'Cd', 'Cd': 'Cd', '48': 'Cd', 48: 'Cd',
        'Calcium': 'Ca', 'Ca': 'Ca', '20': 'Ca', 20: 'Ca',
        'Californium': 'Cf', 'Cf': 'Cf', '98': 'Cf', 98: 'Cf',
        'Carbon': 'C', 'C': 'C', '6': 'C', 6: 'C',
        'Cerium': 'Ce', 'Ce': 'Ce', '58': 'Ce', 58: 'Ce',
        'Cesium': 'Cs', 'Cs': 'Cs', '55': 'Cs', 55: 'Cs',
        'Chlorine': 'Cl', 'Cl': 'Cl', '17': 'Cl', 17: 'Cl',
        'Chromium': 'Cr', 'Cr': 'Cr', '24': 'Cr', 24: 'Cr',
        'Cobalt': 'Co', 'Co': 'Co', '27': 'Co', 27: 'Co',
        'Copper': 'Cu', 'Cu': 'Cu', '29': 'Cu', 29: 'Cu',
        'Curium': 'Cm', 'Cm': 'Cm', '96': 'Cm', 96: 'Cm',
        'Darmstadtium': 'Ds', 'Ds': 'Ds', '110': 'Ds', 110: 'Ds',
        'Dubnium': 'Db', 'Db': 'Db', '105': 'Db', 105: 'Db',
        'Dysprosium': 'Dy', 'Dy': 'Dy', '66': 'Dy', 66: 'Dy',
        'Einsteinium': 'Es', 'Es': 'Es', '99': 'Es', 99: 'Es',
        'Erbium': 'Er', 'Er': 'Er', '68': 'Er', 68: 'Er',
        'Europium': 'Eu', 'Eu': 'Eu', '63': 'Eu', 63: 'Eu',
        'Fermium': 'Fm', 'Fm': 'Fm', '100': 'Fm', 100: 'Fm',
        'Fluorine': 'F', 'F': 'F', '9': 'F', 9: 'F',
        'Francium': 'Fr', 'Fr': 'Fr', '87': 'Fr', 87: 'Fr',
        'Gadolinium': 'Gd', 'Gd': 'Gd', '64': 'Gd', 64: 'Gd',
        'Gallium': 'Ga', 'Ga': 'Ga', '31': 'Ga', 31: 'Ga',
        'Germanium': 'Ge', 'Ge': 'Ge', '32': 'Ge', 32: 'Ge',
        'Gold': 'Au', 'Au': 'Au', '79': 'Au', 79: 'Au',
        'Hafnium': 'Hf', 'Hf': 'Hf', '72': 'Hf', 72: 'Hf',
        'Hassium': 'Hs', 'Hs': 'Hs', '108': 'Hs', 108: 'Hs',
        'Helium': 'He', 'He': 'He', '2': 'He', 2: 'He',
        'Holmium': 'Ho', 'Ho': 'Ho', '67': 'Ho', 67: 'Ho',
        'Hydrogen': 'H', 'H': 'H', '1': 'H', 1: 'H',
        'Indium': 'In', 'In': 'In', '49': 'In', 49: 'In',
        'Iodine': 'I', 'I': 'I', '53': 'I', 53: 'I',
        'Iridium': 'Ir', 'Ir': 'Ir', '77': 'Ir', 77: 'Ir',
        'Iron': 'Fe', 'Fe': 'Fe', '26': 'Fe', 26: 'Fe',
        'Krypton': 'Kr', 'Kr': 'Kr', '36': 'Kr', 36: 'Kr',
        'Lanthanum': 'La', 'La': 'La', '57': 'La', 57: 'La',
        'Lawrencium': 'Lr', 'Lr': 'Lr', '103': 'Lr', 103: 'Lr',
        'Lead': 'Pb', 'Pb': 'Pb', '82': 'Pb', 82: 'Pb',
        'Lithium': 'Li', 'Li': 'Li', '3': 'Li', 3: 'Li',
        'Lutetium': 'Lu', 'Lu': 'Lu', '71': 'Lu', 71: 'Lu',
        'Magnesium': 'Mg', 'Mg': 'Mg', '12': 'Mg', 12: 'Mg',
        'Manganese': 'Mn', 'Mn': 'Mn', '25': 'Mn', 25: 'Mn',
        'Meitnerium': 'Mt', 'Mt': 'Mt', '109': 'Mt', 109: 'Mt',
        'Mendelevium': 'Md', 'Md': 'Md', '101': 'Md', 101: 'Md',
        'Mercury': 'Hg', 'Hg': 'Hg', '80': 'Hg', 80: 'Hg',
        'Molybdenum': 'Mo', 'Mo': 'Mo', '42': 'Mo', 42: 'Mo',
        'Neodymium': 'Nd', 'Nd': 'Nd', '60': 'Nd', 60: 'Nd',
        'Neon': 'Ne', 'Ne': 'Ne', '10': 'Ne', 10: 'Ne',
        'Neptunium': 'Np', 'Np': 'Np', '93': 'Np', 93: 'Np',
        'Nickel': 'Ni', 'Ni': 'Ni', '28': 'Ni', 28: 'Ni',
        'Niobium': 'Nb', 'Nb': 'Nb', '41': 'Nb', 41: 'Nb',
        'Nitrogen': 'N', 'N': 'N', '7': 'N', 7: 'N',
        'Nobelium': 'No', 'No': 'No', '102': 'No', 102: 'No',
        'Osmium': 'Os', 'Os': 'Os', '76': 'Os', 76: 'Os',
        'Oxygen': 'O', 'O': 'O', '8': 'O', 8: 'O',
        'Palladium': 'Pd', 'Pd': 'Pd', '46': 'Pd', 46: 'Pd',
        'Phosphorus': 'P', 'P': 'P', '15': 'P', 15: 'P',
        'Platinum': 'Pt', 'Pt': 'Pt', '78': 'Pt', 78: 'Pt',
        'Plutonium': 'Pu', 'Pu': 'Pu', '94': 'Pu', 94: 'Pu',
        'Polonium': 'Po', 'Po': 'Po', '84': 'Po', 84: 'Po',
        'Potassium': 'K', 'K': 'K', '19': 'K', 19: 'K',
        'Praseodymium': 'Pr', 'Pr': 'Pr', '59': 'Pr', 59: 'Pr',
        'Promethium': 'Pm', 'Pm': 'Pm', '61': 'Pm', 61: 'Pm',
        'Protactinium': 'Pa', 'Pa': 'Pa', '91': 'Pa', 91: 'Pa',
        'Radium': 'Ra', 'Ra': 'Ra', '88': 'Ra', 88: 'Ra',
        'Radon': 'Rn', 'Rn': 'Rn', '86': 'Rn', 86: 'Rn',
        'Rhenium': 'Re', 'Re': 'Re', '75': 'Re', 75: 'Re',
        'Rhodium': 'Rh', 'Rh': 'Rh', '45': 'Rh', 45: 'Rh',
        'Rubidium': 'Rb', 'Rb': 'Rb', '37': 'Rb', 37: 'Rb',
        'Ruthenium': 'Ru', 'Ru': 'Ru', '44': 'Ru', 44: 'Ru',
        'Rutherfordium': 'Rf', 'Rf': 'Rf', '104': 'Rf', 104: 'Rf',
        'Samarium': 'Sm', 'Sm': 'Sm', '62': 'Sm', 62: 'Sm',
        'Scandium': 'Sc', 'Sc': 'Sc', '21': 'Sc', 21: 'Sc',
        'Seaborgium': 'Sg', 'Sg': 'Sg', '106': 'Sg', 106: 'Sg',
        'Selenium': 'Se', 'Se': 'Se', '34': 'Se', 34: 'Se',
        'Silicon': 'Si', 'Si': 'Si', '14': 'Si', 14: 'Si',
        'Silver': 'Ag', 'Ag': 'Ag', '47': 'Ag', 47: 'Ag',
        'Sodium': 'Na', 'Na': 'Na', '11': 'Na', 11: 'Na',
        'Strontium': 'Sr', 'Sr': 'Sr', '38': 'Sr', 38: 'Sr',
        'Sulfur': 'S', 'S': 'S', '16': 'S', 16: 'S',
        'Tantalum': 'Ta', 'Ta': 'Ta', '73': 'Ta', 73: 'Ta',
        'Technetium': 'Tc', 'Tc': 'Tc', '43': 'Tc', 43: 'Tc',
        'Tellurium': 'Te', 'Te': 'Te', '52': 'Te', 52: 'Te',
        'Terbium': 'Tb', 'Tb': 'Tb', '65': 'Tb', 65: 'Tb',
        'Thallium': 'Tl', 'Tl': 'Tl', '81': 'Tl', 81: 'Tl',
        'Thorium': 'Th', 'Th': 'Th', '90': 'Th', 90: 'Th',
        'Thulium': 'Tm', 'Tm': 'Tm', '69': 'Tm', 69: 'Tm',
        'Tin': 'Sn', 'Sn': 'Sn', '50': 'Sn', 50: 'Sn',
        'Titanium': 'Ti', 'Ti': 'Ti', '22': 'Ti', 22: 'Ti',
        'Tungsten': 'W', 'W': 'W', '74': 'W', 74: 'W',
        'Ununbium': 'Uub', 'Uub': 'Uub', '112': 'Uub', 112: 'Uub',
        'Ununhexium': 'Uuh', 'Uuh': 'Uuh', '116': 'Uuh', 116: 'Uuh',
        'Ununoctium': 'Uuo', 'Uuo': 'Uuo', '118': 'Uuo', 118: 'Uuo',
        'Ununpentium': 'Uup', 'Uup': 'Uup', '115': 'Uup', 115: 'Uup',
        'Ununquadium': 'Uuq', 'Uuq': 'Uuq', '114': 'Uuq', 114: 'Uuq',
        'Ununseptium': 'Uus', 'Uus': 'Uus', '117': 'Uus', 117: 'Uus',
        'Ununtrium': 'Uut', 'Uut': 'Uut', '113': 'Uut', 113: 'Uut',
        'Ununium': 'Uuu', 'Uuu': 'Uuu', '111': 'Uuu', 111: 'Uuu',
        'Uranium': 'U', 'U': 'U', '92': 'U', 92: 'U',
        'Vanadium': 'V', 'V': 'V', '23': 'V', 23: 'V',
        'Xenon': 'Xe', 'Xe': 'Xe', '54': 'Xe', 54: 'Xe',
        'Ytterbium': 'Yb', 'Yb': 'Yb', '70': 'Yb', 70: 'Yb',
        'Yttrium': 'Y', 'Y': 'Y', '39': 'Y', 39: 'Y',
        'Zinc': 'Zn', 'Zn': 'Zn', '30': 'Zn', 30: 'Zn',
        'Zirconium': 'Zr', 'Zr': 'Zr', '40': 'Zr', 40: 'Zr',
    }

    _atomic_mass = {
        1: 1.00794,
        2: 4.002602,
        3: 6.941,
        4: 9.012182,
        5: 10.811,
        6: 12.0107,
        7: 14.0067,
        8: 15.9994,
        9: 18.9984032,
        10: 20.1797,
        11: 22.98976928,
        12: 24.3050,
        13: 26.9815386,
        14: 28.0855,
        15: 30.973762,
        16: 32.065,
        17: 35.453,
        18: 39.948,
        19: 39.0983,
        20: 40.078,
        21: 44.955912,
        22: 47.867,
        23: 50.9415,
        24: 51.9961,
        25: 54.938045,
        26: 55.845,
        27: 58.933195,
        28: 58.6934,
        29: 63.546,
        30: 65.409,
        31: 69.723,
        32: 72.64,
        33: 74.92160,
        34: 78.96,
        35: 79.904,
        36: 83.798,
        37: 85.4678,
        38: 87.62,
        39: 88.90585,
        40: 91.224,
        41: 92.906,
        42: 95.94,
        43: 98.,
        44: 101.07,
        45: 102.905,
        46: 106.42,
        47: 107.8682,
        48: 112.411,
        49: 114.818,
        50: 118.710,
        51: 121.760,
        52: 127.60,
        53: 126.904,
        54: 131.293,
        55: 132.9054519,
        56: 137.327,
        57: 138.90547,
        58: 140.116,
        59: 140.90765,
        60: 144.242,
        61: 145.,
        62: 150.36,
        63: 151.964,
        64: 157.25,
        65: 158.92535,
        66: 162.500,
        67: 164.930,
        68: 167.259,
        69: 168.93421,
        70: 173.04,
        71: 174.967,
        72: 178.49,
        73: 180.94788,
        74: 183.84,
        75: 186.207,
        76: 190.23,
        77: 192.217,
        78: 195.084,
        79: 196.966569,
        80: 200.59,
        81: 204.3833,
        82: 207.2,
        83: 208.98040,
        84: 210.,
        85: 210.,
        86: 220.,
        87: 223.,
        88: 226.,
        89: 227.,
        91: 231.03588,
        90: 232.03806,
        93: 237.,
        92: 238.02891,
        95: 243.,
        94: 244.,
        96: 247.,
        97: 247.,
        98: 251.,
        99: 252.,
        100: 257.,
        101: 258.,
        102: 259.,
        103: 262.,
        104: 261.,
        105: 262.,
        106: 266.,
        107: 264.,
        108: 277.,
        109: 268.,
        110: 271.,
        111: 272.,
        112: 285.,
        113: 284.,
        114: 289.,
        115: 288.,
        116: 292.,
        118: 293.,
    }

    _radius_empirical = {
        1: 25,
        2: -1,
        3: 145,
        4: 105,
        5: 85,
        6: 70,
        7: 65,
        8: 60,
        9: 50,
        10: -1,
        11: 180,
        12: 150,
        13: 125,
        14: 110,
        15: 100,
        16: 100,
        17: 100,
        18: 71,
        19: 220,
        20: 180,
        21: 160,
        22: 140,
        23: 135,
        24: 140,
        25: 140,
        26: 140,
        27: 135,
        28: 135,
        29: 135,
        30: 135,
        31: 130,
        32: 125,
        33: 115,
        34: 115,
        35: 115,
        36: -1,
        37: 235,
        38: 200,
        39: 180,
        40: 155,
        41: 145,
        42: 145,
        43: 135,
        44: 130,
        45: 135,
        46: 140,
        47: 160,
        48: 155,
        49: 155,
        50: 145,
        51: 145,
        52: 140,
        53: 140,
        54: -1,
        55: 260,
        56: 215,
        57: 195,
        58: 185,
        59: 185,
        60: 185,
        61: 185,
        62: 185,
        63: 185,
        64: 180,
        65: 175,
        66: 175,
        67: 175,
        68: 175,
        69: 175,
        70: 175,
        71: 175,
        72: 155,
        73: 145,
        74: 135,
        75: 135,
        76: 130,
        77: 135,
        78: 135,
        79: 135,
        80: 150,
        81: 190,
        82: 180,
        83: 160,
        84: 190,
        85: -1,
        86: -1,
        87: -1,
        88: 215,
        89: 195,
        90: 180,
        91: 180,
        92: 175,
        93: 175,
        94: 175,
        95: 175,
        96: -1,
        97: -1,
        98: -1,
        99: -1,
        100: -1,
        101: -1,
        102: -1,
        103: -1,
        104: -1,
        105: -1,
        106: -1,
        107: -1,
        108: -1,
        109: -1,
        110: -1,
        111: -1,
        112: -1,
        113: -1,
        114: -1,
        115: -1,
        116: -1,
        117: -1,
        118: -1,
    }

    _radius_calc = {
        1: 53,
        2: 31,
        3: 167,
        4: 112,
        5: 87,
        6: 67,
        7: 56,
        8: 48,
        9: 42,
        10: 38,
        11: 190,
        12: 145,
        13: 118,
        14: 111,
        15: 98,
        16: 88,
        17: 79,
        18: 71,
        19: 243,
        20: 194,
        21: 184,
        22: 176,
        23: 171,
        24: 166,
        25: 161,
        26: 156,
        27: 152,
        28: 149,
        29: 145,
        30: 142,
        31: 136,
        32: 125,
        33: 114,
        34: 103,
        35: 94,
        36: 88,
        37: 265,
        38: 219,
        39: 212,
        40: 206,
        41: 198,
        42: 190,
        43: 183,
        44: 178,
        45: 173,
        46: 169,
        47: 165,
        48: 161,
        49: 156,
        50: 145,
        51: 133,
        52: 123,
        53: 115,
        54: 108,
        55: 298,
        56: 253,
        57: -1,
        58: -1,
        59: 247,
        60: 206,
        61: 205,
        62: 238,
        63: 231,
        64: 233,
        65: 225,
        66: 228,
        67: -1,
        68: 226,
        69: 222,
        70: 222,
        71: 217,
        72: 208,
        73: 200,
        74: 193,
        75: 188,
        76: 185,
        77: 180,
        78: 177,
        79: 174,
        80: 171,
        81: 156,
        82: 154,
        83: 143,
        84: 135,
        85: -1,
        86: 120,
        87: -1,
        88: -1,
        89: -1,
        90: -1,
        91: -1,
        92: -1,
        93: -1,
        94: -1,
        95: -1,
        96: -1,
        97: -1,
        98: -1,
        99: -1,
        100: -1,
        101: -1,
        102: -1,
        103: -1,
        104: -1,
        105: -1,
        106: -1,
        107: -1,
        108: -1,
        109: -1,
        110: -1,
        111: -1,
        112: -1,
        113: -1,
        114: -1,
        115: -1,
        116: -1,
        117: -1,
        118: -1,
    }

    _radius_vdw = {
        1: 120,
        2: 140,
        3: 182,
        4: 153,
        5: 192,
        6: 170,
        7: 155,
        8: 152,
        9: 147,
        10: 154,
        11: 227,
        12: 173,
        13: 184,
        14: 210,
        15: 180,
        16: 180,
        17: 175,
        18: 188,
        19: 275,
        20: 231,
        21: 211,
        22: -1,
        23: -1,
        24: -1,
        25: -1,
        26: -1,
        27: -1,
        28: 163,
        29: 140,
        30: 139,
        31: 187,
        32: 211,
        33: 185,
        34: 190,
        35: 185,
        36: 202,
        37: 303,
        38: 249,
        39: -1,
        40: -1,
        41: -1,
        42: -1,
        43: -1,
        44: -1,
        45: -1,
        46: 163,
        47: 172,
        48: 158,
        49: 193,
        50: 217,
        51: 206,
        52: 206,
        53: 198,
        54: 216,
        55: 343,
        56: 268,
        57: -1,
        58: -1,
        59: -1,
        60: -1,
        61: -1,
        62: -1,
        63: -1,
        64: -1,
        65: -1,
        66: -1,
        67: -1,
        68: -1,
        69: -1,
        70: -1,
        71: -1,
        72: -1,
        73: -1,
        74: -1,
        75: -1,
        76: -1,
        77: -1,
        78: 175,
        79: 166,
        80: 155,
        81: 196,
        82: 202,
        83: 207,
        84: 197,
        85: 202,
        86: 220,
        87: 348,
        88: 283,
        89: -1,
        90: -1,
        91: -1,
        92: 186,
        93: -1,
        94: -1,
        95: -1,
        96: -1,
        97: -1,
        98: -1,
        99: -1,
        100: -1,
        101: -1,
        102: -1,
        103: -1,
        104: -1,
        105: -1,
        106: -1,
        107: -1,
        108: -1,
        109: -1,
        110: -1,
        111: -1,
        112: -1,
        113: -1,
        114: -1,
        115: -1,
        116: -1,
        117: -1,
        118: -1,
    }

    def Z(self, key):
        """ Return the atomic number based on general input

        Return the atomic number corresponding to the `key` lookup.

        Parameters
        ----------
        key : array_like or str or int
            Uses value to lookup the atomic number in the `PeriodicTable`
            object.

        Returns
        -------
        Z : ndarray or int
            The atomic number corresponding to `key`, if `key` is array_like, so
            will the returned value be.

        Examples
        --------

        >>> 79 == PeriodicTable().Z_int('Au')
        True
        >>> 79 == PeriodicTable().Z('Au')
        True
        >>> 6 == PeriodicTable().Z('Carbon')
        True
        """
        ak = np.asarray([key]).flatten()
        if len(ak) == 1:
            return self._Z_int[ak[0]]
        return np.array([self._Z_int[i] for i in ak], np.int32)

    Z_int = Z

    def Z_label(self, key):
        """ Return the atomic label of the corresponding atom

        Return the atomic short name corresponding to the `key` lookup.

        Parameters
        ----------
        key : array_like or str or int
            Uses value to lookup the atomic short name in the
            `PeriodicTable` object.

        Returns
        -------
        name : ndarra or str
            The atomic short name corresponding to `key`, if `key`
            is array_like, so will the returned value be.
        """
        ak = np.asarray([key]).flatten()
        if len(ak) == 1:
            return self._Z_short[ak[0]]
        return [self._Z_short[i] for i in ak]

    Z_short = Z_label

    def atomic_mass(self, key):
        """ Return the atomic mass of the corresponding atom

        Return the atomic mass corresponding to the `key` lookup.

        Parameters
        ----------
        key : array_like or str or int
            Uses value to lookup the atomic mass in the
            `PeriodicTable` object.

        Returns
        -------
        name : ndarray or float
            The atomic mass in atomic units corresponding to `key`,
            if `key` is array_like, so will the returned value be.
        """
        Z = self.Z_int(key)
        if isinstance(Z, Integral):
            return self._atomic_mass[Z]
        return np.array([self._atomic_mass[i] for i in Z], np.float64)

    def radius(self, key, method='calc'):
        """ Return the atomic radii

        Return the atomic radii.

        Parameters
        ----------
        key : array_like or str or int
            Uses value to lookup the atomic mass in the
            `PeriodicTable` object.
        method : {'calc', 'empirical', 'vdw'}
            There are 3 different radii stored:

             1. ``calc``, the calculated atomic radii
             2. ``empirical``, the empirically found values
             3. ``vdw``, the van-der-Waals found values

        Returns
        -------
        radius : ndarray or float
            The atomic radius in `Ang`
        """
        Z = self.Z_int(key)
        if method == 'calc':
            if isinstance(Z, Integral):
                return self._radius_calc[Z] / 100
            return np.array([self._radius_calc[i]
                             for i in Z], np.float64) / 100
        elif method == 'empirical':
            if isinstance(Z, Integral):
                return self._radius_empirical[Z] / 100
            return np.array([self._radius_empirical[i]
                             for i in Z], np.float64) / 100
        elif method == 'vdw':
            if isinstance(Z, Integral):
                return self._radius_vdw[Z] / 100
            return np.array([self._radius_vdw[i] for i in Z], np.float64) / 100
        raise ValueError(
            'Method option could not be deciphered [calc|empirical|vdw].')
    radii = radius

# Create a local instance of the periodic table to
# faster look up
_ptbl = PeriodicTable()


class AtomMeta(type):
    """ Meta class for key-lookup on the class. """
    def __getitem__(cls, key):
        """ Create a new atom object """
        if isinstance(key, Atom):
            # if the key already is an atomic object
            # return it
            return key
        if isinstance(key, dict):
            # The key is a dictionary, hence
            # we can return the atom directly
            return cls(**key)
        if isinstance(key, list):
            # The key is a list,
            # we need to create a list of atoms
            atm = [cls[k] for k in key]
            return np.array(atm, dtype=Atom)
        if isinstance(key, cls):
            # if the key already is an atomic object
            # return it
            return key
        # Index Z based
        return cls(key)


# Note the with_metaclass which is required for python3 support.
# The designation of metaclass in python3 is actually:
#   class ...(..., metaclass=MetaClass)
# This below construct handles both python2 and python3 cases
class Atom(with_metaclass(AtomMeta, object)):
    """ Atomic information, mass, name number of orbitals and ranges

    Object to handle atomic mass, name, number of orbitals and
    orbital range.

    The `Atom` object handles the atomic species with information
    such as

    * atomic number
    * mass
    * number of orbitals
    * radius of each orbital

    The `Atom` object is `pickle`-able.

    Attributes
    ----------
    Z : int
        atomic number
    R : ndarray
        radius of orbitals belonging to the `Atom`
    orbs : int
        number of orbitals belonging to the `Atom`
    mass : float
        mass of `Atom`

    Parameters
    ----------
    Z : int or str
        key lookup for the atomic specie, `Atom[key]`
    R : array_like or float
        the range of the atomic orbitals
    orbs : int
        number of orbitals attached to this atom
        NOTE: Length of `R` precedes this quantity.
    mass : float, optional
        the atomic mass, if not specified uses the mass from `PeriodicTable`
    tag : str
        arbitrary designation for user handling similar atoms with
        different settings
    """

    def __init__(self, Z, R=None, orbs=None, mass=None, tag=None):
        if isinstance(Z, Atom):
            Z = Z.Z
        if R is None:
            R = -1.
        if orbs is None:
            orbs = 1

        self.Z = _ptbl.Z_int(Z)
        self.orbs = orbs
        try:
            self.orbs = max(len(R), orbs)
        except:
            pass
        self.R = array_fill_repeat(
            np.asarray([R], np.float64).flatten(),
            self.orbs)
        # Save the mass
        self.mass = mass
        if mass is None:
            self.mass = _ptbl.atomic_mass(self.Z)
        if tag is None:
            self.tag = self.symbol
        else:
            self.tag = tag

    def copy(self, Z=None, R=None, orbs=None, mass=None, tag=None):
        """ Return copy of this object """
        return self.__class__(self.Z if Z is None else Z,
                              self.R if R is None else R,
                              self.orbs if orbs is None else orbs,
                              self.mass if mass is None else mass,
                              self.tag if tag is None else tag)

    def radius(self, method='calc'):
        """ Return the atomic radii of the atom (in Ang) 

        See `PeriodicTable.radius` for details on the argument.
        """
        return _ptbl.radius(self.Z, method)
    radii = radius

    @property
    def symbol(self):
        """ Return short atomic name (Au==79). """
        return _ptbl.Z_short(self.Z)

    @property
    def dR(self):
        """ Return the maximum range of orbitals. """
        return np.amax(self.R)

    def scale(self, scale):
        """ Scale the atomic radii and return an equivalent atom.

        Parameters
        ----------
        scale : float
           the scale factor for the atomic radii
        """
        new = self.copy()
        new.R = np.where(new.R > 0, new.R * scale, new.R)
        return new

    def __repr__(self):
        return '{0}, Z: {1:d}, orbs: {2:d}, mass(au): {3:.5f}, dR: {4:.5f}'.format(self.tag, self.Z, self.orbs, self.mass, self.dR)

    def __len__(self):
        """ Return number of orbitals in this atom """
        return self.orbs

    # Check whether they are equal
    def __eq__(a, b):
        """ Return true if the saved quantities are the same """
        if not isinstance(b, Atom):
            return False
        same = a.Z == b.Z
        same &= a.orbs == b.orbs
        same &= a.mass == b.mass
        same &= a.tag == b.tag
        # This prevents an allclose being called with
        # different number of orbitals
        if same:
            same = np.allclose(a.R, b.R)
        return same

    def __ne__(a, b):
        return not (a == b)

    # Create pickling routines
    def __getstate__(self):
        """ Return the state of this object """
        return {'Z': self.Z,
                'orbs': self.orbs,
                'mass': self.mass,
                'tag': self.tag,
                'R': self.R}

    def __setstate__(self, d):
        """ Re-create the state of this object """
        self.__init__(
            d['Z'],
            R=d['R'],
            orbs=d['orbs'],
            mass=d['mass'],
            tag=d['tag'])


class Atoms(object):
    """ A list-like object to contain a list of different atoms with minimum
    data duplication.

    This holds multiple `Atom` objects which are indexed via a species
    index.
    This is convenient when having geometries with millions of atoms
    because it will not duplicate the `Atom` object, only a list index.

    Attributes
    ----------
    atom : `list(Atom)`
        a list of unique atoms in this object
    specie : ndarray(int16)
        a list of unique specie indices
    """

    # Using the slots should make this routine slightly faster.
    __slots__ = ['_atom', '_specie']

    def __init__(self, atom=None, na=None):

        # Default value of the atom object
        if atom is None:
            atom = Atom('H')

        # Correct the atoms input to Atom
        if isinstance(atom, (np.ndarray, list, tuple)):
            # Convert to a list of unique elements
            # We can not use set because that is unordered
            # And we want the same order, always...
            uatom = []
            specie = [0] * len(atom)
            if isinstance(atom[0], Atom):
                for i, a in enumerate(atom):
                    try:
                        s = uatom.index(a)
                    except:
                        s = -1
                    if s < 0:
                        s = len(uatom)
                        uatom.append(a)
                    specie[i] = s

            elif isinstance(atom[0], (_str, Integral)):
                for i, a in enumerate(atom):
                    a = Atom(a)
                    try:
                        s = uatom.index(a)
                    except:
                        s = -1
                    if s < 0:
                        s = len(uatom)
                        uatom.append(a)
                    specie[i] = s

            else:
                raise ValueError('atom keyword was wrong input')

        elif isinstance(atom, (_str, Integral)):
            uatom = [Atom(atom)]
            specie = [0]

        elif isinstance(atom, Atom):
            uatom = [atom]
            specie = [0]

        elif isinstance(atom, Atoms):
            # Ensure we make a copy to not operate
            # on the same data.
            catom = atom.copy()
            uatom = catom.atom[:]
            specie = catom.specie[:]

        else:
            raise ValueError('atom keyword was wrong input')

        # Default for number of atoms
        if na is None:
            na = len(specie)

        # Create atom and species objects
        self._atom = list(uatom)

        self._specie = array_fill_repeat(specie, na, cls=np.int16)

    def copy(self):
        """ Return a copy of this atom """
        atoms = Atoms()
        atoms._atom = [a.copy() for a in self._atom]
        atoms._specie = np.copy(self._specie)
        return atoms

    @property
    def atom(self):
        """ Return the unique atoms list """
        return self._atom

    @property
    def specie(self):
        """ Return the specie list """
        return self._specie

    @property
    def no(self):
        """ Return the total number of orbitals in this list of atoms """
        uorbs = [a.orbs for a in self.atom]
        values, counts = np.unique(self.specie, return_counts=True)
        no = 0
        for v, c in zip(values, counts):
            no += c * uorbs[v]
        return no

    @property
    def orbitals(self):
        """ Return an array of orbitals of the contained objects """
        uorbs = np.array([a.orbs for a in self.atom], np.int32)
        return uorbs[self.specie[:]]

    @property
    def dR(self):
        """ Return an array of masses of the contained objects """
        udR = np.array([a.dR for a in self.atom], np.float64)
        return udR[self.specie[:]]

    @property
    def mass(self):
        """ Return an array of masses of the contained objects """
        umass = np.array([a.mass for a in self.atom], np.float64)
        return umass[self.specie[:]]

    def scale(self, scale):
        """ Scale the atomic radii and return an equivalent atom.

        Parameters
        ----------
        scale : float
           the scale factor for the atomic radii
        """
        atoms = Atoms()
        atoms._atom = [a.scale(scale) for a in self._atom]
        atoms._specie = np.copy(self._specie)
        return atoms

    def index(self, atom):
        """ Return the species index of the atom object """
        for i, a in enumerate(self._atom):
            if a == atom:
                return i
        return -1

    def reorder(self):
        """ Reorders the atoms and species index so that they are ascending """
        smin = np.zeros(len(self.atom), np.int32)
        for i in range(len(self.atom)):
            lst = np.where(self.specie == i)[0]
            if len(lst) == 0:
                # means it is not in use
                smin[i] = len(self.specie)
            else:
                smin[i] = lst[0]

        # Now swap indices into correct place
        isort = np.argsort(smin)
        if np.all(np.diff(isort) == 0):
            # No swaps required
            return self.copy()

        # We need to swap something
        atoms = self.copy()
        for os, ns in zip(range(len(isort)), isort):
            # Reorder the atom array as well.
            atoms._atom[ns] = self._atom[os].copy()
            atoms._specie[self.specie == os] = ns

        return atoms

    def reduce(self):
        """ Returns a new `Atoms` object by removing non-used atoms """
        atoms = self.copy()
        atom = atoms._atom
        specie = atoms._specie

        rem = []
        for i in range(len(self.atom)):
            if len(np.where(specie == i)[0]) == 0:
                rem.append(i)

        # Remove the atoms
        for i in rem[::-1]:
            atom.pop(i)
            specie = np.where(specie > i, specie - 1, specie)

        atoms._atom = atom
        atoms._specie = specie

        return atoms

    def sub(self, atom):
        """ Return a subset of the list """
        atom = ensure_array(atom).flatten()
        atoms = Atoms()
        atoms._atom = self._atom[:]
        atoms._specie = self._specie[atom]
        return atoms

    def cut(self, seps, axis):
        """ Return a subset of the list """
        atom = ensure_array(atom).flatten()
        atoms = Atoms()
        atoms._atom = self._atom[:]
        atoms._specie = self._specie[atom]
        return atoms

    def remove(self, atom):
        """ Remove a set of atoms """
        atms = ensure_array(atom).flatten()
        idx = np.setdiff1d(np.arange(len(self)), atms, assume_unique=True)
        return self.sub(idx)

    def tile(self, reps):
        """ Tile this atom object """
        atoms = self.copy()
        atoms._specie = np.tile(atoms._specie, reps)
        return atoms

    def repeat(self, reps):
        """ Repeat this atom object """
        atoms = self.copy()
        atoms._specie = np.repeat(atoms._specie, reps)
        return atoms

    def swap(self, a, b):
        """ Swaps atoms """
        a = ensure_array(a)
        b = ensure_array(b)
        atoms = self.copy()
        spec = np.copy(atoms._specie)
        atoms._specie[a] = spec[b]
        atoms._specie[b] = spec[a]
        return atoms

    def append(self, other):
        """ Append `other` to this list of atoms and return the appended version """
        if not isinstance(other, Atoms):
            other = Atoms(other)
        else:
            other = other.copy()

        atoms = self.copy()
        spec = np.copy(other._specie)
        for i, atom in enumerate(other.atom):
            if atom not in atoms:
                s = len(atoms.atom)
                atoms._atom.append(atom)
            else:
                s = atoms.index(atom)
            spec = np.where(spec == i, s, spec)
        atoms._specie = np.concatenate((atoms._specie, spec))
        return atoms

    add = append

    def prepend(self, other):
        if not isinstance(other, Atoms):
            other = Atoms(other)
        return other.append(self)

    def reverse(self, atom=None):
        """ Returns a reversed geometry

        Also enables reversing a subset of the atoms.
        """
        atoms = self.copy()
        if atom is None:
            atoms._specie = atoms._specie[::-1]
        else:
            atoms._specie[atom] = atoms._specie[atom[::-1]]
        return atoms

    def insert(self, index, other):
        """ Insert other atoms into the list of atoms at index """
        if isinstance(other, Atom):
            other = Atoms(other)
        else:
            other = other.copy()

        # Create a copy for insertion
        atoms = self.copy()

        spec = other._specie[:]
        for i, atom in enumerate(other.atom):
            if atom not in atoms:
                s = len(atoms.atom)
                atoms._atom.append(atom)
            else:
                s = atoms.index(atom)
            spec = np.where(spec == i, s, spec)
        atoms._specie = np.insert(atoms._specie, index, spec)
        return atoms

    def __repr__(self):
        """ Return the `Atoms` representation """
        s = '{{Atoms({0}):\n'.format(len(self._atom))
        for a, idx in self:
            s += '  ({0}) == [{1}], \n'.format(len(idx), a)
        return s + '}\n'

    def __len__(self):
        """ Return number of atoms in the object """
        return len(self._specie)

    def __iter__(self):
        """ Loop on all atoms with the same specie in order

        This iterator returns two values:
        1. The `Atom` object of the specie
        2. A list of indices with all atoms being that specie
        """
        for s, atom in enumerate(self._atom):
            yield atom, np.where(self.specie == s)[0]

    def __contains__(self, key):
        """ Determine whether the `key` is in the unique atoms list """
        return key in self.atom

    def __getitem__(self, key):
        """ Return an `Atom` object corresponding to the key(s) """
        if isinstance(key, slice):
            if key.step is None:
                nkey = slice(key.start or 0, key.stop or len(self), key.step or 1)
            elif key.step < 0:
                nkey = slice(key.stop or len(self)-1, key.start or -1, key.step)
            return [self.atom[self._specie[s]] for s in range(nkey.start, nkey.stop, nkey.step)]
        elif isinstance(key, (list, tuple, np.ndarray)):
            return [self.atom[self._specie[s]] for s in key]
        else:
            return self.atom[self._specie[key]]

    def __setitem__(self, key, value):
        """ Overwrite an `Atom` object corresponding to the key(s) """

        # First we figure out if this is a new atom
        if isinstance(key, (list, np.ndarray, tuple)):
            atoms = Atoms(value, na=len(key))
        else:
            atoms = Atoms(value)

        # Append the new Atom objects
        for atom, _ in atoms:
            if atom not in self:
                self._atom.append(atom)

        # Now the unique atom list also contains the new atoms
        # We need to re-create the species list
        if isinstance(key, (list, np.ndarray, tuple)):
            for i, j in enumerate(key):
                self._specie[j] = self.index(atoms[i])
        else:
            self._specie[key] = self.index(atoms[0])

    def __eq__(a, b):
        """ Returns true if the contained atoms are the same """
        for atom in a.atom:
            if atom not in b:
                return False
        return True

    # Create pickling routines
    def __getstate__(self):
        """ Return the state of this object """
        return {'atom': self.atom,
                'specie': self.specie}

    def __setstate__(self, d):
        """ Re-create the state of this object """
        self.__init__()
        self._atom = d['atom']
        self._specie = d['specie']
