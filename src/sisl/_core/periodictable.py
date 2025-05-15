# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from itertools import repeat
from numbers import Integral, Real
from typing import Literal, Union

import numpy as np
import numpy.typing as npt

from sisl._internal import set_module

__all__ = ["PeriodicTable"]


@set_module("sisl")
class PeriodicTable:
    r"""Periodic table for creating an `Atom`, or retrieval of atomic information via atomic numbers

    Enables *lookup* of atomic numbers/names/labels to get
    the atomic number.

    Several quantities available to the atomic species are available
    from <https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)>.

    The following values are accessible:

    * atomic mass (in atomic units)
    * empirical atomic radius (in Ang)
    * calculated atomic radius (in Ang)
    * van der Waals atomic radius (in Ang)

    For certain species the above quantities are not available
    and a negative number is returned.

    Examples
    --------
    >>> 79 == PeriodicTable().Z('Au')
    True
    >>> 79 == PeriodicTable().Z_int('Au')
    True
    >>> 'Au' == PeriodicTable().Z_short(79)
    True
    >>> 'Au' == PeriodicTable().Z_label(79)
    True
    >>> 'Au' == PeriodicTable().Z_label('Gold')
    True
    >>> 12.0107 == PeriodicTable().atomic_mass('C')
    True
    >>> 12.0107 == PeriodicTable().atomic_mass(6)
    True
    >>> 12.0107 == PeriodicTable().atomic_mass('Carbon')
    True
    >>> .67 == PeriodicTable().radius('Carbon')
    True
    >>> .67 == PeriodicTable().radius(6,'calc')
    True
    >>> .7  == PeriodicTable().radius(6,'empirical')
    True
    >>> 1.7 == PeriodicTable().radius(6,'vdw')
    True
    """

    # fmt: off
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
    # fmt: on

    @classmethod
    def _sanitize_z(cls, Z):
        Z = np.asarray(Z)
        return Z

    def Z(self, key):
        """Atomic number based on general input

        Return the atomic number corresponding to the `key` lookup.

        Parameters
        ----------
        key : array_like or str or int
            Uses value to lookup the atomic number in the `PeriodicTable`
            object.

        Returns
        -------
        numpy.ndarray or int
            atomic number corresponding to `key`, if `key` is array_like, so
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
        key = self._sanitize_z(key)
        get = self._Z_int.get
        if key.ndim == 0:
            key = key[()]
            return get(key, key)
        return np.asarray(list(map(get, key, key)), dtype=int)

    Z_int = Z

    def Z_label(self, key):
        """Atomic label of the corresponding atom

        Return the atomic short name corresponding to the `key` lookup.

        Parameters
        ----------
        key : array_like or str or int
            Uses value to lookup the atomic short name in the
            `PeriodicTable` object.

        Returns
        -------
        numpy.ndarray or str
            atomic short name corresponding to `key`, if `key`
            is array_like, so will the returned value be.
        """
        key = self._sanitize_z(key)
        get = self._Z_short.get
        if key.ndim == 0:
            return get(key[()], "fa")
        return np.asarray(list(map(get, key, repeat("fa"))), dtype=str)

    Z_short = Z_label

    @classmethod
    def Z_block(cls, Z: Union[int, npt.ArrayLike]):
        """The type-block of the atom in the periodic table.

        Will return one of s, p, d, f or `""`.
        May return `""` if the element isn't found
        in the periodic table.

        Only covers up to Z=118.

        Parameters
        ----------
        Z :
            the atomic number to search for
        """
        row: np.ndarray = cls.Z_row(Z)
        col: np.ndarray = cls.Z_column(Z)

        def conv(row, col):
            if row == 1 or col <= 2:
                return "s"
            if row in (4, 5, 6, 7) and col in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
                return "d"
            if row in (2, 3, 4, 5, 6, 7) and col in (13, 14, 15, 16, 17, 18):
                return "p"
            if row in (6, 7) and col in (
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
            ):
                return "f"
            return ""

        if row.ndim == 0:
            return conv(row[()], col[()])
        return np.asarray(list(map(conv, row, col)), dtype=str)

    @classmethod
    def Z_row(cls, Z: Union[int, npt.ArrayLike]):
        """The row of the atom in the periodic table.

        May return `-1` if the element isn't found
        in the periodic table.

        Only covers up to Z=118.

        Parameters
        ----------
        Z :
            the atomic number to search for
        """
        Z = cls._sanitize_z(Z)

        def conv(Z):
            if Z <= 0:
                return -1
            if Z <= 2:
                return 1
            if Z <= 10:
                return 2
            if Z <= 18:
                return 3
            if Z <= 36:
                return 4
            if Z <= 54:
                return 5
            if Z <= 86:
                return 6
            if Z <= 118:
                return 7
            return -1

        if Z.ndim == 0:
            return conv(Z[()])
        return np.asarray(list(map(conv, Z)), dtype=int)

    @classmethod
    def Z_column(cls, Z: Union[int, npt.ArrayLike]):
        """The column of the atom in the periodic table.

        May return `-1` if the element isn't found
        in the periodic table.

        Only covers up to Z=118.
        """
        Z = cls._sanitize_z(Z)

        def conv(Z):
            if Z in (1, 3, 11, 19, 37, 55, 87):
                return 1
            if Z in (4, 12, 20, 38, 56, 88):
                return 2
            if Z in (21, 39, 71, 103, 57, 89):
                return 3
            if Z in (22, 40, 72, 104, 58, 90):
                return 4
            if Z in (23, 41, 73, 105, 59, 91):
                return 5
            if Z in (24, 42, 74, 106, 60, 92):
                return 6
            if Z in (25, 43, 75, 107, 61, 93):
                return 7
            if Z in (26, 44, 76, 108, 62, 94):
                return 8
            if Z in (27, 45, 77, 109, 63, 95):
                return 9
            if Z in (28, 46, 78, 110, 64, 96):
                return 10
            if Z in (29, 47, 79, 111, 65, 97):
                return 11
            if Z in (30, 48, 80, 112, 66, 98):
                return 12
            if Z in (5, 13, 31, 49, 81, 113, 67, 99):
                return 13
            if Z in (6, 14, 32, 50, 82, 114, 68, 100):
                return 14
            if Z in (7, 15, 33, 51, 83, 115, 69, 101):
                return 15
            if Z in (8, 16, 34, 52, 84, 116, 70, 102):
                return 16
            if Z in (9, 17, 35, 53, 85, 117):
                return 17
            if Z in (2, 10, 18, 36, 54, 86, 118):
                return 18
            return -1

        if Z.ndim == 0:
            return conv(Z[()])
        return np.asarray(list(map(conv, Z)), dtype=int)

    def atomic_mass(self, key):
        """Atomic mass of the corresponding atom

        Return the atomic mass corresponding to the `key` lookup.

        Parameters
        ----------
        key : array_like or str or int
            Uses value to lookup the atomic mass in the
            `PeriodicTable` object.

        Returns
        -------
        numpy.ndarray or float
            atomic mass in atomic units corresponding to `key`,
            if `key` is array_like, so will the returned value be.
        """
        Z = self.Z_int(key)
        get = self._atomic_mass.get
        if isinstance(Z, (Integral, Real)):
            return get(Z, 0.0)
        return np.asarray(list(map(get, Z, repeat(0.0))), dtype=np.float64)

    def radius(self, key, method: Literal["calc", "empirical", "vdw"] = "calc"):
        """Atomic radius using different methods

        Return the atomic radius.

        Parameters
        ----------
        key : array_like or str or int
            Uses value to lookup the atomic mass in the
            `PeriodicTable` object.
        method : {'calc', 'empirical', 'vdw'}
            There are 3 different radii stored:

             1. ``calc``, the calculated atomic radius
             2. ``empirical``, the empirically found values
             3. ``vdw``, the van-der-Waals found values

        Returns
        -------
        numpy.ndarray or float
            atomic radius in `Ang`
        """
        Z = self.Z_int(key)
        func = getattr(self, f"_radius_{method}").get
        if isinstance(Z, (Integral, Real)):
            return func(Z) / 100
        return np.asarray(list(map(func, Z)), dtype=np.float64) / 100
