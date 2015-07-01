"""
Class for retaining information about a single atom
"""
from __future__ import print_function, division

# To check for integers
from numbers import Integral
from sids.geom._help import array_fill_repeat

import numpy as np

class PeriodicTable(object):
    """ 
    Very basic periodic table, not very pretty as it
    was generated using reg-exps.
    """
    _Z_int = {
        'Actinium' : 89 , 'Ac' : 89 , '89' : 89, 89 : 89,
        'Aluminum' : 13 , 'Al' : 13 , '13' : 13, 13 : 13,
        'Americium' : 95 , 'Am' : 95 , '95' : 95, 95 : 95,
        'Antimony' : 51 , 'Sb' : 51 , '51' : 51, 51 : 51,
        'Argon' : 18 , 'Ar' : 18 , '18' : 18, 18 : 18,
        'Arsenic' : 33 , 'As' : 33 , '33' : 33, 33 : 33,
        'Astatine' : 85 , 'At' : 85 , '85' : 85, 85 : 85,
        'Barium' : 56 , 'Ba' : 56 , '56' : 56, 56 : 56,
        'Berkelium' : 97 , 'Bk' : 97 , '97' : 97, 97 : 97,
        'Beryllium' : 4 , 'Be' : 4 , '4' : 4, 4 : 4,
        'Bismuth' : 83 , 'Bi' : 83 , '83' : 83, 83 : 83,
        'Bohrium' : 107 , 'Bh' : 107 , '107' : 107, 107 : 107,
        'Boron' : 5 , 'B' : 5 , '5' : 5, 5 : 5,
        'Bromine' : 35 , 'Br' : 35 , '35' : 35, 35 : 35,
        'Cadmium' : 48 , 'Cd' : 48 , '48' : 48, 48 : 48,
        'Calcium' : 20 , 'Ca' : 20 , '20' : 20, 20 : 20,
        'Californium' : 98 , 'Cf' : 98 , '98' : 98, 98 : 98,
        'Carbon' : 6 , 'C' : 6 , '6' : 6, 6 : 6,
        'Cerium' : 58 , 'Ce' : 58 , '58' : 58, 58 : 58,
        'Cesium' : 55 , 'Cs' : 55 , '55' : 55, 55 : 55,
        'Chlorine' : 17 , 'Cl' : 17 , '17' : 17, 17 : 17,
        'Chromium' : 24 , 'Cr' : 24 , '24' : 24, 24 : 24,
        'Cobalt' : 27 , 'Co' : 27 , '27' : 27, 27 : 27,
        'Copper' : 29 , 'Cu' : 29 , '29' : 29, 29 : 29,
        'Curium' : 96 , 'Cm' : 96 , '96' : 96, 96 : 96,
        'Darmstadtium' : 110 , 'Ds' : 110 , '110' : 110, 110 : 110,
        'Dubnium' : 105 , 'Db' : 105 , '105' : 105, 105 : 105,
        'Dysprosium' : 66 , 'Dy' : 66 , '66' : 66, 66 : 66,
        'Einsteinium' : 99 , 'Es' : 99 , '99' : 99, 99 : 99,
        'Erbium' : 68 , 'Er' : 68 , '68' : 68, 68 : 68,
        'Europium' : 63 , 'Eu' : 63 , '63' : 63, 63 : 63,
        'Fermium' : 100 , 'Fm' : 100 , '100' : 100, 100 : 100,
        'Fluorine' : 9 , 'F' : 9 , '9' : 9, 9 : 9,
        'Francium' : 87 , 'Fr' : 87 , '87' : 87, 87 : 87,
        'Gadolinium' : 64 , 'Gd' : 64 , '64' : 64, 64 : 64,
        'Gallium' : 31 , 'Ga' : 31 , '31' : 31, 31 : 31,
        'Germanium' : 32 , 'Ge' : 32 , '32' : 32, 32 : 32,
        'Gold' : 79 , 'Au' : 79 , '79' : 79, 79 : 79,
        'Hafnium' : 72 , 'Hf' : 72 , '72' : 72, 72 : 72,
        'Hassium' : 108 , 'Hs' : 108 , '108' : 108, 108 : 108,
        'Helium' : 2 , 'He' : 2 , '2' : 2, 2 : 2,
        'Holmium' : 67 , 'Ho' : 67 , '67' : 67, 67 : 67,
        'Hydrogen' : 1 , 'H' : 1 , '1' : 1, 1 : 1,
        'Indium' : 49 , 'In' : 49 , '49' : 49, 49 : 49,
        'Iodine' : 53 , 'I' : 53 , '53' : 53, 53 : 53,
        'Iridium' : 77 , 'Ir' : 77 , '77' : 77, 77 : 77,
        'Iron' : 26 , 'Fe' : 26 , '26' : 26, 26 : 26,
        'Krypton' : 36 , 'Kr' : 36 , '36' : 36, 36 : 36,
        'Lanthanum' : 57 , 'La' : 57 , '57' : 57, 57 : 57,
        'Lawrencium' : 103 , 'Lr' : 103 , '103' : 103, 103 : 103,
        'Lead' : 82 , 'Pb' : 82 , '82' : 82, 82 : 82,
        'Lithium' : 3 , 'Li' : 3 , '3' : 3, 3 : 3,
        'Lutetium' : 71 , 'Lu' : 71 , '71' : 71, 71 : 71,
        'Magnesium' : 12 , 'Mg' : 12 , '12' : 12, 12 : 12,
        'Manganese' : 25 , 'Mn' : 25 , '25' : 25, 25 : 25,
        'Meitnerium' : 109 , 'Mt' : 109 , '109' : 109, 109 : 109,
        'Mendelevium' : 101 , 'Md' : 101 , '101' : 101, 101 : 101,
        'Mercury' : 80 , 'Hg' : 80 , '80' : 80, 80 : 80,
        'Molybdenum' : 42 , 'Mo' : 42 , '42' : 42, 42 : 42,
        'Neodymium' : 60 , 'Nd' : 60 , '60' : 60, 60 : 60,
        'Neon' : 10 , 'Ne' : 10 , '10' : 10, 10 : 10,
        'Neptunium' : 93 , 'Np' : 93 , '93' : 93, 93 : 93,
        'Nickel' : 28 , 'Ni' : 28 , '28' : 28, 28 : 28,
        'Niobium' : 41 , 'Nb' : 41 , '41' : 41, 41 : 41,
        'Nitrogen' : 7 , 'N' : 7 , '7' : 7, 7 : 7,
        'Nobelium' : 102 , 'No' : 102 , '102' : 102, 102 : 102,
        'Osmium' : 76 , 'Os' : 76 , '76' : 76, 76 : 76,
        'Oxygen' : 8 , 'O' : 8 , '8' : 8, 8 : 8,
        'Palladium' : 46 , 'Pd' : 46 , '46' : 46, 46 : 46,
        'Phosphorus' : 15 , 'P' : 15 , '15' : 15, 15 : 15,
        'Platinum' : 78 , 'Pt' : 78 , '78' : 78, 78 : 78,
        'Plutonium' : 94 , 'Pu' : 94 , '94' : 94, 94 : 94,
        'Polonium' : 84 , 'Po' : 84 , '84' : 84, 84 : 84,
        'Potassium' : 19 , 'K' : 19 , '19' : 19, 19 : 19,
        'Praseodymium' : 59 , 'Pr' : 59 , '59' : 59, 59 : 59,
        'Promethium' : 61 , 'Pm' : 61 , '61' : 61, 61 : 61,
        'Protactinium' : 91 , 'Pa' : 91 , '91' : 91, 91 : 91,
        'Radium' : 88 , 'Ra' : 88 , '88' : 88, 88 : 88,
        'Radon' : 86 , 'Rn' : 86 , '86' : 86, 86 : 86,
        'Rhenium' : 75 , 'Re' : 75 , '75' : 75, 75 : 75,
        'Rhodium' : 45 , 'Rh' : 45 , '45' : 45, 45 : 45,
        'Rubidium' : 37 , 'Rb' : 37 , '37' : 37, 37 : 37,
        'Ruthenium' : 44 , 'Ru' : 44 , '44' : 44, 44 : 44,
        'Rutherfordium' : 104 , 'Rf' : 104 , '104' : 104, 104 : 104,
        'Samarium' : 62 , 'Sm' : 62 , '62' : 62, 62 : 62,
        'Scandium' : 21 , 'Sc' : 21 , '21' : 21, 21 : 21,
        'Seaborgium' : 106 , 'Sg' : 106 , '106' : 106, 106 : 106,
        'Selenium' : 34 , 'Se' : 34 , '34' : 34, 34 : 34,
        'Silicon' : 14 , 'Si' : 14 , '14' : 14, 14 : 14,
        'Silver' : 47 , 'Ag' : 47 , '47' : 47, 47 : 47,
        'Sodium' : 11 , 'Na' : 11 , '11' : 11, 11 : 11,
        'Strontium' : 38 , 'Sr' : 38 , '38' : 38, 38 : 38,
        'Sulfur' : 16 , 'S' : 16 , '16' : 16, 16 : 16,
        'Tantalum' : 73 , 'Ta' : 73 , '73' : 73, 73 : 73,
        'Technetium' : 43 , 'Tc' : 43 , '43' : 43, 43 : 43,
        'Tellurium' : 52 , 'Te' : 52 , '52' : 52, 52 : 52,
        'Terbium' : 65 , 'Tb' : 65 , '65' : 65, 65 : 65,
        'Thallium' : 81 , 'Tl' : 81 , '81' : 81, 81 : 81,
        'Thorium' : 90 , 'Th' : 90 , '90' : 90, 90 : 90,
        'Thulium' : 69 , 'Tm' : 69 , '69' : 69, 69 : 69,
        'Tin' : 50 , 'Sn' : 50 , '50' : 50, 50 : 50,
        'Titanium' : 22 , 'Ti' : 22 , '22' : 22, 22 : 22,
        'Tungsten' : 74 , 'W' : 74 , '74' : 74, 74 : 74,
        'Ununbium' : 112 , 'Uub' : 112 , '112' : 112, 112 : 112,
        'Ununhexium' : 116 , 'Uuh' : 116 , '116' : 116, 116 : 116,
        'Ununoctium' : 118 , 'Uuo' : 118 , '118' : 118, 118 : 118,
        'Ununpentium' : 115 , 'Uup' : 115 , '115' : 115, 115 : 115,
        'Ununquadium' : 114 , 'Uuq' : 114 , '114' : 114, 114 : 114,
        'Ununseptium' : 117 , 'Uus' : 117 , '117' : 117, 117 : 117,
        'Ununtrium' : 113 , 'Uut' : 113 , '113' : 113, 113 : 113,
        'Ununium' : 111 , 'Uuu' : 111 , '111' : 111, 111 : 111,
        'Uranium' : 92 , 'U' : 92 , '92' : 92, 92 : 92,
        'Vanadium' : 23 , 'V' : 23 , '23' : 23, 23 : 23,
        'Xenon' : 54 , 'Xe' : 54 , '54' : 54, 54 : 54,
        'Ytterbium' : 70 , 'Yb' : 70 , '70' : 70, 70 : 70,
        'Yttrium' : 39 , 'Y' : 39 , '39' : 39, 39 : 39,
        'Zinc' : 30 , 'Zn' : 30 , '30' : 30, 30 : 30,
        'Zirconium' : 40 , 'Zr' : 40 , '40' : 40, 40 : 40,
        }
    
    _Z_short = {
        'Actinium' : 'Ac' , 'Ac' : 'Ac' , '89' : 'Ac', 89 : 'Ac',
        'Aluminum' : 'Al' , 'Al' : 'Al' , '13' : 'Al', 13 : 'Al',
        'Americium' : 'Am' , 'Am' : 'Am' , '95' : 'Am', 95 : 'Am',
        'Antimony' : 'Sb' , 'Sb' : 'Sb' , '51' : 'Sb', 51 : 'Sb',
        'Argon' : 'Ar' , 'Ar' : 'Ar' , '18' : 'Ar', 18 : 'Ar',
        'Arsenic' : 'As' , 'As' : 'As' , '33' : 'As', 33 : 'As',
        'Astatine' : 'At' , 'At' : 'At' , '85' : 'At', 85 : 'At',
        'Barium' : 'Ba' , 'Ba' : 'Ba' , '56' : 'Ba', 56 : 'Ba',
        'Berkelium' : 'Bk' , 'Bk' : 'Bk' , '97' : 'Bk', 97 : 'Bk',
        'Beryllium' : 'Be' , 'Be' : 'Be' , '4' : 'Be', 4 : 'Be',
        'Bismuth' : 'Bi' , 'Bi' : 'Bi' , '83' : 'Bi', 83 : 'Bi',
        'Bohrium' : 'Bh' , 'Bh' : 'Bh' , '107' : 'Bh', 107 : 'Bh',
        'Boron' : 'B' , 'B' : 'B' , '5' : 'B', 5 : 'B',
        'Bromine' : 'Br' , 'Br' : 'Br' , '35' : 'Br', 35 : 'Br',
        'Cadmium' : 'Cd' , 'Cd' : 'Cd' , '48' : 'Cd', 48 : 'Cd',
        'Calcium' : 'Ca' , 'Ca' : 'Ca' , '20' : 'Ca', 20 : 'Ca',
        'Californium' : 'Cf' , 'Cf' : 'Cf' , '98' : 'Cf', 98 : 'Cf',
        'Carbon' : 'C' , 'C' : 'C' , '6' : 'C', 6 : 'C',
        'Cerium' : 'Ce' , 'Ce' : 'Ce' , '58' : 'Ce', 58 : 'Ce',
        'Cesium' : 'Cs' , 'Cs' : 'Cs' , '55' : 'Cs', 55 : 'Cs',
        'Chlorine' : 'Cl' , 'Cl' : 'Cl' , '17' : 'Cl', 17 : 'Cl',
        'Chromium' : 'Cr' , 'Cr' : 'Cr' , '24' : 'Cr', 24 : 'Cr',
        'Cobalt' : 'Co' , 'Co' : 'Co' , '27' : 'Co', 27 : 'Co',
        'Copper' : 'Cu' , 'Cu' : 'Cu' , '29' : 'Cu', 29 : 'Cu',
        'Curium' : 'Cm' , 'Cm' : 'Cm' , '96' : 'Cm', 96 : 'Cm',
        'Darmstadtium' : 'Ds' , 'Ds' : 'Ds' , '110' : 'Ds', 110 : 'Ds',
        'Dubnium' : 'Db' , 'Db' : 'Db' , '105' : 'Db', 105 : 'Db',
        'Dysprosium' : 'Dy' , 'Dy' : 'Dy' , '66' : 'Dy', 66 : 'Dy',
        'Einsteinium' : 'Es' , 'Es' : 'Es' , '99' : 'Es', 99 : 'Es',
        'Erbium' : 'Er' , 'Er' : 'Er' , '68' : 'Er', 68 : 'Er',
        'Europium' : 'Eu' , 'Eu' : 'Eu' , '63' : 'Eu', 63 : 'Eu',
        'Fermium' : 'Fm' , 'Fm' : 'Fm' , '100' : 'Fm', 100 : 'Fm',
        'Fluorine' : 'F' , 'F' : 'F' , '9' : 'F', 9 : 'F',
        'Francium' : 'Fr' , 'Fr' : 'Fr' , '87' : 'Fr', 87 : 'Fr',
        'Gadolinium' : 'Gd' , 'Gd' : 'Gd' , '64' : 'Gd', 64 : 'Gd',
        'Gallium' : 'Ga' , 'Ga' : 'Ga' , '31' : 'Ga', 31 : 'Ga',
        'Germanium' : 'Ge' , 'Ge' : 'Ge' , '32' : 'Ge', 32 : 'Ge',
        'Gold' : 'Au' , 'Au' : 'Au' , '79' : 'Au', 79 : 'Au',
        'Hafnium' : 'Hf' , 'Hf' : 'Hf' , '72' : 'Hf', 72 : 'Hf',
        'Hassium' : 'Hs' , 'Hs' : 'Hs' , '108' : 'Hs', 108 : 'Hs',
        'Helium' : 'He' , 'He' : 'He' , '2' : 'He', 2 : 'He',
        'Holmium' : 'Ho' , 'Ho' : 'Ho' , '67' : 'Ho', 67 : 'Ho',
        'Hydrogen' : 'H' , 'H' : 'H' , '1' : 'H', 1 : 'H',
        'Indium' : 'In' , 'In' : 'In' , '49' : 'In', 49 : 'In',
        'Iodine' : 'I' , 'I' : 'I' , '53' : 'I', 53 : 'I',
        'Iridium' : 'Ir' , 'Ir' : 'Ir' , '77' : 'Ir', 77 : 'Ir',
        'Iron' : 'Fe' , 'Fe' : 'Fe' , '26' : 'Fe', 26 : 'Fe',
        'Krypton' : 'Kr' , 'Kr' : 'Kr' , '36' : 'Kr', 36 : 'Kr',
        'Lanthanum' : 'La' , 'La' : 'La' , '57' : 'La', 57 : 'La',
        'Lawrencium' : 'Lr' , 'Lr' : 'Lr' , '103' : 'Lr', 103 : 'Lr',
        'Lead' : 'Pb' , 'Pb' : 'Pb' , '82' : 'Pb', 82 : 'Pb',
        'Lithium' : 'Li' , 'Li' : 'Li' , '3' : 'Li', 3 : 'Li',
        'Lutetium' : 'Lu' , 'Lu' : 'Lu' , '71' : 'Lu', 71 : 'Lu',
        'Magnesium' : 'Mg' , 'Mg' : 'Mg' , '12' : 'Mg', 12 : 'Mg',
        'Manganese' : 'Mn' , 'Mn' : 'Mn' , '25' : 'Mn', 25 : 'Mn',
        'Meitnerium' : 'Mt' , 'Mt' : 'Mt' , '109' : 'Mt', 109 : 'Mt',
        'Mendelevium' : 'Md' , 'Md' : 'Md' , '101' : 'Md', 101 : 'Md',
        'Mercury' : 'Hg' , 'Hg' : 'Hg' , '80' : 'Hg', 80 : 'Hg',
        'Molybdenum' : 'Mo' , 'Mo' : 'Mo' , '42' : 'Mo', 42 : 'Mo',
        'Neodymium' : 'Nd' , 'Nd' : 'Nd' , '60' : 'Nd', 60 : 'Nd',
        'Neon' : 'Ne' , 'Ne' : 'Ne' , '10' : 'Ne', 10 : 'Ne',
        'Neptunium' : 'Np' , 'Np' : 'Np' , '93' : 'Np', 93 : 'Np',
        'Nickel' : 'Ni' , 'Ni' : 'Ni' , '28' : 'Ni', 28 : 'Ni',
        'Niobium' : 'Nb' , 'Nb' : 'Nb' , '41' : 'Nb', 41 : 'Nb',
        'Nitrogen' : 'N' , 'N' : 'N' , '7' : 'N', 7 : 'N',
        'Nobelium' : 'No' , 'No' : 'No' , '102' : 'No', 102 : 'No',
        'Osmium' : 'Os' , 'Os' : 'Os' , '76' : 'Os', 76 : 'Os',
        'Oxygen' : 'O' , 'O' : 'O' , '8' : 'O', 8 : 'O',
        'Palladium' : 'Pd' , 'Pd' : 'Pd' , '46' : 'Pd', 46 : 'Pd',
        'Phosphorus' : 'P' , 'P' : 'P' , '15' : 'P', 15 : 'P',
        'Platinum' : 'Pt' , 'Pt' : 'Pt' , '78' : 'Pt', 78 : 'Pt',
        'Plutonium' : 'Pu' , 'Pu' : 'Pu' , '94' : 'Pu', 94 : 'Pu',
        'Polonium' : 'Po' , 'Po' : 'Po' , '84' : 'Po', 84 : 'Po',
        'Potassium' : 'K' , 'K' : 'K' , '19' : 'K', 19 : 'K',
        'Praseodymium' : 'Pr' , 'Pr' : 'Pr' , '59' : 'Pr', 59 : 'Pr',
        'Promethium' : 'Pm' , 'Pm' : 'Pm' , '61' : 'Pm', 61 : 'Pm',
        'Protactinium' : 'Pa' , 'Pa' : 'Pa' , '91' : 'Pa', 91 : 'Pa',
        'Radium' : 'Ra' , 'Ra' : 'Ra' , '88' : 'Ra', 88 : 'Ra',
        'Radon' : 'Rn' , 'Rn' : 'Rn' , '86' : 'Rn', 86 : 'Rn',
        'Rhenium' : 'Re' , 'Re' : 'Re' , '75' : 'Re', 75 : 'Re',
        'Rhodium' : 'Rh' , 'Rh' : 'Rh' , '45' : 'Rh', 45 : 'Rh',
        'Rubidium' : 'Rb' , 'Rb' : 'Rb' , '37' : 'Rb', 37 : 'Rb',
        'Ruthenium' : 'Ru' , 'Ru' : 'Ru' , '44' : 'Ru', 44 : 'Ru',
        'Rutherfordium' : 'Rf' , 'Rf' : 'Rf' , '104' : 'Rf', 104 : 'Rf',
        'Samarium' : 'Sm' , 'Sm' : 'Sm' , '62' : 'Sm', 62 : 'Sm',
        'Scandium' : 'Sc' , 'Sc' : 'Sc' , '21' : 'Sc', 21 : 'Sc',
        'Seaborgium' : 'Sg' , 'Sg' : 'Sg' , '106' : 'Sg', 106 : 'Sg',
        'Selenium' : 'Se' , 'Se' : 'Se' , '34' : 'Se', 34 : 'Se',
        'Silicon' : 'Si' , 'Si' : 'Si' , '14' : 'Si', 14 : 'Si',
        'Silver' : 'Ag' , 'Ag' : 'Ag' , '47' : 'Ag', 47 : 'Ag',
        'Sodium' : 'Na' , 'Na' : 'Na' , '11' : 'Na', 11 : 'Na',
        'Strontium' : 'Sr' , 'Sr' : 'Sr' , '38' : 'Sr', 38 : 'Sr',
        'Sulfur' : 'S' , 'S' : 'S' , '16' : 'S', 16 : 'S',
        'Tantalum' : 'Ta' , 'Ta' : 'Ta' , '73' : 'Ta', 73 : 'Ta',
        'Technetium' : 'Tc' , 'Tc' : 'Tc' , '43' : 'Tc', 43 : 'Tc',
        'Tellurium' : 'Te' , 'Te' : 'Te' , '52' : 'Te', 52 : 'Te',
        'Terbium' : 'Tb' , 'Tb' : 'Tb' , '65' : 'Tb', 65 : 'Tb',
        'Thallium' : 'Tl' , 'Tl' : 'Tl' , '81' : 'Tl', 81 : 'Tl',
        'Thorium' : 'Th' , 'Th' : 'Th' , '90' : 'Th', 90 : 'Th',
        'Thulium' : 'Tm' , 'Tm' : 'Tm' , '69' : 'Tm', 69 : 'Tm',
        'Tin' : 'Sn' , 'Sn' : 'Sn' , '50' : 'Sn', 50 : 'Sn',
        'Titanium' : 'Ti' , 'Ti' : 'Ti' , '22' : 'Ti', 22 : 'Ti',
        'Tungsten' : 'W' , 'W' : 'W' , '74' : 'W', 74 : 'W',
        'Ununbium' : 'Uub' , 'Uub' : 'Uub' , '112' : 'Uub', 112 : 'Uub',
        'Ununhexium' : 'Uuh' , 'Uuh' : 'Uuh' , '116' : 'Uuh', 116 : 'Uuh',
        'Ununoctium' : 'Uuo' , 'Uuo' : 'Uuo' , '118' : 'Uuo', 118 : 'Uuo',
        'Ununpentium' : 'Uup' , 'Uup' : 'Uup' , '115' : 'Uup', 115 : 'Uup',
        'Ununquadium' : 'Uuq' , 'Uuq' : 'Uuq' , '114' : 'Uuq', 114 : 'Uuq',
        'Ununseptium' : 'Uus' , 'Uus' : 'Uus' , '117' : 'Uus', 117 : 'Uus',
        'Ununtrium' : 'Uut' , 'Uut' : 'Uut' , '113' : 'Uut', 113 : 'Uut',
        'Ununium' : 'Uuu' , 'Uuu' : 'Uuu' , '111' : 'Uuu', 111 : 'Uuu',
        'Uranium' : 'U' , 'U' : 'U' , '92' : 'U', 92 : 'U',
        'Vanadium' : 'V' , 'V' : 'V' , '23' : 'V', 23 : 'V',
        'Xenon' : 'Xe' , 'Xe' : 'Xe' , '54' : 'Xe', 54 : 'Xe',
        'Ytterbium' : 'Yb' , 'Yb' : 'Yb' , '70' : 'Yb', 70 : 'Yb',
        'Yttrium' : 'Y' , 'Y' : 'Y' , '39' : 'Y', 39 : 'Y',
        'Zinc' : 'Zn' , 'Zn' : 'Zn' , '30' : 'Zn', 30 : 'Zn',
        'Zirconium' : 'Zr' , 'Zr' : 'Zr' , '40' : 'Zr', 40 : 'Zr',
        }

    _atomic_mass = {
        1 : 1.00794 ,
        2 : 4.002602 ,
        3 : 6.941 ,
        4 : 9.012182 ,
        5 : 10.811 ,
        6 : 12.0107 ,
        7 : 14.0067 ,
        8 : 15.9994 ,
        9 : 18.9984032 ,
        10 : 20.1797 ,
        11 : 22.98976928 ,
        12 : 24.3050 ,
        13 : 26.9815386 ,
        14 : 28.0855 ,
        15 : 30.973762 ,
        16 : 32.065 ,
        17 : 35.453 ,
        18 : 39.948 ,
        19 : 39.0983 ,
        20 : 40.078 ,
        21 : 44.955912 ,
        22 : 47.867 ,
        23 : 50.9415 ,
        24 : 51.9961 ,
        25 : 54.938045 ,
        26 : 55.845 ,
        27 : 58.933195 ,
        28 : 58.6934 ,
        29 : 63.546 ,
        30 : 65.409 ,
        31 : 69.723 ,
        32 : 72.64 ,
        33 : 74.92160 ,
        34 : 78.96 ,
        35 : 79.904 ,
        36 : 83.798 ,
        37 : 85.4678 ,
        38 : 87.62 ,
        39 : 88.90585 ,
        40 : 91.224 ,
        41 : 92.906 ,
        42 : 95.94 ,
        43 : 98. ,
        44 : 101.07 ,
        45 : 102.905 ,
        46 : 106.42 ,
        47 : 107.8682 ,
        48 : 112.411 ,
        49 : 114.818 ,
        50 : 118.710 ,
        51 : 121.760 ,
        52 : 127.60 ,
        53 : 126.904 ,
        54 : 131.293 ,
        55 : 132.9054519 ,
        56 : 137.327 ,
        57 : 138.90547 ,
        58 : 140.116 ,
        59 : 140.90765 ,
        60 : 144.242 ,
        61 : 145. ,
        62 : 150.36 ,
        63 : 151.964 ,
        64 : 157.25 ,
        65 : 158.92535 ,
        66 : 162.500 ,
        67 : 164.930 ,
        68 : 167.259 ,
        69 : 168.93421 ,
        70 : 173.04 ,
        71 : 174.967 ,
        72 : 178.49 ,
        73 : 180.94788 ,
        74 : 183.84 ,
        75 : 186.207 ,
        76 : 190.23 ,
        77 : 192.217 ,
        78 : 195.084 ,
        79 : 196.966569 ,
        80 : 200.59 ,
        81 : 204.3833 ,
        82 : 207.2 ,
        83 : 208.98040 ,
        84 : 210. ,
        85 : 210. ,
        86 : 220. ,
        87 : 223. ,
        88 : 226. ,
        89 : 227. ,
        91 : 231.03588 ,
        90 : 232.03806 ,
        93 : 237. ,
        92 : 238.02891 ,
        95 : 243. ,
        94 : 244. ,
        96 : 247. ,
        97 : 247. ,
        98 : 251. ,
        99 : 252. ,
        100 : 257. ,
        101 : 258. ,
        102 : 259. ,
        103 : 262. ,
        104 : 261. ,
        105 : 262. ,
        106 : 266. ,
        107 : 264. ,
        108 : 277. ,
        109 : 268. ,
        110 : 271. ,
        111 : 272. ,
        112 : 285. ,
        113 : 284. ,
        114 : 289. ,
        115 : 288. ,
        116 : 292. ,
        118 : 293. ,
        }

    def Z_int(self,key):
        """ Returns the Z number """
        ak = np.asarray([key]).flatten()
        if len(ak) == 1: return self._Z_int[ak[0]]
        return np.array([self._Z_int[i] for i in ak],np.int)

    Z = Z_int

    def Z_short(self,key):
        """ Returns the Z name in short """
        ak = np.asarray([key]).flatten()
        if len(ak) == 1: return self._Z_short[ak[0]]
        return np.array([self._Z_short[i] for i in ak],np.int)

    def atomic_mass(self,key):
        Z = self.Z_int(key)
        if isinstance(Z,Integral): return self._atomic_mass[Z]
        return np.array([self._atomic_mass[i] for i in Z],np.float)


# Create a local instance of the periodic table to
# faster look up
_ptbl = PeriodicTable()

class AtomMeta(type):
    def __getitem__(self,key):
        """ Create a new atom object """
        if isinstance(key,Atom):
            # if the key already is an atomic object
            # return it
            return key
        if isinstance(key,dict):
            # The key is a dictionary, hence
            # we can return the atom directly
            return Atom(**key)
        if isinstance(key,list):
            # The key is a list, 
            # we need to create a list of atoms
            atm = [Atom[k] for k in key]
            return atm
        # Index Z based
        return Atom(Z=key)

class Atom(object):
    """
    Atomic object to handle atomic mass, name etc.
    
    This object handles the interaction ranges of the atoms, the
    atomic orbitals etc.

    Parameters
    ----------
    Z     : (1) integer/string
        description of the atom, the atom number or the atom name.
    R     : (-1.) array_like/float
        the range of the atomic orbitals
    orbs  : (1) integer 
        number of orbitals attached to this atom
        NOTE: Length of ``R`` precedes this quantity.
    mass  : (1) float
        the atomic mass (defaults to the periodic table quantity)
    tag   : arbitrary designation for user handling similar atoms with
        different settings.
    """
    def __init__(self,Z,R=-1.,orbs=1,mass=None,tag=None):
        self.Z = _ptbl.Z_int(Z)
        self.orbs = orbs
        try:
            self.orbs = max(len(R),orbs)
        except: pass
        self.R = array_fill_repeat(np.asarray([R],np.float).flatten(),self.orbs)
        # Save the mass
        self.mass = mass
        if mass is None:
            self.mass = _ptbl.atomic_mass(self.Z)
        if tag is None:
            self.tag = self.symbol
        else:
            self.tag = tag

    @property
    def symbol(self):
        return _ptbl.Z_short(self.Z)

    @property
    def dR(self):
        """ Returns the maximum range of orbitals """
        return np.amax(self.R)

    def __repr__(self):
        """ String representation """
        return self.tag + " orbs: "+str(self.orbs) \
            + " mass(au): "+str(self.mass)

    # Check whether they are equal
    def __eq__(a,b):
        """ Returns true if the saved quantities are the same """
        same = a.Z == b.Z
        same &= a.orbs == b.orbs
        same &= a.R == b.R
        same &= a.mass == b.mass
        same &= a.tag == b.tag
        return same

    # Enables easily to create new atoms
    __metaclass__ = AtomMeta

if __name__ == "__main__":
    # Create C
    C = Atom('C',R = [1.,2.])
    print('Default: \n\t' + str(C))

    H = Atom['H']
    print('Default: \n\t' + str(H))

    D = Atom[{'Z':'H','mass':2.001,'tag':'Deuterium'}]
    print('Default: \n\t' + str(D))

