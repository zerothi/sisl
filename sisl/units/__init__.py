"""
Units from various programs

This conversion tool is inspired by the SIESTA fdf-parser in its
group-construct.
"""

__all__ = ['unit_group', 'unit_convert', 'unit_default']


# We do not import anything as it depends on the package.
# Here we only add the conversions according to the
# standard. Other programs may use their units as they
# please with non-standard conversion factors.

_unit_table = {
    'mass' : {
        'DEFAULT' : 'amu',
        'kg' : 1.,
        'g'  : 1.e-3,
        'amu': 1.66054e-27,
        }, 
    'length' : {
        'DEFAULT' : 'Ang',
        'm'    : 1., 
        'cm'   : 0.01, 
        'nm'   : 1.e-9, 
        'Ang'  : 1.e-10, 
        'Bohr' : 5.29177249e-11,
        }, 
    'time' : {
        'DEFAULT' : 'fs',
        's'  : 1. ,
        'fs' : 1.e-15 ,
        'ps' : 1.e-12 ,
        'ns' : 1.e-9 ,
        },
    'energy' : {
        'DEFAULT' : 'eV',
        'J'       : 1., 
        'erg'     : 1.e-7, 
        'eV'      : 1.60217733e-19,
        'meV'     : 1.60217733e-16,
        'Ry'      : 2.1798741e-18,
        'Ha'      : 4.3597482e-18,
        'Hartree' : 4.3597482e-18,
        'K'       : 1.380648780669e-23,
        },
    'force' : {
        'DEFAULT' : 'eV/Ang',
        'N'       : 1.,
        'eV/Ang'  : 1.60217733e-9,
        }
    }


def unit_group(unit, tbl=None):
    """
    Returns the unit group that is associated with
    input unit.

    Parameters
    ----------
    unit : str
      unit, e.g. kg, Ang, eV etc. returns the type of unit it is.

    Examples
    --------
    >>> unit_group('kg')
    'mass'
    >>> unit_group('eV')
    'energy'
    """
    if tbl is None:
        global _unit_table
        tbl = _unit_table

    for k in tbl:
        if unit in tbl[k]:
            return k
    raise ValueError('The unit "'+str(unit)+'" could not be located in the table.')


def unit_default(group, tbl=None):
    """
    Return the default unit in the group `group`.

    Parameters
    ----------
    group, str
       look-up in the table for the default unit.
    """
    if tbl is None:
        global _unit_table
        tbl = _unit_table

    for k in tbl:
        if group == k:
            return tbl[k]['DEFAULT']

    raise UnitError('The unit-group does not exist!')
    

def unit_convert(fr, to, opts={}, tbl=None):
    """
    Returns the factor that takes 'fr' to the units of 'to'.

    Parameters
    ----------
    fr :
      starting unit
    to :
      ending unit
    opts :
      controls whether the unit conversion is in powers or fractional units

    Examples
    -------
    >>> unit_convert('kg','g')
    1000
    >>> unit_convert('eV','J')
    1.60219e-19
    """
    if tbl is None:
        global _unit_table
        tbl = _unit_table

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
    for opt in ['^','power','p']:
        if opt in opts:
            val = val ** opts[opt]
    for opt in ['*','factor','fac']:
        if opt in opts:
            val = val * opts[opt]
    for opt in ['/','divide','div']:
        if opt in opts:
            val = val / opts[opt]

    return val



# # These are older implementations which allows
# # numpy direct calculations (much like Unum package).

# # A single unit-object.
# # Contains functions to compare and convert a unit
# # to another unit.
# class Unit(object):
#     """
#     Container for the unit and the conversion factors etc.
#     This will make it easier to maintain the units, and eventually change the
#     usage.
#     """
#     def __new__(cls,*args,**kwargs):
#         if isinstance(args[0],Unit):
#             return args[0]
#         #print('Creating new unit:',args)
#         obj = object.__new__(cls)
#         if len(args) == 1: # We are creating a unit without a variable name
#             obj.variable = None
#             obj.unit = args[0]
#         else:
#             obj.variable = args[0]
#             # Typical case when passing a unit from another variable...
#             if isinstance(args[1],Unit):
#                 obj.unit = args[1].unit
#             else:
#                 obj.unit = args[1]
        
#         # We need to handle some type of operator definitions
#         # But how to handle them?
#         for op in ['**','^','/','*']:
#             pass
        
#         return obj

#     def type(self):
#         """ Returns the type of unit this is, i.e. energy, length, time, etc. """
#         for k,v in _ConversionTable.iteritems():
#             if self.unit in v: return k

#     def SI(self):
#         """ Returns the SI conversion factor for the unit """
#         for k,v in _ConversionTable.iteritems():
#             if self.variable in v: return v[self.variable]

#     def convert(self,to):
#         """ Convert this unit to another and returns the conversion factor. """
#         u = Unit(to)
#         # This will raise an exception if the units are not of same type...
#         conv = UnitConvert(self.unit,u.unit)
#         #print('Converting:',self.variable,self.unit,u.unit)
#         self.unit = deepcopy(u.unit)
#         return conv

#     def copy(self):
#         """Method for copying the unit """
#         return deepcopy(self)

#     def __repr__(self):
#         """ Return the unit in string format (XML type-like)"""
#         return "<Unit variable='"+str(self.variable)+"' unit='"+str(self.unit)+"'/>"

#     def __eq__(self,other):
#         """ Returns true if the variable is the same as the other """
#         return self.variable == other.variable

#     def __copy__(self):
#         return Unit(copy(self.variable),copy(self.unit))
    
#     def __deepcopy__(self, memo):
#         return Unit(deepcopy(self.variable),deepcopy(self.unit))


# class Units(object):
#     """
#     Container for many units.
#     This will make it easier to maintain the units, and eventually change the
#     usage.
#     """
#     def __new__(cls,*args):
#         # Convert the tuple to a list...
#         obj = object.__new__(cls)
#         # The args are a list of Unit-objects, or a list of pairs which should be converted to a list of units.
#         units = []
#         i = 0
#         while i < len(args):
#             if isinstance(args[i],Unit):
#                 units.append(deepcopy(args[i]))
#             else:
#                 assert i < len(args)-1, 'Can not grap a unit for: ' + str(args[i])
#                 units.append(deepcopy(Unit(args[i],args[i+1])))
#                 i += 1
#             i += 1
#         obj._units = units
#         return obj

#     def append(self,unit):
#         """ Append a unit object """
#         # We cannot have to similar units assigned...
#         if isinstance(unit,Units):
#             for au in unit:
#                 # Use the recursive routine (keep it simple)
#                 self.append(au)
#         else:
#             for u in self:
#                 if u == unit:
#                     raise Exception('Can not append a unit which already exists. Do not assign dublicate variables')
#             self._units.append(deepcopy(unit))

#     def update(self,unit):
#         """ Updates unit object, adds it if it does not exist """
#         if unit is None: return
#         if isinstance(unit,Units):
#             for u in unit:
#                 self.update(u)
#         else:
#             for u in self:
#                 if u.variable == unit.variable:
#                     u.unit = deepcopy(unit.unit)
#                     return
#             self.append(unit)

#     def unit(self,variable):
#         """ Returns the unit object associated with the variable named variable"""
#         # if it is none, return fast.
#         if not variable: return None
#         for i in self:
#             if i.variable == variable:
#                 return i
#         return None

#     def copy(self):
#         """ Copies this unit segment """
#         return deepcopy(self)

#     #################
#     # General routines overwriting python models
#     #################
#     def __len__(self):
#         return len(self._units)

#     def __contains__(self,item):
#         if isinstance(item,Unit):
#             u = Unit(item.variable,None)
#         else:
#             u = Unit(item,None)
#         for unit in self:
#             if u.variable == unit.variable:
#                 return True
#         return False
    
#     def __repr__(self):
#         """ Return the unit in string format (XML type-like)"""
#         tmp = '<Units>'
#         for unit in self:
#             tmp += '\n  ' + str(unit)
#         tmp += '\n</Units>'
#         return tmp

#     def __iter__(self):
#         """ An iterator of the Units collection """
#         for unit in self._units:
#             yield unit

#     def __delitem__(self,variable):
#         """ Remove the variable from the units list. """
#         for i in range(len(self)):
#             if self._units[i].variable == variable:
#                 del self._units[i]
#                 return
                
#     # We need to overwrite the copy mechanisms.
#     # It really is a pain in the ass, but it works.
#     # Luckily all copying need only be refered in the Unit-object.
#     def __copy__(self):
#         units = Units()
#         for unit in self:
#             units.append(copy(unit))
#         return units
    
#     def __deepcopy__(self, memo):
#         units = Units()
#         for unit in self:
#             units.append(deepcopy(unit))
#         return units

#     # Do NOT implement a 'convert' method. It could potentially lead to unexpected behaviour as the
#     # Unit-object needs to handle this....
#     # TODO consider the conversion of a list of Unit-objects via the Units-object.

# class UnitObject(object):
#     """
#     Contains relevant information about units etc.
#     """
#     def convert(self,*units):
#         """
#         Convert all entries in the object to the desired
#         units given by the input.
#         """
#         # Go back in the units variable does not exist.
#         if not '_units' in self.__dict__: return

#         # If it is a Units object, we can simply loop and do the recursive conversion.
#         if isinstance(units[0],Units):
#             for unit in units[0]:
#                 self.convert(unit)
#             return

#         # First convert all variables associated with a type... ('length',etc.)
#         # This well enable one to convert all of length but still have a unit conversion of a
#         # single length variable to another.
#         for unit in units:
#             u = Unit(unit)
#             if not u.variable:
#                 for self_u in self._units:
#                     if self_u.type() == u.type():
#                         self.__dict__[self_u.variable] *= self_u.convert(u)
                        
#         # Now convert the specific requested units.
#         for unit in units:
#             u = Unit(unit)
#             self_u = self.unit(u.variable)
#             if self_u:
#                 self.__dict__[self_u.variable] *= self_u.convert(u)

#     def unit(self,variable):
#         """ Returns the unit that is associated with the variable """
#         return self._units.unit(variable)

#     @property
#     def units(self):
#         """ Returns the units that is associated with the variable """
#         return self._units


# class Variable_ndarray(_np.ndarray):
#     """
#     Numpy array with automatic unit conversion.
    
#     When two arrays are multiplied we can automatically 
#     detect units and convert to the correct units.

#     Creating a variable with Variable_ndarray we gain access
#     to convert which can convert the unit of the variable.
#     """
#     def convert(self,unit):
#         """
#         Convert all entries in the object to the desired
#         units given by the input.
#         """
#         # Go back in the units variable does not exist.
#         if not '_units' in self.__dict__: return

#         # If it is a Units object, 
#         # we can simply loop and do the recursive conversion.
#         if isinstance(unit,Units):
#             for u in unit: 
#                 self.convert(u)
#             return

#         # Ensure that unit is a Unit
#         u = Unit(unit)
        
#         # Loop over all variables in this object.
#         # It only has one
#         for i in self._units:
#             if i.type() == u.type():
#                 self[:] *= i.convert(u)

#     def add_unit(self,var,unit):
#         """ Adds a unit to a variable beloning to the object """
        

#     def unit(self,variable='self'):
#         """ Returns the unit that is associated with the variable """
#         return self._units.unit(variable)

#     @property
#     def units(self):
#         """ Returns the units that is associated with the variable """
#         return self._units

#     @staticmethod
#     def _N(array):
#         return _np.array(array)

#     def __array_finalize__(self,obj):
#         """ Finalize the array with the object """
#         if obj is None: return

#         # Create the default units, we need to copy them, to ensure
#         # that we do not attach the same objects.
#         if hasattr(obj,'_units'):
#             self._units = deepcopy(obj._units)
#         else:
#             self._units = deepcopy(self._UNITS)

#         if hasattr(self,'__variable_finalize__'):
#             self.__variable_finalize__()
