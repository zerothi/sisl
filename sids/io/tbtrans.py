"""
Sile object for reading TBtrans binary files
"""
from __future__ import print_function

# Import sile objects
from sids.io.sile import *

# Import the geometry object
from sids import Geometry, Atom, SuperCell
from sids import Bohr

import numpy as np

__all__ = ['TBtransSile','PHtransSile']


class TBtransSile(NCSile):
    """ TBtrans file object """

    def _data(self,name):
        """ Local method for obtaining the data from the NCSile.
        
        This method checks how the file is access, i.e. whether 
        data is stored in the object or it should be read consequtively.
        """
        if self._access > 0:
            return self.__data[name]
        return self.variables[name][:]

    def _setup(self):
        """ Setup the special object for data containing """
        if self._access > 0:
            self.__data = dict()

            # Fake double calls
            access = self._access
            self._access = 0
            
            # Create elements
            for d in ['cell', 'xa', 'lasto',
                      'a_dev', 'pivot', 
                      'kpt', 'wkpt', 'E']:
                self.__data[d] = self._data(d)

            self._access = access
            
            # Create the geometry in the data file
            self.__data['_geom'] = self.read_geom()


    def read_sc(self):
        """ Returns `SuperCell` object from a .TBT.nc file """
        if not hasattr(self,'fh'):
            with self:
                return self.read_sc()

        cell = np.array(np.copy(self.cell), dtype=np.float64)
        cell.shape = (3,3)

        return SuperCell(cell)
    

    def read_geom(self):
        """ Returns Geometry object from a .TBT.nc file """
        # Quick access to the geometry object
        if self._access > 0:
            return self._data('_geom')
        
        if not hasattr(self,'fh'):
            with self:
                return self.read_geom()

        sc = self.read_sc()

        xyz = np.array(np.copy(self.xa), dtype=np.float64)
        xyz.shape = (-1,3)

        # Create list with correct number of orbitals
        lasto = np.array(np.copy(self.lasto), dtype=np.int32)
        nos = np.append([lasto[0]], np.diff(lasto))
        nos = np.array(nos,np.int32)

        # Default to Hydrogen atom with nos[ia] orbitals
        # This may be counterintuitive but there is no storage of the
        # actual species
        atms = [Atom(Z='H', orbs=o) for o in nos]
        
        # Create and return geometry object
        geom = Geometry(xyz, atoms=atms, sc=sc)
        
        return geom

    
    def write_geom(self):
        """ This does not work """
        raise ValueError(self.__class__.__name__+" can not write a geometry")

    # This class also contains all the important quantities elements of the
    # file.

    @property
    def cell(self):
        """ Unit cell in file """
        return self._data('cell') / Bohr

    @property
    def na(self):
        """ Returns number of atoms in the cell """
        return int(len(self.dimensions['na_u']))
    
    na_u = na

    @property
    def no(self):
        """ Returns number of orbitals in the cell """
        return int(len(self.dimensions['no_u']))

    no_u = no

    @property
    def xa(self):
        """ Atomic coordinates in file """
        return self._data('xa') / Bohr

    xyz = xa

    # Device atoms and other quantities
    @property
    def na_d(self):
        """ Number of atoms in the device region """
        return int(len(self.dimensions['na_d']))

    na_dev = na_d

    @property
    def a_d(self):
        """ Atomic indices (1-based) of device atoms """
        return self._data('a_dev')

    a_dev = a_d

    @property
    def pivot(self):
        """ Pivot table of device orbitals to obtain input sorting """
        return self._data('pivot')
    
    pvt = pivot

    @property
    def lasto(self):
        """ Last orbital of corresponding atom """
        return self._data('lasto')
    
    @property
    def no_d(self):
        """ Number of orbitals in the device region """
        return int(len(self.dimensions['no_d']))

    
    
    @property
    def kpt(self):
        """ Sampled k-points in file """
        return self._data('kpt')

    @property
    def wkpt(self):
        """ Weights of k-points in file """
        return self._data('wkpt')

    @property
    def nkpt(self):
        """ Number of k-points in file """
        return len(self.dimensions['nkpt'])

    @property
    def e(self):
        """ Sampled energy-points in file """
        return self._data('E') / Ry
    E = e

    @property
    def ne(self):
        """ Number of energy-points in file """
        return len(self.dimensions['ne'])

    nE = ne

    @property
    def elecs(self):
        """ List of electrodes """
        elecs = self.groups.keys()

        # in cases of not calculating all 
        # electrode transmissions we must ensure that
        # we add the last one
        var = self.groups[elecs[0]].variables.keys()
        for tvar in var:
            if tvar.endswith('.T'):
                tvar = tvar.split('.')[0]
                if not tvar in elecs:
                    elecs.append(tvar)
        return elecs

    # aliases
    electrodes = elecs
    Electrodes = elecs
    Elecs = elecs

    


class PHtransSile(TBtransSile):
    """ PHtrans file object """
    pass
