"""
Sile object for reading/writing OUT files
"""

from __future__ import print_function, division

import os.path as osp
import numpy as np
import warnings as warn

# Import sile objects
from .sile import SileSiesta
from ..sile import *
from sisl.io._help import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell, Grid

from sisl.utils.cmd import *
from sisl.utils.misc import merge_instances, name_spec

from sisl.units import unit_default, unit_group
from sisl.units.siesta import unit_convert

__all__ = ['outSileSiesta']


Bohr2Ang = unit_convert('Bohr', 'Ang')


class outSileSiesta(SileSiesta):
    """ SIESTA output file object 
    
    This enables reading the output quantities from the SIESTA output.
    """

    @Sile_fh_open
    def read_geom(self, last=True, all=False):
        """ Reads the geometry from the SIESTA output file

        Parameters
        ----------
        last: bool, True
           only read the last geometry 
        all: bool, False
           return a list of all geometries (like an MD)
           If `True` `last` is ignored
        """

        # Read until outcoor is found
        line = self.readline()
        while not 'outcoor' in line:
            line = self.readline()
            if line == '':
                return None

        # Now we have outcoor
        scaled = 'scaled' in line
        fractional = 'fractional' in line
        Ang = 'Ang' in line
        # Else it must be in Bohr

        # Read in data
        xyz = []
        atom = []
        line = self.readline()
        while len(line.strip()) > 0:
            line = line.split()
            xyz.append( [float(x) for x in line[:3]] )
            atom.append(line[3])
            line = self.readline()

        # Now we have the atomic coordinates
        # read in the unit-cell
        # Read until outcell is found
        line = self.readline()
        while not 'outcell: Unit cell vectors' in line:
            line = self.readline()

        # We read the unit-cell vectors (in Ang)
        cell = []
        line = self.readline()
        while len(line.strip()) > 0:
            line = line.split()
            cell.append( [float(x) for x in line[:3]] )
            line = self.readline()
            
        cell = np.array(cell, np.float64)
        xyz = np.array(xyz, np.float64)

        # Now create the geometry
        if scaled:
            # The output file for siesta does not
            # contain the lattice constant.
            # So... :(
            raise ValueError("Could not read the lattice-constant for the scaled geometry")
        elif fractional:
            xyz = xyz[:, 0] * cell[0,:][None,:] + \
                  xyz[:, 1] * cell[1,:][None,:] + \
                  xyz[:, 2] * cell[2,:][None,:]
        elif not Ang:
            xyz *= Bohr2Ang

        geom = Geometry(xyz, atom, sc=cell)
        if all:
            tmp = self.read_geom(last, all)
            if tmp is None:
                return [geom]
            return tmp.extend([geom])
        return geom


    @Sile_fh_open
    def read_force(self, last=True, all=False):
        """ Reads the forces from the SIESTA output file

        Parameters
        ----------
        last: bool, True
           only read the last force
        all: bool, False
           return a list of all forces (like an MD)
           If `True` `last` is ignored
        """

        # Read until outcoor is found
        line = self.readline()
        while not 'siesta: Atomic forces' in line:
            line = self.readline()
            if line == '':
                return None

        F = []
        line = self.readline()
        while not line.startswith('--'):
            F.append( [float(x) for x in line.split()[1:]] )
            line = self.readline()

        F = np.array(F)
        
        if all:
            tmp = self.read_force(last, all)
            if tmp is None:
                return []
            return tmp.extend([F])
        return F


    @Sile_fh_open
    def read_moment(self, orbital=False, quantity='S', last=True, all=False):
        """ Reads the moments from the SIESTA output file
        These will only be present in case of spin-orbit coupling.

        Parameters
        ----------
        orbital: bool, False
           return a table with orbitally resolved
           moments.
        quantity: str, 'S'
           return the spin-moments or the L moments
        last: bool, True
           only read the last force
        all: bool, False
           return a list of all forces (like an MD)
           If `True` `last` is ignored
        """

        # Read until outcoor is found
        line = self.readline()
        while not 'moments: Atomic' in line:
            line = self.readline()
            if line == '':
                return None

        # The moments are printed in SPECIES list
        self.readline() # empty

        na = 0
        # Loop the species
        tbl = []
        # Read the species label
        self.readline() # currently discarded
        while True:
            self.readline() # ""
            self.readline() # Atom    Orb ...
            # Loop atoms in this species list
            while True:
                line = self.readline()
                if line.startswith('Species') or \
                   line.startswith('--'):
                    break
                line = ' '
                atom = []
                ia = 0
                while not line.startswith('--'):
                    line = self.readline().split()
                    if ia == 0:
                        ia = int(line[0])
                    elif ia != int(line[0]):
                        raise ValueError("Error in moments formatting.")
                    # Track maximum number of atoms
                    na = max(ia, na)
                    if quantity == 'S':
                        atom.append( [float(x) for x in line[4:7]] )
                    elif quantity == 'L':
                        atom.append( [float(x) for x in line[7:10]] )
                line = self.readline().split() # Total ...
                if not orbital:
                    ia = int(line[0])
                    if quantity == 'S':
                        atom.append( [float(x) for x in line[4:7]] )
                    elif quantity == 'L':
                        atom.append( [float(x) for x in line[8:11]] )
                tbl.append( (ia, atom) )
            if line.startswith('--'):
                break

        # Sort according to the atomic index
        moments = [] * na

        # Insert in the correct atomic
        for ia, atom in tbl:
            moments[ia-1] = atom

        if not all:
            return np.array(moments)
        return moments


    def read_data(self, *args, **kwargs):
        """ Read specific content in the SIESTA out file 

        The currently implemented things are denoted in
        the parameters list.
        Note that the returned quantities are in the order
        of keywords, so:

        >>> read_data(geometry=True, force=True)
        <geom>, <forces>
        >>> read_data(force=True,geometry=True)
        <forces>, <geom>
       
        Parameters
        ----------
        geom: bool
           return the last geometry in the `outSileSiesta`
        force: bool
           return the last force in the `outSileSiesta`
        moment: bool
           return the last moments in the `outSileSiesta` (only for spin-orbit coupling calculations)
        """
        val = []
        for kw in kwargs:

            if kw == 'geom' and kwargs[kw]:
                val.append(self.read_geom())

            if kw == 'force' and kwargs[kw]:
                val.append(self.read_force())
            
            if kw == 'moment' and kwargs[kw]:
                val.append(self.read_moment())

        if len(val) == 0:
            val = None
        elif len(val) == 1:
            val = val[0]    
        return val
        
        

add_sile('out', outSileSiesta, case=False, gzip=True)

