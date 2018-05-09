"""
Sile object for reading/writing PDB files
"""
from __future__ import print_function

import numpy as np

# Import sile objects
from .sile import *

# Import the geometry object
from sisl import Geometry, SuperCell


__all__ = ['pdbSile']


class pdbSile(Sile):
    """ PDB file object """

    @Sile_fh_open
    def write_supercell(self, sc):
        """ Writes the supercell to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        # Get parameters and append the space group and specification
        args = sc.parameters() + ('P 1', 1)

        #COLUMNS       DATA  TYPE    FIELD          DEFINITION
        #-------------------------------------------------------------
        # 1 -  6       Record name   "CRYST1"
        # 7 - 15       Real(9.3)     a              a (Angstroms).
        #16 - 24       Real(9.3)     b              b (Angstroms).
        #25 - 33       Real(9.3)     c              c (Angstroms).
        #34 - 40       Real(7.2)     alpha          alpha (degrees).
        #41 - 47       Real(7.2)     beta           beta (degrees).
        #48 - 54       Real(7.2)     gamma          gamma (degrees).
        #56 - 66       LString       sGroup         Space  group.
        #67 - 70       Integer       z              Z value.
        self._write(('CRYST1' + '{:9.3f}' * 3 + '{:7.2f}' * 3 + '{:<11s}' + '{:4d}\n').format(*args))

        #COLUMNS        DATA  TYPE     FIELD         DEFINITION
        #----------------------------------------------------------------
        # 1 -  6         Record name   "ORIGXn"      n=1, 2, or 3
        #11 - 20         Real(10.6)    o[n][1]       On1
        #21 - 30         Real(10.6)    o[n][2]       On2
        #31 - 40         Real(10.6)    o[n][3]       On3
        #46 - 55         Real(10.5)    t[n]          Tn
        fmt = 'ORIGX{:1d}   ' + '{:10.6f}' * 3 + '{:10.5f}\n'
        for i in range(3):
            args = [i + 1, 0, 0, 0, sc.origo[i]]
            self._write(fmt.format(*args))

    @Sile_fh_open
    def write_geometry(self, geometry):
        """ Writes the geometry to the contained file

        Parameters
        ----------
        geometry : Geometry
           the geometry to be written
        """
        self.write_supercell(geometry.sc)

        #COLUMNS        DATA  TYPE    FIELD        DEFINITION
        #-------------------------------------------------------------------------------------
        # 1 -  6        Record name   "ATOM  "
        # 7 - 11        Integer       serial       Atom  serial number.
        #13 - 16        Atom          name         Atom name.
        #17             Character     altLoc       Alternate location indicator.
        #18 - 20        Residue name  resName      Residue name.
        #22             Character     chainID      Chain identifier.
        #23 - 26        Integer       resSeq       Residue sequence number.
        #27             AChar         iCode        Code for insertion of residues.
        #31 - 38        Real(8.3)     x            Orthogonal coordinates for X in Angstroms.
        #39 - 46        Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.
        #47 - 54        Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.
        #55 - 60        Real(6.2)     occupancy    Occupancy.
        #61 - 66        Real(6.2)     tempFactor   Temperature  factor.
        #77 - 78        LString(2)    element      Element symbol, right-justified.
        #79 - 80        LString(2)    charge       Charge  on the atom.
        fmt = 'ATOM  {:5d} {:<4s}{:1s}{:<3s} {:1s}{:4d}{:1s}   ' + '{:8.3f}' * 3 + '{:6.2f}' * 2 + ' ' * 10 + '{:2s}' * 2 + '\n'
        xyz = geometry.xyz
        # Current U is used for "UNKNOWN" input. Possibly the user can specify this later.
        for ia in geometry:
            a = geometry.atom[ia]
            args = [ia + 1, a.tag, 'U', 'U1', 'U', 1, 'U', xyz[ia, 0], xyz[ia, 1], xyz[ia, 2], a.q0.sum(), 0., a.symbol, 'U']
            self._write(fmt.format(*args))
        self._write('END\n')

    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)


add_sile('pdb', pdbSile, case=False, gzip=True)
