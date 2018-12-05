"""
Sile object for reading/writing PDB files
"""
from __future__ import print_function

import numpy as np

# Import sile objects
from .sile import *

# Import the geometry object
from sisl import Geometry, SuperCell, Atoms, Atom


__all__ = ['pdbSile']


class pdbSile(Sile):
    """ PDB file object """

    def _setup(self, *args, **kwargs):
        """ Instantiate counters """
        self._model = 1
        self._serial = 1
        self._wrote_header = False

    def _w_sisl(self):
        """ Placeholder for adding the header information """
        if self._wrote_header:
            return
        self._wrote_header = True
        self._write('EXPDTA    {:60s}\n'.format("THEORETICAL MODEL"))
        # Add dates, AUTHOR etc.

    def _w_model(self, start):
        """ Writes the start of the next model """
        if start:
            self._write('MODEL {}\n'.format(self._model))
            self._model += 1
            # Serial counter
            self._serial = 1
        else:
            self._write('ENDMDL\n')

    def _step_record(self, record, reread=True):
        """ Step to a specific record entry in the PDB file """

        found = False
        # The previously read line number
        line = self._line

        while not found:
            l = self.readline()
            if l == '':
                break
            found = l.startswith(record)

        if not found and (l == '' and line > 0) and reread:
            # We may be in the case where the user request
            # reading the same twice...
            # So we need to re-read the file...
            self.fh.close()
            # Re-open the file...
            self._open()

            # Try and read again
            while not found and self._line <= line:
                l = self.readline()
                if l == '':
                    break
                found = l.startswith(record)

        return found, l

    @sile_fh_open()
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

        #COLUMNS         DATA  TYPE    FIELD              DEFINITION
        #------------------------------------------------------------------
        # 1 -  6         Record name   "SCALEn" n=1,  2, or 3
        #11 - 20         Real(10.6)    s[n][1]            Sn1
        #21 - 30         Real(10.6)    s[n][2]            Sn2
        #31 - 40         Real(10.6)    s[n][3]            Sn3
        #46 - 55         Real(10.5)    u[n]               Un
        for i in range(3):
            args = [i + 1] + sc.cell[i, :].tolist() + [0.]
            self._write(('SCALE{:1d}    {:10.6f}{:10.6f}{:10.6f}     {:10.5f}\n').format(*args))

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

    @sile_fh_open()
    def read_supercell(self):
        """ Read supercell from the contained file """
        f, line = self._step_record('CRYST1')

        if not f:
            raise SileError(str(self) + ' does not contain a CRYST1 record.')
        a = float(line[6:15])
        b = float(line[15:24])
        c = float(line[24:33])
        alpha = float(line[33:40])
        beta = float(line[40:47])
        gamma = float(line[47:54])
        cell = SuperCell.tocell([a, b, c, alpha, beta, gamma])

        f, line = self._step_record('SCALE1')
        if f:
            cell[0, :] = float(line[11:20]), float(line[21:30]), float(line[31:40])
            f, line = self._step_record('SCALE2')
            if not f:
                raise SileError(str(self) + ' found SCALE1 but not SCALE2!')
            cell[1, :] = float(line[11:20]), float(line[21:30]), float(line[31:40])
            f, line = self._step_record('SCALE3')
            if not f:
                raise SileError(str(self) + ' found SCALE1 but not SCALE3!')
            cell[2, :] = float(line[11:20]), float(line[21:30]), float(line[31:40])

        origo = np.zeros(3)
        for i in range(3):
            f, line = self._step_record('ORIGX{}'.format(i + 1))
            if f:
                origo[i] = float(line[45:55])

        return SuperCell(cell, origo=origo)

    @sile_fh_open()
    def write_geometry(self, geometry):
        """ Writes the geometry to the contained file

        Parameters
        ----------
        geometry : Geometry
           the geometry to be written
        """
        self.write_supercell(geometry.sc)

        # Start a model
        self._w_model(True)

        # Generically the configuration (model) are non-polymers, hence we use the HETATM type
        atom = 'HETATM'

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
        fmt = '{:<6s}'.format(atom) + '{:5d} {:<4s}{:1s}{:<3s} {:1s}{:4d}{:1s}   ' + '{:8.3f}' * 3 + '{:6.2f}' * 2 + ' ' * 10 + '{:2s}' * 2 + '\n'
        xyz = geometry.xyz
        # Current U is used for "UNKNOWN" input. Possibly the user can specify this later.
        for ia in geometry:
            a = geometry.atoms[ia]
            args = [self._serial, a.tag, 'U', 'U1', 'U', 1, 'U', xyz[ia, 0], xyz[ia, 1], xyz[ia, 2], a.q0.sum(), 0., a.symbol, '0']
            # Step serial
            self._serial += 1
            self._write(fmt.format(*args))

        # End a model
        self._w_model(False)

    @sile_fh_open()
    def read_geometry(self):
        """ Read geometry from the contained file """

        # First we read in the geometry
        sc = self.read_supercell()

        # Try and go to the first model record
        in_model, l = self._step_record('MODEL')

        idx = []
        tags = []
        xyz = []
        Z = []
        if in_model:
            l = self.readline()
            def is_atom(line):
                return l.startswith('ATOM') or l.startswith('HETATM')
            def is_end_model(line):
                return l.startswith('ENDMDL') or l == ''
            while not is_end_model(l):
                if is_atom(l):
                    idx.append(int(l[6:11]))
                    tags.append(l[12:16].strip())
                    xyz.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
                    Z.append(l[76:78].strip())
                l = self.readline()

        # First sort all atoms according to the idx array
        idx = np.array(idx)
        idx = np.argsort(idx)
        xyz = np.array(xyz)[idx, :]
        tags = [tags[i] for i in idx]
        Z = [Z[i] for i in idx]

        # Create the atom list
        atoms = Atoms(Atom(Z[0], tag=tags[0]), na=len(Z))
        for i, a in enumerate(map(Atom, Z, tags)):
            try:
                s = atoms.index(a)
            except:
                s = len(atoms.atom)
                atoms._atom.append(a)
            atoms._specie[i] = s

        return Geometry(xyz, atoms, sc=sc)

    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)


add_sile('pdb', pdbSile, case=False, gzip=True)
