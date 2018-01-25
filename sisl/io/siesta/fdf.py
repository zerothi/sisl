from __future__ import print_function, division

import os.path as osp
import numpy as np
import warnings as warn

# Import sile objects
from sisl._help import _str, ensure_array
from .sile import SileSiesta
from ..sile import *
from sisl.io._help import *

from .binaries import TSHSSileSiesta, DMSileSiesta
from .eig import eigSileSiesta
from .pdos import pdosSileSiesta
from .siesta import ncSileSiesta
from .basis import ionxmlSileSiesta, ionncSileSiesta
from .orb_indx import OrbIndxSileSiesta
from sisl import Geometry, Atom, SuperCell

from sisl.utils.cmd import default_ArgumentParser, default_namespace
from sisl.utils.misc import merge_instances, str_spec

from sisl.unit.siesta import unit_convert, unit_default, unit_group

__all__ = ['fdfSileSiesta']


_LOGICAL_TRUE  = ['.true.', 'true', 'yes', 'y', 't']
_LOGICAL_FALSE = ['.false.', 'false', 'no', 'n', 'f']
_LOGICAL = _LOGICAL_FALSE + _LOGICAL_TRUE

Bohr2Ang = unit_convert('Bohr', 'Ang')


class fdfSileSiesta(SileSiesta):
    """ FDF file object """

    def __init__(self, filename, mode='r', base=None):
        """ Initialize an FDF file from the filename

        By supplying base you can reference files in other directories.
        By default the ``base`` is the directory given in the file name.
        """
        super(fdfSileSiesta, self).__init__(filename, mode=mode)
        if base is None:
            # Extract from filename
            self._directory = osp.dirname(filename)
        else:
            self._directory = base
        if len(self._directory) == 0:
            self._directory = '.'

    def __repr__(self):
        return ''.join([self.__class__.__name__, '(', self.file, ', base=', self._directory, ')'])

    @property
    def file(self):
        """ Return the current file name (without the directory prefix) """
        return self._file

    def _setup(self, *args, **kwargs):
        """ Setup the `fdfSileSiesta` after initialization """
        # These are the comments
        self._comment = ['#', '!', ';']

        # List of parent file-handles used while reading
        # This is because fdf enables inclusion of other files
        self._parent_fh = []
        self._directory = '.'

    def _tofile(self, f):
        """ Make `f` refer to the file with the appropriate base directory """
        return osp.join(self._directory, f)

    @Sile_fh_open
    def includes(self):
        """ Return a list of all files that are *included* or otherwise necessary for reading the fdf file """

        # In FDF files, %include marks files that progress
        # down in a tree structure
        def add(f):
            f = self._tofile(f)
            if f not in includes:
                includes.append(f)
        includes = []
        l = self.readline(_pure=True)
        while l != '':
            ls = l.split()
            if '%include' == ls[0].lower():
                add(ls[1])
            elif '<' in ls:
                add(ls[ls.index('<')+1])
            l = self.readline(_pure=True)
        return includes

    def readline(self, comment=False, _pure=False):
        """ Reads the next line of the file

        Parameters
        ----------
        comment : bool, optional
           allow reading a comment-line.
        """
        # Call the parent readline function
        l = super(fdfSileSiesta, self).readline(comment=comment)

        ls = l.split()
        if len(ls) < 1:
            ls.append('')

        # In FDF files, %include marks files that progress
        # down in a tree structure
        if '%include' == ls[0].lower():
            # Split for reading tree file
            self._parent_fh.append(self.fh)
            self.fh = open(self._tofile(ls[1]), self._mode)
            if _pure:
                # even if returning the include line, we should still open
                # the included file.
                return l
            # Read the following line in the new file
            return self.readline(comment)

        elif '<' in ls:
            # Split for reading tree file
            # There are two cases
            # 1. this line starts with %block
            #    In which case the entire file is read, as is (without comments)
            # 2. a set of labels is specified.
            #    This means that *only* these labels are read from the corresponding
            #    file.
            # However, since we can't know for sure whether this means what the user
            # requests, we will not return such data...
            # We will simply return the line as is.
            pass

        if len(self._parent_fh) > 0 and l == '':
            # l == '' marks the end of the file
            self.fh.close()
            self.fh = self._parent_fh.pop()
            return self.readline(comment)

        return l

    def type(self, key):
        """ Return the type of the fdf-keyword """
        found, fdf = self._read(key)
        if not found:
            return None

        if fdf.startswith('%block'):
            return 'B'

        # Grab the entire line (beside the key)
        fdf = fdf.split()[1:]
        if len(fdf) == 1:
            fdf = fdf[0].lower()
            if fdf in __LOGICAL:
                return 'b'

            if '.' in fdf:
                return 'r'
            return 'i'

        return 'n'

    def key(self, key):
        """ Return the key as written in the fdf-file. If not found, returns `None`. """
        found, fdf = self._read(key)
        if found:
            return fdf.split()[0]
        else:
            return None

    def get(self, key, unit=None, default=None, with_unit=False):
        """ Retrieve fdf-keyword from the file

        Parameters
        ----------
        key : str
            the fdf-label to search for
        """

        # First split into specification and key
        key, tmp_unit = str_spec(key)
        if unit is None:
            unit = tmp_unit

        found, fdf = self._read(key)
        if not found:
            return default

        # The keyword is found...
        fdfl = fdf.lower()
        # We need to check both start and end of block, in case
        # we are "skipping" forward a line and finds endblock
        # first.
        if fdfl.startswith('%block') or fdfl.startswith('%endblock'):
            if fdf.find('<') >= 0:
                # We have a full file to read because the block
                # is from this file
                f = fdf.split('<')[1].replace('\n', '').strip()
                l = open(self._tofile(f), 'r').readlines()
                # Remove all lines starting with a comment
                return [ll for ll in l if not (ll.split()[0][0] in self._comment)]
            found, fdf = self._read_block(key)
            if not found:
                return default
            return fdf

        # We need to process the returned value further.
        fdfl = fdf.split()
        # Check whether this is a logical flag
        if len(fdfl) == 1:
            # This *MUST* be a boolean
            #   SCF.Converge.H
            # defaults to .true.
            return True
        elif fdfl[1] in _LOGICAL_TRUE:
            return True
        elif fdfl[1] in _LOGICAL_FALSE:
            return False

        # It is something different.
        # Try and figure out what it is
        if len(fdfl) == 3:
            # We expect it to be a unit
            if unit is None:
                # Get group of unit
                group = unit_group(fdfl[2])
                # return in default sisl units
                unit = unit_default(group)

            if with_unit and tmp_unit is not None:
                # The user has specifically requested the unit:
                #  key{unit}
                return '{0:.4f} {1}'.format(float(fdfl[1]) * unit_convert(fdfl[2], unit), unit)
            elif not with_unit:
                return float(fdfl[1]) * unit_convert(fdfl[2], unit)

        return ' '.join(fdfl[1:])

    def set(self, key, value, keep=True):
        """ Add the key and value to the FDF file

        Parameters
        ----------
        key : str
           the fdf-key value to be set in the fdf file
        value : str or list of str
           the value of the string. If a `str` is passed a regular
           fdf-key is used, if a `list` it will be a %block.
        keep : bool, optional
           whether old flags will be kept in the fdf file.
        """

        # To set a key we first need to figure out if it is
        # already present, if so, we will add the new key, just above
        # the already present key.

        # 1. find the old value, and thus the file in which it is found
        with self:
            #old_value = self.get(key)
            # Get the file of the containing data
            top_file = self.file

        try:
            while len(self._parent_fh) > 0:
                self.fh.close()
                self.fh = self._parent_fh.pop()
            self.fh.close()
        except:
            # Allowed pass due to pythonic reading
            pass

        # Now we should re-read and edit the file
        lines = open(top_file, 'r').readlines()

        def write(fh, value):
            if value is None:
                return
            if isinstance(value, _str):
                fh.write(' '.join([key, value]))
                if '\n' not in value:
                    fh.write('\n')
            else:
                fh.write('%block ' + key + '\n')
                fh.write(''.join(value))
                fh.write('%endblock ' + key + '\n')

        # Now loop, write and edit
        do_write = True
        with open(top_file, 'w') as fh:
            for line in lines:
                if self.line_has_key(line, key, case=False) and do_write:
                    write(fh, value)
                    if keep:
                        fh.write('# Old value\n')
                        fh.write(line)
                    do_write = False
                else:
                    fh.write(line)

    @staticmethod
    def print(key, value):
        """ Return a string which is pretty-printing the key+value """
        if isinstance(value, list):
            s = '%block ' + key
            # if the value has any new-values
            has_nl = False
            for v in value:
                if '\n' in v:
                    has_nl = True
                    break
            if has_nl:
                # do not skip to next line in next segment
                value[-1].replace('\n', '')
                s += '\n{}'.format(''.join(value))
            else:
                s += '\n{} {}'.format(value[0], '\n'.join(value[1:]))
            s += '%endblock ' + key
        else:
            s = '{} {}'.format(key, value)
        return s

    @Sile_fh_open
    def _read(self, key):
        """ Returns the arguments following the keyword in the FDF file """
        # This routine will simply find a line where key exists
        # However, if the key is not placed according to the
        # fdf specifications we have to search for a new place.
        found, fdf = self.step_to(key, case=False)

        # Easy case when it is not found, anywhere
        # Then, for sure it is not found.
        if not found:
            return False, fdf

        # Check whether the key has the appropriate position in
        # the specification
        fdfs = [f.lower() for f in fdf.split()]

        if fdfs[0] == '%block':
            # It is a block
            if fdfs[1] == key.lower():
                return True, fdf
            # Try again, this may result in an infinite loop
            return self._read_block(key)

        # Else we have to check if '<' is in the list
        if '<' in fdfs:
            # It just have to be left of '<'
            idx = fdfs.index('<')
            if key.lower() in fdfs[:idx]:
                # Create new fdf-file
                f = fdf.split()[idx+1].replace('\n', '').strip()
                sub_fdf = fdfSileSiesta(self._tofile(f), 'r')
                return sub_fdf._read(key)
            # Try again, this may result in an infinite loop
            return self._read(key)

        return True, fdf

    @Sile_fh_open
    def _read_block(self, key, force=False):
        """ Returns the arguments following the keyword in the FDF file """
        k = key.lower()
        f, fdf = self.step_to(k, case=False)
        if not f:
            if force:
                # The user requests that the block *MUST* be found
                raise SileError(('Requested forced block could not be found: ' +
                                 str(key) + '.'), self)
            return False, []  # not found

        # If the block is piped in from another file...
        if '<' in fdf:
            # Create new fdf-file
            sub_fdf = fdfSileSiesta(fdf.split('<')[1].replace('\n', '').strip())
            return sub_fdf._read_block(key, force)

        # Check whether we have accidentially found the endblock construct
        if self.line_has_key(fdf, '%endblock', case=False):
            # Return a re-read
            return self._read_block(key, force=force)

        print('Reading block', key)
        li = []
        while True:
            l = self.readline()
            if self.line_has_key(l, '%endblock', case=False) or \
               self.line_has_key(l, k, case=False):
                return True, li
            # Append list
            li.append(l)
        raise SileError(('Error on reading block: ' + str(key) +
                         ' could not find start/end.'))

    @Sile_fh_open
    def write_supercell(self, sc, fmt='.8f', *args, **kwargs):
        """ Writes the supercell to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        fmt_str = ' {{0:{0}}} {{1:{0}}} {{2:{0}}}\n'.format(fmt)

        # Write out the cell
        self._write('LatticeConstant 1. Ang\n')
        self._write('%block LatticeVectors\n')
        self._write(fmt_str.format(*sc.cell[0, :]))
        self._write(fmt_str.format(*sc.cell[1, :]))
        self._write(fmt_str.format(*sc.cell[2, :]))
        self._write('%endblock LatticeVectors\n')

    @Sile_fh_open
    def write_geometry(self, geom, fmt='.8f', *args, **kwargs):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        self.write_supercell(geom.sc, fmt, *args, **kwargs)

        self._write('\n')
        self._write('NumberOfAtoms {0}\n'.format(geom.na))
        self._write('AtomicCoordinatesFormat Ang\n')
        self._write('%block AtomicCoordinatesAndAtomicSpecies\n')

        fmt_str = ' {{2:{0}}} {{3:{0}}} {{4:{0}}} {{0}} # {{1}}\n'.format(fmt)
        # Count for the species
        for ia, a, isp in geom.iter_species():
            self._write(fmt_str.format(isp + 1, ia + 1, *geom.xyz[ia, :]))
        self._write('%endblock AtomicCoordinatesAndAtomicSpecies\n\n')

        # Write out species
        # First swap key and value
        self._write('NumberOfSpecies {0}\n'.format(len(geom.atom.atom)))
        self._write('%block ChemicalSpeciesLabel\n')
        for i, a in enumerate(geom.atom.atom):
            self._write(' {0} {1} {2}\n'.format(i + 1, a.Z, a.tag))
        self._write('%endblock ChemicalSpeciesLabel\n')

    def read_supercell(self, *args, **kwargs):
        """ Returns `SuperCell` object from the FDF file """
        s = self.get('LatticeConstant', 'Ang')
        if s is None:
            raise SileError('Could not find LatticeConstant in file')

        # Read in cell
        cell = np.empty([3, 3], np.float64)

        lc = self.get('LatticeVectors')
        if lc:
            for i in range(3):
                cell[i, :] = [float(k) for k in lc[i].split()[:3]]
        else:
            lc = self.get('LatticeParameters')
            if lc:
                tmp = [float(k) for k in lc[0].split()[:6]]
                cell = SuperCell.tocell(*tmp)
        if lc is None:
            # the fdf file contains neither the latticevectors or parameters
            raise SileError('Could not find LatticeVectors or LatticeParameters block in file')
        cell *= s

        return SuperCell(cell)

    def _r_supercell_XV(self, *args, **kwargs):
        """ Returns `SuperCell` object from the FDF file """
        f = self.get('SystemLabel', default='siesta')
        sc = None
        if isfile(f + '.XV'):
            sc = XVSileSiesta(f + '.XV').read_supercell()
        return sc

    def read_geometry(self, *args, **kwargs):
        """ Returns Geometry object from the FDF file

        NOTE: Interaction range of the Atoms are currently not read.
        """
        sc = self.read_supercell(*args, **kwargs)

        # No fractional coordinates
        is_frac = False

        # Read atom scaling
        lc = self.get('AtomicCoordinatesFormat', default='Bohr').lower()
        if 'ang' in lc or 'notscaledcartesianang' in lc:
            s = 1.
        elif 'bohr' in lc or 'notscaledcartesianbohr' in lc:
            s = Bohr2Ang
        elif 'scaledcartesian' in lc:
            # the same scaling as the lattice-vectors
            s = self.get('LatticeConstant', 'Ang')
        elif 'fractional' in lc or 'scaledbylatticevectors' in lc:
            # no scaling of coordinates as that is entirely
            # done by the latticevectors
            s = 1.
            is_frac = True

        # If the user requests a shifted geometry
        # we correct for this
        origo = np.zeros([3], np.float64)
        lor = self.get('AtomicCoordinatesOrigin')
        if lor:
            if kwargs.get('origin', True):
                origo = ensure_array(map(float, lor[0].split()[:3])) * s
        # Origo cannot be interpreted with fractional coordinates
        # hence, it is not transformed.

        # Read atom block
        f, atms = self._read_block('AtomicCoordinatesAndAtomicSpecies', force=True)

        # Read number of atoms and block
        na = self.get('NumberOfAtoms')
        if na:
            na = int(na)
        else:
            # We default to the number of elements in the
            # AtomicCoordinatesAndAtomicSpecies block
            na = len(atms)

        # Reduce space if number of atoms specified
        if na != len(atms):
            # align number of atoms and atms array
            atms = atms[:na]

        if na == 0:
            raise ValueError('NumberOfAtoms has been determined to be zero, no atoms.')

        # Create array
        xyz = np.empty([na, 3], np.float64)
        species = np.empty([na], np.int32)
        for ia in range(na):
            l = atms[ia].split()
            xyz[ia, :] = [float(k) for k in l[:3]]
            species[ia] = int(l[3]) - 1
        if is_frac:
            xyz = np.dot(xyz, sc.cell)
        xyz *= s
        xyz += origo

        # Now we read in the species
        ns = int(self.get('NumberOfSpecies', default=0))

        # Read the block (not strictly needed, if so we simply set all atoms to
        # H)
        atom = self.read_basis()
        if atom is None:
            warn.warn('The block ChemicalSpeciesLabel does not exist, cannot determine the basis.')
            # Default atom (hydrogen)
            atom = Atom(1)
            # Force number of species to 1
            ns = 1
        else:
            atom = [atom[i] for i in species]

        # Create and return geometry object
        return Geometry(xyz, atom=atom, sc=sc)

    def _r_geometry_XV(self, *args, **kwargs):
        """ Returns `SuperCell` object from the FDF file """
        f = self.get('SystemLabel', default='siesta')
        geom = None
        if isfile(f + '.XV'):
            basis = self.read_basis()
            if basis is None:
                geom = XVSileSiesta(f + '.XV').read_geometry()
            else:
                geom = XVSileSiesta(f + '.XV').read_geometry(species_Z=True)
                for atom, _ in geom.atom.iter(True):
                    geom.atom.replace(atom, basis[atom.Z-1])
        return geom

    def read_basis(self):
        """ Read the atomic species and figure out the number of atomic orbitals in their basis

        This will try and read the basis from 3 different instances (following is order of preference):

        1. <systemlabel>.nc
        2. <>.ion.nc
        3. <>.ion.xml
        4. <>.ORB_INDX
        """
        basis = self._r_basis_nc()
        if basis is None:
            basis = self._r_basis_ion()
        if basis is None:
            basis = self._r_basis_orb_indx()
        return basis

    def _r_basis_nc(self):
        # Read basis from <>.nc file
        f = self.get('SystemLabel', default='siesta')
        try:
            return ncSileSiesta(f + '.nc').read_basis()
        except:
            pass
        return None

    def _r_basis_ion(self):
        # Read basis from <>.ion.nc file or <>.ion.xml
        spcs = self.get('ChemicalSpeciesLabel')
        if spcs is None:
            spcs = self.get('Chemical_Species_Label')
        if spcs is None:
            # We haven't found the chemical and species label,
            # so return nothing
            return None

        # Now spcs contains the block of the chemicalspecieslabel
        atom = [None] * len(spcs)
        for spc in spcs:
            idx, Z, lbl = spc.split()[:3]
            idx = int(idx) - 1 # F-indexing
            Z = int(Z)
            lbl = lbl.strip()

            # now try and read the basis
            if isfile(lbl + '.ion.nc'):
                atom[idx] = ionncSileSiesta(lbl + '.ion.nc').read_basis()
            elif isfile(lbl + '.ion.xml'):
                atom[idx] = ionxmlSileSiesta(lbl + '.ion.xml').read_basis()
            else:
                # default the atom to not have a range, and no associated orbitals
                atom[idx] = Atom(Z=Z, tag=lbl)
        return atom

    def _r_basis_orb_indx(self):
        f = self.get('SystemLabel', default='siesta')
        return OrbIndxSileSiesta(f + '.ORB_INDX').read_basis()

    def read_density_matrix(self, *args, **kwargs):
        """ Try and read the density matrix by reading the <>.nc """
        sys = self.get('SystemLabel', default='siesta')

        if isfile(sys + '.nc'):
            return ncSileSiesta(sys + '.nc').read_density_matrix()
        elif isfile(sys + '.DM'):
            geom = self.read_geometry()
            DM = DMSileSiesta(sys + '.DM').read_density_matrix()
            if geom.no == DM.no:
                DM._geom = geom
            else:
                warn.warn('The density matrix is read from *.DM without being able to read '
                          'a geometry with the correct orbitals.')
            return DM
        raise RuntimeError("Could not find the density matrix from the *.nc, *.DM.")

    def read_energy_density_matrix(self, *args, **kwargs):
        """ Try and read the energy density matrix by reading the <>.nc """
        sys = self.get('SystemLabel', default='siesta')

        if isfile(sys + '.nc'):
            return ncSileSiesta(sys + '.nc').read_energy_density_matrix()
        raise RuntimeError("Could not find the energy density matrix from the *.nc.")

    def read_hamiltonian(self, *args, **kwargs):
        """ Try and read the Hamiltonian by reading the <>.nc, <>.TSHS files, <>.HSX (in that order) """
        sys = self.get('SystemLabel', default='siesta')

        if isfile(sys + '.nc'):
            return ncSileSiesta(sys + '.nc').read_hamiltonian()
        elif isfile(sys + '.TSHS'):
            # We prefer the atomic positions in the TSHS file, however,
            # the species etc. may not necessarily be good.
            H = TSHSSileSiesta(sys + '.TSHS').read_hamiltonian()
            geom = self.read_geometry()
            for a, s in geom.atom.iter(True):
                if len(s) == 0:
                    continue
                # Only replace if the number of orbitals is correct
                i = s[0]
                if a.no == H.geom.atom[i].no:
                    H.geom.atom.replace(H.geom.atom[i], a)
            return H
        elif isfile(sys + '.HSX'):
            # Read the intrinsic geometry, then HSX will fail
            # if we can't figure out the correct number of orbitals
            geom = self.read_geometry()
            H = HSXSileSiesta(sys + '.HSX').read_hamiltonian(geom=geom)
            return H
        raise RuntimeError("Could not find the Hamiltonian from the *.nc, nor the *.TSHS file.")

    @default_ArgumentParser(description="Manipulate a FDF file.")
    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        import argparse

        # We must by-pass this fdf-file
        import sisl.io.siesta as sis

        # The fdf parser is more complicated

        # It is based on different settings based on the

        sp = p.add_subparsers(help="Determine which part of the fdf-file that should be processed.")

        # Get the label which retains all the sub-modules
        label = self.get('SystemLabel', default='siesta')

        # The default on all sub-parsers are the retrieval and setting

        d = {
            '_fdf': self,
            '_fdf_first': True,
        }
        namespace = default_namespace(**d)

        ep = sp.add_parser('edit',
                           help='Change or read and print data from the fdf file')

        # As the fdf may provide additional stuff, we do not add EVERYTHING from
        # the Geometry class.
        class FDFAdd(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                key = values[0]
                val = values[1]
                if ns._fdf_first:
                    # Append to the end of the file
                    with ns._fdf as fd:
                        fd.write('\n\n# SISL added keywords\n')
                    setattr(ns, '_fdf_first', False)
                ns._fdf.set(key, val)
        ep.add_argument('--set', '-s', nargs=2, metavar=('KEY', 'VALUE'),
                        action=FDFAdd,
                        help='Add a key to the FDF file. If it already exists it will be overwritten')

        class FDFGet(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                # Retrieve the value in standard units
                # Currently, we write out the unit "as-is"
                try:
                    val = ns._fdf.get(value[0], with_unit = True)
                except:
                    val = ns._fdf.get(value[0])
                if val is None:
                    print('# {} is currently not in the FDF file '.format(value[0]))
                    return

                print(ns._fdf.print(value[0], val))

        ep.add_argument('--get', '-g', nargs=1, metavar='KEY',
                        action=FDFGet,
                        help='Print (to stdout) the value of the key in the FDF file.')

        # If the XV file exists, it has precedence
        # of the contained geometry (we will issue
        # a warning in that case)
        f = label + '.XV'
        try:
            if osp.isfile(f):
                geom = sis.XVSileSiesta(f).read_geometry()
                warn.warn("Reading geometry from the XV file instead of the fdf-file!")
            else:
                geom = self.read_geometry()

            tmp_p = sp.add_parser('geom',
                                  help="Edit the contained geometry in the file")
            tmp_p, tmp_ns = geom.ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)
        except:
            # Allowed pass due to pythonic reading
            pass

        f = label + '.bands'
        if osp.isfile(f):
            tmp_p = sp.add_parser('band',
                                  help="Manipulate bands file from the Siesta simulation")
            tmp_p, tmp_ns = sis.bandsSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label + '.PDOS.xml'
        if osp.isfile(f):
            tmp_p = sp.add_parser('pdos',
                                  help="Manipulate PDOS.xml file from the Siesta simulation")
            tmp_p, tmp_ns = sis.pdosSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label + '.EIG'
        if osp.isfile(f):
            tmp_p = sp.add_parser('eig',
                                  help="Manipulate EIG file from the Siesta simulation")
            tmp_p, tmp_ns = sis.eigSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label + '.TBT.nc'
        if osp.isfile(f):
            tmp_p = sp.add_parser('tbt',
                                  help="Manipulate tbtrans output file")
            tmp_p, tmp_ns = sis.tbtncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label + '.TBT.Proj.nc'
        if osp.isfile(f):
            tmp_p = sp.add_parser('tbt-proj',
                                  help="Manipulate tbtrans projection output file")
            tmp_p, tmp_ns = sis.tbtprojncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label + '.PHT.nc'
        if osp.isfile(f):
            tmp_p = sp.add_parser('pht',
                                  help="Manipulate the phtrans output file")
            tmp_p, tmp_ns = sis.phtncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label + '.PHT.Proj.nc'
        if osp.isfile(f):
            tmp_p = sp.add_parser('pht-proj',
                                  help="Manipulate phtrans projection output file")
            tmp_p, tmp_ns = sis.phtprojncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label + '.nc'
        if osp.isfile(f):
            tmp_p = sp.add_parser('nc',
                                  help="Manipulate Siesta NetCDF output file")
            tmp_p, tmp_ns = sis.ncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        return p, namespace


add_sile('fdf', fdfSileSiesta, case=False, gzip=True)
