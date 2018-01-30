from __future__ import print_function, division

import os.path as osp
import numpy as np
import warnings as warn

# Import sile objects
from sisl._help import _str, ensure_array
from .sile import SileSiesta
from ..sile import *
from sisl.io._help import *

from .binaries import TSHSSileSiesta, TSDESileSiesta
from .binaries import DMSileSiesta, HSXSileSiesta
from .eig import eigSileSiesta
from .pdos import pdosSileSiesta
from .siesta import ncSileSiesta
from .basis import ionxmlSileSiesta, ionncSileSiesta
from .orb_indx import OrbIndxSileSiesta
from .xv import XVSileSiesta
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

    def _pushfile(self, f):
        self._parent_fh.append(self.fh)
        self.fh = open(self._tofile(f), self._mode)

    def _popfile(self):
        if len(self._parent_fh) > 0:
            self.fh.close()
            self.fh = self._parent_fh.pop()
            return True
        return False

    def _seek(self):
        """ Closes all files, and starts over from beginning """
        try:
            while len(self._parent_fh) > 1:
                self._popfile()
            self.fh.seek(0)
        except:
            pass

    @Sile_fh_open
    def includes(self):
        """ Return a list of all files that are *included* or otherwise necessary for reading the fdf file """
        self._seek()
        # In FDF files, %include marks files that progress
        # down in a tree structure
        def add(f):
            f = self._tofile(f)
            if f not in includes:
                includes.append(f)
        # List of includes
        includes = []

        l = self.readline()
        while l != '':
            ls = l.split()
            if '%include' == ls[0].lower():
                add(ls[1])
                self._pushfile(ls[1])
            elif '<' in ls:
                # TODO, in principle the < could contain
                # include if this line is not a %block.
                add(ls[ls.index('<')+1])
            l = self.readline()
            while l == '':
                # last line of file
                if self._popfile():
                    l = self.readline()
                else:
                    break

        return includes

    @Sile_fh_open
    def _read_label(self, label):
        """ Try and read the first occurence of a key

        This will take care of blocks, labels and piped in labels

        Parameters
        ----------
        label : str
           label to find in the fdf file
        """
        self._seek()
        def tolabel(label):
            return label.lower().replace('_', '').replace('-', '').replace('.', '')
        labell = tolabel(label)

        def valid_line(line):
            ls = line.strip()
            if len(ls) == 0:
                return False
            return not (ls[0] in self._comment)

        def process_line(line):
            # Split line by spaces
            ls = line.split()
            if len(ls) == 0:
                return None

            # Make a lower equivalent of ls
            lsl = list(map(tolabel, ls))

            # Check if there is a pipe in the line
            if '<' in lsl:
                idx = lsl.index('<')
                # Now there are two cases

                # 1. It is a block, in which case
                #    the full block is piped into the label
                #    %block Label < file
                if lsl[0] == '%block' and lsl[1] == labell:
                    # Correct line found
                    # Read the file content, removing any empty and/or comment lines
                    lines = open(self._tofile(ls[3]), 'r').readlines()
                    return [l.strip() for l in lines if valid_line(l)]

                # 2. There are labels that should be read from a subsequent file
                #    Label1 Label2 < other.fdf
                if labell in lsl[:idx]:
                    # Valid line, read key from other.fdf
                    return fdfSileSiesta(self._tofile(ls[idx+1]), base=self._directory)._read_label(label)

                # It is not in this line, either key is
                # on the RHS of <, or the key could be "block". Say.
                return None

            # The last case is if the label is the first word on the line
            # In that case we have found what we are looking for
            if lsl[0] == labell:
                return (' '.join(ls[1:])).strip()

            elif lsl[0] == '%block':
                if lsl[1] == labell:
                    # Read in the block content
                    lines = []

                    # Now read lines
                    l = self.readline().strip()
                    while not tolabel(l).startswith('%endblock'):
                        if len(l) > 0:
                            lines.append(l)
                        l = self.readline().strip()
                    return lines

            elif lsl[0] == '%include':

                # We have to open a new file
                self._pushfile(ls[1])

            return None

        # Perform actual reading of line
        l = self.readline()
        if len(l) == 0:
            return None
        l = process_line(l)
        while l is None:
            l = self.readline()
            if len(l) == 0:
                return None
            l = process_line(l)

        return l

    def _type(self, value):
        """ Determine the type by the value

        Parameters
        ----------
        value : str or list or numpy.ndarray
            the value to check for fdf-type
        """
        if value is None:
            return None

        if isinstance(value, list):
            # A block, %block ...
            return 'B'

        if isinstance(value, np.ndarray):
            # A list, Label [...]
            return 'a'

        # Grab the entire line (beside the key)
        values = value.split()
        if len(values) == 1:
            fdf = values[0].lower()
            if fdf in _LOGICAL:
                # logical
                return 'b'

            if '.' in fdf:
                # a real number
                return 'r'
            try:
                int(fdf)
                return 'i'
            except:
                pass
            # fall-back to name with everything
        elif len(values) == 2:
            # possibly a physical value
            try:
                float(values[0])
                return 'p'
            except:
                pass

        return 'n'

    @Sile_fh_open
    def type(self, label):
        """ Return the type of the fdf-keyword

        Parameters
        ----------
        label : str
            the label to look-up
        """
        self._seek()
        return self._type(self._read_label(label))

    @Sile_fh_open
    def get(self, label, unit=None, default=None, with_unit=False):
        """ Retrieve fdf-keyword from the file

        Parameters
        ----------
        label : str
            the fdf-label to search for
        unit : str, optional
            unit of the physical quantity to return
        default : *, optional
            if the label is not found, this will be the returned value (default to ``None``)
        with_unit : bool, optional
            whether the physical quantity gets returned with the found unit in the fdf file.

        Returns
        -------
        value : the value of the fdf-label. If the label is a block, a `list` is returned, for
                a real value a `float` (or if the default is of `float`), for an integer, an
                `int` is returned.
        unit : if `with_unit` is true this will contain the associated unit if it is specified

        Examples
        --------
        >>> print(open(...).readlines()) # doctest: +SKIP
        LabeleV 1. eV
        LabelRy 1. Ry
        Label name
        FakeInt 1
        %block Hello
        line 1
        line2
        %endblock
        >>> fdf.get('LabeleV') == 1. # default unit is eV
        >>> fdf.get('LabelRy') == unit.siesta.unit_convert('Ry', 'eV')
        >>> fdf.get('LabelRy', 'Ry') == 1.
        >>> fdf.get('LabelRy', with_unit=True) == (1., 'Ry')
        >>> fdf.get('FakeInt', default='0') == '1'
        >>> fdf.get('LabeleV', with_unit=True) == (1., 'eV')
        >>> fdf.get('Label', with_unit=True) == 'name' # no unit present on line
        >>> fdf.get('Hello') == ['line 1', 'line2']
        """
        # Try and read a line
        value = self._read_label(label)

        # Simply return the default value if not found
        if value is None:
            return default

        # Figure out what it is
        t = self._type(value)

        # We will only do something if it is a real, int, or physical.
        # Else we simply return, as-is
        if t == 'r':
            if default is None:
                return float(value)
            t = type(default)
            return t(value)

        elif t == 'i':
            if default is None:
                return int(value)
            t = type(default)
            return t(value)

        elif t == 'p':
            value = value.split()
            if with_unit:
                # Simply return, as is. Let the user do whatever.
                return float(value[0]), value[1]
            if unit is None:
                default = unit_default(unit_group(value[1]))
            else:
                if unit_group(value[1]) != unit_group(unit):
                    raise ValueError("Requested unit for {} is not the same type. "
                                     "Found/Requested {}/{}'".format(label, value[1], unit))
                default = unit
            return float(value[0]) * unit_convert(value[1], default)

        elif t == 'b':
            return value.lower() in _LOGICAL_TRUE

        return value

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

    @staticmethod
    def _SpGeom_replace_geom(spgeom, geom):
        """ Replace all atoms in spgeom with the atom in geom while retaining the number of orbitals

        Currently we need some way of figuring out whether the number of atoms and orbitals are
        consistent.

        Parameters
        ----------
        spgeom : SparseGeometry
           the sparse object with attached geometry
        geom : Geometry
           geometry to grab atoms from
        full_replace : bool, optional
           whether the full geometry may be replaced in case ``spgeom.na != geom.na && spgeom.no == geom.no``.
           This is required when `spgeom` does not contain information about atoms.
        """
        if spgeom.na != geom.na and spgeom.no == geom.no:
            # In this case we cannot compare individiual atoms # of orbitals.
            # I.e. we
            spgeom._geom = geom
            return True
        elif spgeom.na != geom.na:
            warn.warn('cannot replace geometry due to insufficient information regarding number of '
                      'atoms and orbitals, ensuring correct geometry failed...')

        no_no = spgeom.no == geom.no
        # Loop and make sure the number of orbitals is consistent
        for a, idx in geom.atom.iter(True):
            if len(idx) == 0:
                continue
            Sa = spgeom.geom.atom[idx[0]]
            if Sa.no != a.no:
                # Make sure the atom we replace with retains the same information
                # *except* the number of orbitals.
                a = Atom(a.Z, Sa.orbital, mass=a.mass, tag=a.tag)
            S.geom.atom.replace(idx, a)
        return no_no

    def read_supercell(self, output=False, *args, **kwargs):
        """ Returns SuperCell object by reading fdf or Siesta output related files.

        One can limit the tried files to only one file by passing
        only a single file ending.

        Parameters
        ----------
        output: bool, optional
            whether to read supercell from output files (default to read from
            the fdf file).
        order: list of str, optional
            the order of which to try and read the supercell.
            By default this is ``['XV', 'nc', 'fdf']`` if `output` is true.
            If `order` is present `output` is disregarded.

        Examples
        --------
        >>> fdf = get_sile('RUN.fdf') # doctest: +SKIP
        >>> fdf.read_supercell() # read from fdf
        >>> fdf.read_supercell(True) # read from [XV, nc, fdf]
        >>> fdf.read_supercell(order=['nc']) # read from [nc]
        >>> fdf.read_supercell(True, order=['nc']) # read from [nc]
        """
        if output:
            order = kwargs.pop('order', ['XV', 'nc', 'fdf'])
        else:
            order = kwargs.pop('order', ['fdf'])
        for f in order:
            v = getattr(self, '_r_supercell_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                return v
        return None

    def _r_supercell_fdf(self, *args, **kwargs):
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

    def _r_supercell_nc(self):
        # Read supercell from <>.nc file
        f = self._tofile(self.get('SystemLabel', default='siesta')) + '.nc'
        if isfile(f):
            return ncSileSiesta(f).read_supercell()
        return None

    def _r_supercell_xv(self, *args, **kwargs):
        """ Returns `SuperCell` object from the FDF file """
        f = self._tofile(self.get('SystemLabel', default='siesta')) + '.XV'
        if isfile(f):
            return XVSileSiesta(f).read_supercell()
        return None

    def read_geometry(self, output=False, *args, **kwargs):
        """ Returns Geometry object by reading fdf or Siesta output related files.

        One can limit the tried files to only one file by passing
        only a single file ending.

        Parameters
        ----------
        output: bool, optional
            whether to read geometry from output files (default to read from
            the fdf file).
        order: list of str, optional
            the order of which to try and read the geometry.
            By default this is ``['XV', 'nc', 'fdf']`` if `output` is true
            If `order` is present `output` is disregarded.

        Examples
        --------
        >>> fdf = get_sile('RUN.fdf') # doctest: +SKIP
        >>> fdf.read_geometry() # read from fdf
        >>> fdf.read_geometry(True) # read from [XV, nc, fdf]
        >>> fdf.read_geometry(order=['nc']) # read from [nc]
        >>> fdf.read_geometry(True, order=['nc']) # read from [nc]
        """
        if output:
            order = kwargs.pop('order', ['XV', 'nc', 'fdf'])
        else:
            order = kwargs.pop('order', ['fdf'])
        for f in order:
            v = getattr(self, '_r_geometry_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                return v
        return None

    def _r_geometry_xv(self, *args, **kwargs):
        """ Returns `SuperCell` object from the FDF file """
        f = self._tofile(self.get('SystemLabel', default='siesta')) + '.XV'
        geom = None
        if isfile(f):
            basis = self.read_basis()
            if basis is None:
                geom = XVSileSiesta(f).read_geometry()
            else:
                geom = XVSileSiesta(f).read_geometry(species_Z=True)
                for atom, _ in geom.atom.iter(True):
                    geom.atom.replace(atom, basis[atom.Z-1])
        return geom

    def _r_geometry_nc(self):
        # Read geometry from <>.nc file
        f = self._tofile(self.get('SystemLabel', default='siesta')) + '.nc'
        if isfile(f):
            return ncSileSiesta(f).read_geometry()
        return None

    def _r_geometry_fdf(self, *args, **kwargs):
        """ Returns Geometry object from the FDF file

        NOTE: Interaction range of the Atoms are currently not read.
        """
        sc = self.read_supercell(order=['fdf'])

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
        atms = self.get('AtomicCoordinatesAndAtomicSpecies')
        if atms is None:
            raise SileError('AtomicCoordinatesAndAtomicSpecies block could not be found')

        # Read number of atoms and block
        # We default to the number of elements in the
        # AtomicCoordinatesAndAtomicSpecies block
        na = self.get('NumberOfAtoms', default=len(atms))

        # Reduce space if number of atoms specified
        if na < len(atms):
            # align number of atoms and atms array
            atms = atms[:na]
        elif na > len(atms):
            raise SileError('NumberOfAtoms is larger than the atoms defined in the blocks')
        elif na == 0:
            raise SileError('NumberOfAtoms has been determined to be zero, no atoms.')

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
        ns = self.get('NumberOfSpecies', default=0)

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

    def read_basis(self, *args, **kwargs):
        """ Read the atomic species and figure out the number of atomic orbitals in their basis

        The order of the read is shown below.

        One can limit the tried files to only one file by passing
        only a single file ending.

        Parameters
        ----------
        order: list of str, optional
            the order of which to try and read the basis information.
            By default this is ``['nc', 'ion', 'ORB_INDX']``
        """
        order = kwargs.pop('order', ['nc', 'ion', 'ORB_INDX'])
        for f in order:
            v = getattr(self, '_r_basis_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                return v
        return None

    def _r_basis_nc(self):
        # Read basis from <>.nc file
        f = self._tofile(self.get('SystemLabel', default='siesta')) + '.nc'
        if isfile(f):
            return ncSileSiesta(f).read_basis()
        return None

    def _r_basis_ion(self):
        # Read basis from <>.ion.nc file or <>.ion.xml
        spcs = self.get('ChemicalSpeciesLabel')
        if spcs is None:
            # We haven't found the chemical and species label
            # so return nothing
            return None

        # Now spcs contains the block of the chemicalspecieslabel
        atom = [None] * len(spcs)
        for spc in spcs:
            idx, Z, lbl = spc.split()[:3]
            idx = int(idx) - 1 # F-indexing
            Z = int(Z)
            lbl = lbl.strip()
            f = self._tofile(lbl)

            # now try and read the basis
            if isfile(f + '.ion.nc'):
                atom[idx] = ionncSileSiesta(f + '.ion.nc').read_basis()
            elif isfile(f + '.ion.xml'):
                atom[idx] = ionxmlSileSiesta(f + '.ion.xml').read_basis()
            else:
                # default the atom to not have a range, and no associated orbitals
                atom[idx] = Atom(Z=Z, tag=lbl)
        return atom

    def _r_basis_orb_indx(self):
        f = self._tofile(self.get('SystemLabel', default='siesta')) + '.ORB_INDX'
        if isfile(f):
            return OrbIndxSileSiesta(f).read_basis()
        return None

    def read_density_matrix(self, *args, **kwargs):
        """ Try and read density matrix by reading the <>.nc, <>.TSDE files, <>.DM (in that order)

        One can limit the tried files to only one file by passing
        only a single file ending.

        Parameters
        ----------
        order: list of str, optional
            the order of which to try and read the density matrix
            By default this is ``['nc', 'TSDE', 'DM']``.
        """
        order = kwargs.pop('order', ['nc', 'TSDE', 'DM'])
        for f in order:
            v = getattr(self, '_r_density_matrix_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                return v
        return None

    def _r_density_matrix_nc(self, *args, **kwargs):
        """ Try and read the density matrix by reading the <>.nc """
        f = self._tofile(self.get('SystemLabel', default='siesta')) + '.nc'
        if isfile(f):
            return ncSileSiesta(f).read_density_matrix(*args, **kwargs)
        return None

    def _r_density_matrix_TSDE(self, *args, **kwargs):
        """ Read density matrix from the TSDE file """
        f = self._tofile(self.get('SystemLabel', default='siesta')) + '.TSDE'
        DM = None
        if isfile(f):
            geom = self.read_geometry(True)
            DM = TSDESileSiesta(f).read_density_matrix(*args, **kwargs)
            self._SpGeom_replace_geom(DM, geom)
        return DM

    def _r_density_matrix_DM(self, *args, **kwargs):
        """ Read density matrix from the DM file """
        f = self._tofile(self.get('SystemLabel', default='siesta')) + '.DM'
        DM = None
        if isfile(f):
            geom = self.read_geometry(True)
            DM = DMSileSiesta(f).read_density_matrix(*args, **kwargs)
            self._SpGeom_replace_geom(DM, geom)
        return DM

    def read_energy_density_matrix(self, *args, **kwargs):
        """ Try and read energy density matrix by reading the <>.nc or <>.TSDE files (in that order)

        One can limit the tried files to only one file by passing
        only a single file ending.

        Parameters
        ----------
        order: list of str, optional
            the order of which to try and read the density matrix
            By default this is ``['nc', 'TSDE']``.
        """
        order = kwargs.pop('order', ['nc', 'TSDE'])
        for f in order:
            v = getattr(self, '_r_energy_density_matrix_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                return v
        return None

    def _r_energy_density_matrix_nc(self, *args, **kwargs):
        """ Read energy density matrix by reading the <>.nc """
        f = self._tofile(self.get('SystemLabel', default='siesta')) + '.nc'
        if isfile(f):
            return ncSileSiesta(f).read_energy_density_matrix(*args, **kwargs)
        return None

    def _r_energy_density_matrix_TSDE(self, *args, **kwargs):
        """ Read energy density matrix from the TSDE file """
        f = self._tofile(self.get('SystemLabel', default='siesta')) + '.TSDE'
        DM = None
        if isfile(f):
            geom = self.read_geometry(True)
            DM = TSDESileSiesta(f).read_energy_density_matrix(*args, **kwargs)
            self._SpGeom_replace_geom(DM, geom)
        return DM

    def read_hamiltonian(self, *args, **kwargs):
        """ Try and read the Hamiltonian by reading the <>.nc, <>.TSHS files, <>.HSX (in that order)

        One can limit the tried files to only one file by passing
        only a single file ending.

        Parameters
        ----------
        order: list of str, optional
            the order of which to try and read the Hamiltonian.
            By default this is ``['nc', 'TSHS', 'HSX']``.
        """
        order = kwargs.pop('order', ['nc', 'TSHS', 'HSX'])
        for f in order:
            v = getattr(self, '_r_hamiltonian_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                return v
        return None

    def _r_hamiltonian_nc(self, *args, **kwargs):
        """ Read Hamiltonian from the nc file """
        f = self._tofile(self.get('SystemLabel', default='siesta')) + '.nc'
        if isfile(f):
            return ncSileSiesta(f).read_hamiltonian(*args, **kwargs)
        return None

    def _r_hamiltonian_TSHS(self, *args, **kwargs):
        """ Read Hamiltonian from the TSHS file """
        f = self._tofile(self.get('SystemLabel', default='siesta')) + '.TSHS'
        H = None
        if isfile(f):
            geom = self.read_geometry(True)
            H = TSHSSileSiesta(f).read_hamiltonian(*args, **kwargs)
            self._SpGeom_replace_geom(H, geom)
        return H

    def _r_hamiltonian_HSX(self, *args, **kwargs):
        """ Read Hamiltonian from the HSX file """
        f = self._tofile(self.get('SystemLabel', default='siesta')) + '.HSX'
        H = None
        if isfile(f):
            geom = self.read_geometry(True)
            H = HSXSileSiesta(f).read_hamiltonian(*args, **kwargs)
            self._SpGeom_replace_geom(H, geom)
        return H

    @default_ArgumentParser(description="Manipulate a FDF file.")
    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        import argparse

        # We must by-pass this fdf-file for importing
        import sisl.io.siesta as sis

        # The fdf parser is more complicated

        # It is based on different settings based on the

        sp = p.add_subparsers(help="Determine which part of the fdf-file that should be processed.")

        # Get the label which retains all the sub-modules
        label = self._tofile(self.get('SystemLabel', default='siesta'))

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
                val = ns._fdf.get(value[0], with_unit=True)
                if val is None:
                    print('# {} is currently not in the FDF file '.format(value[0]))
                    return

                if isinstance(val, tuple):
                    print(ns._fdf.print(value[0], '{} {}'.format(*val)))
                else:
                    print(ns._fdf.print(value[0], val))

        ep.add_argument('--get', '-g', nargs=1, metavar='KEY',
                        action=FDFGet,
                        help='Print (to stdout) the value of the key in the FDF file.')

        # If the XV file exists, it has precedence
        # of the contained geometry (we will issue
        # a warning in that case)
        f = label + '.XV'
        try:
            geom = self.read_geometry(True)

            tmp_p = sp.add_parser('geom',
                                  help="Edit the contained geometry in the file")
            tmp_p, tmp_ns = geom.ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)
        except:
            # Allowed pass due to pythonic reading
            pass

        f = label + '.bands'
        if isfile(f):
            tmp_p = sp.add_parser('band',
                                  help="Manipulate bands file from the Siesta simulation")
            tmp_p, tmp_ns = sis.bandsSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label + '.PDOS.xml'
        if isfile(f):
            tmp_p = sp.add_parser('pdos',
                                  help="Manipulate PDOS.xml file from the Siesta simulation")
            tmp_p, tmp_ns = sis.pdosSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label + '.EIG'
        if isfile(f):
            tmp_p = sp.add_parser('eig',
                                  help="Manipulate EIG file from the Siesta simulation")
            tmp_p, tmp_ns = sis.eigSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label + '.TBT.nc'
        if isfile(f):
            tmp_p = sp.add_parser('tbt',
                                  help="Manipulate tbtrans output file")
            tmp_p, tmp_ns = sis.tbtncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label + '.TBT.Proj.nc'
        if isfile(f):
            tmp_p = sp.add_parser('tbt-proj',
                                  help="Manipulate tbtrans projection output file")
            tmp_p, tmp_ns = sis.tbtprojncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label + '.PHT.nc'
        if isfile(f):
            tmp_p = sp.add_parser('pht',
                                  help="Manipulate the phtrans output file")
            tmp_p, tmp_ns = sis.phtncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label + '.PHT.Proj.nc'
        if isfile(f):
            tmp_p = sp.add_parser('pht-proj',
                                  help="Manipulate phtrans projection output file")
            tmp_p, tmp_ns = sis.phtprojncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label + '.nc'
        if isfile(f):
            tmp_p = sp.add_parser('nc',
                                  help="Manipulate Siesta NetCDF output file")
            tmp_p, tmp_ns = sis.ncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        return p, namespace


add_sile('fdf', fdfSileSiesta, case=False, gzip=True)
