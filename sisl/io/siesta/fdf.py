import warnings
from datetime import datetime
import numpy as np
import scipy as sp
from os.path import isfile
import itertools as itools

from ..sile import add_sile, get_sile_class, sile_fh_open, sile_raise_write, SileError
from .sile import SileSiesta
from .._help import *

from sisl._internal import set_module
from sisl import constant
from sisl.unit.siesta import units
import sisl._array as _a
from sisl._indices import indices_only
from sisl.utils.ranges import list2str
from sisl.messages import SislError, info, warn
from sisl.utils.mathematics import fnorm

from .binaries import tshsSileSiesta, tsdeSileSiesta
from .binaries import dmSileSiesta, hsxSileSiesta, onlysSileSiesta
from .eig import eigSileSiesta
from .fc import fcSileSiesta
from .fa import faSileSiesta
from .siesta_grid import gridncSileSiesta
from .siesta_nc import ncSileSiesta
from .basis import ionxmlSileSiesta, ionncSileSiesta
from .orb_indx import orbindxSileSiesta
from .xv import xvSileSiesta
from sisl import Geometry, Orbital, Atom, AtomGhost, Atoms, SuperCell, DynamicalMatrix

from sisl.utils.cmd import default_ArgumentParser, default_namespace
from sisl.utils.misc import merge_instances
from sisl.unit.siesta import unit_convert, unit_default, unit_group

__all__ = ['fdfSileSiesta']


_LOGICAL_TRUE  = ['.true.', 'true', 'yes', 'y', 't']
_LOGICAL_FALSE = ['.false.', 'false', 'no', 'n', 'f']
_LOGICAL = _LOGICAL_FALSE + _LOGICAL_TRUE

Bohr2Ang = unit_convert('Bohr', 'Ang')


def _listify_str(arg):
    if isinstance(arg, str):
        return [arg]
    return arg


def _track(method, msg):
    if method.__self__.track:
        info(f"{method.__self__.__class__.__name__}.{method.__name__}: {msg}")


def _track_file(method, f, msg=None):
    if msg is None:
        if f.is_file():
            msg = f"reading file {f}"
        else:
            msg = f"could not find file {f}"
    if method.__self__.track:
        info(f"{method.__self__.__class__.__name__}.{method.__name__}: {msg}")


@set_module("sisl.io.siesta")
class fdfSileSiesta(SileSiesta):
    """ FDF-input file

    By supplying base you can reference files in other directories.
    By default the ``base`` is the directory given in the file name.

    Parameters
    ----------
    filename: str
       fdf file
    mode : str, optional
       opening mode, default to read-only
    base : str, optional
       base-directory to read output files from.

    Examples
    --------
    >>> fdf = fdfSileSiesta('tmp/RUN.fdf') # reads output files in 'tmp/' folder
    >>> fdf = fdfSileSiesta('tmp/RUN.fdf', base='.') # reads output files in './' folder
    """

    def _setup(self, *args, **kwargs):
        """ Setup the `fdfSileSiesta` after initialization """
        self._comment = ['#', '!', ';']

        # List of parent file-handles used while reading
        # This is because fdf enables inclusion of other files
        self._parent_fh = []

        # Public key for printing information about where stuff comes from
        self.track = kwargs.get("track", False)

    def _pushfile(self, f):
        if self.dir_file(f).is_file():
            self._parent_fh.append(self.fh)
            self.fh = self.dir_file(f).open(self._mode)
        else:
            warn(str(self) + f' is trying to include file: {f} but the file seems not to exist? Will disregard file!')

    def _popfile(self):
        if len(self._parent_fh) > 0:
            self.fh.close()
            self.fh = self._parent_fh.pop()
            return True
        return False

    def _seek(self):
        """ Closes all files, and starts over from beginning """
        try:
            while self._popfile():
                pass
            self.fh.seek(0)
        except:
            pass

    @sile_fh_open()
    def includes(self):
        """ Return a list of all files that are *included* or otherwise necessary for reading the fdf file """
        self._seek()
        # In FDF files, %include marks files that progress
        # down in a tree structure
        def add(f):
            f = self.dir_file(f)
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

    @sile_fh_open()
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
                    lines = self.dir_file(ls[3]).open('r').readlines()
                    return [l.strip() for l in lines if valid_line(l)]

                # 2. There are labels that should be read from a subsequent file
                #    Label1 Label2 < other.fdf
                if labell in lsl[:idx]:
                    # Valid line, read key from other.fdf
                    return fdfSileSiesta(self.dir_file(ls[idx+1]), base=self._directory)._read_label(label)

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
        l = self.readline().split('#')[0]
        if len(l) == 0:
            return None
        l = process_line(l)
        while l is None:
            l = self.readline().split('#')[0]
            if len(l) == 0:
                if not self._popfile():
                    return None
            l = process_line(l)

        return l

    @classmethod
    def _type(cls, value):
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

            try:
                float(fdf)
                if '.' in fdf:
                    # a real number (otherwise an integer)
                    return 'r'
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

    @sile_fh_open()
    def type(self, label):
        """ Return the type of the fdf-keyword

        Parameters
        ----------
        label : str
            the label to look-up
        """
        self._seek()
        return self._type(self._read_label(label))

    @sile_fh_open()
    def get(self, label, default=None, unit=None, with_unit=False):
        """ Retrieve fdf-keyword from the file

        Parameters
        ----------
        label : str
            the fdf-label to search for
        default : optional
            if the label is not found, this will be the returned value (default to ``None``)
        unit : str, optional
            unit of the physical quantity to return
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
        >>> print(open(...).readlines())
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
        >>> fdf.get('LabelRy', unit='Ry') == 1.
        >>> fdf.get('LabelRy', with_unit=True) == (1., 'Ry')
        >>> fdf.get('FakeInt', '0') == '1'
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
                    raise ValueError(f"Requested unit for {label} is not the same type. "
                                     "Found/Requested {value[1]}/{unit}'")
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
           whether old flags will be kept in the fdf file. In this case
           a time-stamp will be written to show when the key was overwritten.
        """

        # To set a key we first need to figure out if it is
        # already present, if so, we will add the new key, just above
        # the already present key.
        top_file = str(self.file)

        # 1. find the old value, and thus the file in which it is found
        with self:
            try:
                self.get(key)
                # Get the file of the containing data
                top_file = str(self.fh.name)
            except:
                pass

        # Ensure that all files are closed
        self._seek()

        # Now we should re-read and edit the file
        lines = open(top_file, 'r').readlines()

        def write(fh, value):
            if value is None:
                return
            fh.write(self.print(key, value))
            if isinstance(value, str) and '\n' not in value:
                fh.write('\n')

        # Now loop, write and edit
        do_write = True
        lkey = key.lower()
        with open(top_file, 'w') as fh:
            for line in lines:
                if self.line_has_key(line, lkey, case=False) and do_write:
                    write(fh, value)
                    if keep:
                        fh.write('# Old value ({})\n'.format(datetime.today().strftime('%Y-%m-%d %H:%M')))
                        fh.write(f'{line}')
                    do_write = False
                else:
                    fh.write(line)
            if do_write:
                write(fh, value)

    @staticmethod
    def print(key, value):
        """ Return a string which is pretty-printing the key+value """
        if isinstance(value, list):
            s = f'%block {key}'
            # if the value has any new-values
            has_nl = False
            for v in value:
                if '\n' in v:
                    has_nl = True
                    break
            if has_nl:
                # copy list, we are going to change it
                value = value[:]
                # do not skip to next line in next segment
                value[-1] = value[-1].replace('\n', '')
                s += '\n{}\n'.format(''.join(value))
            else:
                s += '\n{}\n'.format('\n'.join(value))
            s += f'%endblock {key}'
        else:
            s = f'{key} {value}'
        return s

    @sile_fh_open()
    def write_supercell(self, sc, fmt='.8f', *args, **kwargs):
        """ Writes the supercell to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        fmt_str = ' {{0:{0}}} {{1:{0}}} {{2:{0}}}\n'.format(fmt)

        unit = kwargs.get('unit', 'Ang').capitalize()
        conv = 1.
        if unit in ['Ang', 'Bohr']:
            conv = unit_convert('Ang', unit)
        else:
            unit = 'Ang'

        # Write out the cell
        self._write(f'LatticeConstant 1.0 {unit}\n')
        self._write('%block LatticeVectors\n')
        self._write(fmt_str.format(*sc.cell[0, :] * conv))
        self._write(fmt_str.format(*sc.cell[1, :] * conv))
        self._write(fmt_str.format(*sc.cell[2, :] * conv))
        self._write('%endblock LatticeVectors\n')

    @sile_fh_open()
    def write_geometry(self, geometry, fmt='.8f', *args, **kwargs):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        self.write_supercell(geometry.sc, fmt, *args, **kwargs)

        self._write('\n')
        self._write(f'NumberOfAtoms {geometry.na}\n')
        unit = kwargs.get('unit', 'Ang').capitalize()
        is_fractional = unit in ['Frac', 'Fractional']
        if is_fractional:
            self._write('AtomicCoordinatesFormat Fractional\n')
        else:
            conv = unit_convert('Ang', unit)
            self._write(f'AtomicCoordinatesFormat {unit}\n')
        self._write('%block AtomicCoordinatesAndAtomicSpecies\n')

        n_species = len(geometry.atoms.atom)

        # Count for the species
        if is_fractional:
            xyz = geometry.fxyz
        else:
            xyz = geometry.xyz * conv
            if fmt[0] == '.':
                # Correct for a "same" length of all coordinates
                c_max = len(str((f'{{:{fmt}}}').format(xyz.max())))
                c_min = len(str((f'{{:{fmt}}}').format(xyz.min())))
                fmt = str(max(c_min, c_max)) + fmt
        fmt_str = ' {{3:{0}}} {{4:{0}}} {{5:{0}}} {{0}} # {{1:{1}d}}: {{2}}\n'.format(fmt, len(str(len(geometry))))

        for ia, a, isp in geometry.iter_species():
            self._write(fmt_str.format(isp + 1, ia + 1, a.tag, *xyz[ia, :]))
        self._write('%endblock AtomicCoordinatesAndAtomicSpecies\n\n')

        # Write out species
        # First swap key and value
        self._write(f'NumberOfSpecies {n_species}\n')
        self._write('%block ChemicalSpeciesLabel\n')
        for i, a in enumerate(geometry.atoms.atom):
            if isinstance(a, AtomGhost):
                self._write(' {} {} {}\n'.format(i + 1, -a.Z, a.tag))
            else:
                self._write(' {} {} {}\n'.format(i + 1, a.Z, a.tag))
        self._write('%endblock ChemicalSpeciesLabel\n')

        _write_block = True
        def write_block(atoms, append, write_block):
            if write_block:
                self._write('\n# Constraints\n%block Geometry.Constraints\n')
                write_block = False
            self._write(f' atom [{atoms}]{append}\n')
            return write_block

        for d in range(4):
            append = {0: '', 1: ' 1. 0. 0.', 2: ' 0. 1. 0.', 3: ' 0. 0. 1.'}.get(d)
            n = 'CONSTRAIN' + {0: '', 1: '-x', 2: '-y', 3: '-z'}.get(d)
            if n in geometry.names:
                idx = list2str(geometry.names[n] + 1).replace('-', ' -- ')
                if len(idx) > 200:
                    info(f"{str(self)}.write_geometry will not write the constraints for {n} (too long line).")
                else:
                    _write_block = write_block(idx, append, _write_block)

        if not _write_block:
            self._write('%endblock\n')

    @staticmethod
    def _SpGeom_replace_geom(spgeom, geometry):
        """ Replace all atoms in spgeom with the atom in geometry while retaining the number of orbitals

        Currently we need some way of figuring out whether the number of atoms and orbitals are
        consistent.

        Parameters
        ----------
        spgeom : SparseGeometry
           the sparse object with attached geometry
        geometry : Geometry
           geometry to grab atoms from
        full_replace : bool, optional
           whether the full geometry may be replaced in case ``spgeom.na != geometry.na && spgeom.no == geometry.no``.
           This is required when `spgeom` does not contain information about atoms.
        """
        if spgeom.na != geometry.na and spgeom.no == geometry.no:
            # In this case we cannot compare individiual atoms # of orbitals.
            # I.e. we suspect the incoming geometry to be correct.
            spgeom._geometry = geometry
            return True

        elif spgeom.na != geometry.na:
            warn('cannot replace geometry due to insufficient information regarding number of '
                 'atoms and orbitals, ensuring correct geometry failed...')

        no_no = spgeom.no == geometry.no
        # Loop and make sure the number of orbitals is consistent
        for a, idx in geometry.atoms.iter(True):
            if len(idx) == 0:
                continue
            Sa = spgeom.geometry.atoms[idx[0]]
            if Sa.no != a.no:
                # Make sure the atom we replace with retains the same information
                # *except* the number of orbitals.
                a = a.__class__(a.Z, Sa.orbital, mass=a.mass, tag=a.tag)
            spgeom.geometry.atoms.replace(idx, a)
            spgeom.geometry.reduce()
        return no_no

    def read_supercell_nsc(self, *args, **kwargs):
        """ Read supercell size using any method available

        Raises
        ------
        SislWarning if none of the files can be read
        """
        order = _listify_str(kwargs.pop('order', ['nc', 'ORB_INDX']))
        for f in order:
            v = getattr(self, '_r_supercell_nsc_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                _track(self.read_supercell_nsc, f"found file {f}")
                return v
        warn('number of supercells could not be read from output files. Assuming molecule cell '
             '(no supercell connections)')
        return _a.onesi(3)

    def _r_supercell_nsc_nc(self, *args, **kwargs):
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.nc')
        _track_file(self._r_supercell_nsc_nc, f)
        if f.is_file():
            return ncSileSiesta(f).read_supercell_nsc()
        return None

    def _r_supercell_nsc_orb_indx(self, *args, **kwargs):
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.ORB_INDX')
        _track_file(self._r_supercell_nsc_orb_indx, f)
        if f.is_file():
            return orbindxSileSiesta(f).read_supercell_nsc()
        return None

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
        >>> fdf = get_sile('RUN.fdf')
        >>> fdf.read_supercell() # read from fdf
        >>> fdf.read_supercell(True) # read from [XV, nc, fdf]
        >>> fdf.read_supercell(order=['nc']) # read from [nc]
        >>> fdf.read_supercell(True, order=['nc']) # read from [nc]
        """
        if output:
            order = _listify_str(kwargs.pop('order', ['XV', 'nc', 'fdf']))
        else:
            order = _listify_str(kwargs.pop('order', ['fdf']))
        for f in order:
            v = getattr(self, '_r_supercell_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                _track(self.read_supercell, f"found file {f}")
                return v
        return None

    def _r_supercell_fdf(self, *args, **kwargs):
        """ Returns `SuperCell` object from the FDF file """
        s = self.get('LatticeConstant', unit='Ang')
        if s is None:
            raise SileError('Could not find LatticeConstant in file')

        # Read in cell
        cell = _a.emptyd([3, 3])

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

        # When reading from the fdf, the warning should be suppressed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nsc = self.read_supercell_nsc()

        return SuperCell(cell, nsc=nsc)

    def _r_supercell_nc(self):
        # Read supercell from <>.nc file
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.nc')
        _track_file(self._r_supercell_nc, f)
        if f.is_file():
            return ncSileSiesta(f).read_supercell()
        return None

    def _r_supercell_xv(self, *args, **kwargs):
        """ Returns `SuperCell` object from the FDF file """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.XV')
        _track_file(self._r_supercell_xv, f)
        if f.is_file():
            nsc = self.read_supercell_nsc()
            sc = xvSileSiesta(f).read_supercell()
            sc.set_nsc(nsc)
            return sc
        return None

    def _r_supercell_tshs(self, *args, **kwargs):
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.TSHS')
        _track_file(self._r_supercell_tshs, f)
        if f.is_file():
            return tshsSileSiesta(f).read_supercell()
        return None

    def _r_supercell_onlys(self, *args, **kwargs):
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.onlyS')
        _track_file(self._r_supercell_onlys, f)
        if f.is_file():
            return onlysSileSiesta(f).read_supercell()
        return None

    def read_force(self, *args, **kwargs):
        """ Read forces from the output of the calculation (forces are not defined in the input)

        Parameters
        ----------
        order : list of str, optional
           the order of the forces we are trying to read, default to ``['FA', 'nc']``

        Returns
        -------
        numpy.ndarray : vector with forces for each of the atoms, along each Cartesian direction
        """
        order = _listify_str(kwargs.pop('order', ['FA', 'nc']))
        for f in order:
            v = getattr(self, '_r_force_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                if self.track:
                    info(f"{self.file}(read_force) found in file={f}")
                return v
        return None

    def _r_force_fa(self, *args, **kwargs):
        """ Read forces from the FA file """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.FA')
        if f.is_file():
            return faSileSiesta(f).read_force()
        return None

    def _r_force_fac(self, *args, **kwargs):
        """ Read forces from the FAC file """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.FAC')
        if f.is_file():
            return faSileSiesta(f).read_force()
        return None

    def _r_force_tsfa(self, *args, **kwargs):
        """ Read forces from the TSFA file """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.TSFA')
        if f.is_file():
            return faSileSiesta(f).read_force()
        return None

    def _r_force_tsfac(self, *args, **kwargs):
        """ Read forces from the TSFAC file """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.TSFAC')
        if f.is_file():
            return faSileSiesta(f).read_force()
        return None

    def _r_force_nc(self, *args, **kwargs):
        """ Read forces from the nc file """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.nc')
        if f.is_file():
            return ncSileSiesta(f).read_force()
        return None

    def read_force_constant(self, *args, **kwargs):
        """ Read force constant from the output of the calculation

        Returns
        -------
        force_constant : numpy.ndarray
            vector ``[*, 3, 2, *, 3]``  with force constant element for each of the atomic displacements
        """
        order = _listify_str(kwargs.pop('order', ['nc', 'FC']))
        for f in order:
            v = getattr(self, '_r_force_constant_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                if self.track:
                    info(f"{self.file}(read_force_constant) found in file={f}")
                return v
        return None

    def _r_force_constant_nc(self, *args, **kwargs):
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.nc')
        if f.is_file():
            if not 'FC' in ncSileSiesta(f).groups:
                return None
            fc = ncSileSiesta(f).read_force_constant()
            return fc
        return None

    def _r_force_constant_fc(self, *args, **kwargs):
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.FC')
        if f.is_file():
            na = self.get('NumberOfAtoms', default=None)
            return fcSileSiesta(f).read_force_constant(na=na)
        return None

    def read_fermi_level(self, *args, **kwargs):
        """ Read fermi-level from output of the calculation

        Parameters
        ----------
        order: list of str, optional
            the order of which to try and read the fermi-level.
            By default this is ``['nc', 'TSDE', 'TSHS', 'EIG']``.

        Returns
        -------
        Ef : float
            fermi-level
        """
        order = _listify_str(kwargs.pop('order', ['nc', 'TSDE', 'TSHS', 'EIG']))
        for f in order:
            v = getattr(self, '_r_fermi_level_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                if self.track:
                    info(f"{self.file}(read_fermi_level) found in file={f}")
                return v
        return None

    def _r_fermi_level_nc(self):
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.nc')
        if isfile(f):
            return ncSileSiesta(f).read_fermi_level()
        return None

    def _r_fermi_level_tsde(self):
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.TSDE')
        if isfile(f):
            return tsdeSileSiesta(f).read_fermi_level()
        return None

    def _r_fermi_level_tshs(self):
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.TSHS')
        if isfile(f):
            return tshsSileSiesta(f).read_fermi_level()
        return None

    def _r_fermi_level_eig(self):
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.EIG')
        if isfile(f):
            return eigSileSiesta(f).read_fermi_level()
        return None

    def read_dynamical_matrix(self, *args, **kwargs):
        """ Read dynamical matrix from output of the calculation

        Generally the mass is stored in the basis information output,
        but for dynamical matrices it makes sense to let the user control this,
        e.g. through the fdf file.
        By default the mass will be read from the AtomicMass key in the fdf file
        and _not_ from the basis set information.

        Parameters
        ----------
        order: list of str, optional
            the order of which to try and read the dynamical matrix.
            By default this is ``['nc', 'FC']``.
        cutoff_dist : float, optional
            cutoff value for the distance of the force-constants (everything farther than
            `cutoff_dist` will be set to 0 Ang). Default, no cutoff.
        cutoff : float, optional
            absolute values below the cutoff are considered 0. Defaults to 0. eV/Ang**2.
        trans_inv : bool, optional
            if true (default), the force-constant matrix will be fixed so that translational
            invariance will be enforced
        sum0 : bool, optional
            if true (default), the sum of forces on atoms for each displacement will be
            forced to 0.
        hermitian: bool, optional
            if true (default), the returned dynamical matrix will be hermitian

        Returns
        -------
        dynamic_matrix : DynamicalMatrix
            the dynamical matrix
        """
        order = _listify_str(kwargs.pop('order', ['nc', 'FC']))
        for f in order:
            v = getattr(self, '_r_dynamical_matrix_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                if self.track:
                    info(f"{self.file}(read_dynamical_matrix) found in file={f}")
                return v
        return None

    def _r_dynamical_matrix_fc(self, *args, **kwargs):
        FC = self.read_force_constant(*args, order="FC", **kwargs)
        if FC is None:
            return None
        geom = self.read_geometry()

        basis_fdf = self.read_basis(order="fdf")
        for i, atom in enumerate(basis_fdf):
            geom.atoms.replace(i, atom)

        # Get list of FC atoms
        FC_atoms = _a.arangei(self.get('MD.FCFirst', default=1) - 1, self.get('MD.FCLast', default=geom.na))
        return self._dynamical_matrix_from_fc(geom, FC, FC_atoms, *args, **kwargs)

    def _r_dynamical_matrix_nc(self, *args, **kwargs):
        FC = self.read_force_constant(*args, order=['nc'], **kwargs)
        if FC is None:
            return None
        geom = self.read_geometry(order=['nc'])

        basis_fdf = self.read_basis(order="fdf")
        for i, atom in enumerate(basis_fdf):
            geom.atoms.replace(i, atom)

        # Get list of FC atoms
        # TODO change to read in from the NetCDF file
        FC_atoms = _a.arangei(self.get('MD.FCFirst', default=1) - 1, self.get('MD.FCLast', default=geom.na))
        return self._dynamical_matrix_from_fc(geom, FC, FC_atoms, *args, **kwargs)

    def _dynamical_matrix_from_fc(self, geom, FC, FC_atoms, *args, **kwargs):
        # We have the force constant matrix.
        # Now handle it...
        #  FC(OLD) = (n_displ, 3, 2, na, 3)
        #  FC(NEW) = (n_displ, 3, na, 3)
        # In fact, after averaging this becomes the Hessian
        FC = FC.sum(axis=2) * 0.5
        na_full = FC.shape[2]
        hermitian = kwargs.get("hermitian", True)

        # Figure out the "original" periodic directions
        periodic = geom.nsc > 1

        # Create conversion from eV/Ang^2 to correct units
        # Further down we are multiplying with [1 / amu]
        scale = constant.hbar / units('Ang', 'm') / units('eV amu', 'J kg') ** 0.5

        # Cut-off too small values
        fc_cut = kwargs.get('cutoff', 0.)
        FC = np.where(np.fabs(FC) > fc_cut, FC, 0.)

        # Convert the force constant such that a diagonalization returns eV ^ 2
        # FC is in [eV / Ang^2]

        # Convert the geometry to contain 3 orbitals per atom (x, y, z)
        R = kwargs.get('cutoff_dist', -2.)
        orbs = [Orbital(R / 2, tag=tag) for tag in 'xyz']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for atom, _ in geom.atoms.iter(True):
                new_atom = atom.__class__(atom.Z, orbs, mass=atom.mass, tag=atom.tag)
                geom.atoms.replace(atom, new_atom)

        # Figure out the supercell indices
        # if the displaced atoms equals the length of the geometry
        # it means we are not using a supercell.
        supercell = kwargs.get('supercell', len(geom) != len(FC_atoms))
        if supercell is False:
            supercell = [1] * 3
        elif supercell is True:
            _, supercell = geom.as_primary(FC.shape[0], ret_super=True)
            info("{}.read_dynamical_matrix(FC) guessed on a [{}, {}, {}] "
                 "supercell calculation.".format(str(self), *supercell))
        # Convert to integer array
        supercell = _a.asarrayi(supercell)

        # Reshape to supercell
        FC.shape = (FC.shape[0], 3, *supercell, -1, 3)
        na_fc = len(FC_atoms)
        assert FC.shape[0] == len(FC_atoms)
        assert FC.shape[5] == len(geom) // np.prod(supercell)

        # Now we are in a problem since the tiling of the geometry
        # is not necessarily in x, y, z order.
        # Say for users who did:
        #   geom.tile(*, 2).tile(*, 1).tile(*, 0).write(...)
        # then we need to pivot the data to be consistent with the
        # supercell information
        if np.any(supercell > 1):
            # Re-arange FC before we use _fc_correct
            # Now we need to figure out how the atoms are laid out.
            # It *MUST* either be repeated or tiled (preferentially tiled).

            # We have an actual supercell. Lets try and fix it.
            # First lets recreate the smallest geometry
            sc = geom.sc.cell.copy()
            sc[0, :] /= supercell[0]
            sc[1, :] /= supercell[1]
            sc[2, :] /= supercell[2]

            # Ensure nsc is at least an odd number, later down we will symmetrize the FC matrix
            nsc = supercell + (supercell + 1) % 2
            if R > 0:
                # Correct for the optional radius
                sc_norm = fnorm(sc)
                # R is already "twice" the "orbital" range
                nsc_R = 1 + 2 * np.ceil(R / sc_norm).astype(np.int32)
                for i in range(3):
                    nsc[i] = min(nsc[i], nsc_R[i])
                del nsc_R

            # Construct the minimal unit-cell geometry
            sc = SuperCell(sc, nsc=nsc)
            # TODO check that the coordinates are in the cell
            geom_small = Geometry(geom.xyz[FC_atoms], geom.atoms[FC_atoms], sc)

            # Convert the big geometry's coordinates to fractional coordinates of the small unit-cell.
            isc_xyz = (geom.xyz.dot(geom_small.sc.icell.T) -
                       np.tile(geom_small.fxyz, (np.product(supercell), 1)))

            axis_tiling = []
            offset = len(geom_small)
            for _ in (supercell > 1).nonzero()[0]:
                first_isc = (np.around(isc_xyz[FC_atoms + offset, :]) == 1.).sum(0)
                axis_tiling.append(np.argmax(first_isc))
                # Fix the offset and wrap-around
                offset = (offset * supercell[axis_tiling[-1]]) % na_full

            for i in range(3):
                if not i in axis_tiling:
                    axis_tiling.append(i)

            # Now we have the tiling operation, check it sort of matches
            geom_tile = geom_small.copy()
            for axis in axis_tiling:
                geom_tile = geom_tile.tile(supercell[axis], axis)

            # Proximity check of 0.01 Ang (TODO add this as an argument)
            for ax in range(3):
                daxis = geom_tile.xyz[:, ax] - geom.xyz[:, ax]
                if not np.allclose(daxis, daxis[0], rtol=0., atol=0.01):
                    raise SislError(f"{str(self)}.read_dynamical_matrix(FC) could "
                                    "not figure out the tiling method for the supercell")

            # Convert the FC matrix to a "rollable" matrix
            # This will make it easier to symmetrize
            #  0. displaced atoms
            #  1. x, y, z (displacements)
            #  2. tile-axis_tiling[0]
            #  3. tile-axis_tiling[1]
            #  4. tile-axis_tiling[2]
            #  5. na
            #  6. x, y, z (force components)
            # order of FC is reversed of the axis_tiling (because of contiguous arrays)
            # so reverse
            axis_tiling.reverse()
            FC.shape = (na_fc, 3, *supercell[axis_tiling], -1, 3)

            # now ensure we have the correct order of the supercell
            # If the input supercell is
            # [-2] [-1] [0] [1] [2]
            # we need to convert it to
            #  [0] [1] [2] [3] [4] [5]
            isc_xyz.shape = (*supercell[axis_tiling], na_fc, 3)
            for axis in axis_tiling:
                nroll = isc_xyz[..., axis].min()
                inroll = int(round(nroll))
                if inroll != 0:
                    # offset axis by 2 due to (na_fc, 3, ...)
                    FC = np.roll(FC, inroll, axis=axis + 2)
            FC_atoms -= FC_atoms.min()

            # Now swap the [2, 3, 4] dimensions so that we get in order of lattice vectors
            #  x, y, z
            FC = np.transpose(FC, (0, 1, *(axis_tiling.index(i)+2 for i in range(3)), 5, 6))
            del axis_tiling
            # Now FC is sorted according to the supercell tiling

        # TODO this will probably fail if: FC_atoms.size != FC.shape[5]
        from ._help import _fc_correct
        FC = _fc_correct(FC, trans_inv=kwargs.get("trans_inv", True),
                         sum0=kwargs.get("sum0", True),
                         hermitian=hermitian)

        # Remove ghost-atoms or atoms with 0 mass!
        # TODO check if ghost-atoms should be taken into account in _fc_correct
        idx = (geom.atoms.mass == 0.).nonzero()[0]
        if len(idx) > 0:
            FC = np.delete(FC, idx, axis=5)
            geom = geom.remove(idx)
            geom.set_nsc([1] * 3)
            raise NotImplementedError(f"{self}.read_dynamical_matrix could not reduce geometry "
                                      "since there are atoms with 0 mass.")

        # Now we can build the dynamical matrix (it will always be real)
        na = len(geom)

        if np.all(supercell <= 1):
            # also catches supercell == 0
            D = sp.sparse.lil_matrix((geom.no, geom.no), dtype=np.float64)

            FC = np.squeeze(FC, axis=(2, 3, 4))
            # Instead of doing the sqrt in all D = FC (below) we do it here
            m = scale / geom.atoms.mass ** 0.5
            FC *= m[FC_atoms].reshape(-1, 1, 1, 1) * m.reshape(1, 1, -1, 1)

            j_FC_atoms = FC_atoms
            idx = _a.arangei(len(FC_atoms))
            for ia, fia in enumerate(FC_atoms):

                if R > 0:
                    # find distances between the other atoms to cut-off the distance
                    idx = geom.close(fia, R=R, atoms=FC_atoms)
                    idx = indices_only(FC_atoms, idx)
                    j_FC_atoms = FC_atoms[idx]

                for ja, fja in zip(idx, j_FC_atoms):
                    D[ia*3:(ia+1)*3, ja*3:(ja+1)*3] = FC[ia, :, fja, :]

        else:
            geom = geom_small

            if np.any(np.diff(FC_atoms) != 1):
                raise SislError(f"{self}.read_dynamical_matrix(FC) requires the FC atoms to be consecutive!")

            # Re-order FC matrix so the FC-atoms are first
            if FC.shape[0] != FC.shape[5]:
                ordered = _a.arangei(FC.shape[5])
                ordered = np.concatenate(FC_atoms, np.delete(ordered, FC_atoms))
                FC = FC[:, :, :, :, :, ordered, :]
                FC_atoms = _a.arangei(len(FC_atoms))

            if FC_atoms[0] != 0:
                # TODO we could roll the axis such that the displaced atoms moves into the
                # first elements
                raise SislError(f"{self}.read_dynamical_matrix(FC) requires the displaced atoms to start from 1!")

            # After having done this we can easily mass scale all FC components
            m = scale / geom.atoms.mass ** 0.5
            FC *= m.reshape(-1, 1, 1, 1, 1, 1, 1) * m.reshape(1, 1, 1, 1, 1, -1, 1)

            # Check whether we need to "halve" the equivalent supercell
            # This will be present in calculations done on an even number of supercells.
            # I.e. for 4 supercells
            #  [0] [1] [2] [3]
            # where in the supercell approach:
            #  *[2] [3] [0] [1] *[2]
            # I.e. since we are double counting [2] we will halve it.
            # This is not *exactly* true because depending on the range one should do the symmetry operations.
            # However the FC does not contain such symmetry considerations.
            for i in range(3):
                if supercell[i] % 2 == 1:
                    # We don't need to do anything
                    continue

                # Figure out the supercell to halve
                halve_idx = supercell[i] // 2
                if i == 0:
                    FC[:, :, halve_idx, :, :, :, :] *= 0.5
                elif i == 1:
                    FC[:, :, :, halve_idx, :, :, :] *= 0.5
                else:
                    FC[:, :, :, :, halve_idx, :, :] *= 0.5

            # Now create the dynamical matrix
            # Currently this will be in lil_matrix (changed in the end)
            D = sp.sparse.lil_matrix((geom.no, geom.no_s), dtype=np.float64)

            # When x, y, z are negative we simply look-up from the back of the array
            # which is exactly what is required
            isc_off = geom.sc.isc_off
            nxyz, na = geom.no, geom.na
            dist = geom.rij

            # Now take all positive supercell connections (including inner cell)
            nsc = geom.nsc // 2
            list_nsc = [range(-x, x + 1) for x in nsc]

            iter_FC_atoms = _a.arangei(len(FC_atoms))
            iter_j_FC_atoms = iter_FC_atoms
            for x, y, z in itools.product(*list_nsc):
                isc = isc_off[x, y, z]
                aoff = isc * na
                joff = isc * nxyz
                for ia in iter_FC_atoms:
                    # Reduce second loop based on radius cutoff
                    if R > 0:
                        iter_j_FC_atoms = iter_FC_atoms[dist(ia, aoff + iter_FC_atoms) <= R]

                    for ja in iter_j_FC_atoms:
                        D[ia*3:(ia+1)*3, joff+ja*3:joff+(ja+1)*3] += FC[ia, :, x, y, z, ja, :]

        D = D.tocsr()
        # Remove all zeros
        D.eliminate_zeros()
        D = DynamicalMatrix.fromsp(geom, D)
        if hermitian:
            D.finalize()
            D = (D + D.transpose()) * 0.5

        return D

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
            By default this is ``['XV', 'nc', 'fdf', 'TSHS']`` if `output` is true
            If `order` is present `output` is disregarded.

        Examples
        --------
        >>> fdf = get_sile('RUN.fdf')
        >>> fdf.read_geometry() # read from fdf
        >>> fdf.read_geometry(True) # read from [XV, nc, fdf]
        >>> fdf.read_geometry(order=['nc']) # read from [nc]
        >>> fdf.read_geometry(True, order=['nc']) # read from [nc]
        """
        ##
        # NOTE
        # When adding more capabilities please check the read_geometry(True, order=...) in this
        # code to correct.
        ##
        if output:
            order = _listify_str(kwargs.pop('order', ['XV', 'nc', 'fdf', 'TSHS']))
        else:
            order = _listify_str(kwargs.pop('order', ['fdf']))
        for f in order:
            v = getattr(self, '_r_geometry_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                if self.track:
                    info(f"{self.file}(read_geometry) found in file={f}")
                return v
        return None

    def _r_geometry_xv(self, *args, **kwargs):
        """ Returns `SuperCell` object from the FDF file """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.XV')
        geom = None
        if f.is_file():
            basis = self.read_basis()
            if basis is None:
                geom = xvSileSiesta(f).read_geometry()
            else:
                geom = xvSileSiesta(f).read_geometry(species_Z=True)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    for atom, _ in geom.atoms.iter(True):
                        geom.atoms.replace(atom, basis[atom.Z-1])
                    geom.reduce()
            nsc = self.read_supercell_nsc()
            geom.set_nsc(nsc)
        return geom

    def _r_geometry_nc(self):
        # Read geometry from <>.nc file
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.nc')
        if f.is_file():
            return ncSileSiesta(f).read_geometry()
        return None

    def _r_geometry_tshs(self):
        # Read geometry from <>.TSHS file
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.TSHS')
        if f.is_file():
            # Default to a geometry with the correct atomic numbers etc.
            return tshsSileSiesta(f).read_geometry(geometry=self.read_geometry(False))
        return None

    def _r_geometry_fdf(self, *args, **kwargs):
        """ Returns Geometry object from the FDF file

        NOTE: Interaction range of the Atoms are currently not read.
        """
        sc = self.read_supercell(order='fdf')

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
            s = self.get('LatticeConstant', unit='Ang')
        elif 'fractional' in lc or 'scaledbylatticevectors' in lc:
            # no scaling of coordinates as that is entirely
            # done by the latticevectors
            s = 1.
            is_frac = True

        # If the user requests a shifted geometry
        # we correct for this
        origo = _a.zerosd([3])
        lor = self.get('AtomicCoordinatesOrigin')
        if lor:
            if kwargs.get('origin', True):
                if isinstance(lor, str):
                    origo = lor.lower()
                else:
                    origo = _a.asarrayd(list(map(float, lor[0].split()[:3]))) * s
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
        xyz = _a.emptyd([na, 3])
        species = _a.emptyi([na])
        for ia in range(na):
            l = atms[ia].split()
            xyz[ia, :] = [float(k) for k in l[:3]]
            species[ia] = int(l[3]) - 1
        if is_frac:
            xyz = np.dot(xyz, sc.cell)
        xyz *= s

        # Read the block (not strictly needed, if so we simply set all atoms to H)
        atoms = self.read_basis()
        if atoms is None:
            warn('Block ChemicalSpeciesLabel does not exist, cannot determine the basis (all Hydrogen).')

            # Default atom (hydrogen)
            atoms = Atom(1)
        else:
            atoms = [atoms[i] for i in species]
        atoms = Atoms(atoms, na=len(xyz))

        if isinstance(origo, str):
            opt = origo
            if opt.startswith('cop'):
                origo = sc.cell.sum(0) * 0.5 - np.average(xyz, 0)
            elif opt.startswith('com'):
                # TODO for ghost atoms its mass should not be used
                w = atom.mass
                w /= w.sum()
                origo = sc.cell.sum(0) * 0.5 - np.average(xyz, 0, weights=w)
            elif opt.startswith('min'):
                origo = - np.amin(xyz, 0)
            if len(opt) > 4:
                opt = opt[4:]
                if opt == 'x':
                    origo[1:] = 0.
                elif opt == 'y':
                    origo[[0, 2]] = 0.
                elif opt == 'z':
                    origo[:2] = 0.
                elif opt == 'xy' or opt == 'yx':
                    origo[2] = 0.
                elif opt == 'xz' or opt == 'zx':
                    origo[1] = 0.
                elif opt == 'yz' or opt == 'zy':
                    origo[0] = 0.

        xyz += origo

        # Create and return geometry object
        return Geometry(xyz, atoms, sc=sc)

    def read_grid(self, name, *args, **kwargs):
        """ Read grid related information from any of the output files

        The order of the readed data is shown below.

        One can limit the tried files to only one file by passing
        only a single file ending.

        Parameters
        ----------
        name : str
            name of data to read. The list of names correspond to the
            Siesta output manual (Rho, TotalPotential, etc.), the strings are
            case insensitive.
        order: list of str, optional
            the order of which to try and read the geometry.
            By default this is ``['nc', 'grid.nc', 'bin']`` (bin refers to the binary files)
        """
        order = _listify_str(kwargs.pop('order', ['nc', 'grid.nc', 'bin']))
        for f in order:
            v = getattr(self, '_r_grid_{}'.format(f.lower().replace('.', '_')))(name, *args, **kwargs)
            if v is not None:
                if self.track:
                    info(f"{self.file}(read_grid) found in file={f}")
                return v
        return None

    def _r_grid_nc(self, name, *args, **kwargs):
        # Read grid from the <>.nc file
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.nc')
        if f.is_file():
            # Capitalize correctly
            name = {'rho': 'Rho',
                    'rhoinit': 'RhoInit',
                    'vna': 'Vna',
                    'ioch': 'Chlocal',
                    'chlocal': 'Chlocal',
                    'toch': 'RhoTot',
                    'totalcharge': 'RhoTot',
                    'rhotot': 'RhoTot',
                    'drho': 'RhoDelta',
                    'deltarho': 'RhoDelta',
                    'rhodelta': 'RhoDelta',
                    'vh': 'Vh',
                    'electrostaticpotential': 'Vh',
                    'rhoxc': 'RhoXC',
                    'vt': 'Vt',
                    'totalpotential': 'Vt',
                    'bader': 'RhoBader',
                    'baderrho': 'RhoBader',
                    'rhobader': 'RhoBader'
            }.get(name.lower())
            return ncSileSiesta(f).read_grid(name, **kwargs)
        return None

    def _r_grid_grid_nc(self, name, *args, **kwargs):
        # Read grid from the <>.nc file
        name = {'rho': 'Rho',
                'rhoinit': 'RhoInit',
                'vna': 'Vna',
                'ioch': 'Chlocal',
                'chlocal': 'Chlocal',
                'toch': 'TotalCharge',
                'totalcharge': 'TotalCharge',
                'rhotot': 'TotalCharge',
                'drho': 'DeltaRho',
                'deltarho': 'DeltaRho',
                'rhodelta': 'DeltaRho',
                'vh': 'ElectrostaticPotential',
                'electrostaticpotential': 'ElectrostaticPotential',
                'rhoxc': 'RhoXC',
                'vt': 'TotalPotential',
                'totalpotential': 'TotalPotential',
                'bader': 'BaderCharge',
                'baderrho': 'BaderCharge',
                'rhobader': 'BaderCharge'
        }.get(name.lower()) + '.grid.nc'

        f = self.dir_file(name)
        if f.is_file():
            grid = gridncSileSiesta(f).read_grid(*args, **kwargs)
            grid.set_geometry(self.read_geometry(True))
            return grid
        return None

    def _r_grid_bin(self, name, *args, **kwargs):
        # Read grid from the <>.VT/... file
        name = {'rho': '.RHO',
                'rhoinit': '.RHOINIT',
                'vna': '.VNA',
                'ioch': '.IOCH',
                'chlocal': '.IOCH',
                'toch': '.TOCH',
                'totalcharge': '.TOCH',
                'rhotot': '.TOCH',
                'drho': '.DRHO',
                'deltarho': '.DRHO',
                'rhodelta': '.DRHO',
                'vh': '.VH',
                'electrostaticpotential': '.VH',
                'rhoxc': '.RHOXC',
                'vt': '.VT',
                'totalpotential': '.VT',
                'bader': '.BADER',
                'baderrho': '.BADER',
                'rhobader': '.BADER'
        }.get(name.lower())

        f = self.dir_file(self.get('SystemLabel', default='siesta') + name)
        if f.is_file():
            grid = get_sile_class(f)(f).read_grid(*args, **kwargs)
            grid.set_geometry(self.read_geometry(True))
            return grid
        return None

    def read_basis(self, *args, **kwargs):
        """ Read the atomic species and figure out the number of atomic orbitals in their basis

        The order of the read is shown below.

        One can limit the tried files to only one file by passing
        only a single file ending.

        Parameters
        ----------
        order: list of str, optional
            the order of which to try and read the basis information.
            By default this is ``['nc', 'ion', 'ORB_INDX', 'fdf']``
        """
        order = _listify_str(kwargs.pop('order', ['nc', 'ion', 'ORB_INDX', 'fdf']))
        for f in order:
            v = getattr(self, '_r_basis_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                if self.track:
                    info(f"{self.file}(read_basis) found in file={f}")
                return v
        return None

    def _r_basis_nc(self):
        # Read basis from <>.nc file
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.nc')
        if f.is_file():
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
        atoms = [None] * len(spcs)
        found_one = False
        found_all = True
        for spc in spcs:
            idx, Z, lbl = spc.split()[:3]
            idx = int(idx) - 1 # F-indexing
            Z = int(Z)
            lbl = lbl.strip()
            f = self.dir_file(lbl + ".ext")

            # now try and read the basis
            if f.with_suffix('.ion.nc').is_file():
                atoms[idx] = ionncSileSiesta(f.with_suffix('.ion.nc')).read_basis()
                found_one = True
            elif f.with_suffix('.ion.xml').is_file():
                atoms[idx] = ionxmlSileSiesta(f.with_suffix('.ion.xml')).read_basis()
                found_one = True
            else:
                # default the atom to not have a range, and no associated orbitals
                atoms[idx] = Atom(Z=Z, tag=lbl)
                found_all = False

        if found_one and not found_all:
            warn("Siesta basis information could not read all ion.nc/ion.xml files. "
                 "Only a subset of the basis information is accessible.")
        elif not found_one:
            return None
        return atoms

    def _r_basis_orb_indx(self):
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.ORB_INDX')
        if f.is_file():
            info(f"Siesta basis information is read from {f}, the radial functions are not accessible.")
            return orbindxSileSiesta(f).read_basis(atoms=self._r_basis_fdf())
        return None

    def _r_basis_fdf(self):
        # Read basis from fdf file
        spcs = self.get('ChemicalSpeciesLabel')
        if spcs is None:
            # We haven't found the chemical and species label
            # so return nothing
            return None

        all_mass = self.get('AtomicMass', default=[])
        # default mass
        mass = None

        # Now spcs contains the block of the chemicalspecieslabel
        atoms = [None] * len(spcs)
        for spc in spcs:
            idx, Z, lbl = spc.split()[:3]
            idx = int(idx) - 1 # F-indexing
            Z = int(Z)
            lbl = lbl.strip()

            if len(all_mass) > 0:
                for mass_line in all_mass:
                    s, mass = mass_line.split()
                    if int(s) - 1 == idx:
                        mass = float(mass)
                        break
                else:
                    mass = None

            atoms[idx] = Atom(Z=Z, mass=mass, tag=lbl)
        return atoms

    def _r_add_overlap(self, parent_call, M):
        """ Internal routine to ensure that the overlap matrix is read and added to the matrix `M` """
        try:
            S = self.read_overlap()
            # Check for the same sparsity pattern
            if np.all(M._csr.col == S._csr.col):
                M._csr._D[:, -1] = S._csr._D[:, 0]
            else:
                raise ValueError
        except:
            warn(str(self) + f' could not succesfully read the overlap matrix in {parent_call}.')

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
        order = _listify_str(kwargs.pop('order', ['nc', 'TSDE', 'DM']))
        for f in order:
            DM = getattr(self, '_r_density_matrix_{}'.format(f.lower()))(*args, **kwargs)
            if DM is not None:
                _track(self.read_density_matrix, f"found file {f}")
                return DM
        return None

    def _r_density_matrix_nc(self, *args, **kwargs):
        """ Try and read the density matrix by reading the <>.nc """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.nc')
        _track_file(self._r_density_matrix_nc, f)
        DM = None
        if f.is_file():
            # this *should* also contain the overlap matrix
            DM = ncSileSiesta(f).read_density_matrix(*args, **kwargs)
        return DM

    def _r_density_matrix_tsde(self, *args, **kwargs):
        """ Read density matrix from the TSDE file """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.TSDE')
        _track_file(self._r_density_matrix_tsde, f)
        DM = None
        if f.is_file():
            if 'geometry' not in kwargs:
                # to ensure we get the correct orbital count
                kwargs['geometry'] = self.read_geometry(True, order=['nc', 'TSHS', 'fdf'])
            DM = tsdeSileSiesta(f).read_density_matrix(*args, **kwargs)
            self._r_add_overlap('_r_density_matrix_tsde', DM)
        return DM

    def _r_density_matrix_dm(self, *args, **kwargs):
        """ Read density matrix from the DM file """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.DM')
        _track_file(self._r_density_matrix_dm, f)
        DM = None
        if f.is_file():
            if 'geometry' not in kwargs:
                # to ensure we get the correct orbital count
                kwargs['geometry'] = self.read_geometry(True, order=['nc', 'TSHS', 'fdf'])
            DM = dmSileSiesta(f).read_density_matrix(*args, **kwargs)
            self._r_add_overlap('_r_density_matrix_dm', DM)
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
        order = _listify_str(kwargs.pop('order', ['nc', 'TSDE']))
        for f in order:
            EDM = getattr(self, '_r_energy_density_matrix_{}'.format(f.lower()))(*args, **kwargs)
            if EDM is not None:
                _track(self.read_energy_density_matrix, f"found file {f}")
                return EDM
        return None

    def _r_energy_density_matrix_nc(self, *args, **kwargs):
        """ Read energy density matrix by reading the <>.nc """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.nc')
        _track_file(self._r_energy_density_matrix_nc, f)
        if f.is_file():
            return ncSileSiesta(f).read_energy_density_matrix(*args, **kwargs)
        return None

    def _r_energy_density_matrix_tsde(self, *args, **kwargs):
        """ Read energy density matrix from the TSDE file """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.TSDE')
        _track_file(self._r_energy_density_matrix_tsde, f)
        EDM = None
        if f.is_file():
            if 'geometry' not in kwargs:
                # to ensure we get the correct orbital count
                kwargs['geometry'] = self.read_geometry(True, order=['nc', 'TSHS'])
            EDM = tsdeSileSiesta(f).read_energy_density_matrix(*args, **kwargs)
            self._r_add_overlap('_r_energy_density_matrix_tsde', EDM)
        return EDM

    def read_overlap(self, *args, **kwargs):
        """ Try and read the overlap matrix by reading the <>.nc, <>.TSHS files, <>.HSX, <>.onlyS (in that order)

        One can limit the tried files to only one file by passing
        only a single file ending.

        Parameters
        ----------
        order: list of str, optional
            the order of which to try and read the overlap matrix
            By default this is ``['nc', 'TSHS', 'HSX', 'onlyS']``.
        """
        order = _listify_str(kwargs.pop('order', ['nc', 'TSHS', 'HSX', 'onlyS']))
        for f in order:
            v = getattr(self, '_r_overlap_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                _track(self.read_overlap, f"found file {f}")
                return v
        return None

    def _r_overlap_nc(self, *args, **kwargs):
        """ Read overlap from the nc file """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.nc')
        _track_file(self._r_overlap_nc, f)
        if f.is_file():
            return ncSileSiesta(f).read_overlap(*args, **kwargs)
        return None

    def _r_overlap_tshs(self, *args, **kwargs):
        """ Read overlap from the TSHS file """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.TSHS')
        _track_file(self._r_overlap_tshs, f)
        S = None
        if f.is_file():
            if 'geometry' not in kwargs:
                # to ensure we get the correct orbital count
                kwargs['geometry'] = self.read_geometry(True, order=['nc', 'TSHS'])
            S = tshsSileSiesta(f).read_overlap(*args, **kwargs)
        return S

    def _r_overlap_hsx(self, *args, **kwargs):
        """ Read overlap from the HSX file """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.HSX')
        _track_file(self._r_overlap_hsx, f)
        S = None
        if f.is_file():
            if 'geometry' not in kwargs:
                # to ensure we get the correct orbital count
                kwargs['geometry'] = self.read_geometry(True, order=['nc', 'TSHS', 'fdf'])
            S = hsxSileSiesta(f).read_overlap(*args, **kwargs)
        return S

    def _r_overlap_onlys(self, *args, **kwargs):
        """ Read overlap from the onlyS file """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.onlyS')
        _track_file(self._r_overlap_onlys, f)
        S = None
        if f.is_file():
            if 'geometry' not in kwargs:
                # to ensure we get the correct orbital count
                kwargs['geometry'] = self.read_geometry(True, order=['nc', 'TSHS', 'fdf'])
            S = onlysSileSiesta(f).read_overlap(*args, **kwargs)
        return S

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
        order = _listify_str(kwargs.pop('order', ['nc', 'TSHS', 'HSX']))
        for f in order:
            H = getattr(self, '_r_hamiltonian_{}'.format(f.lower()))(*args, **kwargs)
            if H is not None:
                _track(self.read_hamiltonian, f"found file {f}")
                return H
        return None

    def _r_hamiltonian_nc(self, *args, **kwargs):
        """ Read Hamiltonian from the nc file """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.nc')
        _track_file(self._r_hamiltonian_nc, f)
        if f.is_file():
            return ncSileSiesta(f).read_hamiltonian(*args, **kwargs)
        return None

    def _r_hamiltonian_tshs(self, *args, **kwargs):
        """ Read Hamiltonian from the TSHS file """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.TSHS')
        _track_file(self._r_hamiltonian_tshs, f)
        H = None
        if f.is_file():
            if 'geometry' not in kwargs:
                # to ensure we get the correct orbital count
                kwargs['geometry'] = self.read_geometry(True, order=['nc', 'TSHS'])
            H = tshsSileSiesta(f).read_hamiltonian(*args, **kwargs)
        return H

    def _r_hamiltonian_hsx(self, *args, **kwargs):
        """ Read Hamiltonian from the HSX file """
        f = self.dir_file(self.get('SystemLabel', default='siesta') + '.HSX')
        _track_file(self._r_hamiltonian_hsx, f)
        H = None
        if f.is_file():
            if 'geometry' not in kwargs:
                # to ensure we get the correct orbital count
                kwargs['geometry'] = self.read_geometry(True, order=['nc', 'TSHS', 'fdf'])
            H = hsxSileSiesta(f).read_hamiltonian(*args, **kwargs)
            Ef = self.read_fermi_level()
            if Ef is None:
                info(f"{str(self)}.read_hamiltonian from HSX file failed shifting to the Fermi-level.")
            else:
                H.shift(-Ef)
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
        label = self.get('SystemLabel', default='siesta')
        f_label = label + ".ext"

        def label_file(suffix):
            return self.dir_file(f_label).with_suffix(suffix)

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
                    print(f'# {value[0]} is currently not in the FDF file ')
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
        f = label_file('.XV')
        try:
            geom = self.read_geometry(True)

            tmp_p = sp.add_parser('geom',
                                  help="Edit the contained geometry in the file")
            tmp_p, tmp_ns = geom.ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)
        except:
            # Allowed pass due to pythonic reading
            pass

        f = label_file('.bands')
        if f.is_file():
            tmp_p = sp.add_parser('band',
                                  help="Manipulate bands file from the Siesta simulation")
            tmp_p, tmp_ns = sis.bandsSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label_file('.PDOS.xml')
        if f.is_file():
            tmp_p = sp.add_parser('pdos',
                                  help="Manipulate PDOS.xml file from the Siesta simulation")
            tmp_p, tmp_ns = sis.pdosSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label_file('.EIG')
        if f.is_file():
            tmp_p = sp.add_parser('eig',
                                  help="Manipulate EIG file from the Siesta simulation")
            tmp_p, tmp_ns = sis.eigSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        #f = label + '.FA'
        #if isfile(f):
        #    tmp_p = sp.add_parser('force',
        #                          help="Manipulate FA file from the Siesta simulation")
        #    tmp_p, tmp_ns = sis.faSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
        #    namespace = merge_instances(namespace, tmp_ns)

        f = label_file('.TBT.nc')
        if f.is_file():
            tmp_p = sp.add_parser('tbt',
                                  help="Manipulate tbtrans output file")
            tmp_p, tmp_ns = sis.tbtncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label_file('.TBT.Proj.nc')
        if f.is_file():
            tmp_p = sp.add_parser('tbt-proj',
                                  help="Manipulate tbtrans projection output file")
            tmp_p, tmp_ns = sis.tbtprojncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label_file('.PHT.nc')
        if f.is_file():
            tmp_p = sp.add_parser('pht',
                                  help="Manipulate the phtrans output file")
            tmp_p, tmp_ns = sis.phtncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label_file('.PHT.Proj.nc')
        if f.is_file():
            tmp_p = sp.add_parser('pht-proj',
                                  help="Manipulate phtrans projection output file")
            tmp_p, tmp_ns = sis.phtprojncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label_file('.nc')
        if f.is_file():
            tmp_p = sp.add_parser('nc',
                                  help="Manipulate Siesta NetCDF output file")
            tmp_p, tmp_ns = sis.ncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        return p, namespace


add_sile('fdf', fdfSileSiesta, case=False, gzip=True)
