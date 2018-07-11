from __future__ import print_function, division

import os.path as osp
import warnings
import numpy as np

from sisl.unit import units
import sisl._array as _a
from sisl.messages import SislError, info, warn

from .._help import *
from ..sile import *
from .sile import SileOpenMX

from sisl import Geometry, SphericalOrbital, Atom, SuperCell


__all__ = ['omxSileOpenMX']

_LOGICAL_TRUE  = ['on', 'yes', 'true', '.true.', 'ok']
_LOGICAL_FALSE = ['off', 'no', 'false', '.false.', 'ng']
_LOGICAL = _LOGICAL_FALSE + _LOGICAL_TRUE


class omxSileOpenMX(SileOpenMX):
    r""" OpenMX-input file

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
    >>> omx = omxSileOpenMX('tmp/input.dat') # reads output files in 'tmp/' folder
    >>> omx = omxSileOpenMX('tmp/input.dat', base='.') # reads output files in './' folder

    When using this file in conjunction with the `sgeom` script while your input data-files are
    named *.dat, please do this:

    .. code:: bash

        sgeom input.dat{omx} output.xyz

    which forces the use of the omx file.
    """

    def __init__(self, filename, mode='r', base=None):
        super(omxSileOpenMX, self).__init__(filename, mode=mode)
        if base is None:
            # Extract from filename
            self._directory = osp.dirname(filename)
        else:
            self._directory = base
        if len(self._directory) == 0:
            self._directory = '.'

    def __str__(self):
        return ''.join([self.__class__.__name__, '(', self.file, ', base=', self._directory, ')'])

    @property
    def file(self):
        """ Return the current file name (without the directory prefix) """
        return self._file

    def _setup(self, *args, **kwargs):
        """ Setup the `omxSileOpenMX` after initialization """
        # These are the comments
        self._comment = ['#']

        # List of parent file-handles used while reading
        # This is because fdf enables inclusion of other files
        self._parent_fh = []
        self._directory = '.'

    def _tofile(self, f):
        """ Make `f` refer to the file with the appropriate base directory """
        return osp.join(self._directory, f)

    def _pushfile(self, f):
        if osp.isfile(self._tofile(f)):
            self._parent_fh.append(self.fh)
            self.fh = open(self._tofile(f), self._mode)
        else:
            warn(str(self) + ' is trying to include file: {} but the file seems not to exist? Will disregard file!'.format(f))

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

    @Sile_fh_open
    def _read_key(self, key):
        """ Try and read the first occurence of a key

        This will take care of blocks, labels and piped in labels

        Parameters
        ----------
        key : str
           key to find in the file
        """
        self._seek()
        def tokey(key):
            return key.lower()
        keyl = tokey(key)

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
            lsl = list(map(tokey, ls))

            # The last case is if the key is the first word on the line
            # In that case we have found what we are looking for
            if lsl[0] == keyl:
                return (' '.join(ls[1:])).strip()

            elif lsl[0].startswith('<'):
                # Get key
                lsl_key = lsl[0][1:]
                lsl_end = lsl_key + '>'
                if lsl_key == keyl:
                    # Read in the block content
                    lines = []

                    # Now read lines
                    l = self.readline().strip()
                    while not tokey(l).endswith(lsl_end):
                        if len(l) > 0:
                            lines.append(l)
                        l = self.readline().strip()
                    return lines

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
            the value to check for input-type
        """
        if value is None:
            return None

        if isinstance(value, list):
            # A block, <[name]
            return 'B'

        # Grab the entire line (beside the key)
        values = value.split()
        if len(values) == 1:
            fdf = values[0].lower()
            if fdf in _LOGICAL:
                # logical
                return 'l'

            try:
                float(fdf)
                if '.' in fdf:
                    # a real number (otherwise an integer)
                    return 'r'
                return 'i'
            except:
                pass
            # fall-back to name with everything

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
    def get(self, key, default=None):
        """ Retrieve keyword from the file

        Parameters
        ----------
        key : str
            the key to search for
        default : optional
            if the key is not found, this will be the returned value (default to ``None``)

        Returns
        -------
        value : the value of the key. If the key is a block, a `list` is returned, for
                a real value a `float` (or if the default is of `float`), for an integer, an
                `int` is returned.
        """
        # Try and read a line
        value = self._read_key(key)

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

        elif t == 'l':
            return value.lower() in _LOGICAL_TRUE

        return value

    def read_basis(self, *args, **kwargs):
        """ Reads basis

        Parameters
        ----------
        output: bool, optional
            whether to read supercell from output files (default to read from
            the input file).
        order: list of str, optional
            the order of which to try and read the supercell.
            By default this is ``['dat'/'omx'], `` if `output` is true.
            If `order` is present `output` is disregarded.
        """
        order = kwargs.pop('order', ['dat', 'omx'])
        for f in order:
            v = getattr(self, '_r_basis_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                return v
        return None

    def _r_basis_omx(self):
        ns = self.get('Species.Number', 0)
        data = self.get('Definition.of.Atomic.Species')
        if data is None:
            return None

        if ns == 0:
            ns = len(data)
        data = data[:ns]

        def rf_func(R):
            if R > 0:
                r = np.linspace(0, R, 500)
                f = np.ones(500)
                f[r > R] = 0
                return r, f
            return np.linspace(0, 1., 10), np.zeros(10)

        def decompose_basis(l):
            # Only split once
            Zr, spec = l.split('-', 1)
            idx = 0
            for i, c in enumerate(Zr):
                if c.isdigit():
                    idx = i
                    break
            if idx == 0:
                Z = Zr
                R = -1
            else:
                Z = Zr[:idx]
                R = float(Zr[idx:])

            # Now figure out the orbitals
            orbs = []
            for i, c in enumerate(spec):
                try:
                    l = 'spdfg'.index(c)
                    orbs.extend(SphericalOrbital(l, rf_func(R)).toAtomicOrbital())
                except:
                    pass

            return Z, orbs

        # We are ready to parse
        atom = []
        for dat in data:
            d = dat.split()
            # Figure out the specie
            Z, orbs = decompose_basis(d[1])
            atom.append(Atom(Z, orbs, tag=d[0]))
        return atom

    def read_supercell(self, output=False, *args, **kwargs):
        """ Reads supercell

        One can limit the tried files to only one file by passing
        only a single file ending.

        Parameters
        ----------
        output: bool, optional
            whether to read supercell from output files (default to read from
            the input file).
        order: list of str, optional
            the order of which to try and read the supercell.
            By default this is ``['dat'/'omx'], `` if `output` is true.
            If `order` is present `output` is disregarded.
        """
        if output:
            order = kwargs.pop('order', ['dat', 'omx'])
        else:
            order = kwargs.pop('order', ['dat', 'omx'])
        for f in order:
            v = getattr(self, '_r_supercell_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                return v
        return None

    def _r_supercell_omx(self, *args, **kwargs):
        """ Returns `SuperCell` object from the omx file """
        conv = self.get('Atoms.UnitVectors.Unit', default='Ang')
        if conv.upper() == 'AU':
            conv = units('Bohr', 'Ang')
        else:
            conv = 1.

        # Read in cell
        cell = np.empty([3, 3], np.float64)

        lc = self.get('Atoms.UnitVectors')
        if not lc is None:
            for i in range(3):
                cell[i, :] = [float(k) for k in lc[i].split()[:3]]
        else:
            raise SileError('Could not find Atoms.UnitVectors in file')
        cell *= conv

        return SuperCell(cell)

    _r_supercell_dat = _r_supercell_omx

    def read_geometry(self, output=False, *args, **kwargs):
        """ Returns Geometry object

        One can limit the tried files to only one file by passing
        only a single file ending.

        Parameters
        ----------
        output: bool, optional
            whether to read geometry from output files (default to read from
            the input file).
        order: list of str, optional
            the order of which to try and read the geometry.
            By default this is ``['dat'/'omx']`` if `output` is true
            If `order` is present `output` is disregarded.
        """
        if output:
            order = kwargs.pop('order', ['dat', 'omx'])
        else:
            order = kwargs.pop('order', ['dat', 'omx'])
        for f in order:
            v = getattr(self, '_r_geometry_{}'.format(f.lower()))(*args, **kwargs)
            if v is not None:
                return v
        return None

    def _r_geometry_omx(self, *args, **kwargs):
        """ Returns `Geometry` """
        sc = self.read_supercell(order=['omx'])

        na = self.get('Atoms.Number', default=0)
        conv = self.get('Atoms.SpeciesAndCoordinates.Unit', default='Ang')
        data = self.get('Atoms.SpeciesAndCoordinates')
        if data is None:
            raise SislError('Cannot find key: Atoms.SpeciesAndCoordinates')

        if na == 0:
            # Default to the size of the labels
            na = len(data)

        # Reduce to the number of atoms.
        data = data[:na]

        atoms = self.read_basis(order=['omx'])
        def find_atom(tag):
            if atoms is None:
                return Atom(tag)
            for atom in atoms:
                if atom.tag == tag:
                    return atom
            raise SislError('Error when reading the basis for atomic tag: {}.'.format(tag))

        xyz = []
        atom = []
        for dat in data:
            d = dat.split()
            atom.append(find_atom(d[1]))
            xyz.append(list(map(float, dat.split()[2:5])))
        xyz = _a.arrayd(xyz)

        if conv == 'AU':
            xyz *= units('Bohr', 'Ang')
        elif conv == 'FRAC':
            xyz = np.dot(xyz, sc.cell)

        return Geometry(xyz, atom=atom, sc=sc)

    _r_geometry_dat = _r_geometry_omx


add_sile('omx', omxSileOpenMX, case=False, gzip=True)
