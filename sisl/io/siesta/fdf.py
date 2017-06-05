"""
Sile object for reading/writing FDF files
"""

from __future__ import print_function, division

import os.path as osp
import numpy as np
import warnings as warn

# Import sile objects
from sisl._help import _str
from .sile import SileSiesta
from ..sile import *
from sisl.io._help import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell, Grid

from sisl.utils.cmd import *
from sisl.utils.misc import merge_instances, str_spec

from sisl.units import unit_default, unit_group
from sisl.units.siesta import unit_convert

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

    @property
    def file(self):
        """ Return the current file name (without the directory prefix) """
        return self._file

    def _setup(self):
        """ Setup the `fdfSileSiesta` after initialization """
        # These are the comments
        self._comment = ['#', '!', ';']

        # List of parent file-handles used while reading
        # This is because fdf enables inclusion of other files
        self._parent_fh = []
        self._directory = '.'

    @Sile_fh_open
    def includes(self):
        """ Return a list of all include files """

        includes = [self.fh.name]
        l = self.readline()
        while l != '':
            for inc in self._parent_fh:
                if inc.name not in includes:
                    includes.append(inc.name)

        # Now remove prefixes make it smaller
        includes = [inc.replace(self._directory, '') for inc in includes]
        return includes

    def readline(self, comment=False):
        """ Reads the next line of the file """
        # Call the parent readline function
        l = super(fdfSileSiesta, self).readline(comment=comment)

        # In FDF files, %include marks files that progress
        # down in a tree structure
        if '%include' in l:
            # Split for reading tree file
            self._parent_fh.append(self.fh)
            self.fh = open(self._directory + osp.sep + l.split()[1], self._mode)
            # Read the following line in the new file
            return self.readline(comment)

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
        """ Retrieve fdf-keyword from the file """

        # First split into specification and key
        key, tmp_unit = str_spec(key)
        if unit is None:
            unit = tmp_unit

        found, fdf = self._read(key)
        if not found:
            return default

        # The keyword is found...
        if fdf.startswith('%block'):
            found, fdf = self._read_block(key)
            if not found:
                return default
            else:
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
        key : `str`
           the fdf-key value to be set in the fdf file
        value : `str`/`list`
           the value of the string. If a `str` is passed a regular
           fdf-key is used, if a `list` it will be a %block.
        keep : `bool`
           whether old flags will be kept in the fdf file.
        """

        # To set a key we first need to figure out if it is
        # already present, if so, we will add the new key, just above
        # the already present key.

        # 1. find the old value, and thus the file in which it is found
        with self:
            old_value = self.get(key)
            # Get the file of the containing data
            top_file = self.file

        try:
            while len(self._parent_fh) > 0:
                self.fh.close()
                self.fh = self._parent_fh.pop()
            self.fh.close()
        except:
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
        found, fdf = self.step_to(key, case=False)

        # Check whether the key is piped
        if found and fdf.find('<') >= 0:
            # Create new fdf-file
            sub_fdf = fdfSileSiesta(fdf.split('<')[1].replace('\n', '').strip())
            return sub_fdf._read(key)

        return found, fdf

    @Sile_fh_open
    def _read_block(self, key, force=False):
        """ Returns the arguments following the keyword in the FDF file """
        k = key.lower()
        f, fdf = self.step_to(k, case=False)
        if force and not f:
            # The user requests that the block *MUST* be found
            raise SileError(('Requested forced block could not be found: ' +
                             str(key) + '.'), self)
        if not f:
            return False, []  # not found

        # If the block is piped in from another file...
        if '<' in fdf:
            # Create new fdf-file
            sub_fdf = fdfSileSiesta(fdf.split('<')[1].replace('\n', '').strip())
            with sub_fdf:
                li = []
                line = sub_fdf.readline()
                while line != '':
                    li.append(line.replace('\n', ''))
                    line = sub_fdf.readline()
            return True, li

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
    def write_geometry(self, geom, fmt='.8f', *args, **kwargs):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        # Write out the cell
        self._write('LatticeConstant 1. Ang\n')
        self._write('%block LatticeVectors\n')
        self._write(' {0} {1} {2}\n'.format(*geom.cell[0, :]))
        self._write(' {0} {1} {2}\n'.format(*geom.cell[1, :]))
        self._write(' {0} {1} {2}\n'.format(*geom.cell[2, :]))
        self._write('%endblock LatticeVectors\n\n')
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
        for i, (a, _) in enumerate(geom.atom):
            self._write(' {0} {1} {2}\n'.format(i + 1, a.Z, a.tag))
        self._write('%endblock ChemicalSpeciesLabel\n')

    def read_supercell(self, *args, **kwargs):
        """ Returns `SuperCell` object from the FDF file """
        f, lc = self._read('LatticeConstant')

        s = float(lc.split()[1])
        if 'ang' in lc.lower():
            pass
        elif 'bohr' in lc.lower():
            s *= Bohr2Ang

        # Read in cell
        cell = np.empty([3, 3], np.float64)

        f, lc = self._read_block('LatticeVectors')
        if f:
            for i in range(3):
                cell[i, :] = [float(k) for k in lc[i].split()[:3]]
        else:
            f, lc = self._read_block('LatticeParameters')
            if f:
                tmp = [float(k) for k in lc[0].split()[:6]]
                cell = SuperCell.tocell(*tmp)
        if not f:
            # the fdf file contains neither the latticevectors or parameters
            raise SileError(
                'Could not find Vectors or Parameters block in file')
        cell *= s

        return SuperCell(cell)

    def read_geometry(self, *args, **kwargs):
        """ Returns Geometry object from the FDF file

        NOTE: Interaction range of the Atoms are currently not read.
        """

        f, lc = self._read('LatticeConstant')
        if not f:
            raise ValueError('Could not find LatticeConstant in fdf file.')
        s = float(lc.split()[1])
        if 'ang' in lc.lower():
            pass
        elif 'bohr' in lc.lower():
            s *= Bohr2Ang

        sc = self.read_supercell(*args, **kwargs)

        # No fractional coordinates
        is_frac = False

        # Read atom scaling
        f, lc = self._read('AtomicCoordinatesFormat')
        if not f:
            raise ValueError(
                'Could not find AtomicCoordinatesFormat in fdf file.')
        lc = lc.lower()
        if 'ang' in lc or 'notscaledcartesianang' in lc:
            s = 1.
            pass
        elif 'bohr' in lc or 'notscaledcartesianbohr' in lc:
            s = Bohr2Ang
        elif 'scaledcartesian' in lc:
            # the same scaling as the lattice-vectors
            pass
        elif 'fractional' in lc or 'scaledbylatticevectors' in lc:
            # no scaling of coordinates as that is entirely
            # done by the latticevectors
            s = 1.
            is_frac = True

        # If the user requests a shifted geometry
        # we correct for this
        origo = np.zeros([3], np.float64)
        run = 'origin' in kwargs
        if run:
            run = kwargs['origin']
        if run:
            f, lor = self._read_block('AtomicCoordinatesOrigin')
            if f:
                origo = ensure_array(map(float, lor[0].split()[:3])) * s
        # Origo cannot be interpreted with fractional coordinates
        # hence, it is not transformed.

        # Read atom block
        f, atms = self._read_block(
            'AtomicCoordinatesAndAtomicSpecies', force=True)
        if not f:
            raise ValueError(
                'Could not find AtomicCoordinatesAndAtomicSpecies in fdf file.')

        # Read number of atoms and block
        f, l = self._read('NumberOfAtoms')
        if not f:
            # We default to the number of elements in the
            # AtomicCoordinatesAndAtomicSpecies block
            na = len(atms)
        else:
            na = int(l.split()[1])

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
        f, l = self._read('NumberOfSpecies')
        ns = 0
        if f:
            ns = int(l.split()[1])

        # Read the block (not strictly needed, if so we simply set all atoms to
        # H)
        f, spcs = self._read_block('ChemicalSpeciesLabel')
        if not f:
            f, spcs = self._read_block('Chemical_Species_Label')

        if f:
            # Initialize number of species to
            # the length of the ChemicalSpeciesLabel block
            if ns == 0:
                ns = len(spcs)
            # Pre-allocate the species array
            sp = [None] * ns
            for spc in spcs:
                #  index Z pseudo-tag
                l = spc.split()
                idx = int(l[0]) - 1
                # Insert the atom
                sp[idx] = Atom(Z=int(l[1]), tag=l[2])

            if None in sp:
                idx = sp.index(None) + 1
                raise ValueError(
                    ("Could not populate entire species list. "
                     "Please ensure specie with index {} is present".format(idx)))

            # Create atoms array with species
            atom = [None] * na
            for ia in range(na):
                atom[ia] = sp[species[ia]]

            if None in atom:
                idx = atom.index(None) + 1
                raise ValueError(
                    ("Could not populate entire atomic list list. "
                     "Please ensure atom with index {} is present".format(idx)))

        else:
            # Default atom (hydrogen)
            atom = Atom(1)
            # Force number of species to 1
            ns = 1

        # Create and return geometry object
        return Geometry(xyz, atom=atom, sc=sc)

    @dec_default_AP("Manipulate a FDF file.")
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
            pass

        f = label + '.bands'
        if osp.isfile(f):
            tmp_p = sp.add_parser('band',
                                  help="Manipulate the bands file from the SIESTA simulation")
            tmp_p, tmp_ns = sis.bandsSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label + '.TBT.nc'
        if osp.isfile(f):
            tmp_p = sp.add_parser('tbt',
                                  help="Manipulate the tbtrans output file")
            tmp_p, tmp_ns = sis.tbtncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label + '.TBT.Proj.nc'
        if osp.isfile(f):
            tmp_p = sp.add_parser('tbt-proj',
                                  help="Manipulate the tbtrans projection output file")
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
                                  help="Manipulate the phtrans projection output file")
            tmp_p, tmp_ns = sis.phtprojncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        f = label + '.nc'
        if osp.isfile(f):
            tmp_p = sp.add_parser('nc',
                                  help="Manipulate the SIESTA output file")
            tmp_p, tmp_ns = sis.ncSileSiesta(f).ArgumentParser(tmp_p, *args, **kwargs)
            namespace = merge_instances(namespace, tmp_ns)

        return p, namespace


add_sile('fdf', fdfSileSiesta, case=False, gzip=True)
