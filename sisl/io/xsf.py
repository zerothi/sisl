from __future__ import print_function

import os.path as osp
import numpy as np

# Import sile objects
from .sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl.utils import str_spec


__all__ = ['xsfSile', 'axsfSile']


class xsfSile(Sile):
    """ XSF file for XCrySDen """

    def _setup(self, *args, **kwargs):
        """ Setup the `xsfSile` after initialization """
        self._comment = ['#']
        self._md_steps = kwargs.get('steps', None)
        self._md_index = 0

    def _step_md(self):
        """ Step the MD counter """
        self._md_index += 1

    @sile_fh_open()
    def write_supercell(self, sc, fmt='.8f'):
        """ Writes the supercell to the contained file

        Parameters
        ----------
        sc : SuperCell
           the supercell to be written
        fmt : str, optional
           used format for the precision of the data
        """
        # Implementation notice!
        # The XSF files are compatible with Vesta, but ONLY
        # if there are no empty lines!

        # Check that we can write to the file
        sile_raise_write(self)

        # Write out top-header stuff
        from time import gmtime, strftime
        self._write('# File created by: sisl {}\n#\n'.format(strftime("%Y-%m-%d", gmtime())))

        # Print out the number of ANIMSTEPS (if required)
        if not self._md_steps is None:
            self._write('ANIMSTEPS {}\n'.format(self._md_steps))

        self._write('CRYSTAL\n#\n')

        if self._md_index == 1:
            self._write('# Primitive lattice vectors:\n#\n')
        if self._md_steps is None:
            self._write('PRIMVEC\n')
        else:
            self._write('PRIMVEC {}\n'.format(self._md_index))
        # We write the cell coordinates as the cell coordinates
        fmt_str = '{{:{0}}} '.format(fmt) * 3 + '\n'
        for i in [0, 1, 2]:
            self._write(fmt_str.format(*sc.cell[i, :]))

        # Currently not written (we should convert the unit cell
        # to a conventional cell (90-90-90))
        # It seems this simply allows to store both formats in
        # the same file.
        self._write('#\n# Conventional lattice vectors:\n#\n')
        if self._md_steps is None:
            self._write('CONVVEC\n')
        else:
            self._write('CONVVEC {}\n'.format(self._md_index))
        convcell = sc.toCuboid(True)._v
        for i in [0, 1, 2]:
            self._write(fmt_str.format(*convcell[i, :]))

    @sile_fh_open()
    def write_geometry(self, geometry, fmt='.8f', data=None):
        """ Writes the geometry to the contained file

        Parameters
        ----------
        geometry : Geometry
           the geometry to be written
        fmt : str, optional
           used format for the precision of the data
        data : (geometry.na, 3), optional
           auxiliary data associated with the geometry to be saved
           along side. Internally in XCrySDen this data is named *Forces*
        """
        self._step_md()
        self.write_supercell(geometry.sc, fmt)

        has_data = not data is None
        if has_data:
            data.shape = (-1, 3)

        # The current geometry is currently only a single
        # one, and does not write the convvec
        # Is it a necessity?
        if self._md_index == 1:
            self._write('#\n# Atomic coordinates (in primitive coordinates)\n#\n')
        if self._md_steps is None:
            self._write('PRIMCOORD\n')
        else:
            self._write('PRIMCOORD {}\n'.format(self._md_index))
        self._write('{} {}\n'.format(len(geometry), 1))

        non_valid_Z = (geometry.atoms.Z <= 0).nonzero()[0]
        if len(non_valid_Z) > 0:
            geometry = geometry.remove(non_valid_Z)

        if has_data:
            fmt_str = '{{0:3d}}  {{1:{0}}}  {{2:{0}}}  {{3:{0}}}   {{4:{0}}}  {{5:{0}}}  {{6:{0}}}\n'.format(fmt)
            for ia in geometry:
                tmp = np.append(geometry.xyz[ia, :], data[ia, :])
                self._write(fmt_str.format(geometry.atoms[ia].Z, *tmp))
        else:
            fmt_str = '{{0:3d}}  {{1:{0}}}  {{2:{0}}}  {{3:{0}}}\n'.format(fmt)
            for ia in geometry:
                self._write(fmt_str.format(geometry.atoms[ia].Z, *geometry.xyz[ia, :]))

    @sile_fh_open()
    def read_geometry(self, data=False):
        """ Returns Geometry object from the XSF file

        Parameters
        ----------
        data : bool, optional
           in case the XSF file has auxiliary data, return that as well.
        """
        # Prepare containers...
        cell = np.zeros([3, 3], np.float64)
        cell_set = False
        atom = []
        xyz = []
        na = 0

        line = ' '
        while line != '':
            # skip comments
            line = self.readline()
            key = line.strip()

            # We prefer the primvec
            if key.startswith('CONVVEC') and not cell_set:
                for i in [0, 1, 2]:
                    line = self.readline()
                    cell[i, :] = [float(x) for x in line.split()]

            elif key.startswith('PRIMVEC'):
                cell_set = True
                for i in [0, 1, 2]:
                    line = self.readline()
                    cell[i, :] = [float(x) for x in line.split()]

            elif key.startswith('PRIMCOORD'):
                # First read # of atoms
                line = self.readline().split()
                na = int(line[0])

                # currently line[1] is unused!
                for _ in range(na):
                    line = self.readline().split()
                    atom.append(int(line[0]))
                    xyz.append([float(x) for x in line[1:]])

        xyz = np.array(xyz, np.float64)
        if data:
            dat = None
        if xyz.shape[1] == 6:
            dat = xyz[:, 3:]
            xyz = xyz[:, :3]

        if len(atom) == 0:
            geom = Geometry(xyz, sc=SuperCell(cell))
        elif len(atom) == 1 and atom[0] == -999:
            geom = None
        else:
            geom = Geometry(xyz, atom=atom, sc=SuperCell(cell))

        if data:
            return geom, dat
        return geom

    @sile_fh_open()
    def write_grid(self, *args, **kwargs):
        """ Store grid(s) data to an XSF file

        Examples
        --------
        >>> g1 = Grid(0.1, sc=2.)
        >>> g2 = Grid(0.1, sc=2.)
        >>> get_sile('output.xsf', 'w').write_grid(g1, g2)

        Parameters
        ----------
        *args : Grid
            a list of data-grids to be written to the XSF file.
            Each argument gets the field name ``?grid_<>`` where <> starts
            with the integer 0, and *?* is ``real_``/``imag_`` for complex
            valued grids.
        geometry : Geometry, optional
            the geometry stored in the file, defaults to ``args[0].geometry``
        fmt : str, optional
            floating point format for data (.5e)
        buffersize : int, optional
            size of the buffer while writing the data, (6144)
        """
        sile_raise_write(self)

        geom = kwargs.get('geometry', args[0].geom)
        if geom is None:
            geom = Geometry([0, 0, 0], Atom(-999), sc=args[0].sc)
        self.write_geometry(geom)

        # Buffer size for writing
        buffersize = kwargs.get('buffersize', min(6144, args[0].grid.size))

        # Format for precision
        fmt = kwargs.get('fmt', '.5e')

        self._write('BEGIN_BLOCK_DATAGRID_3D\n')
        name = kwargs.get('name', 'sisl_grid_{}'.format(len(args)))
        # Transfer all spaces to underscores (no spaces allowed)
        self._write(' ' + name.replace(' ', '_') + '\n')
        _v3 = (('{:' + fmt + '} ') * 3).strip() + '\n'

        def write_cell(grid):
            # Now write the grid
            self._write('  {} {} {}\n'.format(*grid.shape))
            self._write('  ' + _v3.format(*grid.origo))
            self._write('  ' + _v3.format(*grid.cell[0, :]))
            self._write('  ' + _v3.format(*grid.cell[1, :]))
            self._write('  ' + _v3.format(*grid.cell[2, :]))

        for i, grid in enumerate(args):
            is_complex = np.iscomplexobj(grid.grid)

            name = kwargs.get('grid' + str(i), str(i))
            if is_complex:
                self._write(' BEGIN_DATAGRID_3D_real_{}\n'.format(name))
            else:
                self._write(' BEGIN_DATAGRID_3D_{}\n'.format(name))

            write_cell(grid)

            # for z
            #   for y
            #     for x
            #       write...
            _fmt = '{:' + fmt + '}\n'
            for x in np.nditer(np.asarray(grid.grid.real.T, order='C').reshape(-1), flags=['external_loop', 'buffered'],
                               op_flags=[['readonly']], order='C', buffersize=buffersize):
                self._write((_fmt * x.shape[0]).format(*x.tolist()))

            self._write(' END_DATAGRID_3D\n')

            # Skip if not complex
            if not is_complex:
                continue
            self._write(' BEGIN_DATAGRID_3D_imag_{}\n'.format(name))
            write_cell(grid)
            for x in np.nditer(np.asarray(grid.grid.imag.T, order='C').reshape(-1), flags=['external_loop', 'buffered'],
                               op_flags=[['readonly']], order='C', buffersize=buffersize):
                self._write((_fmt * x.shape[0]).format(*x.tolist()))

            self._write(' END_DATAGRID_3D\n')

        self._write('END_BLOCK_DATAGRID_3D\n')

    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)

    def ArgumentParser_out(self, p, *args, **kwargs):
        """ Adds arguments only if this file is an output file

        Parameters
        ----------
        p : ``argparse.ArgumentParser``
           the parser which gets amended the additional output options.
        """
        import argparse

        # We will add the vector data
        class VectorNoScale(argparse.Action):

            def __call__(self, parser, ns, no_value, option_string=None):
                setattr(ns, '_vector_scale', False)
        p.add_argument('--no-vector-scale', '-nsv', nargs=0,
                       action=VectorNoScale,
                       help='''Do not modify vector components (same as --vector-scale 1.)''')

        # We will add the vector data
        class VectorScale(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                setattr(ns, '_vector_scale', float(value))
        p.add_argument('--vector-scale', '-sv', metavar='SCALE',
                       action=VectorScale,
                       help='''Scale vector components by this factor.''')

        # We will add the vector data
        class Vectors(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                routine = values.pop(0)

                # Default input file
                input_file = getattr(ns, '_input_file', None)

                # Figure out which of the segments are a file
                for i, val in enumerate(values):
                    if osp.isfile(str_spec(val)[0]):
                        input_file = values.pop(i)
                        break

                # Quick return if there is no input-file...
                if input_file is None:
                    return

                # Try and read the vector
                from sisl.io import get_sile
                input_sile = get_sile(input_file, mode='r')

                vector = None
                if hasattr(input_sile, 'read_{}'.format(routine)):
                    vector = getattr(input_sile, 'read_{}'.format(routine))(*values)

                if vector is None:
                    # Try the read_data function
                    d = {routine: True}
                    vector = input_sile.read_data(*values, **d)

                # Clean the sile
                del input_sile

                if vector is None:
                    # Use title to capitalize
                    raise ValueError('{} could not be read from file: {}.'.format(routine.title(), input_file))

                if len(vector) != len(ns._geometry):
                    raise ValueError('read_{} could read from file: {}, sizes does not conform to geometry.'.format(routine, input_file))
                setattr(ns, '_vector', vector)
        p.add_argument('--vector', '-v', metavar=('DATA', '*ARGS, FILE'), nargs='+',
                       action=Vectors,
                       help='''Adds vector arrows for each atom, first argument is type (force, moment, ...).
If the current input file contains the vectors no second argument is necessary, else 
the file containing the data is required as the last input.

Any arguments inbetween are passed to the `read_data` function (in order).

By default the vectors scaled by 1 / max(|V|) such that the longest vector has length 1.
                       ''')

        # currently adding an argument that is already there does not remove the
        # old one...
        p.add_argument('--out', '-o', nargs=1, action=Out,
                       help='Store the geometry/grid (plus any vector fields) the out file.')


class axsfSile(xsfSile):
    """ AXSF file for XCrySDen

    When creating an AXSF file one should denote how many MD steps to write out:

    >>> axsf = axsfSile('file.axsf', steps=100)
    >>> for i in range(100):
    ...    axsf.write_geometry(geom)
    """

    def _setup(self, *args, **kwargs):
        super(axsfSile, self)._setup(*args, **kwargs)
        # Correct number of steps
        if self._md_steps is None:
            self._md_steps = 1

    write_grid = None


add_sile('xsf', xsfSile, case=False, gzip=True)
add_sile('axsf', axsfSile, case=False, gzip=True)
