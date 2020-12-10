import os.path as osp
from numbers import Integral
import numpy as np

# Import sile objects
from .sile import *

from sisl._internal import set_module
from sisl import Geometry, AtomUnknown, SuperCell
from sisl.utils import str_spec
import sisl._array as _a


__all__ = ['xsfSile', 'axsfSile']


def _get_kw_index(key):
    # Get the integer in a line like 'ATOMS 2', converted to 0-indexing, and with -1 if no int is there
    kl = key.split()
    if len(kl) == 1:
        return -1
    return int(kl[1]) - 1


@set_module("sisl.io")
class xsfSile(Sile):
    """ XSF file for XCrySDen """

    def _setup(self, *args, **kwargs):
        """ Setup the `xsfSile` after initialization """
        self._comment = ['#']

    def _write_key(self, key):
        self._write(key + "\n")

    _write_once = Sile._write

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
        self._write_once('# File created by: sisl {}\n#\n'.format(strftime("%Y-%m-%d", gmtime())))

        self._write_once('CRYSTAL\n#\n')

        self._write_once('# Primitive lattice vectors:\n#\n')
        self._write_key('PRIMVEC')

        # We write the cell coordinates as the cell coordinates
        fmt_str = f'{{:{fmt}}} ' * 3 + '\n'
        for i in [0, 1, 2]:
            self._write(fmt_str.format(*sc.cell[i, :]))

        # Convert the unit cell to a conventional cell (90-90-90))
        # It seems this simply allows to store both formats in
        # the same file.
        self._write_once('#\n# Conventional lattice vectors:\n#\n')
        self._write_key('CONVVEC')
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
        self.write_supercell(geometry.sc, fmt)

        has_data = data is not None
        if has_data:
            data.shape = (-1, 3)

        self._write_once('#\n# Atomic coordinates (in primitive coordinates)\n#\n')
        self._write_key("PRIMCOORD")
        self._write('{} {}\n'.format(len(geometry), 1))

        non_valid_Z = (geometry.atoms.Z <= 0).nonzero()[0]
        if len(non_valid_Z) > 0:
            geometry = geometry.remove(non_valid_Z)

        if has_data:
            fmt_str = (
                '{{0:3d}}  {{1:{0}}}  {{2:{0}}}  {{3:{0}}}   {{4:{0}}}  {{5:{0}}}  {{6:{0}}}\n'
            ).format(fmt)
            for ia in geometry:
                tmp = np.append(geometry.xyz[ia, :], data[ia, :])
                self._write(fmt_str.format(geometry.atoms[ia].Z, *tmp))
        else:
            fmt_str = '{{0:3d}}  {{1:{0}}}  {{2:{0}}}  {{3:{0}}}\n'.format(fmt)
            for ia in geometry:
                self._write(fmt_str.format(geometry.atoms[ia].Z, *geometry.xyz[ia, :]))

    @sile_fh_open()
    def _r_geometry_multiple(self, steps, ret_data=False, squeeze=False):
        asteps = steps
        steps = dict((step, i) for i, step in enumerate(steps))

        # initialize all things
        cell = [None] * len(steps)
        cell_set = [False] * len(steps)
        xyz_set = [False] * len(steps)
        atom = [None for _ in steps]
        xyz = [None for _ in steps]
        data = [None for _ in steps]
        data_set = [not ret_data for _ in steps]

        line = " "
        all_loaded = False

        while line != '' and not all_loaded:
            line = self.readline()

            if line.isspace():
                continue
            kw = line.split()[0]
            if kw not in ("CONVVEC", "PRIMVEC", "PRIMCOORD"):
                continue

            step = _get_kw_index(line)
            if step != -1 and step not in steps:
                continue

            if step not in steps and step == -1:
                step = idstep = istep = None
            else:
                idstep = steps[step]
                istep = idstep

            if kw == "CONVVEC":
                if step is None:
                    if not any(cell_set):
                        cell_set = [True] * len(cell_set)
                    else:
                        continue
                elif cell_set[istep]:
                    continue
                else:
                    cell_set[istep] = True

                icell = _a.zerosd([3, 3])
                for i in range(3):
                    line = self.readline()
                    icell[i] = line.split()
                if step is None:
                    cell = [icell] * len(cell)
                else:
                    cell[istep] = icell

            elif kw == "PRIMVEC":
                if step is None:
                    cell_set = [True] * len(cell_set)
                else:
                    cell_set[istep] = True

                icell = _a.zerosd([3, 3])
                for i in range(3):
                    line = self.readline()
                    icell[i] = line.split()
                if step is None:
                    cell = [icell] * len(cell)
                else:
                    cell[istep] = icell

            elif kw == "PRIMCOORD":
                if step is None:
                    raise ValueError(f"{self.__class__.__name__}"
                        " contains an unindexed (or somehow malformed) 'PRIMCOORD'"
                        " section but you've asked for a particular index. This"
                        f" shouldn't happen. line:\n {line}"
                    )

                iatom = []
                ixyz = []
                idata = []
                line = self.readline().split()
                for _ in range(int(line[0])):
                    line = self.readline().split()
                    if not xyz_set[istep]:
                        iatom.append(int(line[0]))
                        ixyz.append([float(x) for x in line[1:4]])
                    if ret_data and len(line) > 4:
                        idata.append([float(x) for x in line[4:]])
                if not xyz_set[istep]:
                    atom[istep] = iatom
                    xyz[istep] = ixyz
                    xyz_set[istep] = True
                data[idstep] = idata
                data_set[idstep] = True

            all_loaded = all(xyz_set) and all(cell_set) and all(data_set)

        if not all(xyz_set):
            which = [asteps[i] for i in np.flatnonzero(xyz_set)]
            raise ValueError(f"{self.__class__.__name__} file did not contain atom coordinates for the following requested index: {which}")

        if ret_data:
            data = _a.arrayd(data)
            if data.size == 0:
                data.shape = (len(steps), len(xyz[0]), 0)

        xyz = _a.arrayd(xyz)
        cell = _a.arrayd(cell)
        atom = _a.arrayi(atom)

        geoms = []
        for istep in range(len(steps)):
            if len(atom) == 0:
                geoms.append(si.Geometry(xyz[istep], sc=SuperCell(cell[istep])))
            elif len(atom[0]) == 1 and atom[0][0] == -999:
                # should we perhaps do AtomUnknown?
                geoms.append(None)
            else:
                geoms.append(Geometry(xyz[istep], atoms=atom[istep], sc=SuperCell(cell[istep])))

        if squeeze and len(steps) == 1:
            geoms = geoms[0]
            if ret_data:
                data = data[0]

        if ret_data:
            return geoms, data
        return geoms

    def read_geometry(self, ret_data=False):
        """ Geometry contained in file, and optionally the associated data

        Parameters
        ----------
        ret_data : bool, optional
           in case the the file has auxiliary data, return that as well.
        """
        return self._r_geometry_multiple([-1], ret_data=ret_data, squeeze=True)

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

        geom = kwargs.get('geometry', args[0].geometry)
        if geom is None:
            geom = Geometry([0, 0, 0], AtomUnknown(999), sc=args[0].sc)
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
                self._write(f' BEGIN_DATAGRID_3D_real_{name}\n')
            else:
                self._write(f' BEGIN_DATAGRID_3D_{name}\n')

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
            self._write(f' BEGIN_DATAGRID_3D_imag_{name}\n')
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
        ns = kwargs.get("namespace", None)
        if ns is None:
            class _():
                pass
            ns = _()

        # We will add the vector data
        class VectorNoScale(argparse.Action):
            def __call__(self, parser, ns, no_value, option_string=None):
                setattr(ns, "_vector_scale", False)
        p.add_argument("--no-vector-scale", "-nsv", nargs=0,
                       action=VectorNoScale,
                       help="""Do not modify vector components (same as --vector-scale 1.)""")
        # Default to scale the vectors
        setattr(ns, "_vector_scale", True)

        # We will add the vector data
        class VectorScale(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                setattr(ns, '_vector_scale', float(value))
        p.add_argument('--vector-scale', '-sv', metavar='SCALE',
                       action=VectorScale,
                       help="""Scale vector components by this factor.""")

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
                if hasattr(input_sile, f'read_{routine}'):
                    vector = getattr(input_sile, f'read_{routine}')(*values)

                if vector is None:
                    # Try the read_data function
                    d = {routine: True}
                    vector = input_sile.read_data(*values, **d)

                if vector is None and len(values) > 1:
                    # try and see if the first argument is a str, if
                    # so use that as a keyword
                    if isinstance(values[0], str):
                        d = {values[0]: True}
                        vector = input_sile.read_data(*values[1:], **d)

                # Clean the sile
                del input_sile

                if vector is None:
                    # Use title to capitalize
                    raise ValueError('{} could not be read from file: {}.'.format(routine.title(), input_file))

                if len(vector) != len(ns._geometry):
                    raise ValueError(f'read_{routine} could read from file: {input_file}, sizes does not conform to geometry.')
                setattr(ns, '_vector', vector)
        p.add_argument('--vector', '-v', metavar=('DATA', '*ARGS[, FILE]'), nargs='+',
                       action=Vectors,
                       help="""Adds vector arrows for each atom, first argument is type (force, moment, ...).
If the current input file contains the vectors no second argument is necessary, else
the file containing the data is required as the last input.

Any arguments inbetween are passed to the `read_data` function (in order).

By default the vectors scaled by 1 / max(|V|) such that the longest vector has length 1.
                       """)


@set_module("sisl.io")
class axsfSile(xsfSile):
    """ AXSF file for XCrySDen

    When creating an AXSF file one must denote how many geometries to write out.
    It is also necessary to use the axsf in a context manager, otherwise it will
    overwrite itself repeatedly.

    >>> with axsfSile('file.axsf', 'w', steps=100) as axsf:
    ...     for i in range(100):
    ...         axsf.write_geometry(geom)
    """

    def _setup(self, *args, steps=1, **kwargs):
        super()._setup(*args, **kwargs)

        # Index of last written geometry (or current geom when writing one)
        self._geometry_index = -1

        # Total number of geometries intended to be written
        self._geometry_count = steps
        if self._geometry_count < 1 and "w" in self._mode:
            raise ValueError(
                "In write mode, the intended positive number of geometries must be passed in the"
                " `steps` keyword."
            )

    def _incr_index(self):
        """ Increment the geometry index """
        self._geometry_index += 1

    def _write_key(self, key):
        self._write(f"{key} {self._geometry_index + 1}\n")

    def _write_once(self, string):
        if self._geometry_index <= 0:
            self._write(string)

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
        self._incr_index()
        self._write_once(f"ANIMSTEPS {self._geometry_count}\n")
        return super().write_geometry(geometry, fmt=fmt, data=data)

    def read_geometry(self, index=-1, ret_data=False):
        """ Geometries and (possibly) associated data stored in the AXSF file

        Parameters
        ----------
        index : int or iterable of int or None, optional
            The indices to load (0-indexed). If None, load all.
            If an integer is passed, a single Geometry is returned, and the leading dimension on data is removed.
        ret_data : bool, optional
            in case the file has auxiliary data, return that as well.

        Returns
        -------
        geometries : list of Geometry or Geometry
            A list of geometries (or a single Geometries) corresponding to requested indices.
        data : ndarray of shape (nindex, natoms, nperatom) and dtype float64
            Only returned if `data` is True.
        """
        squeeze = isinstance(index, Integral)
        if index is None:
            index = np.arange(self._r_geometry_count())
        else:
            index = _a.arrayi(index).ravel()
            index[index < 0] += self._r_geometry_count()
        return self._r_geometry_multiple(index, ret_data=ret_data, squeeze=squeeze)

    @sile_fh_open()
    def _r_geometry_count(self):
        line = ' '
        while line != '':
            line = self.readline()
            if line.startswith("ANIMSTEPS"):
                return _get_kw_index(line) + 1
        raise ValueError(f"{self.__class__.__name__} did not find 'ANIMSTEPS' in the file...")

    write_grid = None


add_sile('xsf', xsfSile, case=False, gzip=True)
add_sile('axsf', axsfSile, case=False, gzip=True)
