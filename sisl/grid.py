""" Define a grid

This grid is the basis of different used models.
"""
from __future__ import print_function, division

from numbers import Integral

import numpy as np

from .utils import *
from .quaternion import Quaternion
from .supercell import SuperCell, SuperCellChild
from .atom import Atom
from .geometry import Geometry

__all__ = ['Grid', 'sgrid']


class Grid(SuperCellChild):
    """ Object to retain grid information

    This grid object handles cell vectors and divisions of said grid.

    A grid can be periodic and non-periodic.
    """

    # Constant (should never be changed)
    Periodic = 1
    Neumann = 2
    Dirichlet = 3

    def __init__(self, shape=None, bc=None, sc=None, dtype=None, geom=None):
        """ Initialize a `Grid` object.

        Initialize a `Grid` object.

        Parameters
        ----------
        shape : `list of ints`
           the size of each grid dimension
        bc : `int`
           the boundary condition (`Grid.Periodic/Grid.Neumann/Grid.Dirichlet`)
        sc : `SuperCell/list`
           the associated supercell (
        """
        if shape is None:
            shape = [1, 1, 1]
        if bc is None:
            bc = self.Periodic

        self.set_supercell(sc)

        # Create the grid
        self.set_grid(shape, dtype=dtype)

        # Create the grid boundary conditions
        self.set_bc(bc)

        # Create the atomic structure in the grid, if possible
        self.set_geom(geom)

        # If the user sets the super-cell, that has precedence.
        if sc is not None:
            self.geom.set_sc(sc)
            self.set_sc(sc)

    def __getitem__(self, key):
        """ Returns the grid contained """
        return self.grid[key]

    def __setitem__(self, key, val):
        """ Updates the grid contained """
        self.grid[key] = val

    def set_geom(self, geom):
        """ Sets the `Geometry` for the grid.

        Setting the `Geometry` for the grid is a possibility
        to attach atoms to the grid.

        It is not a necessary entity.
        """
        if geom is None:
            # Fake geometry
            self.set_geom(Geometry([0, 0, 0], Atom['H'], sc=self.sc))
        else:
            self.geom = geom
            self.set_sc(geom.sc)

    def interp(self, shape, method='linear', **kwargs):
        """ Returns an interpolated version of the grid

        Parameters
        ----------
        shape : int, array_like
            the new shape of the grid
        method : str
            the method used to perform the interpolation,
            see `scipy.interpolate.interpn` for further details.
        **kwargs :
            optional arguments passed to the interpolation algorithm
            The interpolation routine is `scipy.interpolate.interpn`
        """
        # Get current grid spacing
        dold = (
            np.linspace(0, 1, self.shape[0]),
            np.linspace(0, 1, self.shape[1]),
            np.linspace(0, 1, self.shape[2])
        )

        # Interpolate
        from scipy.interpolate import interpn

        # Create new grid
        grid = self.__class__(shape, bc=np.copy(self.bc), sc=self.sc.copy())
        # Clean-up to reduce memory
        del grid.grid

        # Create new mesh-grid
        dnew = np.concatenate(np.meshgrid(
            np.linspace(0, 1, shape[0]),
            np.linspace(0, 1, shape[1]),
            np.linspace(0, 1, shape[2])), axis=0)
        dnew.shape = (-1, 3)

        grid.grid = interpn(dold, self.grid, dnew, method=method, **kwargs)
        # immediately delete the dnew (which is VERY large)
        del dold, dnew
        # Ensure that the grid has the correct shape
        grid.grid.shape = tuple(shape)

        return grid

    @property
    def size(self):
        """ Returns size of the grid """
        return np.prod(self.grid.shape)

    @property
    def shape(self):
        """ Returns the shape of the grid """
        return self.grid.shape

    @property
    def dtype(self):
        """ Returns the data-type of the grid """
        return self.grid.dtype

    def set_grid(self, shape, dtype=None):
        """ Create the internal grid of certain size.
        """
        if dtype is None:
            dtype = np.float64
        self.grid = np.zeros(shape, dtype=dtype)

    def set_bc(self, boundary=None, a=None, b=None, c=None):
        """ Set the boundary conditions on the grid
        boundary: [3], integer, optional
           boundary condition for all boundaries (or the same for all)
        a: integer, optional
           boundary condition for the first unit-cell vector direction
        b: integer, optional
           boundary condition for the second unit-cell vector direction
        c: integer, optional
           boundary condition for the third unit-cell vector direction
        """
        if not boundary is None:
            if isinstance(boundary, Integral):
                self.bc = np.array([boundary] * 3, np.int32)
            else:
                self.bc = np.asarray(boundary, np.int32)
        if not a is None:
            self.bc[0] = a
        if not b is None:
            self.bc[1] = b
        if not c is None:
            self.bc[2] = c

    # Aliases
    set_boundary = set_bc
    set_boundary_condition = set_bc

    def copy(self):
        """
        Returns a copy of the object.
        """
        grid = self.__class__(np.copy(self.shape), bc=np.copy(self.bc),
                              dtype=self.dtype,
                              geom=self.geom.copy())
        grid.grid[:, :, :] = self.grid[:, :, :]
        return grid

    def swapaxes(self, a, b):
        """ Returns Grid with swapped axis

        If ``swapaxes(0,1)`` it returns the 0 in the 1 values.
        """
        # Create index vector
        idx = np.arange(3)
        idx[b] = a
        idx[a] = b
        s = np.copy(self.shape)
        grid = self.__class__(s[idx], bc=self.bc[idx],
                              sc=self.sc.swapaxes(a, b), dtype=self.dtype,
                              geom=self.geom.copy())
        # We need to force the C-order or we loose the contiguity
        grid.grid = np.copy(np.swapaxes(self.grid, a, b), order='C')
        return grid

    @property
    def dcell(self):
        """ Returns the delta-cell """
        # Calculate the grid-distribution
        dcell = np.empty([3, 3], np.float64)
        shape = self.shape
        dcell[0, :] = self.cell[0, :] / shape[0]
        dcell[1, :] = self.cell[1, :] / shape[1]
        dcell[2, :] = self.cell[2, :] / shape[2]
        return dcell

    @property
    def dvol(self):
        """ Returns the delta-volume """
        return self.sc.vol / self.size

    def cross_section(self, idx, axis):
        """ Takes a cross-section of the grid along axis ``axis``

        Remark: This API entry might change to handle arbitrary
        cuts via rotation of the axis """
        idx = np.array(idx, np.int32).flatten()
        # First calculate the new shape
        shape = list(self.shape)
        cell = np.copy(self.cell)
        # Down-scale cell
        cell[axis, :] /= shape[axis]
        shape[axis] = 1
        grid = self.__class__(shape, bc=np.copy(self.bc), geom=self.geom.copy())
        # Update cell shape (the cell is smaller now)
        grid.set_sc(cell)

        if axis == 0:
            grid.grid[:, :, :] = self.grid[idx, :, :]
        elif axis == 1:
            grid.grid[:, :, :] = self.grid[:, idx, :]
        elif axis == 2:
            grid.grid[:, :, :] = self.grid[:, :, idx]
        else:
            raise ValueError('Unknown axis specification in cross_section')

        return grid

    def sum(self, axis):
        """ Returns the grid summed along axis ``axis``. """
        # First calculate the new shape
        shape = list(self.shape)
        cell = np.copy(self.cell)
        # Down-scale cell
        cell[axis, :] /= shape[axis]
        shape[axis] = 1

        grid = self.__class__(shape, bc=np.copy(self.bc), geom=self.geom.copy())
        # Update cell shape (the cell is smaller now)
        grid.set_sc(cell)

        # Calculate sum (retain dimensions)
        grid.grid[:, :, :] = np.sum(self.grid, axis=axis, keepdims=True)
        return grid

    def average(self, axis):
        """ Returns the average grid along direction ``axis`` """
        n = self.shape[axis]
        g = self.sum(axis)
        g /= float(n)
        return g

    # for compatibility
    mean = average

    def remove_part(self, idx, axis, above):
        """ Removes parts of the grid via above/below designations.

        Works exactly opposite to `sub_part`

        Parameters
        ----------
        idx : array_like
           the indices of the grid axis ``axis`` to be removed
           for ``above=True`` grid[:idx,...]
           for ``above=False`` grid[idx:,...]
        axis : int
           the axis segment from which we retain the indices `idx`
        above: bool
           if `True` will retain the grid:
              `grid[:idx,...]`
           else it will retain the grid:
              `grid[idx:,...]`
        """
        return self.sub_part(idx, axis, not above)

    def sub_part(self, idx, axis, above):
        """ Retains parts of the grid via above/below designations.

        Works exactly opposite to `remove_part`

        Parameters
        ----------
        idx : array_like
           the indices of the grid axis ``axis`` to be retained
           for ``above=True`` grid[idx:,...]
           for ``above=False`` grid[:idx,...]
        axis : int
           the axis segment from which we retain the indices ``idx``
        above: bool
           if ``True`` will retain the grid:
              ``grid[idx:,...]``
           else it will retain the grid:
              ``grid[:idx,...]``
        """
        if above:
            sub = np.arange(idx, self.shape[axis])
        else:
            sub = np.arange(0, idx)
        return self.sub(sub, axis)

    def sub(self, idx, axis):
        """ Retains certain indices from a specified axis.

        Works exactly opposite to `remove`.

        Parameters
        ----------
        idx : array_like
           the indices of the grid axis ``axis`` to be retained
        axis : int
           the axis segment from which we retain the indices ``idx``
        """
        idx = np.array([idx], np.int32).flatten()
        uidx = np.unique(np.clip(idx, 0, self.shape[axis] - 1))

        # Calculate new shape
        shape = list(self.shape)
        cell = np.copy(self.cell)
        old_N = shape[axis]

        # Calculate new shape
        shape[axis] = len(idx)
        if shape[axis] < 1:
            raise ValueError('You cannot retain no indices.')

        # Down-scale cell
        cell[axis, :] = cell[axis, :] / old_N * shape[axis]

        grid = self.__class__(shape, bc=np.copy(self.bc), geom=self.geom.copy())
        # Update cell shape (the cell is smaller now)
        grid.set_sc(cell)

        # Remove the indices
        # First create the opposite, index
        if axis == 0:
            grid.grid[:, :, :] = self.grid[idx, :, :]
        elif axis == 1:
            grid.grid[:, :, :] = self.grid[:, idx, :]
        elif axis == 2:
            grid.grid[:, :, :] = self.grid[:, :, idx]

        return grid

    def remove(self, idx, axis):
        """ Removes certain indices from a specified axis.

        Works exactly opposite to `sub`.

        Parameters
        ----------
        idx : array_like
           the indices of the grid axis ``axis`` to be removed
        axis : int
           the axis segment from which we remove all indices ``idx``
        """
        uidx = np.unique(np.clip(idx, 0, self.shape[axis] - 1))
        ret_idx = np.setdiff1d(
            np.arange(self.shape[axis]),
            uidx, assume_unique=True)
        return self.sub(ret_idx, axis)

    def index(self, coord, axis=None):
        """ Returns the index along axis ``axis`` where ``coord`` exists

        Parameters
        ----------
        coord : array_like / float
           the coordinate of the axis
        axis : int
           the axis direction of the index
        """

        # if the axis is none, we do this for all axes
        if axis is None:
            rcell = self.rcell / (2. * np.pi)
            # Loop over each direction
            idx = np.empty([3], np.int32)
            for i in [0, 1, 2]:
                # get the coordinate along the direction of the cell vector
                c = np.dot(rcell[i, :], coord) * self.cell[i, :]
                # Calculate the index corresponding to this placement
                idx[i] = self.index(c, i)
            return idx

        # Ensure a 1D array
        ac = np.atleast_1d(np.asarray(coord, np.float64))

        # Calculate the index of the coord in the cell
        dax = self.dcell[axis, :]

        # Calculate how many indices are required to fulfil
        # the correct line cut
        return int(np.rint((np.sum(ac ** 2) / np.sum(dax ** 2)) ** .5))

    def append(self, other, axis):
        """ Appends other `Grid` to this grid along axis

        """
        shape = list(self.shape)
        shape[axis] += other.shape[axis]
        return self.__class__(shape, bc=np.copy(self.bc),
                              geom=self.geom.append(other.geom, axis))

    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads grid from the `Sile` using `read_grid`

        Parameters
        ----------
        sile : Sile, str
            a `Sile` object which will be used to read the grid
            if it is a string it will create a new sile using `get_sile`.
        * : args passed directly to ``read_grid(,**)``
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            return sile.read_grid(*args, **kwargs)
        else:
            return get_sile(sile).read_grid(*args, **kwargs)

    def write(self, sile):
        """ Writes grid to the `Sile` using `write_grid`

        Parameters
        ----------
        sile : Sile, str
            a `Sile` object which will be used to write the grid
            if it is a string it will create a new sile using `get_sile`
        """

        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            sile.write_grid(self)
        else:
            get_sile(sile, 'w').write_grid(self)

    def __repr__(self):
        """ Representation of object """

        return 'Grid[{} {} {}]'.format(*self.shape)

    def _check_compatibility(self, other, msg):
        """ Internal check for asserting two grids are commensurable """
        if self == other:
            return True
        s1 = repr(self)
        s2 = repr(other)
        raise ValueError('Grids are not compatible, ' +
                         s1 + '-' + s2 + '. ', msg)

    def _compatible_copy(self, other, *args, **kwargs):
        """ Returns a copy of self with an additional check of commensurable """
        if isinstance(other, Grid):
            if self._check_compatibility(other, *args, **kwargs):
                pass
        return self.copy()

    def __eq__(self, other):
        """ Returns true if the two grids are commensurable

        There will be no check of the values _on_ the grid. """
        a = np.array
        return bool(np.all(a(self.shape, np.int32) == a(other.shape, np.int32)))

    def __ne__(self, other):
        """ Returns whether two grids have the same shape """
        return not (self == other)

    def __add__(self, other):
        """ Returns a new grid with the addition of two grids

        Returns same shape with same cell as the first"""
        if isinstance(other, Grid):
            grid = self._compatible_copy(other, 'they cannot be added')
            grid.grid = self.grid + other.grid
        else:
            grid = self.copy()
            grid.grid = self.grid + other
        return grid

    def __iadd__(self, other):
        """ Returns a new grid with the addition of two grids

        Returns same shape with same cell as the first"""
        if isinstance(other, Grid):
            self._check_compatibility(other, 'they cannot be added')
            self.grid += other.grid
        else:
            self.grid += other
        return self

    def __sub__(self, other):
        """ Returns a new grid with the difference of two grids

        Returns same shape with same cell as the first"""
        if isinstance(other, Grid):
            grid = self._compatible_copy(other, 'they cannot be subtracted')
            grid.grid = self.grid - other.grid
        else:
            grid = self.copy()
            grid.grid = self.grid - other
        return grid

    def __isub__(self, other):
        """ Returns a same grid with the difference of two grids

        Returns same shape with same cell as the first"""
        if isinstance(other, Grid):
            self._check_compatibility(other, 'they cannot be subtracted')
            self.grid -= other.grid
        else:
            self.grid -= other
        return self

    def __div__(self, other):
        return self.__truediv__(other)

    def __idiv__(self, other):
        return self.__itruediv__(other)

    def __truediv__(self, other):
        if isinstance(other, Grid):
            grid = self._compatible_copy(other, 'they cannot be divided')
            grid.grid = self.grid / other.grid
        else:
            grid = self.copy()
            grid.grid = self.grid / other
        return grid

    def __itruediv__(self, other):
        if isinstance(other, Grid):
            self._check_compatibility(other, 'they cannot be divided')
            self.grid /= other.grid
        else:
            self.grid /= other
        return self

    def __mul__(self, other):
        if isinstance(other, Grid):
            grid = self._compatible_copy(other, 'they cannot be multiplied')
            grid.grid = self.grid * other.grid
        else:
            grid = self.copy()
            grid.grid = self.grid * other
        return grid

    def __imul__(self, other):
        if isinstance(other, Grid):
            self._check_compatibility(other, 'they cannot be multiplied')
            self.grid *= other.grid
        else:
            self.grid *= other
        return self

    @classmethod
    def _ArgumentParser_args_single(cls):
        """ Returns the options for `Grid.ArgumentParser` in case they are the only options """
        return {'limit_arguments': False,
                'short': True,
                'positional_out': True,
            }

    # Hook into the Grid class to create
    # an automatic ArgumentParser which makes actions
    # as the options are read.
    def ArgumentParser(self, parser=None, *args, **kwargs):
        """ Create and return a group of argument parsers which manipulates it self `Grid`. 

        Parameters
        ----------
        parser: ArgumentParser, None
           in case the arguments should be added to a specific parser. It defaults
           to create a new.
        limit_arguments: bool, True
           If `False` additional options will be created which are similar to other options.
           For instance `--repeat-x` which is equivalent to `--repeat x`.
        short: bool, False
           Create short options for a selected range of options
        positional_out: bool, False
           If `True`, adds a positional argument which acts as --out. This may be handy if only the geometry is in the argument list.
        """

        limit_args = kwargs.get('limit_arguments', True)
        short = kwargs.get('short', False)

        def opts(*args):
            if short:
                return args
            return [args[0]]

        # We limit the import to occur here
        import argparse

        if parser is None:
            p = argparse.ArgumentParser("Manipulate a Grid object in sisl.")
        else:
            p = parser

        # The first thing we do is adding the Grid to the NameSpace of the
        # parser.
        # This will enable custom actions to interact with the grid in a
        # straight forward manner.
        d = {
            "_grid": self.copy(),
            "_stored_grid": False,
        }
        namespace = default_namespace(**d)

        # Define actions
        class SetGeometry(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                ns._geometry = Geometry.read(value)
                ns._grid.set_geom(ns._geometry)
        p.add_argument(*opts('--geometry', '-G'), action=SetGeometry,
                       help='Define the geometry attached to the Grid.')

        # Define size of grid
        class InterpGrid(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                ns._grid = ns._grid.interp([int(x) for x in values])
        p.add_argument(*opts('--interp'), nargs=3,
                       action=InterpGrid,
                       help='Interpolate the grid.')

        # substract another grid
        # They *MUST* be conmensurate.
        class DiffGrid(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                grid = Grid.read(value)
                ns._grid -= grid
                del grid
        p.add_argument(*opts('--diff', '-d'), action=DiffGrid,
                       help='Subtract another grid (they must be commensurate).')

        class AverageGrid(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                ns._grid = ns._grid.average(direction(value))
        p.add_argument(*opts('--average'), metavar='DIR',
                       action=AverageGrid,
                       help='Take the average of the grid along DIR.')

        # Create-subsets of the grid
        class SubDirectionGrid(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                # The unit-cell direction
                axis = direction(values[0])
                # Figure out whether this is a fractional or
                # distance in Ang
                is_frac = 'f' in values[1]
                rng = strseq(float, values[1].replace('f', ''))
                if isinstance(rng, tuple):
                    if is_frac:
                        t = [ns._grid.cell[axis, :] * r for r in rng]
                        rng = tuple(rng)
                    # we have bounds
                    idx1 = ns._grid.index(rng[0], axis=axis)
                    idx2 = ns._grid.index(rng[1], axis=axis)
                    ns._grid = ns._grid.sub(range(idx1, idx2+1), d)
                    return
                elif rng < 0.:
                    if is_frac:
                        rng = ns._grid.cell[axis, :] * abs(rng)
                    b = False
                else:
                    if is_frac:
                        rng = ns._grid.cell[axis, :] * rng
                    b = True
                idx = ns._grid.index(rng, axis=axis)
                ns._grid = ns._grid.sub_part(idx, axis, b)
        p.add_argument(*opts('--sub'), nargs=2, metavar=('DIR', 'COORD'),
                       action=SubDirectionGrid,
                       help='Reduce the grid by taking a subset of the grid (along DIR).')

        # Create-subsets of the grid
        class RemoveDirectionGrid(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                # The unit-cell direction
                axis = direction(values[0])
                # Figure out whether this is a fractional or
                # distance in Ang
                is_frac = 'f' in values[1]
                rng = strseq(float, values[1].replace('f', ''))
                if isinstance(rng, tuple):
                    raise NotImplementedError('Can not figure out how to apply mid-removal of grids.')
                elif rng < 0.:
                    if is_frac:
                        rng = ns._grid.cell[axis, :] * abs(rng)
                    b = True
                else:
                    if is_frac:
                        rng = ns._grid.cell[axis, :] * rng
                    b = False
                idx = ns._grid.index(rng, axis=axis)
                ns._grid = ns._grid.sub_part(idx, axis, b)
        p.add_argument(*opts('--remove'), nargs=2, metavar=('DIR', 'COORD'),
                       action=RemoveDirectionGrid,
                       help='Reduce the grid by removing a subset of the grid (along DIR).')

        # Define size of grid
        class PrintInfo(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                print(ns._grid)
        p.add_argument(*opts('--info'), nargs=0,
                       action=PrintInfo,
                       help='Print, to stdout, some regular information about the grid.')

        class Out(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                if value is None:
                    return
                if len(value) == 0:
                    return
                ns._grid.write(value[0])
                # Issue to the namespace that the geometry has been written, at least once.
                ns._stored_grid = True
        p.add_argument(*opts('--out', '-o'), nargs=1, action=Out,
                       help='Store the grid (at its current invocation) to the out file.')

        # If the user requests positional out arguments, we also add that.
        if kwargs.get('positional_out', False):
            p.add_argument('out', nargs='*', default=None,
                           action=Out,
                           help='Store the grid (at its current invocation) to the out file.')

        # We have now created all arguments
        return p, namespace


def sgrid(grid=None, argv=None, ret_grid=False):
    """ Main script for sgrid script. 

    This routine may be called with `argv` and/or a `Sile` which is the grid at hand.

    Parameters
    ----------
    grid : `Grid`/`BaseSile`
       this may either be the grid, as-is, or a `Sile` which contains
       the grid.
    argv : `list of str`
       the arguments passed to sgeom
    ret_grid : `bool` (`False`)
       whether the function should return the grid
    """
    import sys
    import os.path as osp
    import argparse

    from sisl.io import get_sile, BaseSile

    # The file *MUST* be the first argument
    # (except --help|-h)

    # We cannot create a separate ArgumentParser to retrieve a positional arguments
    # as that will grab the first argument for an option!

    # Start creating the command-line utilities that are the actual ones.
    description = """
This manipulation utility is highly advanced and one should note that the ORDER of
options is determining the final structure. For instance:

   {0} ElectrostaticPotential.grid.nc --diff Other.grid.nc --sub z 0.:0.2f

is NOT equivalent to:

   {0} ElectrostaticPotential.grid.nc --sub z 0.:0.2f --diff Other.grid.nc

This may be unexpected but enables one to do advanced manipulations.
    """.format(osp.basename(sys.argv[0]))

    if argv is not None:
        if len(argv) == 0:
            argv = ['--help']
    elif len(sys.argv) == 1:
        # no arguments
        # fake a help
        argv = ['--help']
    else:
        argv = sys.argv[1:]

    # Ensure that the arguments have pre-pended spaces
    argv = cmd.argv_negative_fix(argv)

    p = argparse.ArgumentParser('Manipulates real-space grids in commonly encounterd files.',
                           formatter_class=argparse.RawDescriptionHelpFormatter,
                           description=description)

    # First read the input "Sile"
    if grid is None:
        argv, input_file = cmd.collect_input(argv)
        try:
            grid = get_sile(input_file).read_grid()
        except:
            grid = Grid([10, 10, 10])

    elif isinstance(grid, Grid):
        # Do nothing, the geometry is already created
        argv = ['fake.grid.nc'] + argv
        pass

    elif isinstance(grid, BaseSile):
        try:
            grid = sile.read_grid()
            # Store the input file...
            input_file = grid.file
        except Exception as E:
            grid = Grid([10, 10, 10])
        argv = ['fake.grid.nc'] + argv

    # Do the argument parser
    p, ns = grid.ArgumentParser(p, **grid._ArgumentParser_args_single())

    # Now the arguments should have been populated
    # and we will sort out if the input options
    # is only a help option.
    try:
        if not hasattr(ns, '_input_file'):
            setattr(ns, '_input_file', input_file)
    except:
        pass

    # Now try and figure out the actual arguments
    p, ns, argv = cmd.collect_arguments(argv, input=False,
                                        argumentparser=p,
                                        namespace=ns)

    # We are good to go!!!
    args = p.parse_args(argv, namespace=ns)
    g = args._grid

    if not args._stored_grid:
        # We should write out the information to the stdout
        # This is merely for testing purposes and may not be used for anything.
        print(g)

    if ret_grid:
        return g
    return 0
