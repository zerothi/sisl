""" Define a grid

This grid is the basis of different used models.
"""
from __future__ import print_function, division

from numbers import Integral

import numpy as np

from .quaternion import Quaternion
from .supercell import SuperCellChild
from .atom import Atom
from .geometry import Geometry

__all__ = ['Grid']


# Default nsc variable
_nsc = np.array([1]*3,np.int32)
_size = np.array([0]*3,np.int32)
_bc = 1 # Periodic (Grid.Periodic)
_dtype = np.float64

class Grid(SuperCellChild):
    """ Object to retain grid information

    This grid object handles cell vectors and divisions of said grid.

    A grid can be periodic and non-periodic.
    """

    # Constant (should never be changed)
    Periodic = 1
    Neumann = 2
    Dirichlet = 3
    
    def __init__(self,size=_size,bc=_bc,sc=None,dtype=_dtype,geom=None):
        """ Initialize a `Grid` object.
        
        Initialize a `Grid` object.
        """

        self.set_supercell(sc)

        # Create the grid
        self.set_grid(size,dtype=dtype)

        # Create the grid boundary conditions
        self.set_bc(bc)

        # Create the atomic structure in the grid, if possible
        self.set_geom(geom)

        
    @property
    def size(self):
        """ Returns size of the grid """
        return self.grid.shape

    
    def set_grid(self,size,dtype=_dtype):
        """ Create the internal grid of certain size.
        """
        self.grid = np.empty(size,dtype=dtype)


    def set_bc(self,boundary=None,a=None,b=None,c=None):
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
            if isinstance(boundary,Integral):
                self.bc = np.array([boundary]*3,np.int32)
            else:
                self.bc = np.asarray(boundary,np.int32)
        if not a is None: self.bc[0] = a
        if not b is None: self.bc[1] = b
        if not c is None: self.bc[2] = c
        
    # Aliases
    set_boundary = set_bc
    set_boundary_condition = set_bc

    def set_geom(self,geom):
        """ Sets the `Geometry` for the grid.

        Setting the `Geometry` for the grid is a possibility
        to attach atoms to the grid.

        It is not a necessary entity.
        """
        if geom is None:
            # Fake geometry
            self.set_geom(Geometry([0,0,0],Atom['H'],sc=self.sc))
        else:
            self.geom = geom
            self.set_sc(geom.sc)
                          

    def copy(self):
        """
        Returns a copy of the object.
        """
        grid = self.__class__(np.copy(self.size), bc=np.copy(self.bc),
                              sc=self.sc.copy())
        grid.grid[:,:,:] = self.grid[:,:,:]
        return grid


    def swapaxes(self,a,b):
        """ Returns Grid with swapped axis
        
        If ``swapaxes(0,1)`` it returns the 0 in the 1 values.
        """
        # Create index vector
        idx = np.arange(3)
        idx[b] = a
        idx[a] = b
        s = np.copy(self.size)
        grid = self.__class__(s[idx], bc=self.bc[idx],
                              sc=self.sc.swapaxes(a,b))
        grid.grid = np.copy(np.swapaxes(self.grid,a,b))
        return grid


    @property
    def dcell(self):
        """ Returns the delta-cell """
        # Calculate the grid-distribution
        g_size = self.size
        dcell = _np.empty([3,3],np.float64)
        for ix in range(3):
            dcell[ix,:] = self.cell[ix,:] / g_size[ix]
        return dcell
            
    def append(self,other,axis):
        """ Appends other `Grid` to this grid along axis

        """
        size = np.copy(self.size)
        size[axis] += other.size[axis]
        return self.__class__(size, bc = np.copy(self.bc),
                              sc=self.sc.append(other.sc,axis))
    
    def write(self,sile):
        """ Writes grid to the ``Sile`` using ``sile.write_grid``

        Parameters
        ----------
        sile : Sile, str
            a ``Sile`` object which will be used to write the grid
            if it is a string it will create a new sile using ``get_sile``
        """

        # This only works because, they *must*
        # have been imported previously
        from sids.io import get_sile, BaseSile
        if isinstance(sile,BaseSile):
            sile.write_grid(self)
        else:
            get_sile(sile,'w').write_grid(self)


    def __repr__(self):
        """ Representation of object """
        return 'Grid[{} {} {}]'.format(*self.size)

    
if __name__ == "__main__":
    pass
