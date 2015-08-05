""" Define a grid

This grid is the basis of different used models.
"""
from numbers import Integral

import numpy as np

from .quaternion import Quaternion
from .supercell import SuperCellChild

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
    
    def __init__(self,size=_size,bc=_bc,sc=None):
        """ Initialize a `Grid` object.
        
        Initialize a `Grid` object.
        """

        if sc is None:
            # Create fake super-cell of zero size
            self.set_supercell(SuperCell([1.,1.,1.]))
        else:
            self.set_supercell(sc)

        # Create the grid
        self.set_grid(size)

        # Create the grid boundary conditions
        self.set_bc(bc)
        

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


    def copy(self):
        """
        Returns a copy of the object.
        """
        return self.__class__(np.copy(self.size), bc=np.copy(self.bc),
                              sc=self.sc.copy())


    def swapaxes(self,a,b):
        """ Returns Grid with swapped axis
        
        If ``swapaxes(0,1)`` it returns the 0 in the 1 values.
        """
        # Create index vector
        idx = np.arange(3)
        idx[b] = a
        idx[a] = b
        return self.__class__(self.size[idx], bc=self.bc[idx],
                              sc=self.sc.swapaxes(a,b))

            
    def append(self,other,axis):
        """ Appends other `Grid` to this grid along axis

        """
        size = np.copy(self.grid.shape)
        size[axis] += other.grid.shape[axis]
        return self.__class__(size, bc = np.copy(self.bc),
                              sc=self.sc.append(other.sc,axis))


if __name__ == "__main__":
    pass
