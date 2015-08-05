""" Define a supercell 

This class is the basis of many different objects.
"""
from __future__ import print_function, division

import numpy as np

from .quaternion import Quaternion

__all__ = ['SuperCell','SuperCellChild']


# Default nsc variable
_nsc = np.array([1]*3,np.int32)
_dtype = np.float64

class SuperCell(object):
    """ Object to retain a super-cell and its nested values.

    This supercell object handles cell vectors and its supercell mirrors.
    """

    # We limit the scope of this SuperCell object.
    __slots__ = ['cell','vol','nsc','n_s','sc_off']


    def __init__(self,cell,nsc=_nsc):
        """ Initialize a `SuperCell` object from initial quantities
        
        Initialize a `SuperCell` object with cell information
        and number of supercells in each direction.
        """
        # Store the actual 3D cell
        self.cell = np.asarray(cell,np.float64)
        if len(self.cell.shape) == 1:
            # The cell is a diagonal entry cell
            self.cell = np.diag(self.cell)

        # Set the volume
        self.vol = np.abs(np.dot(self.cell[0,:],
                                 np.cross(self.cell[1,:], self.cell[2,:])
                                 )
                          )

        # Set the super-cell
        self.set_nsc(nsc=nsc)


    def set_nsc(self,nsc=None,a=None,b=None,c=None):
        """ Sets the number of supercells in the 3 different cell directions

        nsc: [3], integer, optional
           number of supercells in each direction
        a: integer, optional
           number of supercells in the first unit-cell vector direction
        b: integer, optional
           number of supercells in the second unit-cell vector direction
        c: integer, optional
           number of supercells in the third unit-cell vector direction
        """
        if not nsc is None:
            self.nsc = np.asarray(nsc,np.int32)
        if a: self.nsc[0] = a
        if b: self.nsc[1] = b
        if c: self.nsc[2] = c
        # Correct for misplaced number of unit-cells
        for i in range(3):
            if self.nsc[i] == 0: self.nsc[i] = 1
        if np.sum(self.nsc % 2) != 3 :
            raise ValueError("Supercells has to be of un-even size. The primary cell counts "+
                             "one, all others count 2")

        # We might use this very often, hence we store it
        self.n_s = np.prod(self.nsc)
        self.sc_off = np.zeros([self.n_s,3],np.int32)

        n = self.nsc
        # We define the following ones like this:
        i = n[0] // 2 ; x = range(-i,i+1)
        i = n[1] // 2 ; y = range(-i,i+1)
        i = n[2] // 2 ; z = range(-i,i+1)
        i = 0
        for iz in z:
            for iy in y:
                for ix in x:
                    if ix == 0 and iy == 0 and iz == 0:
                        continue
                    # Increment index
                    i += 1
                    # The offsets for the supercells in the
                    # sparsity pattern
                    self.sc_off[i,0] = ix
                    self.sc_off[i,1] = iy
                    self.sc_off[i,2] = iz

    # Aliases
    set_supercell = set_nsc


    def copy(self):
        """
        Returns a copy of the object.
        """
        return self.__class__(np.copy(self.cell), nsc = np.copy(self.nsc))

    
    def swapaxes(self,a,b):
        """ Returns `SuperCell` with swapped axis
        
        If `swapaxes(0,1)` it returns the 0 in the 1 values.
        """
        # Create index vector
        idx = np.arange(3)
        idx[b] = a
        idx[a] = b
        return self.__class__(self.cell[idx,:], nsc = self.nsc[idx])

            
    def rotate(self,angle,v,degree=False):
        """ 
        Rotates the geometry, in-place by the angle around the vector

        Per default will the entire geometry be rotated, such that everything
        is aligned as before rotation.

        However, by supplying ``only='cell|xyz'`` one can designate which
        part of the geometry that will be rotated.
        
        Parameters
        ----------
        angle : float
             the angle in radians of which the geometry should be rotated
        v     : array_like [3]
             the vector around the rotation is going to happen
             v = [1,0,0] will rotate in the ``yz`` plane
        degree : bool
             Whether the angle is in radians (False) or in degrees (True)
        """
        q = Quaternion(angle,v,degree=degree)
        q /= q.norm() # normalize the quaternion
        cell = q.rotate( self.cell )
        return self.__class__(cell, nsc = np.copy(self.nsc))


    def offset(self,isc=[0,0,0]):
        """ Returns the supercell offset of the supercell index
        """
        return self.cell[0,:] * isc[0] + \
            self.cell[1,:] * isc[1] + \
            self.cell[2,:] * isc[2]

    
    def sc_index(self,sc_off):
        """ Returns the integer index in the sc_off list that corresponds to `sc_off`

        Returns the integer for the supercell
        """
        for i in range(self.n_s):
            if sc_off[0] == self.sc_off[i,0] and \
               sc_off[1] == self.sc_off[i,1] and \
               sc_off[2] == self.sc_off[i,2]:
                return i
        #idx = np.where(self.sc_off[:,0] == sc_off[0])[0]
        #if len(idx) > 0:
        #    idx = idx[np.where(self.sc_off[idx,1] == sc_off[1])[0]]
        #if len(idx) > 0:
        #    idx = idx[np.where(self.sc_off[idx,2] == sc_off[2])[0]]
        #if len(idx) == 1:
        #    return idx[0]
        raise Exception('Could not find supercell index, number of super-cells not big enough')


    def cut(self,seps,axis):
        """ Cuts the cell into several different sections.
        """
        cell = np.copy(self.cell)
        cell[axis,:] /= seps
        return self.__class__(cell, np.copy(self.nsc))
        
    
    def append(self,other,axis):
        """ Appends other `SuperCell` to this grid along axis

        """
        cell = np.copy(self.cell)
        cell[axis,:] += other.cell[axis,:]
        return self.__class__(cell, nsc = np.copy(self.nsc) )

    
    def translate(self,v):
        """ Appends additional space in the SuperCell object
        """
        # check which cell vector resembles v the most,
        # use that
        cell = np.copy(self.cell)
        p = np.empty([3],np.float64)
        for i in range(3):
            p[i] = abs(np.sum(cell[i,:] * v)) / np.sum(cell[i,:]**2)**.5
        cell[np.argmax(p),:] += v
        return self.__class__(cell, np.copy(self.nsc))


    def center(self,axis=None):
        """ Returns center of the `SuperCell`, possibly with respect to an axis
        """
        if axis is None:
            return np.mean(self.cell,axis=0)
        return self.cell[axis,:] / 2


class SuperCellChild(object):
    """ Class to be inherited by using the `self.sc` as a `SuperCell` object

    Initialize by a `SuperCell` object and get access to several different
    routines directly related to the `SuperCell` class.
    """

    def set_supercell(self,sc):
        """ Overwrites the local supercell """
        self.sc = sc

    @property
    def vol(self):
        """ Returns the inherent `SuperCell` objects `vol` """
        return self.sc.vol
        
    @property
    def cell(self):
        """ Returns the inherent `SuperCell` objects `cell` """
        return self.sc.cell

    @property
    def n_s(self):
        """ Returns the inherent `SuperCell` objects `n_s` """
        return self.sc.n_s

    @property
    def nsc(self):
        """ Returns the inherent `SuperCell` objects `nsc` """
        return self.sc.nsc

    @property
    def sc_off(self):
        """ Returns the inherent `SuperCell` objects `sc_off` """
        return self.sc.sc_off


    def sc_index(self,*args,**kwargs):
        """ Call local `SuperCell` object `sc_index` function """
        return self.sc.sc_index(*args,**kwargs)

    
if __name__ == "__main__":
    pass
