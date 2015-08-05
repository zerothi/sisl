"""
Geometry class to retain the atomic structure.
"""
from __future__ import print_function, division

# To check for integers
from numbers import Integral
from math import acos, pi
import sys

import numpy as np

from .quaternion import Quaternion
from .supercell import SuperCell, SuperCellChild
from .atom import Atom
from ._help import array_fill_repeat

__all__ = ['Geometry']


# Local default variables for the __init__ of the
# Geometry class
_H = Atom['H']
_nsc = np.array([1]*3,np.int32)


class Geometry(SuperCellChild):
    """
    Geometry object handling atomic coordinates in a supercell
    fashion.

    Every geometry deals with these information:
      - cell, the unit cell of the geometry
      - xyz, the atomic coordinates
      - atoms, the atoms associated with the coordinates
    
    All lengths are assumed to be in units of Angstrom, however, as
    long as units are kept same the exact units are irrespective.
    
    Parameters
    ----------
    sc  : `SuperCell`
    xyz   : array_like
        atomic coordinates in a Nx3 matrix.
        ``xyz[i,:]`` is the atomic coordinate of the i'th atom.
        This atomic coordinate need not be inside the unit-cell.
    atoms : (Atom['H']), list of atoms associated
        These atoms constitute the atomic ranges, weights, orbitals
        etc. for the geometry.

    Attributes
    ----------
    sc : The supercell object
    na/len(self) : integer
        Number of atoms.
    xyz  : (na,3) ndarray
        Atomic coordinates.
    atoms  : (na)
        The atomic objects associated with each atom.
    no: sum([a.orbs for a in self.atoms])
        Total number of orbitals 
    dR   : float np.max([a.dR for a in self.atoms])
        Maximum orbital range.
    nsc  : (3) ndarray
        Total number of supercells in each direction
    cell : (3,3) ndarray
        Cell vectors, property locating the `sc.cell` variable
    """

    # The length conversion factor
    Length = 0.529177

    def __init__(self,xyz,atoms=_H,sc=None):

        if sc is None:
            # Create fake super-cell of zero size
            self.set_supercell(SuperCell([1.,1.,1.]))
        else:
            self.set_supercell(sc)

        # Create the geometry coordinate
        self.xyz = np.asarray(xyz)
        self.xyz.shape = (-1, 3)
        self.na = len(self.xyz)

        # Correct the atoms input to Atom
        if isinstance(atoms, list):
            if isinstance(atoms[0],str):
                atoms = np.array([Atom[a] for a in atoms])
        elif isinstance(atoms, str):
            atoms = np.array([Atom[atoms]])

        # Create atom objects
        self.atoms = array_fill_repeat(atoms, self.na, cls=Atom)

        # Store maximum interaction range
        if isinstance(atoms, Atom):
            self.dR = atoms.dR
        else:
            self.dR = np.amax([a.dR for a in atoms])

        # Get total number of orbitals
        orbs = np.array([a.orbs for a in self.atoms], np.int32)

        # Get total number of orbitals
        self.no = np.sum(orbs)

        # Create local lasto
        lasto = np.append(np.array(0,np.int32), orbs)
        self.lasto = np.cumsum(lasto)

        
    def __len__(self):
        """ Return number of atoms in this geometry """
        return self.na


    def write(self,sile):
        """ Writes a geometry to the ``sile`` as implemented in the ``sile.write_geom``
        method """
        sile.write_geom(self)


    def __repr__(self):
        """ Representation of the object """
        spec = self._species_order()
        s = '{{na: {0}, no: {1}, species:\n {{ n: {2},\n   '.format(self.na,self.no,len(spec))
        for z in spec:
            s += '[{0}], '.format(str(spec[z][1]))
        return s[:-2] + '\n }},\n nsc: [{1}, {2}, {3}], dR: {0}\n}}'.format(self.dR,*self.nsc)

    
    def iter_species(self):
        """ 
        Returns an iterator over all atoms and species as a tuple in this geometry
        
         >>> for ia,a,idx_specie in self.iter_species():

        with ``ia`` being the atomic index, ``a`` the ``Atom`` object, ``idx_specie``
        is the index of the species
        """
        # Count for the species
        spec = []
        for ia,a in enumerate(self.atoms):
            if not a.tag in spec:
                spec.append(a.tag)
                yield ia,a,len(spec) - 1
            else:
                # It must already exist in the species list
                yield ia,a,spec.index(a.tag)


    def iter_linear(self):
        """
        Returns an iterator for simple linear ranges.

        This iterator is the same as:

          >>> for ia in range(len(self)):
          >>>    <do something>
        or equivalently
          >>> for ia in self:
          >>>    <do something>
        """
        for ia in range(len(self)):
            yield ia

    __iter__ = iter_linear


    def iter_block(self,iR=10,dR=None):
        """ 
        Returns an iterator for performance critical looping.
        
        Parameters
        ----------
        iR  : (10) integer
            the number of ``dR`` ranges taken into account when doing the iterator
        dR  : (self.dR), float
            enables overwriting the local dR quantity.
 
        Returns two lists with [0] being a list of atoms to be looped and [1] being the atoms that 
        need searched.

        NOTE: This requires that dR has been set correctly as the maximum interaction range.

        I.e. the loop would look like this:
        
        >>> for ias, idxs in Geometry.iter_block():
        >>>    for ia in ias:
        >>>        idx_a = dev.close(ia, dR = dR, idx = idxs)

        This iterator is intended for systems with more than 1000 atoms.

        Remark that the iterator used is non-deterministic, i.e. any two iterators need
        not return the same atoms in any way.
        """

        # We implement yields as we can then do nested iterators
        # create a boolean array
        na = len(self)
        not_passed = np.empty(na,dtype='b')
        not_passed[:] = True
        not_passed_N = na

        if dR is None:
            # The boundaries (ensure complete overlap)
            dr = ( self.dR * (iR - 1), self.dR * (iR+.1))
        else:
            dr = (      dR * (iR - 1),      dR * (iR+.1))


        # loop until all passed are true
        while not_passed_N > 0:
            
            # Take a random non-passed element
            all_true = np.where(not_passed)[0]
            # Shuffle should increase the chance of hitting a
            # completely "fresh" segment, thus we take the most 
            # atoms at any single time.
            # Shuffling will cut down needed iterations.
            np.random.shuffle(all_true)
            idx = all_true[0]
            del all_true

            # Now we have found a new index, from which
            # we want to create the index based stuff on
            
            # get all elements within two radii
            all_idx = self.close(idx, dR = dr )

            # Get unit-cell atoms
            all_idx[0] = self.sc2uc(all_idx[0],uniq=True)
            # First extend the search-space (before reducing)
            all_idx[1] = self.sc2uc(np.append(all_idx[1],all_idx[0]),uniq=True)

            # Only select those who have not been runned yet
            all_idx[0] = all_idx[0][np.where(not_passed[all_idx[0]])[0]]
            if len(all_idx[0]) == 0:
                raise ValueError('Internal error, please report to the developers')

            # Tell the next loop to skip those passed
            not_passed[all_idx[0]] = False
            # Update looped variables
            not_passed_N -= len(all_idx[0])

            # Now we want to yield the stuff revealed
            # all_idx[0] contains the elements that should be looped
            # all_idx[1] contains the indices that can be searched
            yield all_idx[0], all_idx[1]

        if np.any(not_passed):
            raise ValueError('Error on iterations. Not all atoms has been visited.')
    

    @property
    def no_s(self):
        """ Number of supercell orbitals """
        return self.no * self.n_s


    def sub(self,atoms,cell=None):
        """
        Returns a subset of atoms from the geometry.

        Indices passed *MUST* be unique.

        Parameters
        ----------
        atoms  : array_like
            indices of all atoms to be removed.
        cell   : (``self.cell`), array_like, optional
            the new associated cell of the geometry
        """
        atms = np.asarray([atoms],np.int32).flatten()
        if cell is None: 
            return self.__class__(self.xyz[atoms,:],
                                  atoms=[self.atoms[i] for i in atms], sc=self.sc.copy())
        return self.__class__(self.xyz[atoms,:],
                            atoms=[self.atoms[i] for i in atms], sc=cell)


    def cut(self,seps,axis):
        """
        Returns a subset of atoms from the geometry by cutting the 
        geometry into ``seps`` parts along the direction ``axis``.
        It will then _only_ return the first cut.
        
        This will effectively change the unit-cell in the ``axis`` as-well
        as removing ``self.na_u/seps`` atoms.
        It requires that ``self.na_u % seps == 0``.

        REMARK: You need to ensure that all atoms within the first 
        cut out region are within the primary unit-cell.

        Doing ``geom.cut(2,1).tile(reps=2,axis=1)``, could for symmetric setups,
        be equivalent to a no-op operation. A `UserWarning` will be issued
        if this is not the case.

        Parameters
        ----------
        axis  : integer
           the axis that will be cut
        seps  : integer, optional
           number of times the structure will be cut.
        """
        if self.na % seps != 0:
            raise ValueError('The system cannot be cut into {0} different '+
                             'pieces. Please check your geometry and input.'.format(seps))
        # Cut down cell
        sc = self.sc.cut(seps,axis)
        new = self.sub(np.arange(self.na//seps), cell=sc)
        if not np.allclose(new.tile(seps, axis).xyz, self.xyz):
            warnings.warn('The cut structure cannot be re-created by tiling', UserWarning) 
        return new


    def _species_order(self):
        """ Returns dictionary with species indices for the atoms.
        They will be populated in order of appearence"""

        # Count for the species
        spec = {}
        ispec = 0
        for a in self.atoms:
            if not a.tag is None:
                if not a.tag in spec:
                    ispec += 1
                    spec[a.tag] = (ispec,a)
            elif not a.Z in spec:
                ispec += 1
                spec[a.Z] = (ispec,a)
        return spec


    def copy(self):
        """
        Returns a copy of the object.
        """
        return self.__class__(np.copy(self.xyz),
                              atoms=self.atoms, sc=self.sc.copy())


    def remove(self,atoms):
        """
        Remove atoms from the geometry.

        Indices passed *MUST* be unique.

        Parameters
        ----------
        atoms  : array_like
            indices of all atoms to be removed.
        """
        idx = np.setdiff1d(np.arange(self.na),atoms,assume_unique=True)
        return self.sub(idx)


    def tile(self,reps,axis):
        """ 
        Returns a geometry tiled, i.e. copied.

        The atomic indices are retained for the base structure.

        Parameters
        ----------
        reps  : number of tiles (repetitions)
        axis  : direction of tiling 
                  0, 1, 2 according to the cell-direction

        Examples
        --------
        >>> geom = Geometry(cell=[[1.,0,0],[0,1.,0.],[0,0,1.]],xyz=[[0,0,0],[0.5,0,0]])
        >>> g = geom.tile(2,axis=0)
        >>> print(g.xyz)
        [[ 0.   0.   0. ]
         [ 0.5  0.   0. ]
         [ 1.   0.   0. ]
         [ 1.5  0.   0. ]]
        >>> g = geom.tile(2,0).tile(2,axis=1)
        >>> print(g.xyz)
        [[ 0.   0.   0. ]
         [ 0.5  0.   0. ]
         [ 1.   0.   0. ]
         [ 1.5  0.   0. ]
         [ 0.   1.   0. ]
         [ 0.5  1.   0. ]
         [ 1.   1.   0. ]
         [ 1.5  1.   0. ]]

        """
        # We need a double copy as we want to re-calculate after
        # enlarging cell
        sc = self.sc.copy()
        sc.cell[axis,:] *= reps ; sc = sc.copy()
        # Pre-allocate geometry
        # Our first repetition *must* be with
        # the later coordinate
        # Copy the entire structure
        xyz = np.tile(self.xyz,(reps,1))
        # Single cell displacements
        dx = np.dot(np.arange(reps)[:,None],self.cell[axis,:][None,:])
        # Correct the unit-cell offsets
        xyz[0:self.na*reps,:] += np.repeat(dx,self.na,axis=0)
        # Create the geometry and return it (note the smaller atoms array
        # will also expand via tiling)
        return self.__class__(xyz, atoms=self.atoms, sc=sc)


    def repeat(self,reps,axis):
        """
        Returns a geometry repeated, i.e. copied in a special way.

        The atomic indices are *NOT* retained for the base structure.

        The expansion of the atoms are basically performed using this
        algorithm:
          ja = 0
          for ia in range(self.na):
              for id,r in args:
                 for i in range(r):
                    ja = ia + cell[id,:] * i

        This method allows to utilise Bloch's theorem when creating
        tight-binding parameter sets for TBtrans.

        For geometries with a single atom this routine returns the same as
        ``self.tile``.

        It is adviced to only use this for electrode Bloch's theorem
        purposes as ``self.tile`` is faster.
        
        Parameters
        ----------
        reps  : number of repetitions
        axis  : direction of repetition
                  0, 1, 2 according to the cell-direction

        Examples
        --------
        >>> geom = Geometry(cell=[[1.,0,0],[0,1.,0.],[0,0,1.]],xyz=[[0,0,0],[0.5,0,0]])
        >>> g = geom.repeat(2,axis=0)
        >>> print(g.xyz)
        [[ 0.   0.   0. ]
         [ 1.   0.   0. ]
         [ 0.5  0.   0. ]
         [ 1.5  0.   0. ]]
        >>> g = geom.repeat(2,0).repeat(2,1)
        >>> print(g.xyz)
        [[ 0.   0.   0. ]
         [ 1.   0.   0. ]
         [ 0.   1.   0. ]
         [ 1.   1.   0. ]
         [ 0.5  0.   0. ]
         [ 1.5  0.   0. ]
         [ 0.5  1.   0. ]
         [ 1.5  1.   0. ]]

        """
        # Figure out the size
        sc = self.sc.copy()
        sc.cell[axis,:] *= reps ; sc = sc.copy()
        # Pre-allocate geometry
        na = self.na * reps
        xyz = np.zeros([na,3],np.float64)
        atoms = [None for i in range(na)]
        dx = np.dot(np.arange(reps)[:,None],self.cell[axis,:][None,:])
        # Start the repetition
        ja = 0
        for ia in range(self.na):
            # Single atom displacements
            # First add the basic atomic coordinate,
            # then add displacement for each repetition.
            xyz[ja:ja+reps,:] = self.xyz[ia,:][None,:] + dx[:,:]
            for i in range(reps):
                atoms[ja+i] = self.atoms[ia]
            ja += reps
        # Create the geometry and return it
        return self.__class__(xyz, atoms=atoms, sc=sc)

    
    def rotate(self,angle,v,only='cell+xyz',degree=False):
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
        only  : ('cell+xyz'), str, optional
             which coordinate subject should be rotated,
             if ``cell`` is in this string the cell will be rotated
             if ``xyz`` is in this string the coordinates will be rotated
        """
        q = Quaternion(angle,v,degree=degree)
        q /= q.norm() # normalize the quaternion
        if 'cell' in only:
            sc = self.sc.rotate(angle,v,degree=degree)
        else:
            sc = self.sc.copy()
        
        xyz = np.copy(self.xyz)
        if 'xyz' in only: xyz = q.rotate(xyz)
        return self.__class__(xyz, atoms=self.atoms, sc=sc)


    def rotate_miller(self,m,v):
        """ Align Miller direction along ``v`` 

        Rotate geometry and cell such that the Miller direction 
        points along the Cartesian vector ``v``.
        """
        # Create normal vector to miller direction and cartesian
        # direction
        cp = np.array([m[1]*v[2]-m[2]*v[1],
                       m[2]*v[0]-m[0]*v[2],
                       m[0]*v[1]-m[1]*v[0]],np.float64)
        cp /= np.sum(cp**2) ** .5

        lm = np.array(m,np.float64)
        lm /= np.sum(lm**2) ** .5
        lv = np.array(v,np.float64)
        lv /= np.sum(lv**2) ** .5

        # Now rotate the angle between them
        a = acos( np.sum(lm*lv) )
        return self.rotate(a,cp)
        

    def translate(self,v,atoms=None,cell=False):
        """ Translates the geometry by ``v``

        One can translate a subset of the atoms by supplying ``atoms``.

        Returns a copy of the structure translated by ``v``.
        """
        g = self.copy()
        if atoms is None:
            g.xyz[:,:] += np.asarray(v,g.xyz.dtype)[None,:]
        else:
            g.xyz[atoms,:] += np.asarray(v,g.xyz.dtype)[None,:]
        if cell:
            g.sc = g.sc.translate(v)
        return g


    def swapaxes(self,a,b,swap='cell+xyz'):
        """ Returns geometry with swapped axis
        
        If ``swapaxes(0,1)`` it returns the 0 and 1 values
        swapped in the ``cell`` variable.
        """
        xyz = np.copy(self.xyz)
        if 'xyz' in swap:
            xyz[:,a] = self.xyz[:,b]
            xyz[:,b] = self.xyz[:,a]
        cell = np.copy(self.cell)
        if 'cell' in swap:
            sc = self.sc.swapaxes(a,b)
        else:
            sc = self.sc.copy()
        return self.__class__(xyz, atoms=np.copy(self.atoms), sc=sc)

    
    def center(self,atoms=None,which='xyz'):
        """ Returns the center of the geometry 
        By specifying ``which`` one can control whether it should be:
        ``xyz``|``position: Center of coordinates (default)
        ``mass``: Center of mass
        ``cell``: Center of cell
        """
        if 'cell' in which:
            return self.sc.center()
        if atoms is None:
            g = self
        else:
            g = self.sub(atoms)
        if 'mass' in which:
            # Create list of masses
            mass = np.array([atm.mass for atm in g.atoms])
            return np.dot(mass,g.xyz) / np.sum(mass)
        if not ('xyz' in which or 'position' in which):
            raise ValueError('Unknown ``which``, not one of [xyz,position,mass,cell]')
        return np.mean(g.xyz,axis=0)


    def append(self,other,axis):
        """
        Appends structure along ``axis``. This will automatically
        add the ``self.cell[axis,:]`` to all atomic coordiates in the 
        ``other`` structure before appending.

        The basic algorithm is this:
        
          >>> oxa = other.xyz + self.cell[axis,:][None,:]
          >>> self.xyz = np.append(self.xyz,oxa)
          >>> self.cell[axis,:] += other.cell[axis,:]
          >>> self.lasto = np.append(self.lasto,other.lasto)

        NOTE: The cell appended is only in the axis that
        is appended, which means that the other cell directions
        need not conform.

        Parameters
        ----------
        other : Geometry
            Other geometry class which needs to be appended
        axis  : int
            Cell direction to which the ``other`` geometry should be
            appended.
        """
        xyz = np.append(self.xyz,
                       self.cell[axis,:][None,:] + other.xyz,
                       axis=0)
        atoms = np.append(self.atoms,other.atoms)
        sc = self.sc.append(other.sc,axis)
        return self.__class__(xyz, atoms=atoms, sc=sc)


    def reverse(self,atoms=None):
        """ Returns a reversed geometry

        Also enables reversing a subset
        """
        if atoms is None:
            xyz = self.xyz[::-1,:]
            atms = self.atoms[::-1]
        else:
            xyz = np.copy(self.xyz)
            xyz[atoms,:] = self.xyz[atoms[::-1],:]
            atms = np.copy(self.atoms)
            atms[atoms] = atms[atoms][::-1]
        return self.__class__(xyz, atoms=atms, sc=self.sc.copy())

    
    def mirror(self,atoms=None):
        """ Mirrors the structure around the center of the atoms """
        c = self.center(atoms=atoms)
        g = self.translate(-c)
        g.xyz *= -1
        g = g.translate(c)
        return self.__class__(g.xyz, atoms=g.atoms, sc=self.sc.copy())
        
    
    def insert(self,atom,other):
        """ Inserts other atoms right before index

        We insert the `other` ``Geometry`` before obj
        """
        xyz = np.insert(self.xyz,atom, other.xyz, axis=0)
        atoms = np.insert(self.atoms, atom, other.atoms)
        return self.__class__(xyz, atoms=atoms, sc=self.sc.copy())


    def coords(self,isc=[0,0,0],idx=None):
        """
        Returns the coordinates of a given super-cell index

        Parameters
        ----------
        isc   : array_like
            Returns the atomic coordinates shifted according to the integer
            parts of the cell.
        idx   : int/array_like
            Only return the coordinates of these indices

        Examples
        --------
        
        >>> geom = Geometry(cell=[[1.,0,0],[0,1.,0.],[0,0,1.]],xyz=[[0,0,0],[0.5,0,0]])
        >>> print(geom.coords(isc=[1,0,0])
        [[ 1.   0.   0. ]
         [ 1.5  0.   0. ]]

        """
        offset = self.sc.offset(isc)
        if idx is None:
            return self.xyz + offset[None,:]
        else:
            return self.xyz[idx,:] + offset[None,:]


    def axyzsc(self,ia):
        return self.coords(self.a2isc(ia),self.sc2uc(ia))


    def close_sc(self,xyz_ia,isc=[0,0,0],dR=None,idx=None,ret_coord=False,ret_dist=False):
        """
        Calculates which atoms are close to some atom or point
        in space, only returns so relative to a super-cell.

        This returns a set of atomic indices which are within a 
        sphere of radius ``dR``.

        If dR is a tuple/list/array it will return the indices:
        in the ranges:
           ( x <= dR[0] , dR[0] < x <= dR[1], dR[1] < x <= dR[2] )

        Parameters
        ----------
        xyz_ia    : coordinate/index
            Either a point in space or an index of an atom.
            If an index is passed it is the equivalent of passing
            the atomic coordinate ``self.close_sc(self.xyz[xyz_ia,:])``.
        isc       : ([0,0,0]), array_like, optional
            The super-cell which the coordinates are checked in.
        dR        : (None), float/tuple of float
            The radii parameter to where the atomic connections are found.
            If ``dR`` is an array it will return the indices:
            in the ranges:
               ``( x <= dR[0] , dR[0] < x <= dR[1], dR[1] < x <= dR[2] )``
            If a single float it will return:
               ``x <= dR``
        idx       : (None), array_like
            List of atoms that will be considered. This can
            be used to only take out a certain atoms.
        ret_coord : (False), boolean
            If true this method will return the coordinates 
            for each of the couplings.
        ret_dist : (False), boolean
            If true this method will return the distance
            for each of the couplings.
        """

        if dR is None:
            ddR = np.array([self.dR],np.float64)
        else:
            ddR = np.array([dR],np.float64).flatten()

        if isinstance(xyz_ia,Integral):
            off = self.xyz[xyz_ia,:]
            # Get atomic coordinate in principal cell
            dxa = self.coords(isc=isc,idx=idx) - off[None,:]
        else:
            off = xyz_ia
            # The user has passed a coordinate
            dxa = self.coords(isc=isc,idx=idx) - off[None,:]

        ret_special = ret_coord or ret_dist

        # Retrieve all atomic indices which are closer
        # than our delta-R
        # The linear algebra norm function could be used, but it
        # has a lot of checks, hence we do it manually
        #xaR = np.linalg.norm(dxa,axis=-1)
        xaR = (dxa[:,0]**2+dxa[:,1]**2+dxa[:,2]**2) ** .5
        ix = np.where(xaR <= ddR[-1])[0]
        if ret_coord:
            xa = dxa[ix,:] + off[None,:]
        if ret_dist:
            d = xaR[ix]
        del dxa # just because these arrays could be very big...

        # Check whether we only have one range to check.
        # If so, we need not reduce the index space
        if len(ddR) == 1:
            if idx is None:
                ret = [ix]
            else:
                ret = [idx[ix]]
            if ret_coord: ret.append(xa)
            if ret_dist: ret.append(d,)
            if ret_special: return ret
            return ret[0]

        if np.any(np.diff(ddR) < 0.):
            raise ValueError('Proximity checks for several quantities '+ \
                                 'at a time requires ascending dR values.')

        # Reduce search space!
        # The more neigbours you wish to find the faster this becomes
        # We only do "one" heavy duty search,
        # then we immediately reduce search space to this subspace
        xaR = xaR[ix]
        tidx = np.where(xaR <= ddR[0])[0]
        if idx is None:
            ret = [ [ix[tidx]] ]
        else:
            ret = [ [idx[ix[tidx]]] ]
        i = 0
        if ret_coord: 
            c = i + 1
            i += 1
            ret.append([xa[tidx]])
        if ret_dist: 
            d = i + 1
            i += 1
            ret.append([d[tidx]])
        for i in range(1,len(ddR)):
            # Search in the sub-space
            # Notice that this sub-space reduction will never
            # allow the same indice to be in two ranges (due to
            # numerics)
            tidx = np.where(np.logical_and(ddR[i-1] < xaR,xaR <= ddR[i]))[0]
            if idx is None:
                ret[0].append(ix[tidx])
            else:
                ret[0].append(idx[ix[tidx]])
            if ret_coord: ret[c].append(xa[tidx])
            if ret_dist: ret[d].append(d[tidx])
        if ret_special: return ret
        return ret[0]

    
    def close(self,xyz_ia,dR=None,idx=None,ret_coord=False,ret_dist=False):
        """
        Returns supercell atomic indices for all atoms connecting to ``xyz_ia``

        This heavily relies on the ``self.close_sc`` method.

        Note that if a connection is made in a neighbouring super-cell
        then the atomic index is shifted by the super-cell index times
        number of atoms.
        This allows one to decipher super-cell atoms from unit-cell atoms.

        Parameters
        ----------
        xyz_ia  : coordinate/index
            Either a point in space or an index of an atom.
            If an index is passed it is the equivalent of passing
            the atomic coordinate ``self.close_sc(self.xyz[xyz_ia,:])``.
        dR      : (None), float/tuple of float
            The radii parameter to where the atomic connections are found.
            If ``dR`` is an array it will return the indices:
            in the ranges:
               ``( x <= dR[0] , dR[0] < x <= dR[1], dR[1] < x <= dR[2] )``
            If a single float it will return:
               ``x <= dR``
        idx     : (None), array_like
            List of indices for atoms that are to be considered
        ret_coord : (False), boolean
            If true this method will return the coordinates 
            for each of the couplings.
        ret_dist : (False), boolean
            If true this method will return the distances from the ``xyz_ia`` 
            for each of the couplings.
        """

        ret = [None]
        i = 0
        if ret_coord: 
            c = i + 1
            i += 1
            ret.append(None)
        if ret_dist: 
            d = i + 1
            i += 1
            ret.append(None)
        ret_special = ret_coord or ret_dist
        for s in range(self.n_s):
            na = self.na * s
            sret = self.close_sc(xyz_ia,self.sc.sc_off[s,:],dR=dR,idx=idx,ret_coord=ret_coord,ret_dist=ret_dist)
            if not ret_special: sret = (sret,)
            if isinstance(sret[0],list):
                # we have a list of arrays
                if ret[0] is None:
                    ret[0] = [x + na for x in sret[0]]
                    if ret_coord: ret[c] = sret[c]
                    if ret_dist: ret[d] = sret[d]
                else:
                    for i,x in enumerate(sret[0]):
                        ret[0][i] = np.append(ret[0][i],x + na)
                        if ret_coord: ret[c][i] = np.vstack((ret[c][i],sret[c][i]))
                        if ret_dist: ret[d][i] = np.hstack((ret[d][i],sret[d][i]))
            elif len(sret[0]) > 0:
                # We can add it to the list
                # We add the atomic offset for the supercell index
                if ret[0] is None:
                    ret[0] = sret[0] + na
                    if ret_coord: ret[c] = sret[c]
                    if ret_dist: ret[d] = sret[d]
                else:
                    ret[0] = np.append(ret[0],sret[0] + na)
                    if ret_coord: ret[c] = np.vstack((ret[c],sret[c]))
                    if ret_dist: ret[d] = np.hstack((ret[d],sret[d]))
        if ret_special: return ret
        return ret[0]

    # Hence ``close_all`` has exact meaning
    # but ``close`` is shorten and retains meaning
    close_all = close


    def a2o(self,ia,all=False):
        """
        Returns an orbital index of the first orbital of said atom.
        This is particularly handy if you want to create
        TB models with more than one orbital per atom.

        Parameters
        ----------
        ia : list, int
             Atomic indices
        all: False, bool
             `False`, return only the first orbital corresponding to the atom,
             `True`, returns list of the full atom
        """
        if not all:
            return self.lasto[ia % self.na] + (ia // self.na) * self.no
        ob = self.a2o(ia)
        oe = self.a2o(np.asarray(ia,np.int32)+1)
        # Create ranges
        o = np.empty([np.sum(oe-ob)],np.int32)
        n = 0
        for i in range(len(ob)):
            o[n:n+oe[i]-ob[i]] = np.arange(ob[i],oe[i],np.int32)
            n += oe[i]-ob[i]
        return o


    def o2a(self,io):
        """
        Returns an atomic index corresponding to the orbital indicies.

        This is a particurlaly slow algorithm.

        Parameters
        ----------
        io: list, int
             List of indices to return the atoms for
        """
        rlasto = self.lasto[::-1]
        iio = np.asarray([io % self.no]).flatten()
        a = [self.na - np.argmax(rlasto <= i) for i in iio]
        return np.asarray(a) + ( io // self.no ) * self.na


    def sc2uc(self,atoms,uniq=False):
        """ Returns atoms from super-cell indices to unit-cell indices, possibly removing dublicates """
        if uniq: return np.unique( atoms % self.na )
        return atoms % self.na
    asc2uc = sc2uc


    def osc2uc(self,orbs,uniq=False):
        """ Returns orbitals from super-cell indices to unit-cell indices, possibly removing dublicates """
        if uniq: return np.unique( orbs % self.no )
        return orbs % self.no


    def a2isc(self,a):
        """
        Returns the super-cell index for a specific atom

        Hence one can easily figure out the supercell
        """
        idx = np.where( a < self.na * np.arange(1,self.n_s+1) )[0][0]
        return self.sc.sc_off[idx,:]


    def o2isc(self,o):
        """
        Returns the super-cell index for a specific orbital.

        Hence one can easily figure out the supercell
        """
        idx = np.where( o < self.no * np.arange(1,self.n_s+1) )[0][0]
        return self.sc.sc_off[idx,:]

    
if __name__ == '__main__':
    import math as m
    from .geom.default import diamond
    
    # Get a diamond
    dia = diamond()

    # Print all closest atoms
    print('Atom')
    for sc in [1,3]:
        dia.sc.set_nsc(nsc=[sc]*3)
        print(dia.close(0,dia.dR))

    # Print all closest atoms and distances
    print('\nAtom and distance')
    for sc in [1,3]:
        dia.sc.set_nsc(nsc=[sc]*3)
        print(dia.close(0,dia.dR,ret_dist=True))

    # Print all closest atoms and coords
    print('\nAtom and coords')
    for sc in [1,3]:
        dia.sc.set_nsc(nsc=[sc]*3)
        print(dia.close(0,dia.dR,ret_coord=True))

    # Print all closest atoms, coords and distances
    print('\nAtom and coords and distances')
    for sc in [1,3]:
        dia.sc.set_nsc(nsc=[sc]*3)
        print(dia.close(0,dia.dR,ret_coord=True,ret_dist=True))
    print("\n")


    print('\nOrbital indices')
    print(dia.a2o(0))
    print(dia.a2o(1))

    # Lets try and create a big one and cut it
    big = dia.tile(3,1).tile(3,axis=0)
    print('\nBig stuff')
    print(big)
    half = big.cut(3,axis=0)
    print('\nSmall stuff')
    print(half)


    big = dia.tile(10,1).tile(10,0)
    print('\nIterable loop: '+str(len(big)))
    na = 0
    for ia in big:
        na += 1
    print('Completed with: '+str(na))

    big = dia.tile(10,1).tile(10,0)
    print('\nIterable loop: '+str(len(big)))
    na = 0
    for ias, idxs in big.iter_block(5):
        na += len(ias)
    print('Completed with: '+str(na))

    # Try the rotation
    rot = dia.copy()
    print(rot.cell,rot.xyz)
    rot = rot.rotate(m.pi/4,[1,0,0])
    print(rot.cell,rot.xyz)

    # Try the rotation
    rot = dia.copy()
    print(rot.cell,rot.xyz)
    rot = rot.rotate(m.pi/4,[1,0,0],only='cell')
    print(rot.cell,rot.xyz)

    # Try and align Miller indices
    fcc = Geometry(np.zeros([3]),atoms=Atom['Fe'],
                   sc=SuperCell([[ 0.5, 0.5, 0.5],
                                 [ 0.5,-0.5, 0.5],
                                 [ 0.5, 0.5,-0.5]]))
    print(fcc.atoms)
    print(fcc.cell)
    rot = fcc.rotate_miller([1,1,1],[0,0,1]).swapaxes(0,2)
    print(rot.cell)

    # Try the passing of an actual SuperCell
    new = Geometry(np.zeros([3]),atoms=Atom['Fe'],sc=fcc.sc)
    print(new)

