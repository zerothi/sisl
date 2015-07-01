"""
Geometry class to retain the atomic structure.
"""
from __future__ import print_function, division

# To check for integers
from numbers import Integral

# Helpers
from sids.geom._help import array_fill_repeat

# The atom model
from sids.geom.atom import Atom
from sids.geom.quaternion import Quaternion

import numpy as np

class Geometry(object):
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
    cell  : array_like
        cell vectors in a 3x3 matrix.
        ``cell[i,:]`` is the cell vector along the i'th direction.
    xyz   : array_like
        atomic coordinates in a Nx3 matrix.
        ``xyz[i,:]`` is the atomic coordinate of the i'th atom.
        This atomic coordinate need not be inside the unit-cell.
    atoms : (Atom['H']), list of atoms associated
        These atoms constitute the atomic ranges, weights, orbitals
        etc. for the geometry.
    nsc   : (0,0,0) array_like, integer
        Number of supercells in the geometry.
        We default it to be a molecule, hence no interaction.
        NOTE That this quantity is the actual number of supercells.
        Each direction can be any of these numbers:
          0/1, 3, 5, ...,
        as one starts with the primary cell, any other super-cell accounts
        for both the positive and negative direction, hence two additional 
        supercells per additional principal layer in any-one direction.

    Attributes
    ----------
    cell : (3,3) ndarray
        Cell vectors.
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
    """
    def __init__(self,cell,xyz,atoms=Atom['H'],nsc=np.array([1,1,1],np.int)):
        self.cell = np.asarray(cell)
        self.xyz = np.asarray(xyz)
        self.xyz.shape = (-1,3)
        self.na = len(self.xyz)

        # Correct the atoms input to Atom
        if isinstance(atoms,list):
            if isinstance(atoms[0],basestring):
                atoms = [Atom[a] for a in atoms]
        elif isinstance(atoms,basestring):
            atoms = Atom[atoms]

        # Create atom objects
        self.atoms = array_fill_repeat(atoms,self.na)
        
        # Store maximum interaction range
        if isinstance(atoms,Atom):
            self.dR = atoms.dR
        else:
            self.dR = max(*[a.dR for a in atoms])

        # Get total number of orbitals
        orbs = np.array([a.orbs for a in self.atoms],np.int)

        # Get total number of orbitals
        self.no = np.sum(orbs)

        # Create local lasto
        lasto = np.append(np.array(0,np.int),orbs)
        self.lasto = np.cumsum(lasto)

        # Set number of super-cells
        self.set_supercell(nsc=nsc)


    def __len__(self):
        """ Return number of atoms in this geometry """
        return self.na


    def write(self,ObjSile):
        """ Writes a geometry to the ``ObjSile`` as implemented in the ``ObjSile.write_geom``
        method """
        ObjSile.write_geom(self)


    def set_supercell(self,nsc=None,a=None,b=None,c=None):
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
            self.nsc = np.asarray(nsc)
        if a: self.nsc[0] = a
        if b: self.nsc[1] = b
        if c: self.nsc[2] = c
        # Correct for misplaced number of unit-cells
        for i in range(3):
            if self.nsc[i] == 0: self.nsc[i] = 1
        if np.sum(self.nsc % 2) != 3 :
            raise ValueError("Supercells has to be of un-even size. The primary cell counts "+
                             "one, all others count 2")

        self.isc_off = np.zeros([np.prod(self.nsc),3],np.int)

        n = self.nsc
        # We define the following ones like this:
        x = range(-n[0]//2+1,n[0]//2+1)
        y = range(-n[1]//2+1,n[1]//2+1)
        z = range(-n[2]//2+1,n[2]//2+1)
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
                    self.isc_off[i,0] = ix
                    self.isc_off[i,1] = iy
                    self.isc_off[i,2] = iz


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

          >>> for ia in xrange(len(self)):
          >>>    <do something>
        """
        for ia in xrange(len(self)):
            yield ia

    __iter__ = iter_linear

    def iter_block(self,iR=15):
        """ 
        Returns an iterator for performance critical looping.
        
        Parameters
        ----------
        iR  : (15) integer
            the number of ``dR`` ranges taken into account when doing the iterator
 
        Returns two lists with [0] being a list of atoms to be looped and [1] being the atoms that 
        need searched.

        NOTE: This requires that dR has been set correctly as the maximum interaction range.

        I.e. the loop would look like this:
        
        >>> for ias, idxs in Geometry.iter():
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

        # The boundaries (ensure complete overlap)
        dR = ( self.dR * (iR - 1), self.dR * (iR+.1))

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
            all_idx = self.close(idx, dR = dR )

            # Get unit-cell atoms
            all_idx[0] = self.sc2uc(all_idx[0])
            # First extend the search-space (before reducing)
            all_idx[1] = self.sc2uc(np.append(all_idx[1],all_idx[0]))

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
        return self.no * np.prod(self.nsc)


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
        if cell is None: cell = np.copy(self.cell)
        return self.__class__(cell,np.copy(self.xyz[atoms,:]),
                              atoms = self.atoms[atoms], nsc = self.nsc)


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
        seps  : (2), integer, optional
           number of times the structure will be cut.
        """
        if self.na % seps != 0:
            raise ValueError('The system cannot be cut into {0} different '+
                             'pieces. Please check your geometry and input.'.format(seps))
        # Cut down cell
        cell = np.copy(self.cell)
        cell[axis,:] /= seps
        new = self.sub(np.arange(self.na//seps),cell=cell)
        if not np.allclose(new.tile(seps,axis).xyz,self.xyz):
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
        return self.__class__(np.copy(self.cell),np.copy(self.xyz),
                               atoms = self.atoms, nsc = np.copy(self.nsc) )


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
        cell = np.copy(self.cell)
        cell[axis,:] *= reps
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
        return self.__class__(cell,xyz,atoms = self.atoms,nsc = np.copy(self.nsc))


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
        cell = np.copy(self.cell)
        cell[axis,:] *= reps
        # Pre-allocate geometry
        na = self.na * reps
        xyz = np.zeros([na,3],np.float)
        atoms = [None for i in range(na)]
        dx = np.dot(np.arange(reps)[:,None],self.cell[axis,:][None,:])
        # Start the repetition
        ja = 0
        for ia in xrange(self.na):
            # Single atom displacements
            # First add the basic atomic coordinate,
            # then add displacement for each repetition.
            xa[ja:ja+reps,:] = self.xyz[ia,:][None,:] + dx[:,:]
            Z[ja:ja+reps] = self.atoms[ia]
            ja += reps
        # Create the geometry and return it
        return self.__class__(cell,xyz,atoms=atoms,nsc=np.copy(self.nsc))


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
        cell = np.copy(self.cell)
        cell[axis,:] += other.cell[axis,:]
        return self.__class__(cell,xyz,atoms=atoms,nsc=np.copy(self.nsc))


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
        offset = self.cell[0,:] * isc[0] + \
            self.cell[1,:] * isc[1] + \
            self.cell[2,:] * isc[2]
        if idx is None:
            return self.xyz + offset[None,:]
        else:
            return self.xyz[idx,:] + offset[None,:]


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
            ddR = np.array([self.dR],np.float)
        else:
            ddR = np.array([dR],np.float).flatten()

        if isinstance(xyz_ia,(int,np.int,np.int16,np.int32)):
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
        for s in xrange(np.prod(self.nsc)):
            na = self.na * s
            sret = self.close_sc(xyz_ia,self.isc_off[s,:],dR=dR,idx=idx,ret_coord=ret_coord,ret_dist=ret_dist)
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

    def a2o(self,ia):
        """
        Returns an orbital index of the first orbital of said atom.
        This is particularly handy if you want to create
        TB models with more than one orbital per atom.

        Parameters
        ----------
        ia : list, int
             Atomic indices

        """
        return self.lasto[ia % self.na] + (ia // self.na) * self.no


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


    def sc2uc(self,atoms):
        """ Returns atoms from super-cell indices to unit-cell indices (removing dublicates) """
        return np.unique(atoms % self.na)
    asc2uc = sc2uc


    def sc_index(self,isc):
        """
        Returns the geometry index for the supercell
        corresponding to isc ([ix,iy,iz])
        """
        asc = np.asarray(isc,np.int)
        for i in xrange(self.isc_off.shape[0]):
            if np.all(self.isc_off[i,:] == asc): return i
        raise Exception('Could not find supercell index, number of super-cells not big enough')


    def a2isc(self,a):
        """
        Returns the super-cell index for a specific atom

        Hence one can easily figure out the supercell
        """
        idx = np.where( a < self.na * np.arange(1,np.product(self.nsc)+1) )[0][0]
        return self.isc_off[idx,:]


    def o2isc(self,o):
        """
        Returns the super-cell index for a specific orbital.

        Hence one can easily figure out the supercell
        """
        idx = np.where( o < self.no * np.arange(1,np.product(self.nsc)+1) )[0][0]
        return self.isc_off[idx,:]

    def rotate(self,angle,v,only='cell+xyz'):
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
        q = Quaternion(angle,v)
        q /= q.norm() # normalize
        if 'cell' in only:
            self.cell = q.rotate(self.cell)
        if 'xyz' in only:
            self.xyz = q.rotate(self.xyz)


if __name__ == '__main__':
    import math as m
    from .default import diamond
    
    # Get a diamond
    dia = diamond()

    # Print all closest atoms
    print('Atom')
    for sc in [1,3]:
        dia.set_supercell(nsc=[sc]*3)
        print(dia.close(0,dia.dR))

    # Print all closest atoms and distances
    print('\nAtom and distance')
    for sc in [1,3]:
        dia.set_supercell(nsc=[sc]*3)
        print(dia.close(0,dia.dR,ret_dist=True))

    # Print all closest atoms and coords
    print('\nAtom and coords')
    for sc in [1,3]:
        dia.set_supercell(nsc=[sc]*3)
        print(dia.close(0,dia.dR,ret_coord=True))

    # Print all closest atoms, coords and distances
    print('\nAtom and coords and distances')
    for sc in [1,3]:
        dia.set_supercell(nsc=[sc]*3)
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
        print(len(ias))
    print('Completed with: '+str(na))

    # Try the rotation
    rot = dia.copy()
    print(rot.cell,rot.xyz)
    rot.rotate(m.pi/4,[1,0,0])
    print(rot.cell,rot.xyz)

    # Try the rotation
    rot = dia.copy()
    print(rot.cell,rot.xyz)
    rot.rotate(m.pi/4,[1,0,0],only='cell')
    print(rot.cell,rot.xyz)

