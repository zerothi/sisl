"""
Tight-binding class to create tight-binding models.
"""
from __future__ import print_function, division

import warnings
from numbers import Integral

import numpy as np
import scipy.linalg as sli


from sids import Atom, Geometry, Quaternion

__all__ = ['TightBinding']


class TightBinding(object):
    """
    The defining tight-binding model for constructing models
    in python for easy manipulation in tbtrans.

    This class allows the creation of sparse tight-binding
    models on extreme scales.

    Any tight-binding model has an underlying geometry which
    is a requirement for the construction of a full tight-binding
    parameter set.
    """

    # The energy conversion factor
    Energy = 13.60580

    def __init__(self,geom,*args,**kwargs):
        """Create tight-binding model from geometry

        Initializes a tight-binding model using the ``geom`` object
        as the underlying geometry for the tight-binding parameters.
        """
        self.geom = geom

        self.reset(**kwargs)

    ######### Definitions of overrides ############
    def __len__(self):
        """ Returns number of non-zero elements in the model """
        return self._nnzs

    def __repr__(self):
        """ Representation of the tight-binding model """
        s = self.geom.__repr__()
        return s + '\nNumber of non-zero elements {0}'.format(len(self))

    def __getitem__(self,key):
        """ Return tight-binding parameters for the index

        Returns the tight-binding parameters for the index.
        """
        if isinstance(key,Integral):
            # We allow the retrieval of the full row
            i = key
            ind = self.ptr[i]
            n = self.ncol[i]
            return self.col[ind:ind+n],self._TB[ind:ind+n,:]
        
        # It must be both row and column
        i, j = key
        ind = np.where(self.col[self.ptr[i]:self.ptr[i]+self.ncol[i]] == j)[0]
        if len(ind) > 0:
            return self._TB[self.ptr[i]+ind[0],:]
        else:
            # this is a zero element
            return np.array([0., 0.],self._TB.dtype)
    
    def __getattr__(self,attr):
        """ Returns the attributes from the underlying geometry
        
        Any attribute not found in the tight-binding model will
        be looked up in the underlying geometry.
        """
        return getattr(self.geom,attr)

    def __setitem__(self,key,val):
        """ Sparse creation of the sparsity pattern

        Override set item for slicing operations and enables easy 
        setting of tight-binding parameters in a sparse matrix

        It does allow fancy slicing in both dimensions with limited usability
        
        Ok, it is not pretty, it is not fast, but it works!
        """
        # unpack index
        i , j = key
        # Copy over the values of the tight-binding model
        v = np.array([val],self._TB.dtype).flatten()
        if not isinstance(i,Integral):
            # Recursively handle index,index = val
            # designators.
            if len(i) > 1: 
                for ii in i:
                    self[ii,j] = val
                return
            i = int(i[0])

        # step pointer of all above this
        ptr  = self.ptr[i]
        ncol = self.ncol[i]
        # Create the column indices in a strict manner
        jj = np.array([j],np.int32).flatten()
        lj = jj.shape[0]
        if ncol > 0:
            # Checks whether any values in either array exists
            # if so we remove those from the jj
            idx = np.intersect1d(jj,self.col[ptr:ptr+ncol],assume_unique=True)
        else:
            idx = []
            
        if len(idx) > 0:
            
            # Here we truncate jj to the "new" values,
            # this allows us to both overwrite and add new values to the
            # sparsity pattern (simultaneously)
            jj = np.setdiff1d(jj, idx, assume_unique=True)
            lj = jj.shape[0]

            # the values corresponding to idx already exists,
            # we overwrite that value
            if isinstance(j,Integral):
                ix = np.where(j == self.col[ptr:ptr+ncol])[0][0]
                self._TB[ptr+ix,:] = v
            else:
                # remember that idx is the intersection values
                for ij in idx:
                    ix = np.where(ij == self.col[ptr:ptr+ncol])[0][0]
                    self._TB[ptr+ix,:] = v

            # if no new values are left we return immediately
            if lj == 0: return

        while self.ptr[i+1] - ptr < ncol + lj:
            # Ensure that it is not-set as finalized
            # There is no need to set it all the time.
            # Simply because the first call to finalize 
            # will reduce the sparsity pattern, which 
            # on first expansion calls this part.
            self._finalized = False

            # Expand size of the sparsity pattern
            # We add 10 new elements on each extension
            # This may be too reductionists, however, 
            # it should happen rarely
            self.ptr[i+1:] += 10
            self.col = np.insert(self.col,ptr+ncol,np.empty([10],np.int32))
            self._TB = np.insert(self._TB,ptr+ncol,np.empty([10,2],np.int32),axis=0)

        # Step to the placement of the new values
        ptr += ncol
        # set current value
        self.col[ptr:ptr+lj]   = jj
        self._TB[ptr:ptr+lj,:] = v
        # Append the new columns
        self.ncol[i] += lj
        # Increment number of non-zero elements
        self._nnzs += lj

    ############# DONE creating easy overrides #################

    def reset(self,nc=None,dtype=np.float64):
        """
        The sparsity pattern is cleaned and every thing 
        is reset. 

        The object will be the same as if it had been
        initialized with the same geometry as it were
        created with.
        """
        # I know that this is not the most efficient way to
        # access a C-array, however, for constructing a
        # sparse pattern, it should be faster if memory elements
        # are closer... 
        # Hence, this choice of having H and S like this

        # We check the first atom and its neighbours, we then
        # select max(5,len(nc) * 4)
        if nc is None:
            nc = self.geom.close(0)
            nc = max(5,len(nc) * 4)

        # Reset the sparsity pattern
        self.ncol = np.zeros([self.geom.no],np.int32)
        # Create the interstitial pointer for each orbital
        self.ptr = np.cumsum(np.array([nc] * (self.geom.no+1),np.int32)) - nc
        self._nnzs = 0
        self.col = np.empty([self.ptr[-1]],np.int32)
        
        # Denote the tight-binding model as _not_ finalized
        # Before saving, or anything being done, it _has_ to be
        # finalized.
        self._finalized = False

        # Initialize TB size
        # NOTE, this is not zeroed!
        self._TB = np.empty([self.ptr[-1],2],dtype=dtype)

        
    def finalize(self):
        """ Finalizes the tight-binding model

        Finalizes the tight-binding model so that no new sparse
        elements can be added. 

        Sparse elements can still be changed.
        """
        if self._finalized: return
        
        self._finalized = True
        ptr = self.ncol[0]
        if np.unique(self.col[:ptr]).shape[0] != ptr:
            raise ValueError('You cannot have two hoppings between the same '+
                             'orbitals, something has went terribly wrong.')
        
        if self.no > 1:
            # We truncate all the connections
            for io in range(1,self.no):
                cptr = self.ptr[io]
                # Update actual pointer position
                self.ptr[io] = ptr
                no = self.ncol[io]
                if no == 0: continue # no non-assigned elements
                self.col[ptr:ptr+no]  = self.col[cptr:cptr+no]
                self._TB[ptr:ptr+no,:] = self._TB[cptr:cptr+no,:]
                # we also assert no two connections
                if np.unique(self.col[ptr:ptr+no]).shape[0] != no:
                    raise ValueError('You cannot have two hoppings between the same '+
                                     'orbitals, something has went terribly wrong.')
                ptr += no
        
        # Correcting the size of the pointer array
        self.ptr[self.no] = ptr
        if ptr != self._nnzs:
            raise ValueError('Final size in the tight-binding finalization '+
                             'went wrong.')
        
        # Truncate values to correct size
        self._TB = self._TB[:self._nnzs,:]
        self.col = self.col[:self._nnzs]
        
        # Sort the indices, this is not strictly required, but
        # it should speed up things.
        for io in range(self.no):
            ptr = self.ptr[io]
            no  = self.ncol[io]
            if no == 0: continue
            # Sort the indices
            si = np.argsort(self.col[ptr:ptr+no])
            self.col[ptr:ptr+no]  = self.col[ptr+si]
            self._TB[ptr:ptr+no,:] = self._TB[ptr+si,:]

    @property
    def nnzs(self):
        """ Returns number of non-zero elements in the tight-binding model """
        return self._nnzs

    def tocsr(self,k=None):
        """ Returns ``scipy.sparse`` matrices for the tight-binding model

        Returns a CSR sparse matrix for both the Hamiltonian
        and the overlap matrix using the scipy package.
        
        This method depends on scipy.
        """
        # Ensure completeness
        self.finalize()

        # Create csr sparse formats.
        # We import here as the user might not want to
        # rely on this feature.
        from scipy.sparse import csr_matrix
        
        if k is None:
            kw = {'shape': (self.no,self.no_s),
                  'dtype': self._TB.dtype }
            if self._TB.shape[1] == 1:
                return csr_matrix((self._TB[:,0],self.col,self.ptr),**kw)
            return (csr_matrix((self._TB[:,0],self.col,self.ptr),**kw), \
                        csr_matrix((self._TB[:,1],self.col,self.ptr),**kw))
        else:
            k = np.asarray(k,np.float64)
            import scipy.linalg as sla
            # Setup the Hamiltonian for this k-point
            Hfull, Sfull = self.tocsr()

            s = (self.no,self.no)

            # Create k-space Hamiltonian
            H = csr_matrix(s,dtype=np.complex)
            S = csr_matrix(s,dtype=np.complex)
            
            # Get the reciprocal lattice vectors dotted with k
            rcell = sla.inv(self.cell.copy())
            kr = np.dot(np.asarray(k),rcell) * np.pi * 2.
            for si in range(self.sc.n_s):
                isc = self.sc_off[si,:]
                phase = np.exp(-1j*np.dot(kr,np.dot(self.cell,isc)))
                H += Hfull[:,si*self.no:(si+1)*self.no] * phase
                S += Sfull[:,si*self.no:(si+1)*self.no] * phase
            del Hfull, Sfull
            return (H,S)

    def eigh(self,k=None,atoms=None,*args,**kwargs):
        """ Returns the eigenvalues of the tight-binding model

        Setup the Hamiltonian and overlap matrix with respect to
        the given k-point, then reduce the space to the specified atoms
        and calculate the eigenvalues.

        All subsequent arguments gets passed directly to ``scipy.linalg.eigh``
        """
        H, S = self.tocsr(k=k)
        # Reduce sparsity pattern
        if not atoms is None:
            orbs = self.a2o(atoms)
            # Reduce space
            H = H[orbs,orbs]
            S = S[orbs,orbs]
        return sli.eigh(H.todense(),S.todense(),*args,**kwargs)

    def cut(self,seps,axis):
        """ Cuts the tight-binding model into different parts.

        Creates a tight-binding model by retaining the parameters
        for the cut-out region, possibly creating a super-cell.

        Parameters
        ----------
        seps  : integer, optional
           number of times the structure will be cut.
        axis  : integer
           the axis that will be cut
        """
        # Create new geometry
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Create new cut geometry
            geom = self.geom.cut(seps,axis)
            # Check whether the warning exists
            if len(w) > 0:
                if issubclass(w[-1].category,UserWarning):
                    raise ValueError('You cannot cut a tight-binding model '+
                                     'if the structure cannot be recreated using tiling constructs.')
        
        # Now we need to re-create the tight-binding model
        H, S = self.tocsr()
        # they are created similarly, hence the following
        # should keep their order

        # First we need to figure out how long the interaction range is
        # in the cut-direction
        # We initialize to be the same as the parent direction
        nsc = np.copy(self.nsc) // 2
        nsc[axis] = 0 # we count the new direction
        isc = np.zeros([3],np.int32)
        isc[axis] -= 1
        out = False
        while not out:
            # Get supercell index
            isc[axis] += 1
            try:
                idx = self.sc_index(isc)
            except: 
                break

            # Figure out if the Hamiltonian has interactions
            # to ``isc``
            sub = H[0:geom.no,idx*self.no:(idx+1)*self.no].indices[:]
            if len(sub) == 0: break

            c_max = np.amax(sub)
            # Count the number of cells it interacts with
            i = (c_max % self.no) // geom.no
            ic = idx * self.no
            for j in range(i):
                idx = ic + geom.no * j
                # We need to ensure that every "in between" index exists
                # if it does not we discard those indices
                if len(np.where( 
                        np.logical_and(idx <= sub, 
                                       sub < idx + geom.no)
                        )[0]) == 0:
                    i = j - 1
                    out = True
                    break
            nsc[axis] = isc[axis] * seps + i
            
            if out:
                warnings.warn('Cut the connection at {0} in direction {1}.'.format(nsc[axis],axis), UserWarning) 

        # Update number of super-cells
        nsc[:] = nsc[:] * 2 + 1
        geom.sc.set_nsc(nsc)

        # Now we have a correct geometry, and 
        # we are now ready to create the sparsity pattern
        # Reduce the sparsity pattern, first create the new one
        tb = self.__class__(geom,nc=np.amax(self.ncol))

        def sco2sco(M,o,m,seps,axis):
            # Converts an o from M to m
            isc = np.copy( M.o2isc(o) )
            isc[axis] = isc[axis] * seps
            # Correct for cell-offset
            isc[axis] = isc[axis] + (o % M.no) // m.no
            # find the equivalent cell in m
            try:
                # If a fail happens it is due to a discarded
                # interaction across a non-interacting region
                return ( o % m.no, 
                        m.sc_index( isc) * m.no, 
                        m.sc_index(-isc) * m.no)
            except:
                return None, None, None

        # Copy elements
        for jo in range(geom.no):

            # make smaller cut
            sH = H[jo,:]
            sS = S[jo,:]

            for io, iH in zip(sH.indices,sH.data):
                # Get the equivalent orbital in the smaller cell
                o, ofp, ofm = sco2sco(self.geom,io,tb.geom,seps,axis)
                if o is None: continue
                tb[jo,o+ofp] = iH, S[jo,io]
                tb[o,jo+ofm] = iH, S[jo,io]

        return tb

    @classmethod
    def sp2tb(cls,geom,H,S):
        """ Returns a tight-binding model from a preset H, S and Geometry """

        # Calculate number of connections
        nc = 0
        H = H.tocsr()
        for i in range(geom.no):
            nc = max(nc,H[i,:].getnnz())
        H = H.tocoo()
        
        tb = cls(geom,nc=nc)

        # Copy data to the model
        for jo,io,h in zip(H.row,H.col,H.data):
            tb[jo,io] = (h,S[jo,io])

        return tb

    def write(self,sile):
        """ Writes a tight-binding model to the ``sile`` as implemented in the ``ObjSile.write_tb``
        method """
        self.finalize()

        # This only works because, they *must*
        # have been imported previously
        from sids.io import get_sile, BaseSile
        if isinstance(sile,BaseSile):
            sile.write_tb(self)
        else:
            get_sile(sile,'w').write_tb(self)

        
if __name__ == "__main__":
    import datetime
    from sids.geom import graphene

    # Create graphene unit-cell
    gr = graphene()
    gr.sc.set_nsc([3,3,1])

    # Create nearest-neighbour tight-binding
    # graphene lattice constant 1.42
    dR = ( 0.1 , 1.5 )
    on = (0.,1.)
    nn = (-0.5,0.)
    
    tb = TightBinding(gr)
    for ia in tb.geom:
        idx_a = tb.geom.close(ia,dR=dR)
        tb[ia,idx_a[0]] = on
        tb[ia,idx_a[1]] = nn
    print(len(tb))
    tb.finalize()
    print('H\n',tb.tocsr()[0])
    print('H\n',tb.tocsr(k=[.25,.25,0])[0])
    print('eig\n',tb.eigh(k=[3./4,.5,0],eigvals_only=True))


    print('\nCheck expansion')
    tb = TightBinding(gr)
    tb.reset(1) # force only one connection (this should force expansion)
    for ia in tb.geom:
        idx_a = tb.geom.close(ia,dR=dR)
        tb[ia,idx_a[0]] = on
        tb[ia,idx_a[1]] = nn
    print(len(tb))
    print('H\n',tb.tocsr()[0])
    print('H\n',tb.tocsr(k=[.25,.25,0])[0])

    # Lets try and create a huge sample
    print('\n\nStarting time... '+str(datetime.datetime.now().time()))
    tb = TightBinding(gr.tile(41,0).tile(41,1))
    for ias, idxs in tb.iter_block(13):
        for ia in ias:
            idx_a = tb.geom.close(ia,dR=dR)
            tb[ia,idx_a[0]] = on
            tb[ia,idx_a[1]] = nn
    print(len(tb))
    print('Ending time... '+str(datetime.datetime.now().time()))

    
