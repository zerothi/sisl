"""
Tight-binding class to create tight-binding models.
"""
from __future__ import print_function, division

from numbers import Integral

# The atom model
from sids.geom import Atom, Geometry, Quaternion

import numpy as np

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

    def __init__(self,geom):
        """Create tight-binding model from geometry

        Initializes a tight-binding model using the ``geom`` object
        as the underlying geometry for the tight-binding parameters.
        """
        self.geom = geom

        self.reset()

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
        jj = np.array([j],np.int).flatten()
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

        # As nnzs is deleted when the object has been finalized 
        # this line should error out in case one tries to
        # alter the sparsity pattern after finalization
        m = self.ptr[i+1] - ptr
        if m < ncol + lj:
            print('Setting illegal number of parameters in sparsity pattern')
            raise ValueError('Have you changed the sparsity pattern while editing '+
                             'the TB parameters? This is not allowed.\n'+
                             'Try resetting the model with larger connectivity')

        # Step to the placement of the new values
        ptr += ncol
        # set current value
        self.col[ptr:ptr+lj]  = jj
        self._TB[ptr:ptr+lj,:] = v
        # Append the new columns
        self.ncol[i] += lj
        # Increment number of non-zero elements
        self._nnzs += lj

    ############# DONE creating easy overrides #################

    def reset(self,dtype=np.float64,nc=None):
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
        # select a minimun of max(5,nc * 4)
        if nc is None:
            nc = self.geom.close_all(0)
            nc = max(5,len(nc) * 4)

        # Reset the sparsity pattern
        self.ncol = np.zeros([self.geom.no],np.int)
        # Create the interstitial pointer for each orbital
        self.ptr = np.cumsum(np.array([nc] * (self.geom.no+1),np.int)) - nc
        self._nnzs = 0
        self.col = np.empty([self.ptr[-1]],np.int)
        
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
            for io in xrange(1,self.no):
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
        
        # Sort the indices (THIS IS A REQUIREMENT!)
        for io in xrange(self.no):
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

        kw = {'shape': (self.no,self.no_s),
              'dtype': self._TB.dtype }
        
        if k is None:
            if self._TB.shape[1] == 1:
                return csr_matrix((self._TB[:,0],self.col,self.ptr),**kw)
            return (csr_matrix((self._TB[:,0],self.col,self.ptr),**kw), \
                        csr_matrix((self._TB[:,1],self.col,self.ptr),**kw))
        else:
            k = np.asarray(k,np.float64)
            import scipy.linalg as sla
            # Setup the Hamiltonian for this k-point
            Hfull,Sfull = self.tocsr()

            del kw['shape']
            s = (self.no,self.no)
            # Create k-space Hamiltonian
            H = csr_matrix(s,**kw)
            S = csr_matrix(s,**kw)
            
            # Get the reciprocal lattice vectors dotted with k
            rcell = sla.inv(self.cell.copy())
            kr = np.dot(np.asarray(k),rcell) * np.pi * 2.
            for si in range(np.product(self.nsc)):
                isc = self.isc_off[si,:]
                phase = np.exp(1j*np.dot(kr,np.dot(self.cell,isc)))
                H += Hfull[:,si*self.no:(si+1)*self.no] * phase
                S += Sfull[:,si*self.no:(si+1)*self.no] * phase
            del Hfull, Sfull
            return (H,S)

        
if __name__ == "__main__":
    import datetime
    from sids.geom import graphene

    # Create graphene unit-cell
    gr = graphene()
    gr.set_supercell([3,3,1])

    # Create nearest-neighbour tight-binding
    # graphene lattice constant 1.42
    dR = ( 0.1 , 1.5 )
    on = (0.,1.)
    nn = (-0.5,0.)
    
    tb = TightBinding(gr)
    for ia in tb.geom:
        idx_a = tb.close_all(ia,dR=dR)
        tb[ia,idx_a[0]] = on
        tb[ia,idx_a[1]] = nn
    print(len(tb))
    tb.finalize()
    print('H\n',tb.tocsr()[0])
    print('H\n',tb.tocsr(k=[.5,.5,0])[0])


    # Lets try and create a huge sample
    print('Starting time... '+str(datetime.datetime.now().time()))
    tb = TightBinding(gr.tile(71,0).tile(71,1))
    for ias, idxs in tb.iter_block(13):
        for ia in ias:
            idx_a = tb.close(ia,dR=dR)
            tb[ia,idx_a[0]] = on
            tb[ia,idx_a[1]] = nn
    print(len(tb))
    print('Ending time... '+str(datetime.datetime.now().time()))
