# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
""" Eigenchannel calculator for any number of electrodes

Developer: Nick Papior
Contact: nickpapior <at> gmail.com
sisl-version: >=0.11.0
tbtrans-version: >=siesta-4.1.5

This eigenchannel calculater uses TBtrans output to calculate the eigenchannels
for N-terminal systems. In the future this will get transferred to the TBtrans code
but for now this may be used for arbitrary geometries.

It requires two inputs and has several optional flags.

- The siesta.TBT.nc file which contains the geometry that is to be calculated for
  The reason for using the siesta.TBT.nc file is the ease of use:

    The siesta.TBT.nc contains electrode atoms and device atoms. Hence it
    becomes easy to read in the electrode atomic positions.
    Note that since you'll always do a 0 V calculation this isn't making
    any implications for the requirement of the TBT.nc file.
"""
from numbers import Integral

import numpy as np
from numpy import einsum
from numpy import conjugate as conj
from scipy.sparse import csr_matrix

import sisl as si
from sisl import _array as _a
from sisl.linalg import *
from sisl.utils.misc import PropertyDict


arange = _a.arangei
indices_only = si._indices.indices_only

__all__ = ['PivotSelfEnergy', 'DeviceGreen']


def dagger(M):
    return conj(M.T)


def get_maxerrr(u):
    inner = conj(u.T) @ u
    np.fill_diagonal(inner, inner.diagonal() - 1.)
    a = np.absolute(inner)
    aidx = np.argmax(a)
    uidx = np.argmax(np.absolute(u))
    return a.max(), a[:, 0], np.unravel_index(aidx, a.shape), u.ravel()[uidx], np.unravel_index(uidx, a.shape)


def gram_schmidt(u, modified=True):
    """ Assumes u is in fortran indexing as returned from eigh

    Gram-Schmidt orthogonalization is not always a good idea.

    1. When some of the states die out the precision of the norm
       becomes extremely important and quite often it will blow up.

    2. DOS normalization will be lost if GS is done.

    3. It is not clear whether GS done in each block or at the end
       is the best choice.
    """
    # first normalize
    norm = np.empty(u.shape[1], dtype=si._help.dtype_complex_to_real(u.dtype))

    # we know that u[:, -1] is the largest eigenvector, so we use that
    # as the default
    if modified:
        for i in range(u.shape[1] - 2, -1, -1):
            norm[i+1] = (conj(u[:, i+1]) @ u[:, i+1]).real
            cu = conj(u[:, i])
            for j in range(u.shape[1] - 1, i, -1):
                u[:, i] -= (cu @ u[:, j]) * u[:, j] / norm[j]

    else:
        for i in range(u.shape[1] - 2, -1, -1):
            norm[i+1] = (conj(u[:, i+1]) @ u[:, i+1]).real
            u[:, i] -= (((conj(u[:, i]) @ u[:, i+1:]) / norm[i+1:]).reshape(1, -1) * u[:, i+1:]).sum(1)


class PivotSelfEnergy(si.physics.SelfEnergy):
    """ Container for the self-energy object

    This may either be a `tbtsencSileTBtrans`, a `tbtgfSileTBtrans` or a sisl.SelfEnergy objectfile
    """

    def __init__(self, name, se, pivot=None):
        # Name of electrode
        self.name = name

        # File containing the self-energy
        # This may be either of:
        #  tbtsencSileTBtrans
        #  tbtgfSileTBtrans
        #  SelfEnergy object (for direct calculation)
        self._se = se
        if isinstance(se, si.io.tbtrans.tbtsencSileTBtrans):
            def se_func(*args, **kwargs):
                return se.self_energy(self.name, *args, **kwargs)
            def scat_func(*args, **kwargs):
                return se.scattering_matrix(self.name, *args, **kwargs)
        else:
            def se_func(*args, **kwargs):
                return se.self_energy(*args, **kwargs)
            def scat_func(*args, **kwargs):
                return se.scattering_matrix(*args, **kwargs)

        if pivot is None:
            if isinstance(se, si.io.tbtrans.tbtsencSileTBtrans):
                pivot = se

        # Store the pivoting for faster indexing

        # Pivoting indices for the self-energy for the device region
        # but with respect to the full system size
        self.pvt = pivot.pivot(name).reshape(-1, 1)

        # Pivoting indices for the self-energy for the device region
        # but with respect to the device region only
        self.pvt_dev = pivot.pivot(name, in_device=True).reshape(-1, 1)

        # the pivoting in the downfolding region (with respect to the full
        # system size)
        self.pvt_down = pivot.pivot_down(name).reshape(-1, 1)

        # Retrieve BTD matrices for the corresponding electrode
        self.btd = pivot.btd(name).reshape(-1, 1)

        # Get the individual matrices
        cbtd = np.cumsum(self.btd)
        pvt_btd = []
        o = 0
        for i in cbtd:
            # collect the pivoting indices for the downfolding
            pvt_btd.append(self.pvt_down[o:i, 0])
            o += i
        self.pvt_btd = np.concatenate(pvt_btd).reshape(-1, 1)
        self.pvt_btd_sort = np.arange(o)

        self._func = (
            se_func,
            scat_func
        )

    def __len__(self):
        return len(self.pvt_dev)

    def index_slice(self, in_device=True):
        if in_device:
            return self.pvt_dev, self.pvt_dev.T
        return self.pvt, self.pvt.T

    def btd_slices(self, sort=True):
        """ BTD slices for down-folding the self-energies

        This is *not* related to the device region.
        """
        if sort:
            return [(i, i.T) for i in self.pvt_btd_sort]
        return [(i, i.T) for i in self.pvt_btd]

    def self_energy(self, *args, **kwargs):
        return self._func[0](*args, **kwargs)

    def scattering_matrix(self, *args, **kwargs):
        return self._func[1](*args, **kwargs)


class BTD:
    """ Container class that holds a BTD matrix """

    def __init__(self, btd):
        self._btd = btd
        # diagonal
        self._M11 = [None] * len(btd)
        # above
        self._M10 = [None] * (len(btd)-1)
        # below
        self._M12 = [None] * (len(btd)-1)

    @property
    def btd(self):
        return self._btd

    def diagonal(self):
        return np.concatenate([M.diagonal() for M in self._M11])

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, j = key
            if i == j:
                return self._M11[i]
            elif i - 1 == j: # (i, i-1)
                return self._M10[j]
            elif i + 1 == j: # (i, i+1)
                return self._M12[i]
            raise IndexError(f"{self.__class__.__name__} does not have index ({i},{j}), only (i,i+-1) are allowed.")
        raise ValueError(f"{self.__class__.__name__} index retrieval must be done with a tuple.")

    def __setitem__(self, key, M):
        if isinstance(key, tuple):
            i, j = key
            if i == j:
                self._M11[i] = M
            elif i - 1 == j: # (i, i-1)
                self._M10[j] = M
            elif i + 1 == j: # (i, i+1)
                self._M12[i] = M
            elif not np.allclose(M, 0.):
                raise IndexError(f"{self.__class__.__name__} does not have index ({i},{j}); only (i,i+-1) are allowed.")
            assert M.shape == (self.btd[j], self.btd[i])
        else:
            raise ValueError(f"{self.__class__.__name__} index retrieval must be done with a tuple.")


class DeviceGreen:
    """

    Basic usage:

    .. code::

       import sisl
       from sisl_toolbox.btd import *
       left = PivotSelfEnergy("Left", sisl.get_sile("siesta.TBT.SE.nc"))
       right = PivotSelfEnergy("Right", sisl.get_sile("siesta.TBT.SE.nc"))
       H = sisl.Hamiltonian.read("DEVICE.nc")
       G = DeviceGreen(H, [left, right], tbt)
       G.prepare(0.1, [0.1, 0.1, 0.1])
       G.green()

    """

    # TODO we should speed this up by overwriting A with the inverse once
    #      calculated. We don't need it at that point.
    #      That would probably require us to use a method to retrieve
    #      the elements which determines if it has been calculated or not.

    def __init__(self, H, elec, pivot):
        """ Create Green function with Hamiltonian and BTD matrix elements """
        self.H = H

        # Store electrodes (for easy retrieval of the SE)
        # There may be no electrodes
        self.elec = elec

        self.pvt = pivot.pivot()
        self.btd = pivot.btd()

        # Create BTD indices
        self.btd_cum = np.cumsum(self.btd)
        cumbtd = np.append(0, self.btd_cum)

        self.btd_idx = [self.pvt[cumbtd[i]:cumbtd[i + 1]]
                        for i in range(len(self.btd))]

        self._data = PropertyDict()

    def __len__(self):
        return len(self.pvt)

    def _elec(self, elec):
        """ Convert a string electrode to the proper linear index """
        if isinstance(elec, str):
            for iel, el in enumerate(self.elec):
                if el.name == elec:
                    return iel
        return elec

    def _elec_name(self, elec):
        """ Convert an electrode index or str to the name of the electrode """
        if isinstance(elec, str):
            return elec
        return self.elec[elec].name

    def prepare(self, E, k=(0, 0, 0), eta=0.0):
        self._data.E = E
        self._data.k = _a.asarrayd(k)
        self._data.eta = eta

        # Prepare the Green function calculation
        inv_G = self.H.Sk(k) * (E + 1j * eta) - self.H.Hk(k)

        # Create all self-energies (and store the Gamma's)
        gamma = []
        for elec in self.elec:
            # Insert values
            SE = elec.self_energy(E, k)
            inv_G[elec.pvt, elec.pvt.T] -= SE
            gamma.append(elec.se2scat(SE))
        self._data.gamma = gamma

        # Now reduce the sparse matrix to the device region (plus do the pivoting)
        inv_G = inv_G[self.pvt, :][:, self.pvt]

        nb = len(self.btd)

        # Now we have all needed to calculate the inverse parts of the Green function
        A = [None] * nb
        B = [1] * nb
        C = [1] * nb

        # Now we can calculate everything
        cbtd = self.btd_cum
        btd = self.btd

        sl0 = arange(0, cbtd[0]).reshape(-1, 1)
        slp = arange(cbtd[0], cbtd[1]).reshape(1, -1)
        # initial matrix A and C
        A[0] = inv_G[sl0, sl0.T].toarray()
        C[1] = inv_G[sl0, slp].toarray()
        for b, bs in enumerate(btd[1:-1], 1):
            # rotate slices
            sln = sl0.T
            sl0 = slp.T
            slp = arange(cbtd[b], cbtd[b+1]).reshape(1, -1)

            B[b-1] = inv_G[sl0, sln].toarray()
            A[b] = inv_G[sl0, sl0.T].toarray()
            C[b+1] = inv_G[sl0, slp].toarray()
        # and final matrix A and B
        A[-1] = inv_G[slp.T, slp].toarray()
        B[-2] = inv_G[slp.T, sl0.T].toarray()

        # clean-up, not used anymore
        del inv_G

        self._data.A = A
        self._data.B = B
        self._data.C = C

        # Now do propagation forward, tilde matrices
        tX = [0] * nb
        tY = [0] * nb
        # \tilde Y
        tY[1] = solve(A[0], C[1])
        # \tilde X
        tX[-2] = solve(A[-1], B[-2])
        for n in range(2, nb):
            p = nb - n - 1
            # \tilde Y
            tY[n] = solve(A[n-1] - B[n-2] @ tY[n-1], C[n], overwrite_a=True)
            # \tilde X
            tX[p] = solve(A[p+1] - C[p+2] @ tX[p+1], B[p], overwrite_a=True)

        self._data.tX = tX
        self._data.tY = tY

    def green(self, format='array'):
        format = format.lower()
        if format in ('array', 'dense'):
            return self._green_array()
        elif format == 'sparse':
            return self._green_sparse()
        elif format == 'btd':
            return self._green_btd()
        raise ValueError(f"{self.__class__.__name__}.green 'format' not valid input [array,sparse,btd]")

    def _green_array(self):
        n = len(self.pvt)
        G = np.empty([n, n], dtype=self._data.A[0].dtype)

        btd = self.btd
        nb = len(btd)
        nbm1 = nb - 1
        sumbs = 0
        A = self._data.A
        B = self._data.B
        C = self._data.C
        tX = self._data.tX
        tY = self._data.tY
        for b, bs in enumerate(btd):
            bsn = btd[b - 1]
            if b < nbm1:
                bsp = btd[b + 1]

            sl0 = slice(sumbs, sumbs + bs)

            # Calculate diagonal part
            if b == 0:
                G[sl0, sl0] = inv_destroy(A[b] - C[b + 1] @ tX[b])
            elif b == nbm1:
                G[sl0, sl0] = inv_destroy(A[b] - B[b - 1] @ tY[b])
            else:
                G[sl0, sl0] = inv_destroy(A[b] - B[b - 1] @ tY[b] - C[b + 1] @ tX[b])

            # Do above
            next_sum = sumbs
            slp = sl0
            for a in range(b - 1, 0, -1):
                # Calculate all parts above
                sla = slice(next_sum - btd[a], next_sum)
                G[sla, sl0] = - tY[a + 1] @ G[slp, sl0]
                slp = sla
                next_sum -= btd[a]

            sl0 = slice(sumbs, sumbs + bs)

            # Step block
            sumbs += bs

            # Do below
            next_sum = sumbs
            slp = sl0
            for a in range(b + 1, nb - 1):
                # Calculate all parts above
                sla = slice(next_sum, next_sum + btd[a])
                G[sla, sl0] = - tX[a - 1] @ G[slp, sl0]
                slp = sla
                next_sum += btd[a]

        return G

    def _green_btd(self):
        btd = self.btd
        G = BTD(btd)
        nbm1 = len(btd) - 1
        A = self._data.A
        B = self._data.B
        C = self._data.C
        tX = self._data.tX
        tY = self._data.tY
        for b, bs in enumerate(btd):
            # Calculate diagonal part
            if b == 0:
                G11 = inv_destroy(A[b] - C[b + 1] @ tX[b])
                G[b, b]
            elif b == nbm1:
                G11 = inv_destroy(A[b] - B[b - 1] @ tY[b])
            else:
                G11 = inv_destroy(A[b] - B[b - 1] @ tY[b] - C[b + 1] @ tX[b])

            # Do above
            G[b, b] = G11
            if b > 0:
                G[b, b-1] = - tY[b] @ G11
            if b < nbm1:
                G[b, b+1] = - tX[b] @ G11

        return G

    def _green_sparse(self):
        n = len(self.pvt)

        # create a sparse matrix
        G = self.H.Sk(format='csr', dtype=self._data.A[0].dtype)
        # pivot the matrix
        G = G[self.pvt, :][:, self.pvt]

        # Get row and column entries
        ncol = np.diff(G.indptr)
        row = (ncol > 0).nonzero()[0]
        # Now we have [0 0 0 0 1 1 1 1 2 2 ... no-1 no-1]
        row = np.repeat(row.astype(np.int32, copy=False), ncol[row])
        col = G.indices

        def get_idx(row, col, row_b, col_b=None):
            if col_b is None:
                col_b = row_b
            idx = (row_b[0] <= row).nonzero()[0]
            idx = idx[row[idx] < row_b[1]]
            idx = idx[col_b[0] <= col[idx]]
            return idx[col[idx] < col_b[1]]

        btd = self.btd
        nb = len(btd)
        nbm1 = nb - 1
        A = self._data.A
        B = self._data.B
        C = self._data.C
        tX = self._data.tX
        tY = self._data.tY
        sumbsn, sumbs, sumbsp = 0, 0, 0
        for b, bs in enumerate(btd):
            sumbsp = sumbs + bs
            if b < nbm1:
                bsp = btd[b + 1]

            # Calculate diagonal part
            if b == 0:
                GM = inv_destroy(A[b] - C[b + 1] @ tX[b])
            elif b == nbm1:
                GM = inv_destroy(A[b] - B[b - 1] @ tY[b])
            else:
                GM = inv_destroy(A[b] - B[b - 1] @ tY[b] - C[b + 1] @ tX[b])

            # get all entries where G is non-zero
            idx = get_idx(row, col, (sumbs, sumbsp))
            G.data[idx] = GM[row[idx] - sumbs, col[idx] - sumbs]

            # check if we should do block above
            if b > 0:
                idx = get_idx(row, col, (sumbsn, sumbs), (sumbs, sumbsp))
                if len(idx) > 0:
                    G.data[idx] = -(tY[b] @ GM)[row[idx] - sumbsn, col[idx] - sumbs]

            # check if we should do block below
            if b < nbm1:
                idx = get_idx(row, col, (sumbsp, sumbsp + bsp), (sumbs, sumbsp))
                if len(idx) > 0:
                    G.data[idx] = -(tX[b] @ GM)[row[idx] - sumbsp, col[idx] - sumbs]

            bsn = bs
            sumbsn = sumbs
            sumbs += bs

        return G

    def spectral(self, elec, format='array', method='column'):
        elec = self._elec(elec)
        format = format.lower()
        method = method.lower()
        if format in ('array', 'dense'):
            if method == 'column':
                return self._spectral_column(elec)
            elif method == 'propagate':
                return self._spectral_propagate(elec)
        raise ValueError(f"{self.__class__.__name__}.spectral format/method not recognized.")

    def _spectral_column(self, elec):
        # To calculate the full A we simply calculate the
        # G column where the electrode resides
        nb = len(self.btd)
        nbm1 = nb - 1

        # These are the indices in the device (after pivoting)
        # So they refer to the
        idx = self.elec[elec].pvt_dev

        # Find parts we need to calculate
        block1 = (idx.min() < self.btd_cum).nonzero()[0][0]
        block2 = (idx.max() < self.btd_cum).nonzero()[0][0]
        if block1 == block2:
            blocks = [block1]
        else:
            blocks = [block1, block2]

        # We can only have 2 consecutive blocks for
        # a Gamma, so same for BTD
        assert len(blocks) <= 2

        n = len(self)
        G = np.empty([n, len(idx)], dtype=self._data.A[0].dtype)

        c = np.append(0, self.btd_cum)
        A = self._data.A
        B = self._data.B
        C = self._data.C
        tX = self._data.tX
        tY = self._data.tY
        for b in blocks:
            # Find the indices in the block
            i = idx[c[b] <= idx]
            i = i[i < c[b + 1]].astype(np.int32)

            c_idx = _a.arangei(c[b], c[b + 1]).reshape(-1, 1)
            b_idx = indices_only(c_idx.ravel(), i)

            if b == blocks[0]:
                r_idx = np.arange(len(b_idx))
            else:
                r_idx = np.arange(len(idx) - len(b_idx), len(idx))

            sl = slice(c[b], c[b + 1])
            if b == 0:
                G[sl, r_idx] = inv_destroy(A[b] - C[b + 1] @ tX[b])[:, b_idx]
            elif b == nbm1:
                G[sl, r_idx] = inv_destroy(A[b] - B[b - 1] @ tY[b])[:, b_idx]
            else:
                G[sl, r_idx] = inv_destroy(A[b] - B[b - 1] @ tY[b] - C[b + 1] @ tX[b])[:, b_idx]

            if len(blocks) == 1:
                break

            # Now calculate the thing (above below)
            sl = slice(c[b], c[b + 1])
            if b == blocks[0]:
                # Calculate below
                slp = slice(c[b + 1], c[b + 2])
                G[slp, r_idx] = - tX[b] @ G[sl, r_idx]
            else:
                # Calculate above
                slp = slice(c[b - 1], c[b])
                G[slp, r_idx] = - tY[b] @ G[sl, r_idx]

        # Now we can calculate the Gf column above
        b = blocks[0]
        slp = slice(c[b], c[b + 1])
        for b in range(blocks[0] - 1, -1, -1):
            sl = slice(c[b], c[b + 1])
            G[sl, :] = - tY[b + 1] @ G[slp, :]
            slp = sl

        # All blocks below
        b = blocks[-1]
        slp = slice(c[b], c[b + 1])
        for b in range(blocks[-1] + 1, nb):
            sl = slice(c[b], c[b + 1])
            G[sl, :] = - tX[b - 1] @ G[slp, :]
            slp = sl

        # Now calculate the full spectral function
        return G @ self._data.gamma[elec] @ dagger(G)

    def _spectral_propagate(self, elec):
        nb = len(self.btd)
        nbm1 = nb - 1

        # First we need to calculate diagonal blocks of the spectral matrix
        blocks, A = self._green_diag_block(self.elec[elec].pvt_dev.ravel())
        A = A @ self._data.gamma[elec] @ dagger(A)

        # Allocate space for the full matrix
        S = np.empty([len(self), len(self)], dtype=A.dtype)

        c = np.append(0, self.btd_cum)
        S[c[blocks[0]]:c[blocks[-1]+1], c[blocks[0]]:c[blocks[-1]+1]] = A
        del A

        # now loop backwards
        tX = self._data.tX
        tY = self._data.tY

        def left(i, j):
            """ Calculate the next block LEFT of block (i,j) """
            if j <= 0:
                return
            #print('left', i, j)
            ij = slice(c[i], c[i+1]), slice(c[j], c[j+1])
            ijm1 = ij[0], slice(c[j-1], c[j])
            S[ijm1] = - S[ij] @ dagger(tY[j])

        def right(i, j):
            """ Calculate the next block RIGHT of block (i,j) """
            if nbm1 <= j:
                return
            #print('right', i, j)
            ij = slice(c[i], c[i+1]), slice(c[j], c[j+1])
            ijp1 = ij[0], slice(c[j+1], c[j+2])
            S[ijp1] = - S[ij] @ dagger(tX[j])

        def above(i, j):
            """ Calculate the next block ABOVE of block (i,j) """
            if i <= 0:
                return
            #print('above', i, j)
            ij = slice(c[i], c[i+1]), slice(c[j], c[j+1])
            im1j = slice(c[i-1], c[i]), ij[1]
            S[im1j] = - tY[i] @ S[ij]
            del ij, im1j
            above(i-1, j)

        def below(i, j):
            """ Calculate the next block BELOW of block (i,j) """
            if nbm1 <= i:
                return
            #print('below', i, j)
            ij = slice(c[i], c[i+1]), slice(c[j], c[j+1])
            ip1j = slice(c[i+1], c[i+2]), ij[1]
            S[ip1j] = - tX[i] @ S[ij]
            del ij, ip1j
            below(i+1, j)

        if len(blocks) == 1:
            for b in range(blocks[0], -1, -1):
                left(blocks[0], b)
                above(blocks[0], b)
                below(blocks[0], b)

            # to grab first block on the right
            right(blocks[0], blocks[0])
            for b in range(blocks[0] + 1, nb):
                right(blocks[0], b)
                above(blocks[0], b)
                below(blocks[0], b)
        else:
            for b in range(blocks[0], -1, -1):
                left(blocks[0], b)
                above(blocks[0], b)
                left(blocks[1], b)
                below(blocks[1], b)

            # calculating everything above/below
            # the 2nd block
            above(blocks[0], blocks[1])
            below(blocks[1], blocks[1])
            # to grab first blocks on the right
            right(blocks[0], blocks[1])
            right(blocks[1], blocks[1])
            for b in range(blocks[1] + 1, nb):
                right(blocks[0], b)
                above(blocks[0], b)
                right(blocks[1], b)
                below(blocks[1], b)
        return S

    def _scattering_state_reduce(self, elec, DOS, U, cutoff):
        """ U on input is a fortran-index as returned from eigh """
        # Select only the first N components where N is the
        # number of orbitals in the electrode (there can't be
        # any more propagating states anyhow).
        N = len(self.elec[elec].pvt_dev)
        # this assumes DOS is ordered
        DOS = DOS[-N:]
        U = U[:, -N:]

        if cutoff > 0:
            idx = (DOS > cutoff).nonzero()[0]
            DOS = DOS[idx]
            U = U[:, idx]

        return DOS, U

    def scattering_state_from_spectral(self, A, elec, cutoff=0., method='full', *args, **kwargs):
        """ On entry `A` contains the spectral function in format appropriate for the method.

        This routine will change the values in `A`. So retain a copy if needed.
        """
        elec = self._elec(elec)
        method = method.lower()
        if method == 'full':
            return self._scattering_state_from_spectral_full(A, elec, cutoff, *args, **kwargs)
        elif method == 'propagate':
            return self._scattering_state_from_spectral_propagate(A, elec, cutoff, *args, **kwargs)
        raise ValueError(f"{self.__class__.__name__}.scattering_state_from_spectral method is not [full,propagate]")

    def _scattering_state_from_spectral_full(self, A, elec, cutoff):
        # add something to the diagonal (improves diag precision for small states)
        np.fill_diagonal(A, A.diagonal() + 0.1)

        # Now diagonalize A
        DOS, A = eigh_destroy(A)
        # backconvert diagonal
        DOS -= 0.1
        # TODO check with overlap convert with correct magnitude (Tr[A] / 2pi)
        DOS /= 2 * np.pi
        DOS, A = self._scattering_state_reduce(elec, DOS, A, cutoff)

        data = self._data
        info = dict(
            elec=self._elec_name(elec),
            E=data.E,
            k=data.k,
            eta=data.eta,
            cutoff=cutoff
        )

        # always have the first state with the largest values
        return si.physics.StateCElectron(A.T[::-1], DOS[::-1], self, **info)

    def _scattering_state_from_spectral_propagate(self, blocks_A, elec, cutoff):
        blocks, U = blocks_A

        # add something to the diagonal (improves diag precision)
        np.fill_diagonal(U, U.diagonal() + 0.1)

        # Calculate eigenvalues
        DOS, U = eigh_destroy(U)
        # backconvert diagonal
        DOS -= 0.1
        # TODO check with overlap convert with correct magnitude (Tr[A] / 2pi)
        DOS /= 2 * np.pi

        # Remove states for cutoff and size
        # Since there cannot be any addition of states later, we
        # can do the reduction here.
        DOS, U = self._scattering_state_reduce(elec, DOS, U, cutoff)

        nb = len(self.btd)
        u = [None] * nb
        u[blocks[0]] = U[:self.btd[blocks[0]], :]
        if len(blocks) > 1:
            u[blocks[1]] = U[self.btd[blocks[0]]:, :]

        # Clean up
        del U

        # Propagate U in the full BTD matrix
        t = self._data.tY
        for b in range(blocks[0], 0, -1):
            u[b - 1] = - t[b] @ u[b]

        t = self._data.tX
        for b in range(blocks[-1], nb - 1):
            u[b + 1] = - t[b] @ u[b]

        # Now the full U is created (C-order), but the DOS is *not* correct
        u = np.concatenate(u).T

        # reflects the DOS in the electrode region
        # this is:
        #    Diag[u^H u]
        unorm = einsum('ij,ij->i', conj(u), u).real
        DOS *= unorm

        # We could check the orthogonality via this:
        #  unorm = conj(u) @ u.T
        #  np.fill_diagonal(unorm, 1j * unorm.diagonal().imag)
        #  max_non_ortho = np.absolute(unorm).max()
        # Since we don't know the real part, we retain the imaginary values.
        # For a non-propagated version, the above should yield a
        # diagonal matrix with 1's.
        # Also, note the above mentioning of the Gram-Schmidt orthogonalization.

        # And now rescale the eigenvectors for unity
        u /= unorm.reshape(-1, 1) ** 0.5

        # We then need to sort again since the eigenvalues may change
        idx = np.argsort(-DOS)

        # Now we have the full u, create it and transpose to get it in C indexing
        data = self._data
        info = dict(
            elec=self._elec_name(elec),
            E=data.E,
            k=data.k,
            eta=data.eta,
            cutoff=cutoff
        )
        return si.physics.StateCElectron(u[idx], DOS[idx], self, **info)

    def scattering_state(self, elec, cutoff=0., method='full', *args, **kwargs):
        elec = self._elec(elec)
        method = method.lower()
        if method == 'full':
            return self._scattering_state_full(elec, cutoff, *args, **kwargs)
        elif method == 'propagate':
            return self._scattering_state_propagate(elec, cutoff, *args, **kwargs)
        raise ValueError(f"{self.__class__.__name__}.scattering_state method is not [full,propagate]")

    def _scattering_state_full(self, elec, cutoff=0., **kwargs):
        A = self.spectral(elec, **kwargs)
        return self._scattering_state_from_spectral_full(A, elec, cutoff)

    def _scattering_state_propagate(self, elec, cutoff=0):
        # First we need to calculate diagonal blocks of the spectral matrix
        # This is basically the same thing as calculating the Gf column
        # But only in the 1/2 diagonal blocks of Gf
        blocks, A = self._green_diag_block(self.elec[elec].pvt_dev.ravel())

        # Calculate the spectral function only for the blocks that host the
        # scattering matrix
        A = A @ self._data.gamma[elec] @ dagger(A)
        return self._scattering_state_from_spectral_propagate((blocks, A), elec, cutoff)

    def eigen_channel(self, state, elec_to):
        if isinstance(elec_to, (Integral, str)):
            elec_to = [elec_to]
        elec_to = [self._elec(e) for e in elec_to]

        # Retrive the scattering states `A`
        A = state.state
        # The sign shouldn't really matter since the states should always
        # have a finite DOS, however, for completeness sake we retain the sign.
        sqDOS = (np.sign(state.c) * np.sqrt(np.fabs(state.c))).reshape(-1, 1)

        # create shorthands
        elec = self.elec
        G = self._data.gamma

        # Create the first electrode
        el = elec_to[0]
        idx = elec[el].pvt_dev.ravel()
        u = A[:, idx] * sqDOS
        # the summed transmission matrix
        Ut = u @ G[el] @ dagger(u)
        for el in elec_to[1:]:
            idx = elec[el].pvt_dev.ravel()
            u = A[:, idx] * sqDOS
            Ut += u @ G[el] @ dagger(u)

        # TODO currently a factor depends on what is used
        #      in `scattering_states`, so go check there.
        #      The resulting Ut should have a factor: 1 / 2pi ** 0.5
        #      When the states DOS values (`state.c`) has the factor 1 / 2pi
        #      then `u` has the correct magnitude and all we need to do is to add the factor 2pi
        # diagonalize the transmission matrix tt
        tt, Ut = eigh_destroy(Ut)
        tt *= 2 * np.pi

        info = {**state.info}
        info["elec_to"] = [self._elec_name(e) for e in elec_to]

        # Backtransform U to form the eigenchannels
        return si.physics.StateCElectron((Ut.T @ A)[::-1, :],
                                         tt[::-1], self, **info)

    def _green_diag_block(self, idx):
        nb = len(self.btd)
        nbm1 = nb - 1

        # Find parts we need to calculate
        block1 = (idx.min() < self.btd_cum).nonzero()[0][0]
        block2 = (idx.max() < self.btd_cum).nonzero()[0][0]
        if block1 == block2:
            blocks = [block1]
        else:
            blocks = list(range(block1, block2+1))
        assert len(blocks) <= 2

        n = self.btd[blocks].sum()
        G = np.empty([n, len(idx)], dtype=self._data.A[0].dtype)

        btd = self.btd
        c = np.append(0, self.btd_cum)
        A = self._data.A
        B = self._data.B
        C = self._data.C
        tX = self._data.tX
        tY = self._data.tY
        for b in blocks:
            # Find the indices in the block
            i = idx[c[b] <= idx].copy()
            i = i[i < c[b + 1]].astype(np.int32)

            c_idx = _a.arangei(c[b], c[b + 1]).reshape(-1, 1)
            b_idx = indices_only(c_idx.ravel(), i)
            # Subtract the first block to put it only in the sub-part
            c_idx -= c[blocks[0]]

            if b == blocks[0]:
                sl = slice(0, btd[b])
                r_idx = np.arange(len(b_idx))
            else:
                sl = slice(btd[blocks[0]], btd[blocks[0]] + btd[b])
                r_idx = np.arange(len(idx) - len(b_idx), len(idx))

            if b == 0:
                G[sl, r_idx] = inv_destroy(A[b] - C[b + 1] @ tX[b])[:, b_idx]
            elif b == nbm1:
                G[sl, r_idx] = inv_destroy(A[b] - B[b - 1] @ tY[b])[:, b_idx]
            else:
                G[sl, r_idx] = inv_destroy(A[b] - B[b - 1] @ tY[b] - C[b + 1] @ tX[b])[:, b_idx]

            if len(blocks) == 1:
                break

            # Now calculate the thing (below/above)
            if b == blocks[0]:
                # Calculate below
                slp = slice(btd[b], btd[b] + btd[blocks[1]])
                G[slp, r_idx] = - tX[b] @ G[sl, r_idx]
            else:
                # Calculate above
                slp = slice(0, btd[blocks[0]])
                G[slp, r_idx] = - tY[b] @ G[sl, r_idx]

        return blocks, G

    def reset(self):
        self._data = PropertyDict()
