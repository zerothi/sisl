"""
Wrapper routines for creating partial function calls with other defaults
"""

import functools as ftool
import numpy as np
import numpy.linalg as nl
import scipy as sp
import scipy.linalg as sl
import scipy.sparse.linalg as ssl

_partial = ftool.partial

# Create __all__
__all__ = []

# Create all partial objects for creating arrays
zerosi = _partial(np.zeros, dtype=np.int32)
zerosl = _partial(np.zeros, dtype=np.int64)
zerosf = _partial(np.zeros, dtype=np.float32)
zerosd = _partial(np.zeros, dtype=np.float64)
__all__ += ['zeros' + s for s in 'ilfd']

onesi = _partial(np.ones, dtype=np.int32)
onesl = _partial(np.ones, dtype=np.int64)
onesf = _partial(np.ones, dtype=np.float32)
onesd = _partial(np.ones, dtype=np.float64)
__all__ += ['ones' + s for s in 'ilfd']

emptyi = _partial(np.empty, dtype=np.int32)
emptyl = _partial(np.empty, dtype=np.int64)
emptyf = _partial(np.empty, dtype=np.float32)
emptyd = _partial(np.empty, dtype=np.float64)
__all__ += ['empty' + s for s in 'ilfd']

arrayi = _partial(np.array, dtype=np.int32)
arrayl = _partial(np.array, dtype=np.int64)
arrayf = _partial(np.array, dtype=np.float32)
arrayd = _partial(np.array, dtype=np.float64)
__all__ += ['array' + s for s in 'ilfd']

asarrayi = _partial(np.asarray, dtype=np.int32)
asarrayl = _partial(np.asarray, dtype=np.int64)
asarrayf = _partial(np.asarray, dtype=np.float32)
asarrayd = _partial(np.asarray, dtype=np.float64)
__all__ += ['asarray' + s for s in 'ilfd']

sumi = _partial(np.sum, dtype=np.int32)
suml = _partial(np.sum, dtype=np.int64)
sumf = _partial(np.sum, dtype=np.float32)
sumd = _partial(np.sum, dtype=np.float64)
__all__ += ['sum' + s for s in 'ilfd']

cumsumi = _partial(np.cumsum, dtype=np.int32)
cumsuml = _partial(np.cumsum, dtype=np.int64)
cumsumf = _partial(np.cumsum, dtype=np.float32)
cumsumd = _partial(np.cumsum, dtype=np.float64)
__all__ += ['cumsum' + s for s in 'ilfd']

arangei = _partial(np.arange, dtype=np.int32)
arangel = _partial(np.arange, dtype=np.int64)
arangef = _partial(np.arange, dtype=np.float32)
aranged = _partial(np.arange, dtype=np.float64)
__all__ += ['arange' + s for s in 'ilfd']


# Create all partial objects for linear algebra
solve = _partial(sl.solve, check_finite=False, overwrite_a=False, overwrite_b=False)
solve_herm = _partial(sl.solve, check_finite=False, overwrite_a=False, overwrite_b=False, assume_a='her')
solve_sym = _partial(sl.solve, check_finite=False, overwrite_a=False, overwrite_b=False, assume_a='sym')
__all__ += ['solve', 'solve_herm', 'solve_sym']

solve_destroy = _partial(sl.solve, check_finite=False, overwrite_a=True, overwrite_b=True)
solve_herm_destroy = _partial(sl.solve, check_finite=False, overwrite_a=True, overwrite_b=True, assume_a='her')
solve_sym_destroy = _partial(sl.solve, check_finite=False, overwrite_a=True, overwrite_b=True, assume_a='sym')
__all__ += ['solve_destroy', 'solve_herm_destroy', 'solve_sym_destroy']

# Inversion of matrix
inv = _partial(sl.inv, check_finite=False, overwrite_a=False)
__all__ += ['inv']
inv_destroy = _partial(sl.inv, check_finite=False, overwrite_a=True)
__all__ += ['inv_destroy']

# Solve eigenvalue problem
eig = _partial(sl.eig, check_finite=False, overwrite_a=False, overwrite_b=False)
eig_left = _partial(sl.eig, check_finite=False, overwrite_a=False, overwrite_b=False, left=True)
eig_right = _partial(sl.eig, check_finite=False, overwrite_a=False, overwrite_b=False, right=True)
__all__ += ['eig', 'eig_left', 'eig_right']

# Solve eigenvalue problem
eig_destroy = _partial(sl.eig, check_finite=False, overwrite_a=True, overwrite_b=True)
eig_left_destroy = _partial(sl.eig, check_finite=False, overwrite_a=True, overwrite_b=True, left=True)
eig_right_destroy = _partial(sl.eig, check_finite=False, overwrite_a=True, overwrite_b=True, right=True)
__all__ += ['eig_destroy', 'eig_left_destroy', 'eig_right_destroy']

# Solve symmetric/hermitian eigenvalue problem (generic == no overwrite)
eigh = _partial(sl.eigh, check_finite=False, overwrite_a=False, overwrite_b=False, turbo=True)
eigh_dc = _partial(sl.eigh, check_finite=False, overwrite_a=False, overwrite_b=False, turbo=True)
eigh_qr = _partial(sl.eigh, check_finite=False, overwrite_a=False, overwrite_b=False, turbo=False)
__all__ += ['eigh', 'eigh_dc', 'eigh_qr']

# Solve symmetric/hermitian eigenvalue problem (allow overwrite)
eigh_destroy = _partial(sl.eigh, check_finite=False, overwrite_a=True, overwrite_b=True, turbo=True)
eigh_dc_destroy = _partial(sl.eigh, check_finite=False, overwrite_a=True, overwrite_b=True, turbo=True)
eigh_qr_destroy = _partial(sl.eigh, check_finite=False, overwrite_a=True, overwrite_b=True, turbo=False)
__all__ += ['eigh_destroy', 'eigh_dc_destroy', 'eigh_qr_destroy']


# Sparse linalg routines

# Solve eigenvalue problem
eigs = ssl.eigs
__all__ += ['eigs']

# Solve symmetric/hermitian eigenvalue problem (generic == no overwrite)
eigsh = ssl.eigsh
__all__ += ['eigsh']
