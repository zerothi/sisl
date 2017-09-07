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


def _append(name, suffix=('i', 'l', 'f', 'd')):
    return [name + s for s in suffix]

# Create all partial objects for creating arrays
zerosi = _partial(np.zeros, dtype=np.int32)
zerosl = _partial(np.zeros, dtype=np.int64)
zerosf = _partial(np.zeros, dtype=np.float32)
zerosd = _partial(np.zeros, dtype=np.float64)
__all__ += _append('zeros')

onesi = _partial(np.ones, dtype=np.int32)
onesl = _partial(np.ones, dtype=np.int64)
onesf = _partial(np.ones, dtype=np.float32)
onesd = _partial(np.ones, dtype=np.float64)
__all__ += _append('ones')

emptyi = _partial(np.empty, dtype=np.int32)
emptyl = _partial(np.empty, dtype=np.int64)
emptyf = _partial(np.empty, dtype=np.float32)
emptyd = _partial(np.empty, dtype=np.float64)
__all__ += _append('empty')

arrayi = _partial(np.array, dtype=np.int32)
arrayl = _partial(np.array, dtype=np.int64)
arrayf = _partial(np.array, dtype=np.float32)
arrayd = _partial(np.array, dtype=np.float64)
__all__ += _append('array')

asarrayi = _partial(np.asarray, dtype=np.int32)
asarrayl = _partial(np.asarray, dtype=np.int64)
asarrayf = _partial(np.asarray, dtype=np.float32)
asarrayd = _partial(np.asarray, dtype=np.float64)
__all__ += _append('asarray')

sumi = _partial(np.sum, dtype=np.int32)
suml = _partial(np.sum, dtype=np.int64)
sumf = _partial(np.sum, dtype=np.float32)
sumd = _partial(np.sum, dtype=np.float64)
__all__ += _append('sum')

cumsumi = _partial(np.cumsum, dtype=np.int32)
cumsuml = _partial(np.cumsum, dtype=np.int64)
cumsumf = _partial(np.cumsum, dtype=np.float32)
cumsumd = _partial(np.cumsum, dtype=np.float64)
__all__ += _append('cumsum')

arangei = _partial(np.arange, dtype=np.int32)
arangel = _partial(np.arange, dtype=np.int64)
arangef = _partial(np.arange, dtype=np.float32)
aranged = _partial(np.arange, dtype=np.float64)
__all__ += _append('arange')


# Create all partial objects for linear algebra
solve = _partial(sl.solve, check_finite=False, overwrite_a=False, overwrite_b=False)
solve_herm = _partial(sl.solve, check_finite=False, overwrite_a=False, overwrite_b=False, assume_a='her')
solve_sym = _partial(sl.solve, check_finite=False, overwrite_a=False, overwrite_b=False, assume_a='sym')
__all__ += _append('solve', ['', '_herm', '_sym'])

solve_destroy = _partial(sl.solve, check_finite=False, overwrite_a=True, overwrite_b=True)
solve_herm_destroy = _partial(sl.solve, check_finite=False, overwrite_a=True, overwrite_b=True, assume_a='her')
solve_sym_destroy = _partial(sl.solve, check_finite=False, overwrite_a=True, overwrite_b=True, assume_a='sym')
__all__ += _append('solve_', ['destroy', 'herm_destroy', 'sym_destroy'])

# Inversion of matrix
inv = _partial(sl.inv, check_finite=False, overwrite_a=False)
inv_destroy = _partial(sl.inv, check_finite=False, overwrite_a=True)
__all__ += _append('inv', ['', '_destroy'])

# Solve eigenvalue problem
eig = _partial(sl.eig, check_finite=False, overwrite_a=False, overwrite_b=False)
eig_left = _partial(sl.eig, check_finite=False, overwrite_a=False, overwrite_b=False, left=True)
eig_right = _partial(sl.eig, check_finite=False, overwrite_a=False, overwrite_b=False, right=True)
__all__ += _append('eig', ['', '_left', '_right'])

# Solve eigenvalue problem
eig_destroy = _partial(sl.eig, check_finite=False, overwrite_a=True, overwrite_b=True)
eig_left_destroy = _partial(sl.eig, check_finite=False, overwrite_a=True, overwrite_b=True, left=True)
eig_right_destroy = _partial(sl.eig, check_finite=False, overwrite_a=True, overwrite_b=True, right=True)
__all__ += _append('eig_', ['destroy', 'left_destroy', 'right_destroy'])

# Solve symmetric/hermitian eigenvalue problem (generic == no overwrite)
eigh = _partial(sl.eigh, check_finite=False, overwrite_a=False, overwrite_b=False, turbo=True)
eigh_dc = _partial(sl.eigh, check_finite=False, overwrite_a=False, overwrite_b=False, turbo=True)
eigh_qr = _partial(sl.eigh, check_finite=False, overwrite_a=False, overwrite_b=False, turbo=False)
__all__ += _append('eigh', ['', '_dc', '_qr'])

# Solve symmetric/hermitian eigenvalue problem (allow overwrite)
eigh_destroy = _partial(sl.eigh, check_finite=False, overwrite_a=True, overwrite_b=True, turbo=True)
eigh_dc_destroy = _partial(sl.eigh, check_finite=False, overwrite_a=True, overwrite_b=True, turbo=True)
eigh_qr_destroy = _partial(sl.eigh, check_finite=False, overwrite_a=True, overwrite_b=True, turbo=False)
__all__ += _append('eigh_', ['destroy', 'dc_destroy', 'qr_destroy'])


# Sparse linalg routines

# Solve eigenvalue problem
eigs = ssl.eigs
__all__ += ['eigs']

# Solve symmetric/hermitian eigenvalue problem (generic == no overwrite)
eigsh = ssl.eigsh
__all__ += ['eigsh']
