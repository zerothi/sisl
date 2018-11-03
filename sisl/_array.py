from functools import partial as _partial

import numpy as np

__all__ = []


def _append(name, suffix=('i', 'l', 'f', 'd')):
    return [name + s for s in suffix]


def _append_c(name, suffix=('i', 'l', 'f', 'd', 'c', 'z')):
    return _append(name, suffix)

# Create all partial objects for creating arrays
zerosi = _partial(np.zeros, dtype=np.int32)
zerosl = _partial(np.zeros, dtype=np.int64)
zerosf = _partial(np.zeros, dtype=np.float32)
zerosd = _partial(np.zeros, dtype=np.float64)
zerosc = _partial(np.zeros, dtype=np.complex64)
zerosz = _partial(np.zeros, dtype=np.complex128)
__all__ += _append_c('zeros')

onesi = _partial(np.ones, dtype=np.int32)
onesl = _partial(np.ones, dtype=np.int64)
onesf = _partial(np.ones, dtype=np.float32)
onesd = _partial(np.ones, dtype=np.float64)
onesc = _partial(np.ones, dtype=np.complex64)
onesz = _partial(np.ones, dtype=np.complex128)
__all__ += _append_c('ones')

emptyi = _partial(np.empty, dtype=np.int32)
emptyl = _partial(np.empty, dtype=np.int64)
emptyf = _partial(np.empty, dtype=np.float32)
emptyd = _partial(np.empty, dtype=np.float64)
emptyc = _partial(np.empty, dtype=np.complex64)
emptyz = _partial(np.empty, dtype=np.complex128)
__all__ += _append_c('empty')

arrayi = _partial(np.array, dtype=np.int32)
arrayl = _partial(np.array, dtype=np.int64)
arrayf = _partial(np.array, dtype=np.float32)
arrayd = _partial(np.array, dtype=np.float64)
arrayc = _partial(np.array, dtype=np.complex64)
arrayz = _partial(np.array, dtype=np.complex128)
__all__ += _append_c('array')

asarray = np.asarray
asarrayi = _partial(np.asarray, dtype=np.int32)
asarrayl = _partial(np.asarray, dtype=np.int64)
asarrayf = _partial(np.asarray, dtype=np.float32)
asarrayd = _partial(np.asarray, dtype=np.float64)
__all__ += _append('asarray') + ['asarray']

fromiteri = _partial(np.fromiter, dtype=np.int32)
fromiterl = _partial(np.fromiter, dtype=np.int64)
fromiterf = _partial(np.fromiter, dtype=np.float32)
fromiterd = _partial(np.fromiter, dtype=np.float64)
__all__ += _append('fromiter')

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
arangec = _partial(np.arange, dtype=np.complex64)
arangez = _partial(np.arange, dtype=np.complex128)
__all__ += _append_c('arange')

prodi = _partial(np.prod, dtype=np.int32)
prodl = _partial(np.prod, dtype=np.int64)
prodf = _partial(np.prod, dtype=np.float32)
prodd = _partial(np.prod, dtype=np.float64)
__all__ += _append('prod')

# Create all partial objects for creating arrays
fulli = _partial(np.full, dtype=np.int32)
fulll = _partial(np.full, dtype=np.int64)
fullf = _partial(np.full, dtype=np.float32)
fulld = _partial(np.full, dtype=np.float64)
fullc = _partial(np.full, dtype=np.complex64)
fullz = _partial(np.full, dtype=np.complex128)
__all__ += _append_c('full')


del _append_c, _append
