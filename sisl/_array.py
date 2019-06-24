from functools import partial as _partial

import numpy as np

__all__ = []


def _append(name, suffix='ilfd'):
    return [name + s for s in suffix]


# Create all partial objects for creating arrays
zerosi = _partial(np.zeros, dtype=np.int32)
zerosl = _partial(np.zeros, dtype=np.int64)
zerosf = _partial(np.zeros, dtype=np.float32)
zerosd = _partial(np.zeros, dtype=np.float64)
zerosc = _partial(np.zeros, dtype=np.complex64)
zerosz = _partial(np.zeros, dtype=np.complex128)
__all__ += _append('zeros', 'ilfdcz')

onesi = _partial(np.ones, dtype=np.int32)
onesl = _partial(np.ones, dtype=np.int64)
onesf = _partial(np.ones, dtype=np.float32)
onesd = _partial(np.ones, dtype=np.float64)
onesc = _partial(np.ones, dtype=np.complex64)
onesz = _partial(np.ones, dtype=np.complex128)
__all__ += _append('ones', 'ilfdcz')

emptyi = _partial(np.empty, dtype=np.int32)
emptyl = _partial(np.empty, dtype=np.int64)
emptyf = _partial(np.empty, dtype=np.float32)
emptyd = _partial(np.empty, dtype=np.float64)
emptyc = _partial(np.empty, dtype=np.complex64)
emptyz = _partial(np.empty, dtype=np.complex128)
__all__ += _append('empty', 'ilfdcz')

arrayi = _partial(np.array, dtype=np.int32)
arrayl = _partial(np.array, dtype=np.int64)
arrayf = _partial(np.array, dtype=np.float32)
arrayd = _partial(np.array, dtype=np.float64)
arrayc = _partial(np.array, dtype=np.complex64)
arrayz = _partial(np.array, dtype=np.complex128)
__all__ += _append('array', 'ilfdcz')

asarray = np.asarray
asarrayi = _partial(np.asarray, dtype=np.int32)
asarrayl = _partial(np.asarray, dtype=np.int64)
asarrayf = _partial(np.asarray, dtype=np.float32)
asarrayd = _partial(np.asarray, dtype=np.float64)
asarrayc = _partial(np.asarray, dtype=np.complex64)
asarrayz = _partial(np.asarray, dtype=np.complex128)
__all__ += _append('asarray', 'ilfdcz') + ['asarray']

fromiteri = _partial(np.fromiter, dtype=np.int32)
fromiterl = _partial(np.fromiter, dtype=np.int64)
fromiterf = _partial(np.fromiter, dtype=np.float32)
fromiterd = _partial(np.fromiter, dtype=np.float64)
fromiterc = _partial(np.fromiter, dtype=np.complex64)
fromiterz = _partial(np.fromiter, dtype=np.complex128)
__all__ += _append('fromiter', 'ilfdcz')

sumi = _partial(np.sum, dtype=np.int32)
suml = _partial(np.sum, dtype=np.int64)
sumf = _partial(np.sum, dtype=np.float32)
sumd = _partial(np.sum, dtype=np.float64)
sumc = _partial(np.sum, dtype=np.complex64)
sumz = _partial(np.sum, dtype=np.complex128)
__all__ += _append('sum', 'ilfdcz')

cumsumi = _partial(np.cumsum, dtype=np.int32)
cumsuml = _partial(np.cumsum, dtype=np.int64)
cumsumf = _partial(np.cumsum, dtype=np.float32)
cumsumd = _partial(np.cumsum, dtype=np.float64)
cumsumc = _partial(np.cumsum, dtype=np.complex64)
cumsumz = _partial(np.cumsum, dtype=np.complex128)
__all__ += _append('cumsum', 'ilfdcz')

arangei = _partial(np.arange, dtype=np.int32)
arangel = _partial(np.arange, dtype=np.int64)
arangef = _partial(np.arange, dtype=np.float32)
aranged = _partial(np.arange, dtype=np.float64)
arangec = _partial(np.arange, dtype=np.complex64)
arangez = _partial(np.arange, dtype=np.complex128)
__all__ += _append('arange', 'ilfdcz')

prodi = _partial(np.prod, dtype=np.int32)
prodl = _partial(np.prod, dtype=np.int64)
prodf = _partial(np.prod, dtype=np.float32)
prodd = _partial(np.prod, dtype=np.float64)
prodc = _partial(np.prod, dtype=np.complex64)
prodz = _partial(np.prod, dtype=np.complex128)
__all__ += _append('prod', 'ilfdcz')

# Create all partial objects for creating arrays
fulli = _partial(np.full, dtype=np.int32)
fulll = _partial(np.full, dtype=np.int64)
fullf = _partial(np.full, dtype=np.float32)
fulld = _partial(np.full, dtype=np.float64)
fullc = _partial(np.full, dtype=np.complex64)
fullz = _partial(np.full, dtype=np.complex128)
__all__ += _append('full', 'ilfdcz')

linspacef = _partial(np.linspace, dtype=np.float32)
linspaced = _partial(np.linspace, dtype=np.float64)
linspacec = _partial(np.linspace, dtype=np.complex64)
linspacez = _partial(np.linspace, dtype=np.complex128)
__all__ += _append('linspace', 'fdcz')

del _append, _partial
