from functools import partial as _partial

import numpy as np

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

# Create all partial objects for creating arrays
fulli = _partial(np.full, dtype=np.int32)
fulll = _partial(np.full, dtype=np.int64)
fullf = _partial(np.full, dtype=np.float32)
fulld = _partial(np.full, dtype=np.float64)
__all__ += _append('full')
