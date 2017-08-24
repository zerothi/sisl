"""
Wrapper routines for creating partial function calls with other defaults
"""

import functools as ftool
import numpy as np

# Create all partial objects
zerosi = ftool.partial(np.zeros, dtype=np.int32)
zerosl = ftool.partial(np.zeros, dtype=np.int64)
zerosf = ftool.partial(np.zeros, dtype=np.float32)
zerosd = ftool.partial(np.zeros, dtype=np.float64)

emptyi = ftool.partial(np.empty, dtype=np.int32)
emptyl = ftool.partial(np.empty, dtype=np.int64)
emptyf = ftool.partial(np.empty, dtype=np.float32)
emptyd = ftool.partial(np.empty, dtype=np.float64)

arrayi = ftool.partial(np.array, dtype=np.int32)
arrayl = ftool.partial(np.array, dtype=np.int64)
arrayf = ftool.partial(np.array, dtype=np.float32)
arrayd = ftool.partial(np.array, dtype=np.float64)

asarrayi = ftool.partial(np.asarray, dtype=np.int32)
asarrayl = ftool.partial(np.asarray, dtype=np.int64)
asarrayf = ftool.partial(np.asarray, dtype=np.float32)
asarrayd = ftool.partial(np.asarray, dtype=np.float64)

sumi = ftool.partial(np.sum, dtype=np.int32)
suml = ftool.partial(np.sum, dtype=np.int64)
sumf = ftool.partial(np.sum, dtype=np.float32)
sumd = ftool.partial(np.sum, dtype=np.float64)

cumsumi = ftool.partial(np.cumsum, dtype=np.int32)
cumsuml = ftool.partial(np.cumsum, dtype=np.int64)
cumsumf = ftool.partial(np.cumsum, dtype=np.float32)
cumsumd = ftool.partial(np.cumsum, dtype=np.float64)

arangei = ftool.partial(np.arange, dtype=np.int32)
arangel = ftool.partial(np.arange, dtype=np.int64)
arangef = ftool.partial(np.arange, dtype=np.float32)
aranged = ftool.partial(np.arange, dtype=np.float64)
