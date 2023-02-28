from numpy import __version__

if tuple(map(int, __version__.split('.'))) >= (1, 21, 0):
    # NDArray entered in 1.21.
    # numpy.typing entered in 1.20.0
    # we have numpy typing
    from numpy.typing import *
else:
    ArrayLike = "ArrayLike"
    NDArray = "NDArray"
    DTypeLike = "DTypeLike"
