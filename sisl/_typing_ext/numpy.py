from sisl.utils.misc import PropertyDict
import numpy as np

__all__ = ["npt"]

if tuple(map(int, np.__version__.split('.'))) >= (1, 21, 0):
    # NDArray entered in 1.21.
    # numpy.typing entered in 1.20.0
    # we have numpy typing
    import numpy.typing as npt
else:
    npt = PropertyDict()
    npt.ArrayLike = "ArrayLike"
    npt.NDArray = "NDArray"
    np.DTypeLike = "DTypeLike"
