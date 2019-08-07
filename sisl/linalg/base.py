from functools import partial as _partial

# Create a _copy_ of the scipy.linalg.solve routine and implement
# our own refine keyword.
import numpy as np
from numpy import atleast_1d, atleast_2d
from scipy.linalg.blas import get_blas_funcs
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.misc import LinAlgError, _datacopied
from scipy._lib._util import _asarray_validated

import scipy.linalg as sl
import scipy.sparse.linalg as ssl


__all__ = ['linalg_info']


# Placeholder for basic linear algebra methods
# I.e. when fetching the same method over and over
# we should be able to reduce the overhead by retrieving the intrinsic version.
_linalg_info_dtype = {
    np.float32: 'f4',
    np.float64: 'f8',
    np.complex64: 'c8',
    np.complex128: 'c16',
    'f4': 'f4',
    'f8': 'f8',
    'c8': 'c8',
    'c16': 'c16',
}
_linalg_info_base = {}
# Initialize the base-dtype dicts
for _, item in _linalg_info_dtype.items():
    _linalg_info_base[item] = {}


def linalg_info(method, dtype, method_dict=_linalg_info_base, dtype_dict=_linalg_info_dtype):
    """ Faster BLAS/LAPACK methods to be returned without too many lookups an array checks

    Parameters
    ----------
    method : str
       BLAS/LAPACK instance to retrieve
    dtype : numpy.dtype
       matrix corresponding data-type

    Returns
    -------
    Function to call corresponding to method `method` in precision `dtype`.

    Raises
    ------
    ValueError: if the corresponding method is not present
    """
    # dtype as string
    dtype_str = dtype_dict[dtype]

    # Get dictionary for methods
    m_dict = method_dict[dtype_str]

    # Check if it exists
    if method in m_dict:
        return m_dict[method]

    # Get the corresponding method and store it before returning it
    try:
        func = get_lapack_funcs(method, dtype=dtype)
    except ValueError as e:
        if 'LAPACK function' in str(e):
            func = get_blas_funcs(method, dtype=dtype)
        else:
            raise e
    m_dict[method] = func
    return func


def _compute_lwork(routine, *args, **kwargs):
    """ See scipy.linalg.lapack._compute_lwork """
    wi = routine(*args, **kwargs)
    if len(wi) < 2:
        raise ValueError('')
    info = wi[-1]
    if info != 0:
        raise ValueError("Internal work array size computation failed: "
                         "%d" % (info,))

    lwork = [w.real for w in wi[:-1]]

    dtype = getattr(routine, 'dtype', None)
    if dtype == np.float32 or dtype == np.complex64:
        # Single-precision routine -- take next fp value to work
        # around possible truncation in LAPACK code
        lwork = np.nextafter(lwork, np.inf, dtype=np.float32)

    lwork = np.array(lwork, np.int64)
    if np.any(np.logical_or(lwork < 0, lwork > np.iinfo(np.int32).max)):
        raise ValueError("Too large work array required -- computation cannot "
                         "be performed with standard 32-bit LAPACK.")
    lwork = lwork.astype(np.int32)
    if lwork.size == 1:
        return lwork[0]
    return lwork


def inv(a, overwrite_a=False):
    """
    Inverts a matrix

    Parameters
    ----------
    a : (N, N) array_like
       the matrix to be inverted.
    overwrite_a : bool, optional
       whether we are allowed to overwrite the matrix `a`

    Returns
    -------
    x : (N, N) numpy.ndarray
        The inverted matrix
    """
    a1 = atleast_2d(_asarray_validated(a, check_finite=False))

    overwrite_a = overwrite_a or _datacopied(a1, a)

    if a1.shape[0] != a1.shape[1]:
        raise ValueError('Input a needs to be a square matrix.')

    getrf, getri, getri_lwork = get_lapack_funcs(('getrf', 'getri',
                                                  'getri_lwork'), (a1,))
    lu, piv, info = getrf(a1, overwrite_a=overwrite_a)
    if info == 0:
        lwork = _compute_lwork(getri_lwork, a1.shape[0])
        lwork = int(1.01 * lwork)
        x, info = getri(lu, piv, lwork=lwork, overwrite_lu=True)
    if info > 0:
        raise LinAlgError("Singular matrix")
    elif info < 0:
        raise ValueError('illegal value in %d-th argument of internal '
                         'getrf|getri' % -info)
    return x


def solve(a, b, overwrite_a=False, overwrite_b=False):
    """
    Solve a linear system ``a x = b``

    Parameters
    ----------
    a : (N, N) array_like
       left-hand-side matrix
    b : (N, NRHS) array_like
       right-hand-side matrix
    overwrite_a : bool, optional
       whether we are allowed to overwrite the matrix `a`
    overwrite_b : bool, optional
       whether we are allowed to overwrite the matrix `b`

    Returns
    -------
    x : (N, NRHS) numpy.ndarray
        solution matrix
    """
    a1 = atleast_2d(_asarray_validated(a, check_finite=False))
    b1 = atleast_1d(_asarray_validated(b, check_finite=False))
    n = a1.shape[0]

    overwrite_a = overwrite_a or _datacopied(a1, a)
    overwrite_b = overwrite_b or _datacopied(b1, b)

    if a1.shape[0] != a1.shape[1]:
        raise ValueError('LHS needs to be a square matrix.')

    if n != b1.shape[0]:
        # Last chance to catch 1x1 scalar a and 1D b arrays
        if not (n == 1 and b1.size != 0):
            raise ValueError('Input b has to have same number of rows as '
                             'input a')

    # regularize 1D b arrays to 2D
    if b1.ndim == 1:
        if n == 1:
            b1 = b1[None, :]
        else:
            b1 = b1[:, None]
        b_is_1D = True
    else:
        b_is_1D = False

    gesv = get_lapack_funcs('gesv', (a1, b1))
    _, _, x, info = gesv(a1, b1, overwrite_a=overwrite_a, overwrite_b=overwrite_b)
    if info > 0:
        raise LinAlgError("Singular matrix")
    elif info < 0:
        raise ValueError('illegal value in %d-th argument of internal '
                         'gesv' % -info)
    if b_is_1D:
        return x.ravel()

    return x


def _append(name, suffix):
    return [name + s for s in suffix]

# Solving a linear system
solve_destroy = _partial(solve, overwrite_a=True, overwrite_b=True)
__all__ += _append('solve', ['', '_destroy'])

# Inversion of matrix
inv_destroy = _partial(inv, overwrite_a=True)
__all__ += _append('inv', ['', '_destroy'])

# Solve eigenvalue problem
eig = _partial(sl.eig, check_finite=False, overwrite_a=False, overwrite_b=False)
eig_left = _partial(sl.eig, check_finite=False, overwrite_a=False, overwrite_b=False, left=True)
eig_right = _partial(sl.eig, check_finite=False, overwrite_a=False, overwrite_b=False, right=True)
__all__ += _append('eig', ['', '_left', '_right'])

eig_destroy = _partial(sl.eig, check_finite=False, overwrite_a=True, overwrite_b=True)
eig_left_destroy = _partial(sl.eig, check_finite=False, overwrite_a=True, overwrite_b=True, left=True)
eig_right_destroy = _partial(sl.eig, check_finite=False, overwrite_a=True, overwrite_b=True, right=True)
__all__ += _append('eig_', ['destroy', 'left_destroy', 'right_destroy'])

eigvals = _partial(sl.eigvals, check_finite=False, overwrite_a=False)
eigvals_destroy = _partial(sl.eigvals, check_finite=False, overwrite_a=True)
__all__ += _append('eigvals', ['', '_destroy'])

# Solve symmetric/hermitian eigenvalue problem (generic == no overwrite)
eigh = _partial(sl.eigh, check_finite=False, overwrite_a=False, overwrite_b=False, turbo=True)
eigh_dc = eigh
eigh_qr = _partial(sl.eigh, check_finite=False, overwrite_a=False, overwrite_b=False, turbo=False)
__all__ += _append('eigh', ['', '_dc', '_qr'])

# Solve symmetric/hermitian eigenvalue problem (allow overwrite)
eigh_destroy = _partial(sl.eigh, check_finite=False, overwrite_a=True, overwrite_b=True, turbo=True)
eigh_dc_destroy = eigh_destroy
eigh_qr_destroy = _partial(sl.eigh, check_finite=False, overwrite_a=True, overwrite_b=True, turbo=False)
__all__ += _append('eigh_', ['destroy', 'dc_destroy', 'qr_destroy'])

eigvalsh = _partial(sl.eigvalsh, check_finite=False, overwrite_a=False, overwrite_b=False, turbo=True)
eigvalsh_dc = eigvalsh
eigvalsh_qr = _partial(sl.eigvalsh, check_finite=False, overwrite_a=False, overwrite_b=False, turbo=False)
__all__ += _append('eigvalsh', ['', '_dc', '_qr'])

eigvalsh_destroy = _partial(sl.eigvalsh, check_finite=False, overwrite_a=True, overwrite_b=True, turbo=True)
eigvalsh_dc_destroy = eigvalsh_destroy
eigvalsh_qr_destroy = _partial(sl.eigvalsh, check_finite=False, overwrite_a=True, overwrite_b=True, turbo=False)
__all__ += _append('eigvalsh_', ['destroy', 'dc_destroy', 'qr_destroy'])

# SVD problem
svd = _partial(sl.svd, check_finite=False, overwrite_a=False)
svd_destroy = _partial(sl.svd, check_finite=False, overwrite_a=True)
__all__ += _append('svd', ['', '_destroy'])

# Determinants
det = _partial(sl.det, check_finite=False, overwrite_a=False)
det_destroy = _partial(sl.det, check_finite=False, overwrite_a=True)
__all__ += _append('det', ['', '_destroy'])

# Sparse linalg routines

# Solve eigenvalue problem
eigs = ssl.eigs
__all__ += ['eigs']

# Solve symmetric/hermitian eigenvalue problem (generic == no overwrite)
eigsh = ssl.eigsh
__all__ += ['eigsh']
