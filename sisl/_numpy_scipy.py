"""
Wrapper routines for creating partial function calls with other defaults
"""

import warnings
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
try:
    # Figure out if the solve routine has a `refine` keyword which determines
    # the used LAPACK routine (SV vs. SVX)
    # We do NOT want to use SVX because they are rediculously slow for NRSH ~ N
    sl.solve(np.random.rand(2, 2), np.random.rand(2, 2), refine=False)
    _solve = sl.solve
except:
    # Create a _copy_ of the scipy.linalg.solve routine and implement
    # our own refine keyword.
    from numpy import atleast_1d, atleast_2d
    from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork
    from scipy.linalg.misc import LinAlgError, _datacopied
    from scipy.linalg.decomp import _asarray_validated

    def _solve(a, b, sym_pos=False, lower=False, overwrite_a=False,
               overwrite_b=False, debug=None, check_finite=True, assume_a='gen',
               transposed=False, refine=False):
        """
        Solves the linear equation set ``a * x = b`` for the unknown ``x``
        for square ``a`` matrix.

        If the data matrix is known to be a particular type then supplying the
        corresponding string to ``assume_a`` key chooses the dedicated solver.
        The available options are

        ===================  ========
         generic matrix       'gen'
         symmetric            'sym'
         hermitian            'her'
         positive definite    'pos'
        ===================  ========

        If omitted, ``'gen'`` is the default structure.

        The datatype of the arrays define which solver is called regardless
        of the values. In other words, even when the complex array entries have
        precisely zero imaginary parts, the complex solver will be called based
        on the data type of the array.

        Parameters
        ----------
        a : (N, N) array_like
            Square input data
        b : (N, NRHS) array_like
            Input data for the right hand side.
        sym_pos : bool, optional
            Assume `a` is symmetric and positive definite. This key is deprecated
            and assume_a = 'pos' keyword is recommended instead. The functionality
            is the same. It will be removed in the future.
        lower : bool, optional
            If True, only the data contained in the lower triangle of `a`. Default
            is to use upper triangle. (ignored for ``'gen'``)
        overwrite_a : bool, optional
            Allow overwriting data in `a` (may enhance performance).
            Default is False.
        overwrite_b : bool, optional
            Allow overwriting data in `b` (may enhance performance).
            Default is False.
        check_finite : bool, optional
            Whether to check that the input matrices contain only finite numbers.
            Disabling may give a performance gain, but may result in problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs.
        assume_a : str, optional
            Valid entries are explained above.
        transposed: bool, optional
            If True, depending on the data type ``a^T x = b`` or ``a^H x = b`` is
            solved (only taken into account for ``'gen'``).
        refine : bool, optional
            If True the ``svx`` LAPACK routines are used which produces more precise
            solutions at the expense of more computations. For ``NRHS << N`` this
            has minimal influence of the performance, whereas for ``NRHS`` close to 
            ``N`` it can hurt performance. Exploration whether ``refine=True`` is
            required for your particular application may be desirable.

        Returns
        -------
        x : (N, NRHS) ndarray
            The solution array.

        Raises
        ------
        ValueError
            If size mismatches detected or input a is not square.
        LinAlgError
            If the matrix is singular.
        RuntimeWarning
            If an ill-conditioned input a is detected.

        Examples
        --------
        Given `a` and `b`, solve for `x`:

        >>> a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
        >>> b = np.array([2, 4, -1])
        >>> from scipy import linalg
        >>> x = linalg.solve(a, b)
        >>> x
        array([ 2., -2.,  9.])
        >>> np.dot(a, x) == b
        array([ True,  True,  True], dtype=bool)

        To check whether the `refine` flag improves accuracy:

        >>> import numpy as np
        >>> np.random.seed(123789)
        >>> N = 100
        >>> a = np.random.rand(N, N)
        >>> b = np.random.rand(N, N)
        >>> from scipy import linalg
        >>> x_fast = linalg.solve(a, b)
        >>> x_refined = linalg.solve(a, b, refine=True)
        >>> np.linalg.norm(np.dot(a, x_fast) - np.dot(a, x_refined), 'fro')
        5.18303768429e-13

        Notes
        -----
        If the input `b` matrix is a 1D array with N elements, when supplied
        together with an NxN input `a`, it is assumed as a valid column vector
        despite the apparent size mismatch. This is compatible with the
        numpy.dot() behavior and the returned result is still 1D array.

        The generic, symmetric, hermitian and positive definite solutions are
        obtained via calling ?GESV, ?SYSV, ?HESV, and ?POSV routines of
        LAPACK respectively for ``refine=False``, else ?GESVX, ?SYSVX, ?HESVX,
        and ?POSVX are called.
        """
        # Flags for 1D or nD right hand side
        b_is_1D = False
        b_is_ND = False

        a1 = atleast_2d(_asarray_validated(a, check_finite=check_finite))
        b1 = atleast_1d(_asarray_validated(b, check_finite=check_finite))
        n = a1.shape[0]

        overwrite_a = overwrite_a or _datacopied(a1, a)
        overwrite_b = overwrite_b or _datacopied(b1, b)

        if a1.shape[0] != a1.shape[1]:
            raise ValueError('Input a needs to be a square matrix.')

        if n != b1.shape[0]:
            # Last chance to catch 1x1 scalar a and 1D b arrays
            if not (n == 1 and b1.size != 0):
                raise ValueError('Input b has to have same number of rows as '
                                 'input a')

        # accomodate empty arrays
        if b1.size == 0:
            return np.asfortranarray(b1.copy())

        # regularize 1D b arrays to 2D and catch nD RHS arrays
        if b1.ndim == 1:
            if n == 1:
                b1 = b1[None, :]
            else:
                b1 = b1[:, None]
            b_is_1D = True
        elif b1.ndim > 2:
            b_is_ND = True

        r_or_c = complex if np.iscomplexobj(a1) else float

        # Backwards compatibility - old keyword, has precedence
        if sym_pos:
            assume_a = 'pos'

        if assume_a in ('gen', 'sym', 'her', 'pos'):
            _structure = assume_a
        else:
            raise ValueError('{} is not a recognized matrix structure'
                             ''.format(assume_a))

        # Deprecate keyword "debug"
        if debug is not None:
            warnings.warn('Use of the "debug" keyword is deprecated '
                          'and this keyword will be removed in the future '
                          'versions of SciPy.', DeprecationWarning)

        if refine:
            if _structure == 'gen':
                gesvx = get_lapack_funcs('gesvx', (a1, b1))
                trans_conj = 'N'
                if transposed:
                    trans_conj = 'T' if r_or_c is float else 'H'
                (_, _, _, _, _, _, _,
                 x, rcond, _, _, info) = gesvx(a1, b1,
                                               trans=trans_conj,
                                               overwrite_a=overwrite_a,
                                               overwrite_b=overwrite_b)
            elif _structure == 'sym':
                sysvx, sysvx_lw = get_lapack_funcs(('sysvx', 'sysvx_lwork'), (a1, b1))
                lwork = _compute_lwork(sysvx_lw, n, lower)
                _, _, _, _, x, rcond, _, _, info = sysvx(a1, b1, lwork=lwork,
                                                         lower=lower,
                                                         overwrite_a=overwrite_a,
                                                         overwrite_b=overwrite_b)
            elif _structure == 'her':
                hesvx, hesvx_lw = get_lapack_funcs(('hesvx', 'hesvx_lwork'), (a1, b1))
                lwork = _compute_lwork(hesvx_lw, n, lower)
                _, _, x, rcond, _, _, info = hesvx(a1, b1, lwork=lwork,
                                                   lower=lower,
                                                   overwrite_a=overwrite_a,
                                                   overwrite_b=overwrite_b)
            else:
                posvx = get_lapack_funcs('posvx', (a1, b1))
                _, _, _, _, _, x, rcond, _, _, info = posvx(a1, b1,
                                                            lower=lower,
                                                            overwrite_a=overwrite_a,
                                                            overwrite_b=overwrite_b)

            # Unlike ?xxSV, ?xxSVX writes the solution x to a separate array, and
            # overwrites b with its scaled version which is thrown away. Thus, the
            # solution does not admit the same shape with the original b. For
            # backwards compatibility, we reshape it manually.
            if b_is_ND:
                x = x.reshape(*b1.shape, order='F')

        else:
            # Signal unknowing condition number (rcond always positive, hence negative makes
            # no sense)
            # Regardless, this value will not be used because info <= n
            rcond = -1.
            if _structure == 'gen':
                gesv = get_lapack_funcs('gesv', (a1, b1))
                _, _, x, info = gesv(a1, b1,
                                     overwrite_a=overwrite_a,
                                     overwrite_b=overwrite_b)
            elif _structure == 'sym':
                sysv, sysv_lw = get_lapack_funcs(('sysv', 'sysv_lwork'), (a1, b1))
                lwork = _compute_lwork(sysv_lw, n, lower)
                _, _, x, info = sysv(a1, b1, lwork=lwork,
                                     lower=lower,
                                     overwrite_a=overwrite_a,
                                     overwrite_b=overwrite_b)
            elif _structure == 'her':
                hesv, hesv_lw = get_lapack_funcs(('hesv', 'hesv_lwork'), (a1, b1))
                lwork = _compute_lwork(hesv_lw, n, lower)
                _, _, x, info = hesv(a1, b1, lwork=lwork,
                                     lower=lower,
                                     overwrite_a=overwrite_a,
                                     overwrite_b=overwrite_b)
            else:
                posv = get_lapack_funcs('posv', (a1, b1))
                _, x, info = posv(a1, b1,
                                  lower=lower,
                                  overwrite_a=overwrite_a,
                                  overwrite_b=overwrite_b)

        if b_is_1D:
            x = x.ravel()

        if info == 0:
            return x
        elif info < 0:
            raise ValueError('LAPACK reported an illegal value in {}-th argument'
                             '.'.format(-info))
        elif 0 < info <= n:
            raise LinAlgError('Matrix is singular.')
        elif info > n:
            warnings.warn('scipy.linalg.solve\nIll-conditioned matrix detected.'
                          ' Result is not guaranteed to be accurate.\nReciprocal'
                          ' condition number: {}'.format(rcond), RuntimeWarning)

solve = _partial(_solve, check_finite=False, overwrite_a=False, overwrite_b=False, refine=False)
solve_herm = _partial(_solve, check_finite=False, overwrite_a=False, overwrite_b=False, refine=False, assume_a='her')
solve_sym = _partial(_solve, check_finite=False, overwrite_a=False, overwrite_b=False, refine=False, assume_a='sym')
__all__ += _append('solve', ['', '_herm', '_sym'])

solve_destroy = _partial(_solve, check_finite=False, overwrite_a=True, overwrite_b=True, refine=False)
solve_herm_destroy = _partial(_solve, check_finite=False, overwrite_a=True, overwrite_b=True, refine=False, assume_a='her')
solve_sym_destroy = _partial(_solve, check_finite=False, overwrite_a=True, overwrite_b=True, refine=False, assume_a='sym')
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
eigh_dc = eigh
eigh_qr = _partial(sl.eigh, check_finite=False, overwrite_a=False, overwrite_b=False, turbo=False)
__all__ += _append('eigh', ['', '_dc', '_qr'])

# Solve symmetric/hermitian eigenvalue problem (allow overwrite)
eigh_destroy = _partial(sl.eigh, check_finite=False, overwrite_a=True, overwrite_b=True, turbo=True)
eigh_dc_destroy = eigh_destroy
eigh_qr_destroy = _partial(sl.eigh, check_finite=False, overwrite_a=True, overwrite_b=True, turbo=False)
__all__ += _append('eigh_', ['destroy', 'dc_destroy', 'qr_destroy'])

# SVD problem
svd = _partial(sl.svd, check_finite=False, overwrite_a=False)
svd_destroy = _partial(sl.svd, check_finite=False, overwrite_a=True)
__all__ += _append('svd', ['', '_destroy'])


# Sparse linalg routines

# Solve eigenvalue problem
eigs = ssl.eigs
__all__ += ['eigs']

# Solve symmetric/hermitian eigenvalue problem (generic == no overwrite)
eigsh = ssl.eigsh
__all__ += ['eigsh']
