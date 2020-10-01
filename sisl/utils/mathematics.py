from numpy import dot, sqrt, square
from numpy import cos, sin, arctan2, arccos
from numpy import asarray, take, delete, empty
from numpy import concatenate, argsort

from sisl import _array as _a
from sisl._indices import indices_le

__all__ = ["fnorm", "fnorm2", "expand", "orthogonalize"]
__all__ += ["spher2cart", "cart2spher", "spherical_harm"]
__all__ += ["curl"]


def fnorm(array, axis=-1):
    r""" Fast calculation of the norm of a vector

    Parameters
    ----------
    array : (..., *)
       the vector/matrix to perform the norm on, norm performed along `axis`
    axis : int, optional
       the axis to take the norm against, default to last axis.
    """
    return sqrt(square(array).sum(axis))


def fnorm2(array, axis=-1):
    r""" Fast calculation of the squared norm of a vector

    Parameters
    ----------
    array : (..., *)
       the vector/matrix to perform the squared norm on, norm performed along `axis`
    axis : int, optional
       the axis to take the norm against, default to last axis.
    """
    return square(array).sum(axis)


def expand(vector, length):
    r""" Expand `vector` by `length` such that the norm of the vector is increased by `length`

    The expansion of the vector can be written as:

    .. math::
        V' = V + \hat V l

    Parameters
    ----------
    vector : array_like
        original vector
    length : float
        the length to be added along the vector

    Returns
    -------
    new_vector : the new vector with increased length
    """
    return vector * (1 + length / fnorm(vector))


def orthogonalize(ref, vector):
    r""" Ensure `vector` is orthogonal to `ref`, `vector` must *not* be parallel to `ref`.

    Enable an easy creation of a vector orthogonal to a reference vector. The length of the vector
    is not necessarily preserved (if they are not orthogonal).

    The orthogonalization is performed by:

    .. math::
       V_{\perp} = V - \hat R (\hat R \cdot V)

    which is subtracting the projected part from :math:`V`.

    Parameters
    ----------
    ref : array_like
       reference vector to make `vector` orthogonal too
    vector : array_like
       the vector to orthogonalize, must have same dimension as `ref`

    Returns
    -------
    ortho : the orthogonalized vector

    Raises
    ------
    ValueError : if `vector` is parallel to `ref`
    """
    ref = asarray(ref).ravel()
    nr = fnorm(ref)
    vector = asarray(vector).ravel()
    d = dot(ref, vector) / nr
    if abs(1. - abs(d) / fnorm(vector)) < 1e-7:
        raise ValueError(f"orthogonalize: requires non-parallel vectors to perform an orthogonalization: ref.vector = {d}")
    return vector - ref * d / nr


def spher2cart(r, theta, phi):
    r""" Convert spherical coordinates to cartesian coordinates

    Parameters
    ----------
    r : array_like
       radius
    theta : array_like
       azimuthal angle in the :math:`x-y` plane
    phi : array_like
       polar angle from the :math:`z` axis
    """
    rx = r * cos(theta) * sin(phi)
    R = _a.emptyd(rx.shape + (3, ))
    R[..., 0] = rx
    del rx
    R[..., 1] = r * sin(theta) * sin(phi)
    R[..., 2] = r * cos(phi)
    return R


def cart2spher(r, theta=True, cos_phi=False, maxR=None):
    r""" Transfer a vector to spherical coordinates with some possible differences

    Parameters
    ----------
    r : array_like
       the cartesian vectors
    theta : bool, optional
       if ``True`` also calculate the theta angle and return it
    cos_phi : bool, optional
       if ``True`` return :math:`\cos(\phi)` rather than :math:`\phi` which may
       be useful in some subsequent mathematical calculations
    maxR : float, optional
       cutoff of the spherical coordinate calculations. If ``None``, calculate
       and return for all.

    Returns
    -------
    n : int
       number of total points, only for `maxR` different from ``None``
    idx : numpy.ndarray
       indices of points with ``r <= maxR``
    r : numpy.ndarray
       radius in spherical coordinates, only for `maxR` different from ``None``
    theta : numpy.ndarray
       angle in the :math:`x-y` plane from :math:`x` (azimuthal)
       Only calculated if input `theta` is ``True``, otherwise None is returned.
    phi : numpy.ndarray
       If `cos_phi` is ``True`` this is :math:`\cos(\phi)`, otherwise
       :math:`\phi` is returned (the polar angle from the :math:`z` axis)
    """
    r = _a.asarrayd(r).reshape(-1, 3)
    if r.shape[-1] != 3:
        raise ValueError("Vector does not end with shape 3.")
    n = r.shape[0]
    if maxR is None:
        rr = sqrt(square(r).sum(1))
        if theta:
            theta = arctan2(r[:, 1], r[:, 0])
        else:
            theta = None
        if cos_phi:
            phi = r[:, 2] / rr
        else:
            phi = arccos(r[:, 2] / rr)
        phi[rr == 0.] = 0.
        return rr, theta, phi

    rr = square(r).sum(1)
    idx = indices_le(rr, maxR ** 2)
    r = take(r, idx, 0)
    rr = sqrt(take(rr, idx))
    if theta:
        theta = arctan2(r[:, 1], r[:, 0])
    else:
        theta = None
    if cos_phi:
        phi = r[:, 2] / rr
    else:
        phi = arccos(r[:, 2] / rr)
    # Typically there will be few rr==0. values, so no need to
    # create indices
    phi[rr == 0.] = 0.
    return n, idx, rr, theta, phi


def spherical_harm(m, l, theta, phi):
    r""" Calculate the spherical harmonics using :math:`Y_l^m(\theta, \varphi)` with :math:`\mathbf R\to \{r, \theta, \varphi\}`.

    .. math::
        Y^m_l(\theta,\varphi) = (-1)^m\sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}}
             e^{i m \theta} P^m_l(\cos(\varphi))

    which is the spherical harmonics with the Condon-Shortley phase.

    Parameters
    ----------
    m : int
       order of the spherical harmonics
    l : int
       degree of the spherical harmonics
    theta : array_like
       angle in :math:`x-y` plane (azimuthal)
    phi : array_like
       angle from :math:`z` axis (polar)
    """
    # Probably same as:
    #return (-1) ** m * ( (2*l+1)/(4*pi) * factorial(l-m) / factorial(l+m) ) ** 0.5 \
    #    * lpmv(m, l, cos(theta)) * exp(1j * m * phi)
    return sph_harm(m, l, theta, phi) * (-1) ** m


def curl(m, axis=-2, axisv=-1):
    r""" Determine the curl of a matrix `m` where `m` contains the differentiated quantites along `axisv`.

    The curl is calculated as:

    .. math::
       \mathrm{curl} \mathbf M|_x &= \frac{\partial\mathbf M_z}{\partial y} - \frac{\partial\mathbf M_y}{\partial z}
       \\
       \mathrm{curl} \mathbf M|_y &= \frac{\partial\mathbf M_x}{\partial z} - \frac{\partial\mathbf M_z}{\partial x}
       \\
       \mathrm{curl} \mathbf M|_z &= \frac{\partial\mathbf M_y}{\partial x} - \frac{\partial\mathbf M_x}{\partial y}

    where the `axis` are the :math:`\partial x` axis and `axisv` are the :math:`\partial M_x` axis.

    Parameters
    ----------
    m : numpy.ndarray
       matrix to calculate the curl of
    axis : int, optional
       axis that contains the direction vectors, this dimension is removed from the returned curl
    axisv : int, optional
       axis that contains the differentiated vectors

    Returns
    -------
    curl : the curl of the matrix shape of `m` without axis `axis` 
    """
    if m.shape[axis] != 3:
        raise ValueError("curl requires 3 vectors to calculate the curl of!")
    elif m.shape[axisv] != 3:
        raise ValueError("curl requires the vectors to have 3 components!")

    # Check that no two axis are used for the same thing
    axis %= m.ndim
    axisv %= m.ndim
    if axis == axisv:
        raise ValueError("curl requires axis and axisv to be different axes")

    # Create lists for correct slices
    slx = [slice(None) for _ in m.shape]
    sly = slx[:]
    slz = slx[:]
    vx = slx[:]
    vy = slx[:]
    vz = slx[:]
    slx[axis] = 0
    sly[axis] = 1
    slz[axis] = 2

    # Prepare the curl elements
    vx[axisv] = 0
    vy[axisv] = 1
    vz[axisv] = 2
    vx.pop(axis)
    vy.pop(axis)
    vz.pop(axis)

    slx = tuple(slx)
    sly = tuple(sly)
    slz = tuple(slz)
    vx = tuple(vx)
    vy = tuple(vy)
    vz = tuple(vz)

    # Create curl by removing the v dimension
    curl = empty(delete(m.shape, axis), dtype=m.dtype)
    curl[vx] = m[sly][vz] - m[slz][vy]
    curl[vy] = m[slz][vx] - m[slx][vz]
    curl[vz] = m[slx][vy] - m[sly][vx]
    return curl


def intersect_and_diff_sets(a, b):
    """See numpy.intersect1d(a, b, assume_unique=True, return_indices=True).
    In addition to that, this function also returns the indices in a and b which
    are *not* in the intersection.
    This saves a bit compared to doing np.delete() afterwards.
    """
    aux = concatenate((a, b))
    aux_sort_indices = argsort(aux, kind='mergesort')
    aux = aux[aux_sort_indices]
    # find elements that are the same in both arrays
    # after sorting we should have at most 2 same elements
    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    aover = aux_sort_indices[:-1][mask]
    bover = aux_sort_indices[1:][mask] - a.size

    nobuddy_lr = concatenate([[True], ~mask, [True]])
    no_buddy = nobuddy_lr[:-1]  # no match left
    no_buddy &= nobuddy_lr[1:]  # no match right

    aonly = (aux_sort_indices < a.size)
    bonly = ~aonly
    aonly &= no_buddy
    bonly &= no_buddy
    # the below is for some reason slower even though its only two ops
    # aonly &= no_buddy
    # bonly = aonly ^ no_buddy

    aonly = aux_sort_indices[aonly]
    bonly = aux_sort_indices[bonly] - a.size

    return int1d, aover, bover, aonly, bonly
