"""
Define a common VASP Sile
"""
import sisl._array as _a

from ..sile import Sile, SileCDF, SileBin

__all__ = ['SileVASP', 'SileCDFVASP', 'SileBinVASP']


def _geometry_sort(geometry, ret_index=False):
    r""" Order atoms in geometry according to species such that all of one specie is consecutive

    When creating VASP input files (`poscarSileVASP` for instance) the equivalent
    ``POTCAR`` file needs to contain the pseudos for each specie as they are provided
    in blocks.

    I.e. for a geometry like this:
    .. code::

        [Atom(6), Atom(4), Atom(6)]

    the resulting ``POTCAR`` needs to contain the pseudo for Carbon twice.

    This method will re-order atoms according to the species"

    Parameters
    ----------
    geometry : Geometry
       geometry to be re-ordered
    ret_index : bool, optional
       return sorted indices

    Returns
    -------
    geometry: reordered geometry
    """
    na = len(geometry)
    idx = _a.emptyi(na)

    ia = 0
    for _, idx_s in geometry.atoms.iter(species=True):
        idx[ia:ia + len(idx_s)] = idx_s
        ia += len(idx_s)

    assert ia == na

    if ret_index:
        return geometry.sub(idx), idx
    return geometry.sub(idx)


class SileVASP(Sile):
    geometry_sort = staticmethod(_geometry_sort)


class SileCDFVASP(SileCDF):
    geometry_sort = staticmethod(_geometry_sort)


class SileBinVASP(SileBin):
    geometry_sort = staticmethod(_geometry_sort)
