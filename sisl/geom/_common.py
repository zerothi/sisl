import numpy as np

__all__ = ["geometry_define_nsc", "geometry2uc"]


def geometry_define_nsc(geometry, periodic=(True, True, True)):
    """Define the number of supercells for a geometry based on the periodicity """
    if np.all(geometry.maxR(True) > 0.):
        geometry.optimize_nsc()
        if not periodic[0]:
            geometry.set_nsc(a=1)
        if not periodic[1]:
            geometry.set_nsc(b=1)
        if not periodic[2]:
            geometry.set_nsc(c=1)
    else:
        nsc = [3 if p else 1 for p in periodic]
        geometry.set_nsc(nsc)


def geometry2uc(geometry, dx=1e-8):
    """ Translate the geometry to the unit cell by first shifting `dx` """
    geometry = geometry.move(dx).translate2uc().move(-dx)
    geometry.xyz[geometry.xyz < 0] = 0
    return geometry
