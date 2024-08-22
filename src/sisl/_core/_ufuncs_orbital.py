# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from sisl._ufuncs import register_sisl_dispatch

from .orbital import (
    AtomicOrbital,
    HydrogenicOrbital,
    Orbital,
    SphericalOrbital,
    _ExponentialOrbital,
)

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(Orbital, module="sisl")
def scale(orbital: Orbital, scale: float) -> Orbital:
    """Scale the orbital by extending R by `scale`"""
    R = orbital.R * scale
    if R < 0:
        R = -1.0
    return orbital.__class__(R, orbital.q0, orbital.tag)


@register_sisl_dispatch(Orbital, module="sisl")
def copy(orbital: Orbital) -> Orbital:
    """Create an exact copy of this object"""
    return orbital.__class__(orbital.R, orbital.q0, orbital.tag)


@register_sisl_dispatch(SphericalOrbital, module="sisl")
def copy(orbital: SphericalOrbital) -> SphericalOrbital:
    """Create an exact copy of this object"""
    return orbital.__class__(
        orbital.l, orbital._radial, R=orbital.R, q0=orbital.q0, tag=orbital.tag
    )


@register_sisl_dispatch(AtomicOrbital, module="sisl")
def copy(orbital: AtomicOrbital) -> AtomicOrbital:
    """Create an exact copy of this object"""
    return orbital.__class__(
        n=orbital.n,
        l=orbital.l,
        m=orbital.m,
        zeta=orbital.zeta,
        P=orbital.P,
        spherical=orbital.orb.copy(),
        q0=orbital.q0,
        tag=orbital.tag,
    )


@register_sisl_dispatch(HydrogenicOrbital, module="sisl")
def copy(orbital: HydrogenicOrbital) -> HydrogenicOrbital:
    """Create an exact copy of this object"""
    return orbital.__class__(
        orbital.n,
        orbital.l,
        orbital.m,
        orbital._Z,
        R=orbital.R,
        q0=orbital.q0,
        tag=orbital.tag,
    )


@register_sisl_dispatch(_ExponentialOrbital, module="sisl")
def copy(orbital: _ExponentialOrbital) -> _ExponentialOrbital:
    """Create an exact copy of this object"""
    return orbital.__class__(
        n=orbital.n,
        l=orbital.l,
        m=orbital.m,
        alpha=orbital.alpha[:],
        coeff=orbital.coeff[:],
        R=orbital.R,
        q0=orbital.q0,
        tag=orbital.tag,
    )
