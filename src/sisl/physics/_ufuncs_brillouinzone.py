# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Optional

import numpy as np

from sisl._ufuncs import register_sisl_dispatch

from .brillouinzone import BandStructure, BrillouinZone, MonkhorstPack

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(BrillouinZone, module="sisl.physics")
def copy(bz: BrillouinZone, parent=None):
    """Create a copy of this object, optionally changing the parent

    Parameters
    ----------
    parent : optional
       change the parent
    """
    if parent is None:
        parent = bz.parent
    out = bz.__class__(parent, bz._k.copy(), bz.weight.copy())
    return out


@register_sisl_dispatch(MonkhorstPack, module="sisl.physics")
def copy(mp: MonkhorstPack, parent=None):
    """Create a copy of this object, optionally changing the parent

    Parameters
    ----------
    parent : optional
       change the parent
    """
    if parent is None:
        parent = mp.parent
    out = mp.__class__(
        parent, mp._diag, mp._displ, mp._size, mp._centered, mp._trs >= 0
    )
    # this is required due to replace calls
    out._k = mp._k.copy()
    out._w = mp._w.copy()
    return out


@register_sisl_dispatch(BandStructure, module="sisl.physics")
def copy(bs: BandStructure, parent=None):
    """Create a copy of this object, optionally changing the parent

    Parameters
    ----------
    parent : optional
       change the parent
    """
    if parent is None:
        parent = bs.parent
    out = bs.__class__(
        parent, bs.points.copy(), bs.divisions.copy(), bs.names[:], jump_dk=bs._jump_dk
    )
    return out
