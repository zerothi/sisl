# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from copy import copy as _copy

import numpy as np

from sisl._ufuncs import register_sisl_dispatch
from sisl.typing import SileLike

from ._brillouinzone_adaptive import GaussKronrodGrid
from .brillouinzone import BrillouinZone

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(BrillouinZone, module="sisl.physics")
def copy(bz: BrillouinZone, parent=None) -> BrillouinZone:
    """Create a copy of this object, optionally changing the parent

    Parameters
    ----------
    parent : optional
       change the parent
    """
    out = bz.__class__.__new__(bz.__class__)
    out.__dict__.update(
        (key, _copy(value)) for key, value in bz.__dict__.items() if key != "parent"
    )
    out.set_parent(bz.parent if parent is None else parent)
    return out


_copy_bz = copy


@register_sisl_dispatch(GaussKronrodGrid, module="sisl.physics")
def copy(gk: GaussKronrodGrid, parent=None) -> GaussKronrodGrid:
    """Create a copy of this object, optionally changing the parent

    The copy has no result of `integrate`. Thus `last_bz` requires a new `integrate`.

    Parameters
    ----------
    parent : optional
       change the parent
    """
    out = _copy_bz(gk, parent)
    out._last = None
    return out


@register_sisl_dispatch(BrillouinZone, module="sisl.physics")
def write(bz: BrillouinZone, sile: SileLike, *args, **kwargs) -> None:
    """Writes k-points to a `~sisl.io.tableSile`.

    This allows one to pass a `tableSile` or a file-name.
    """
    from sisl.io import tableSile

    kw = np.concatenate((bz.k, bz.weight.reshape(-1, 1)), axis=1)
    if isinstance(sile, tableSile):
        sile.write_data(kw.T, *args, **kwargs)
    else:
        with tableSile(sile, "w") as fh:
            fh.write_data(kw.T, *args, **kwargs)
