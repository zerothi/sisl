# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from sisl._ufuncs import register_sisl_dispatch
from sisl.typing import SileLike

from .energydensitymatrix import EnergyDensityMatrix

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(EnergyDensityMatrix, module="sisl.physics")
def write(edm: EnergyDensityMatrix, sile: SileLike, *args, **kwargs) -> None:
    """Writes an energy density matrix to the `Sile` as implemented in the :code:`Sile.write_energy_density_matrix` method"""
    # This only works because, they *must*
    # have been imported previously
    from sisl.io import BaseSile, get_sile

    if isinstance(sile, BaseSile):
        sile.write_energy_density_matrix(edm, *args, **kwargs)
    else:
        with get_sile(sile, mode="w") as fh:
            fh.write_energy_density_matrix(edm, *args, **kwargs)
