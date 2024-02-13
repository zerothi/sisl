# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from sisl._ufuncs import register_sisl_dispatch
from sisl.typing import SileLike

from .hamiltonian import Hamiltonian

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(Hamiltonian, module="sisl.physics")
def write(H: Hamiltonian, sile: SileLike, *args, **kwargs) -> None:
    """Writes a Hamiltonian to the `Sile` as implemented in the :code:`Sile.write_hamiltonian` method"""
    # This only works because, they *must*
    # have been imported previously
    from sisl.io import BaseSile, get_sile

    if isinstance(sile, BaseSile):
        sile.write_hamiltonian(H, *args, **kwargs)
    else:
        with get_sile(sile, mode="w") as fh:
            fh.write_hamiltonian(H, *args, **kwargs)
