# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

from ..plotutils import random_color
from .data_source import DataSource

# from ..processors.orbital import reduce_orbital_data, get_orbital_request_sanitizer


class OrbitalData(DataSource):
    pass


def style_fatbands(data, groups=[{}]):
    # Get the function that is going to convert our request to something that can actually
    # select orbitals from the xarray object.
    _sanitize_request = get_orbital_request_sanitizer(
        data,
        gens={
            "color": lambda req: req.get("color") or random_color(),
        },
    )

    styled = reduce_orbital_data(
        data,
        groups,
        orb_dim="orb",
        spin_dim="spin",
        sanitize_group=_sanitize_request,
        group_vars=("color", "dash"),
        groups_dim="group",
        drop_empty=True,
        spin_reduce=np.sum,
    )

    return styled  # .color


class FatbandsData(OrbitalData):
    function = staticmethod(style_fatbands)
