# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from sisl_toolbox.models._base import BaseModel

__all__ = ["GrapheneModel"]


class GrapheneModel(BaseModel):
    # copy the dispatcher method
    ref = BaseModel.ref.copy()

    # A graphene model is generally made of 1-3 nearest neighbor
    # couplings
    # The distances are here kept
    @classmethod
    def distance(cls, n=1, a=1.42):
        """Return the distance to the nearest neighbor according to the bond-length `a`

        Currently only up to 3rd nearest neighbor is implemneted

        Parameters
        ----------
        n : int, optional
           the nearest neighbor, 1 means first nearest neighbor, and 0 means it self
        a : float, optional
           the bond length of the intrinsic graphene lattice
        """
        dist = {
            0: 0.0,
            1: a,
            2: a * 3**0.5,
            3: a * 2,
        }
        return dist[n]
