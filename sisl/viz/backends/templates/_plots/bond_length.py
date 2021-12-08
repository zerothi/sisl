# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from .geometry import GeometryBackend

from ....plots import BondLengthMap


class BondLengthMapBackend(GeometryBackend):
    """Draws a bond length map provided by `BondLengthMap`

    The flow is exactly the same as `GeometryPlot`, in fact this class might only be extended to
    manipulate color bars or things like that. Otherwise, if you already have a `MyGeometryBackend`,
    you can just create a bond length map backend like

    ```
    class MyBondLengthMapBackend(BondLengthMapBackend, MyGeometryBackend):
        pass
    ```

    """

    def draw_1D(self, backend_info, **kwargs):
        return NotImplementedError("1D representations of bond length maps are not implemented")

BondLengthMap.backends.register_template(BondLengthMapBackend)
