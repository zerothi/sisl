# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import re
import numpy as np

from .basic import OptionsInput


class GeomAxisSelect(OptionsInput):

    _default = {
        "params": {
            "placeholder": "Choose axis...",
            "options": [
                {'label': ax, 'value': ax} for ax in ["x", "y", "z", "-x", "-y", "-z", "a", "b", "c", "-a", "-b", "-c"]
            ],
            "isMulti": True,
            "isClearable": False,
            "isSearchable": True,
        }
    }

    def _sanitize_axis(self, ax):
        if isinstance(ax, str):
            if re.match("[+-]?[012]", ax):
                ax = ax.replace("0", "a").replace("1", "b").replace("2", "c")
            ax = ax.lower().replace("+", "")
        elif isinstance(ax, int):
            ax = 'abc'[ax]
        elif isinstance(ax, (list, tuple)):
            ax = np.array(ax)

        # Now perform some checks
        invalid = True
        if isinstance(ax, str):
            invalid = not re.match("-?[xyzabc]", ax)
        elif isinstance(ax, np.ndarray):
            invalid = ax.shape != (3,)

        if invalid:
            raise ValueError(f"Incorrect axis passed. Axes must be one of [+-]('x', 'y', 'z', 'a', 'b', 'c', '0', '1', '2', 0, 1, 2)" +
                             " or a numpy array/list/tuple of shape (3, )")

        return ax

    def parse(self, val):
        if isinstance(val, str):
            val = re.findall("[+-]?[xyzabc012]", val)
        return [self._sanitize_axis(ax) for ax in val]
