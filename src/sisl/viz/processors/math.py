# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np


def normalize(data, vmin=0, vmax=1):
    """Normalize data to [vmin, vmax] range.

    Parameters
    ----------
    data : array_like
        Data to normalize.
    vmin : float, optional
        Minimum value of normalized data.
    vmax : float, optional
        Maximum value of normalized data.

    Returns
    -------
    data : array_like
        Normalized data.
    """
    data = np.asarray(data)
    data_min = np.min(data)
    data_max = np.max(data)
    return vmin + (vmax - vmin) * (data - data_min) / (data_max - data_min)
