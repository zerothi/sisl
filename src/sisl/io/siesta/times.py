# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import xarray as xr

from sisl._array import arrayd, arrayi
from sisl._internal import set_module

from ..sile import add_sile, sile_fh_open
from .sile import SileSiesta

__all__ = ["timesSileSiesta"]


@set_module("sisl.io.siesta")
class timesSileSiesta(SileSiesta):
    """Time information from the Siesta run

    The timings are stored in either the ``TIMES`` or
    the ``<SystemLabel>.times`` files.
    """

    _info_attributes_ = [
        dict(
            name="processors",
            searcher=r"^timer: Number of nodes",
            parser=lambda attr, instance, match: int(match.string.split()[-1]),
            not_found="error",
        ),
        dict(
            name="threads",
            searcher=r"^timer: Number of threads per node",
            parser=lambda attr, instance, match: int(match.string.split()[-1]),
            default=1,
            not_found="info",
        ),
        dict(
            name="processor_reference",
            searcher=r"^timer: Times refer to node",
            parser=lambda attr, instance, match: int(match.string.split()[-1]),
            not_found="error",
        ),
        dict(
            name="wall_clock",
            searcher=r"^timer: Total elapsed wall-clock",
            parser=lambda attr, instance, match: float(match.string.split()[-1]),
            not_found="error",
        ),
    ]

    @sile_fh_open(True)
    def read_data(self) -> xarray.DataArray:
        r"""Returns data associated with the times file

        Returns
        -------
        times : xarray.DataArray
            containing timings.
        """
        rl = self.readline
        # first line is empty
        rl()

        line = rl()  # Number of nodes
        line = rl()  # Busiest calculating
        line = rl()  # Times refer to node
        line = rl()  # Total elapsed

        line = rl()  # CPU times
        for _ in range(4):
            rl()

        self.step_to("Program")
        # header
        rl()

        # now we read content
        routines = []
        calls = []
        comm = []
        times = []
        imbalance = []

        while len(cols := rl().split()) > 1:
            if cols[0] == "MPI":
                break
            routines.append(cols[0])
            calls.append(cols[1])
            comm.append(cols[2])
            times.append(cols[4])
            imbalance.append(cols[6])

        def ar(ar):
            return ar.reshape(1, -1)

        dims = ["parallel", "routine"]
        data = arrayd([calls, comm, times, imbalance]).T
        data_vars = dict(
            calls=(dims, ar(arrayi(calls))),
            comm=(dims, ar(arrayd(comm))),
            time=(dims, ar(arrayd(times))),
            imbalance=(dims, ar(arrayd(imbalance))),
        )

        coords = dict(
            processors=("parallel", [self.info.processors]),
            threads=("parallel", [self.info.threads]),
            routine=routines,
        )

        data = xr.Dataset(
            data_vars=data_vars, coords=coords, attrs=dict(file=f"{self!s}", unit="s")
        )
        data = data.set_index(parallel=["processors", "threads"])

        return data


# This will both allow TIMES + siesta.times
add_sile("TIMES", timesSileSiesta, case=False, gzip=True)
