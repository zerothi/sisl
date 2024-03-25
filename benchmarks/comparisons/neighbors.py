# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import timeit
from collections import defaultdict

from ase.neighborlist import neighbor_list

from sisl.geom import NeighborFinder, fcc, graphene_nanoribbon


def quadratic(geom):
    neighs = []
    for at in geom:
        neighs.append(geom.close(at, R=1.5))


what = "fcc"
times = defaultdict(list)
na = []

# Iterate over multiple tiles
for tiles in range(1, 11):
    print(tiles)

    if what == "fcc":
        geom = fcc(1.5, "C").tile(3 * tiles, 1).tile(3 * tiles, 2).tile(3 * tiles, 0)
    elif what == "ribbon":
        geom = graphene_nanoribbon(9).tile(tiles, 0)

    na.append(geom.na)

    # Compute with ASE
    ase_time = timeit.timeit(
        lambda: neighbor_list("ij", geom.to.ase(), 1.5, self_interaction=False),
        number=1,
    )
    times["ASE"].append(ase_time)

    for bs in [4, 8, 12]:
        print("  bs = ", bs)

        # Compute with this implementation (different variants)
        my_time = timeit.timeit(
            lambda: NeighborFinder(geom, R=1.5, bin_size=bs).find_neighbors(
                None, as_pairs=True
            ),
            number=1,
        )
        times[f"(PAIRS) [{bs}]"].append(my_time)

        my_time = timeit.timeit(
            lambda: NeighborFinder(geom, R=1.5, bin_size=bs).find_neighbors(
                None, as_pairs=False
            ),
            number=1,
        )
        times[f"(NEIGHS) [{bs}]"].append(my_time)

        my_time = timeit.timeit(
            lambda: NeighborFinder(geom, R=1.5, bin_size=bs).find_neighbors(
                None, as_pairs=False
            ),
            number=1,
        )
        times[f"(NEIGHS) [{bs}]"].append(my_time)

        my_time = timeit.timeit(
            lambda: NeighborFinder(geom, R=1.5, bin_size=bs).find_all_unique_pairs(),
            number=1,
        )
        times[f"(UNIQUE PAIRS) [{bs}]"].append(my_time)

    # Compute with quadratic search
    # quadratic_time = timeit.timeit(lambda: quadratic(geom), number=1)
    # times["QUADRATIC"].append(quadratic_time)

# Plotting
import plotly.graph_objs as go

fig = go.Figure()
for key, time in times.items():
    fig.add_scatter(x=na, y=time, name=key)

fig.update_layout(
    xaxis_title="Number of atoms",
    yaxis_title="Time (s)",
    title=f"Finding neighbors on {what}",
    yaxis_showgrid=True,
    xaxis_showgrid=True,
)
fig.write_image(f"neighbors_{what}.png")
fig.show()
