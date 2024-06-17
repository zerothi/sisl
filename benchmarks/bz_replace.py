#!/usr/bin/env python
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# This benchmark creates a very large graphene flake and uses construct
# to create it.

# This benchmark may be called using:
#
#  python $0
#
# and it may be post-processed using
#
#  python stats.py $0.profile
#
from __future__ import annotations

import cProfile
import pstats
import sys

import numpy as np
from tqdm import tqdm

import sisl

if len(sys.argv) > 1:
    nlvls = int(sys.argv[1])
else:
    nlvls = 2
ndim = 2
print(f"dimensions = {ndim}")
print(f"levels = {nlvls}")


# this will create 10x10 points
if nlvls > 2:
    nks = [40 for _ in range(nlvls)]
else:
    nks = [50 for _ in range(nlvls)]

if ndim == 1:

    def get_nk(nk):
        return [nk, 1, 1]

    ns = [2 for _ in range(nlvls)]
else:

    def get_nk(nk):
        return [nk, nk, 1]

    if nlvls > 2:
        ns = [50 for _ in range(nlvls)]
    else:
        ns = [100 for _ in range(nlvls)]


# Replace many k-points
def yield_kpoint(bz, n):
    yield from np.unique(np.random.randint(len(bz), size=n))[::-1]


# Replacement function


def add_levels(bz, nks, ns, fast=False, as_index=False, debug=False):
    """Add different levels according to the length of `ns`"""
    global nlvls

    lvl = nlvls - len(nks)
    nreps = 0

    if fast:
        # we need to copy the bz since for each ik, the new_bz gets
        # changed in add_levels.
        # If there was only 1 ik per level, then all would work fine.
        bz = bz.copy()

    from io import StringIO

    s = StringIO()

    def print_s(force=True):
        nonlocal s
        out = s.getvalue()
        spaces = " " * (lvl * 2)
        out = spaces + out.replace("\n", f"\n{spaces}")
        if force:
            print(out)

        # reset s
        s = StringIO()

    if debug:
        print(f"lvl = {lvl}", file=s)
        print_s()

    if len(nks) > 0:
        # calculate the size of the current BZ
        dsize = bz._size / bz._diag

        # pop the last items
        nk = get_nk(nks[-1])
        n = ns[-1]
        assert n < len(bz), "Too loong n"

        iks = [ik for ik in yield_kpoint(bz, n)]

        if debug:
            print("size", bz._size, file=s)
            print("dsize", dsize, file=s)
            print("iks", iks, file=s)
            print("len(bz): ", len(bz), file=s)
            print("ks:", file=s)
            print(bz.k[iks], file=s)
            print("weights:", bz.weight.min(), bz.weight[iks].sum(), file=s)
            print(bz.weight[iks], file=s)
            print("sum(weights): ", bz.weight.sum(), file=s)
            print_s()

        # create the single monkhorst pack we will use for replacements
        if fast:
            new_bz = sisl.MonkhorstPack(bz.parent, nk, size=dsize, trs=False)
            new, reps = add_levels(
                new_bz, nks[:-1], ns[:-1], fast, as_index, debug=debug
            )

            if as_index:
                bz.replace(iks, new, displacement=True, as_index=True)
            else:
                bz.replace(bz.k[iks], new, displacement=True, as_index=False)

            nreps += 1 + reps

        else:
            if lvl == 0:
                iks = tqdm(iks, desc=f"lvl {lvl}")
            for ik in iks:
                k = bz.k[ik]
                if debug:
                    print(f"ik = {ik}", file=s)
                    print(f"k = {k}", file=s)
                    print(f"wk = {bz.weight[ik]}", file=s)
                    print_s()

                # Recursively add a new level
                # create the single monkhorst pack we will use for replacements
                new_bz = sisl.MonkhorstPack(
                    bz.parent, nk, size=dsize, trs=False, displacement=k
                )
                new, reps = add_levels(
                    new_bz, nks[:-1], ns[:-1], fast, as_index, debug=debug
                )

                # calculate number of replaced k-points
                if debug:
                    bz_nk = len(bz)

                if debug:
                    print(f"ik = {ik}", file=s)
                    print(f"k = {k}", file=s)
                    print(f"wk = {bz.weight[ik]}", file=s)
                    print_s()

                if False:
                    import matplotlib.pyplot as plt

                    plt.figure()
                    plt.scatter(bz.k[:, 0], bz.k[:, 1])
                    plt.title(f"{lvl} and {ik}")
                    plt.show()

                if as_index:
                    bz.replace(ik, new, displacement=fast, as_index=True)
                else:
                    bz.replace(k, new, displacement=fast)

                if debug:
                    rep_nk = len(new) - (len(bz) - bz_nk)
                    print("replaced k-points ", rep_nk, file=s)
                    print_s()
                    # print(len(bz)*4 * 8 / 1024**3)

                del new

                nreps += 1 + reps
            del new_bz

    return bz, nreps


gr = sisl.geom.graphene(1.44)
H = sisl.Hamiltonian(gr)
H.construct([[0, 1.44], [0, -2.7]])

# Now create the k-input
# this is number of k-points
trs = False
bz = sisl.MonkhorstPack(H, get_nk(nks[-1]), trs=trs)
debug = False

# Now add *many* points
for fast, as_index in [(True, False), (True, True), (False, False), (False, True)]:
    # Always fix the random seed to make each profiling concurrent
    np.random.seed(1234567890)
    b = bz.copy()

    print(f"running fast={fast}  as_index={as_index}")
    pr = cProfile.Profile()
    pr.enable()
    _, nreps = add_levels(b, nks, ns, fast, as_index, debug=debug)
    pr.disable()

    stat = pstats.Stats(pr)
    # We sort against total-time
    stat.sort_stats("tottime")
    # Only print the first 20% of the routines.
    stat.print_stats("sisl", 0.2)
