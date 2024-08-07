#!/usr/bin/env python
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# This benchmark creates a Hamiltonian and does eigenstate DOS + PDOS
# calculations.

# This benchmark may be called using:
#
#  python -m cProfile -o $0.profile $0
#
# and it may be post-processed using
#
#  python stats.py $0.profile
#

from __future__ import annotations

import os
import sys

import numpy as np

import sisl

if len(sys.argv) > 1:
    N = int(sys.argv[1])
else:
    N = 10
print(f"orbitals = {N*2}")
if len(sys.argv) > 2:
    nk = int(sys.argv[2])
else:
    nk = 100
print(f"nk = {nk}")

# Always fix the random seed to make each profiling concurrent
np.random.seed(1234567890)

gr = sisl.geom.graphene()
H = sisl.Hamiltonian(gr)
H.construct([(0.1, 1.44), (0.0, -2.7)])
H.finalize()
H = H.tile(N, 0)
bz = sisl.MonkhorstPack(H, [nk, nk, 1], trs=False)

nprocs = int(os.environ.get("SISL_NUM_PROCS", 1))
print(f"nprocs = {nprocs}")
if nprocs > 1:
    par = bz.apply.renew(pool=nprocs)
else:
    par = bz.apply.renew(eta=True)

E = np.linspace(-2, 2, 200)


def wrap_DOS(es):
    return es.DOS(E)


dos = par.ndarray.eigenstate(wrap=wrap_DOS)
# dos = par.average.eigenstate(wrap=wrap_DOS)
