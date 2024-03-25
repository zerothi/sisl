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

import sisl

pr = cProfile.Profile()
pr.disable()

if len(sys.argv) > 1:
    N = int(sys.argv[1])
else:
    N = 200
print(f"N = {N}")

# Always fix the random seed to make each profiling concurrent
np.random.seed(1234567890)

gr = sisl.geom.graphene(orthogonal=True)
H = sisl.Hamiltonian(gr)
H.construct([(0.1, 1.44), (0.0, -2.7)])
pr.enable()
H.repeat(N, 0).repeat(N, 1)
H.finalize()
pr.disable()
pr.dump_stats(f"{sys.argv[0]}.profile")


stat = pstats.Stats(pr)
# We sort against total-time
stat.sort_stats("tottime")
# Only print the first 20% of the routines.
stat.print_stats("sisl", 0.2)
