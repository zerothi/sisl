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

randint = np.random.randint

if len(sys.argv) > 1:
    N = int(sys.argv[1])
else:
    N = 200
if len(sys.argv) > 2:
    frac = float(sys.argv[2])
else:
    frac = 0.1
if len(sys.argv) > 3:
    ND = int(sys.argv[3])
else:
    ND = 1
print(f"N = {N}")
print(f"sparsity = {frac}")
print(f"dimensions = {ND}")

# Always fix the random seed to make each profiling consistent
np.random.seed(1234567890)

n = int(N * frac)
sp0 = sisl.SparseCSR((N, N, ND), dtype=np.int32, nnzpr=n)
sp1 = sisl.SparseCSR((N, N, ND), dtype=np.int32, nnzpr=n)
for r in range(N):
    dat = randint(0, N, n)
    sp0[r, dat] = 1
    dat = randint(0, N, n)
    sp1[r, dat] = 1

# Now start profiling
pr.enable()
out = sp0 + sp1
pr.disable()
pr.dump_stats(f"{sys.argv[0]}.profile")


stat = pstats.Stats(pr)
# We sort against total-time
stat.sort_stats("tottime")
# Only print the first 20% of the routines.
stat.print_stats("sisl", 0.2)
