#!/usr/bin/env python
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# This benchmark creates a very large graphene flake and uses construct
# to create it.

# This benchmark may be called using:
#
#  python -m cProfile -o $0.profile $0
#
# and it may be post-processed using
#
#  python stats.py $0.profile
#

from __future__ import annotations

import sys

import sisl

method = "cube"
if "cube" in sys.argv:
    method = "cube"
    sys.argv.remove("cube")
elif "sphere" in sys.argv:
    method = "sphere"
    sys.argv.remove("sphere")

if len(sys.argv) > 1:
    N = int(sys.argv[1])
else:
    N = 200
print(f"N = {N}")

gr = sisl.geom.graphene(orthogonal=True).tile(N, 0).tile(N, 1)
H = sisl.Hamiltonian(gr)
H.construct([(0.1, 1.44), (0.0, -2.7)], method=method, eta=True)
H.finalize()
