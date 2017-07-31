#!/usr/bin/env python

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

import sys
import sisl
import numpy as np

if len(sys.argv) > 1:
    N = int(sys.argv[1])
else:
    N = 200
print("N = {}".format(N))

gr = sisl.geom.graphene(orthogonal=True).tile(N, 0).tile(N, 1)
H = sisl.Hamiltonian(gr)
H.construct([(0.1, 1.44), (0., -2.7)], method='cube', eta=True)
