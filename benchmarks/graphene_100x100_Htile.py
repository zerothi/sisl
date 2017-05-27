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

import sisl
import numpy as np

# Always fix the random seed to make each profiling concurrent
np.random.seed(1234567890)

gr = sisl.geom.graphene(orthogonal=True)
H = sisl.Hamiltonian(gr)
H.construct([(0.1, 1.44), (0., -2.7)], eta=True)
H.tile(50, 0).tile(100, 1)
