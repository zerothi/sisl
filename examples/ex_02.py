#!/usr/bin/env python
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# This example creates the tight-binding Hamiltonian
# for graphene with on-site energy 0, and hopping energy
# -2.7 eV using a simpler method to construct the
# Hamiltonian.

from __future__ import annotations

import sisl

bond = 1.42
# Construct the atom with the appropriate orbital range
# Note the 0.01 which is for numerical accuracy.
C = sisl.Atom(6, R=bond + 0.01)
# Create graphene unit-cell
gr = sisl.geom.graphene(bond, C)

# Create the tight-binding Hamiltonian
H = sisl.Hamiltonian(gr)

# Create function to be passed to the construct method.
# This method is *much* faster for large scale simulations.
func = H.create_construct([0.1 * bond, bond + 0.01], [0.0, -2.7])
H.construct(func)

# Calculate eigenvalues at K-point
print(H.eigh([2.0 / 3, 1.0 / 3, 0.0]))
