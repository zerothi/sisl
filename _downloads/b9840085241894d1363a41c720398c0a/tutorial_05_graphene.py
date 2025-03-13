# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import sisl as si

graphene = si.geom.graphene()

# Generate Hamiltonian
H = si.Hamiltonian(graphene)

# Show the initial state of the Hamiltonian
print(H)

H[0, 1] = 2.7
H[0, 1, (-1, 0)] = 2.7
H[0, 1, (0, -1)] = 2.7
H[1, 0] = 2.7
H[1, 0, (1, 0)] = 2.7
H[1, 0, (0, 1)] = 2.7

# Show that we indeed have added some code
print(H)

# Create band-structure for the supercell.
band = si.BandStructure(
    H, [[0.0, 0.0], [2.0 / 3, 1.0 / 3], [0.5, 0.5], [0.0, 0.0]], 300
)

# Calculate eigenvalues of the band-structure
eigs = band.apply.ndarray.eigh()

# Plot them
import matplotlib.pyplot as plt

plt.figure()
plt.title("Bandstructure of graphene, nearest neighbor")
plt.xlabel("k")
plt.ylabel("Eigenvalue")

# Generate linear-k for plotting (ensures correct spacing)
lband = band.lineark()
for i in range(eigs.shape[1]):
    plt.plot(lband, eigs[:, i])
plt.savefig("05_graphene_bs.png")
