from sisl import *

graphene = geom.graphene()

# Generate Hamiltonian
H = Hamiltonian(graphene)

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
band = BandStructure(H, [[0., 0.], [2./3, 1./3], [0.5, 0.5], [0., 0.]], 300)

# Calculate eigenvalues of the band-structure
eigs = band.eigh()

# Plot them
import matplotlib.pyplot as plt

plt.figure()
plt.title('Bandstructure of graphene, nearest neighbour')
plt.xlabel('k')
plt.ylabel('Eigenvalue')

# Generate linear-k for plotting (ensures correct spacing)
lband = band.lineark()
for i in range(eigs.shape[1]):
    plt.plot(lband, eigs[:, i])
plt.savefig('05_graphene_bs.png')
