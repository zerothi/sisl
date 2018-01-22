from sisl import *

# Generate square lattice with nearest neighbour couplings
square = Geometry([[0.5, 0.5, 0]], sc=SuperCell([1, 1, 10], [3, 3, 1]))

# Generate Hamiltonian
H = Hamiltonian(square)

# Show the initial state of the Hamiltonian
print(H)

# Specify matrix elements (on-site and coupling elements)
H[0, 0] = -4.
H[0, 0, (1, 0)] = 1.
H[0, 0, (-1, 0)] = 1.
H[0, 0, (0, 1)] = 1.
H[0, 0, (0, -1)] = 1.

# Show that we indeed have added some code
print(H)

# Create band-structure for the supercell.
band = BandStructure(H, [[0., 0.], [0.5, 0.], [0.5, 0.5], [0., 0.]], 300)

# Calculate eigenvalues of the band-structure
eigs = band.eigh()

# Plot them
import matplotlib.pyplot as plt

plt.figure()
plt.title('Bandstructure of square, nearest neighbour')
plt.xlabel('k')
plt.ylabel('Eigenvalue')

# Generate linear-k for plotting (ensures correct spacing)
lband = band.lineark()
for i in range(eigs.shape[1]):
    plt.plot(lband, eigs[:, i])
plt.savefig('05_square_bs.png')
