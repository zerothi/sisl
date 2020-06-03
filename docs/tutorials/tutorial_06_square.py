from sisl import *

# Generate square lattice with nearest neighbour couplings
Hydrogen = Atom(1, R=1.)
square = Geometry([[0.5, 0.5, 0]], Hydrogen,
                  sc=SuperCell([1, 1, 10], [3, 3, 1]))

# Generate Hamiltonian
H = Hamiltonian(square)

# Show the initial state of the Hamiltonian
print(H)

# Specify matrix elements
for ias, idxs in square.iter_block():
    for ia in ias:
        idx_a = square.close(ia, R=[0.1, 1.1], atoms=idxs)
        H[ia, idx_a[0]] = -4.
        H[ia, idx_a[1]] = 1.

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
plt.savefig('06_square_bs.png')
