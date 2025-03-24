`WideBandSE.self_energy` changed its behavior of ``eta``

Now, the first argument is the *energy*, from which only
the imaginary value will be used.
Also, the energy will be scaled with `np.pi` to get the
*correct* integration.

Additionally, when the WideBandSE object is instantiated with
a sparsematrix, it will use the overlap matrix to account
for this.
