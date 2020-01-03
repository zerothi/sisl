import numpy as np
from .sparse import SparseOrbitalBZ

__all__ = ['Overlap']


class Overlap(SparseOrbitalBZ):
    """ Sparse Overlap matrix object

    The Overlap object contains orbital overlaps. It should be used when the overlaps are not associated with
    another physical object such as a Hamiltonian, as is the case with eg. Siesta onlyS outputs.
    When the overlap is associated with a Hamiltonian, then this object should not be used as the overlap is stored
    in the Hamiltonian itself.

    Parameters
    ----------
    geometry : Geometry
      parent geometry to create a density matrix from. The density matrix will
      have size equivalent to the number of orbitals in the geometry
    dtype : np.dtype, optional
      data type contained in the density matrix.
    nnzpr : int, optional
      number of initially allocated memory per orbital in the density matrix.
      For increased performance this should be larger than the actual number of entries
      per orbital.
    """

    def __init__(self, geometry, dtype=None, nnzpr=None, **kwargs):
        """ Initialize Overlap """
        kwargs.update({"orthogonal": True})  # Avoid the dim += 1 in super
        super().__init__(geometry, 1, np.float64, nnzpr, **kwargs)
        self._orthogonal = False
        self._reset()

    def _reset(self):
        super()._reset()
        self.S_idx = 0
        self.Sk = self.Pk
        self.dSk = self.dPk
        self.ddSk = self.ddPk

    @classmethod
    def fromsp(cls, geometry, P=None, S=None, **kwargs):
        r""" Create an Overlap object from a preset `Geometry` and a sparse matrix

        The passed sparse matrix is in one of `scipy.sparse` formats.

        Note that the method for creating Overlap objects is (nearly) identical to eg. Hamiltonians, but you may not
        pass any 'P' matrices. Instead you must use the `S` keyword argument.

        Parameters
        ----------
        geometry : Geometry
           geometry to describe the new sparse geometry
        S : scipy.sparse, required, keyword only
           The sparse matrix in a `scipy.sparse` format.
        **kwargs : optional
           any arguments that are directly passed to the ``__init__`` method
           of the class.

        Returns
        -------
        Overlap
             a new Overlap object
        """
        if S is None:
            raise TypeError("fromsp() is missing 1 required keyword argument: 'S'")
        if P is not None and len(P) != 0:
            raise TypeError("Cannot create an Overlap object with anything other than S")
        return super().fromsp(geometry, P=[S], S=None, **kwargs)

    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads Overlap from `Sile` using `read_overlap`.

        Parameters
        ----------
        sile : Sile, str or pathlib.Path
            a `Sile` object which will be used to read the Overlap
            and the overlap matrix (if any)
            if it is a string it will create a new sile using `get_sile`.
        * : args passed directly to ``read_overlap(,**)``
        """
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            return sile.read_overlap(*args, **kwargs)
        else:
            with get_sile(sile) as fh:
                return fh.read_overlap(*args, **kwargs)

    def write(self, sile, *args, **kwargs):
        """ Writes a Hamiltonian to the `Sile` as implemented in the :code:`Sile.write_hamiltonian` method """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            sile.write_overlap(self, *args, **kwargs)
        else:
            with get_sile(sile, 'w') as fh:
                fh.write_overlap(self, *args, **kwargs)
