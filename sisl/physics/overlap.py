from __future__ import print_function, division

from .sparse import SparseOrbitalBZ

__all__ = ['Overlap']


class Overlap(SparseOrbitalBZ):
    """ Sparse overlap matrix object

    The Overlap object contains orbital overlaps. It should be used when the overlaps are not associated with
    another physical object such as a Hamiltonian, as is the case with eg. Siesta onlyS outputs.
    When the overlap is associated with a Hamiltonian, then this object should not be used as the overlap is stored
    in the Hamiltonian itself.

    Parameters
    ----------
    geometry : Geometry
      parent geometry to create an overlap matrix from. The overlap matrix will
      have size equivalent to the number of orbitals in the geometry.
    dtype : np.dtype, optional
      data type contained in the matrix.
    nnzpr : int, optional
      number of initially allocated memory per orbital in the matrix.
      For best performance this should be larger or equal to the actual number of entries
      per orbital.
    """

    def __init__(self, geometry, dtype=None, nnzpr=None, **kwargs):
        """ Initialize Overlap """
        kwargs["orthogonal"] = True  # Avoid the dim += 1 in super
        kwargs["dim"] = 1
        super(Overlap, self).__init__(geometry, dtype=dtype, nnzpr=nnzpr, **kwargs)

    def _reset(self):
        super(Overlap, self)._reset()
        self._orthogonal = False
        self.S_idx = 0
        self.Sk = self._Sk
        self.dSk = self._dPk
        self.ddSk = self._ddPk

    @classmethod
    def fromsp(cls, geometry, S, **kwargs):
        r""" Create an Overlap object from a preset `Geometry` and a sparse matrix

        The passed sparse matrix is in one of `scipy.sparse` formats.

        Note that the method for creating Overlap objects is (nearly) identical to eg. Hamiltonians, but may only be passed a single matrix.

        Parameters
        ----------
        geometry : Geometry
           geometry to describe the new sparse geometry
        S : scipy.sparse
           The sparse matrix in a `scipy.sparse` format.
        **kwargs : optional
           any arguments that are directly passed to the ``__init__`` method
           of the class. `dtype` and `nnzpr` are overridden.

        Returns
        -------
        Overlap
             a new Overlap object
        """
        # Using S explicitly in the argument ensures users will not pass it through
        # kwargs, if they do, an error will be raised.
        return super(Overlap, cls).fromsp(geometry, P=[S], S=None, **kwargs)

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
        """ Writes the Overlap to the `Sile` as implemented in the :code:`Sile.write_overlap` method """
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            sile.write_overlap(self, *args, **kwargs)
        else:
            with get_sile(sile, 'w') as fh:
                fh.write_overlap(self, *args, **kwargs)
