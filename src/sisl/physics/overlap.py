# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

from sisl._internal import set_module

from .sparse import SparseOrbitalBZ

__all__ = ["Overlap"]


@set_module("sisl.physics")
class Overlap(SparseOrbitalBZ):
    r"""Sparse overlap matrix object

    The Overlap object contains orbital overlaps. It should be used when the overlaps are not associated with
    another physical object such as a Hamiltonian, as is the case with eg. Siesta onlyS outputs.
    When the overlap is associated with a Hamiltonian, then this object should not be used as the overlap is stored
    in the Hamiltonian itself.

    Parameters
    ----------
    geometry : Geometry
      parent geometry to create an overlap matrix from. The overlap matrix will
      have size equivalent to the number of orbitals in the geometry.
    dim : int, optional
      number of dimensions used to store the overlap matrix
    dtype : np.dtype, optional
      data type contained in the matrix.
    nnzpr : int, optional
      number of initially allocated memory per orbital in the matrix.
      For best performance this should be larger or equal to the actual number of entries
      per orbital.
    """

    def __init__(self, geometry, dim=1, dtype=None, nnzpr=None, **kwargs):
        r"""Initialize Overlap"""
        # Since this *is* the overlap matrix, we should never use the
        # orthogonal keyword
        kwargs["orthogonal"] = True
        super().__init__(geometry, dim, np.float64, nnzpr, **kwargs)
        self._reset()

    def _reset(self):
        super()._reset()
        self.Sk = self._Pk
        self.dSk = self._dPk
        self.ddSk = self._ddPk

    @property
    def S(self):
        r"""Access the overlap elements"""
        self._def_dim = 0
        return self

    @classmethod
    def fromsp(cls, geometry, P, **kwargs):
        r"""Create an Overlap object from a preset `Geometry` and a sparse matrix

        The passed sparse matrix is in one of `scipy.sparse` formats.

        Note that the method for creating Overlap objects is (nearly) identical to eg. Hamiltonians, but may only be passed a single matrix.

        Parameters
        ----------
        geometry : Geometry
           geometry to describe the new sparse geometry
        P : list of scipy.sparse or scipy.sparse
           the new sparse matrices that are to be populated in the sparse
           matrix
        **kwargs : optional
           any arguments that are directly passed to the ``__init__`` method
           of the class.

        Returns
        -------
        Overlap
             a new Overlap object
        """
        # Using S explicitly in the argument ensures users will not pass it through
        # kwargs, if they do, an error will be raised.
        return super().fromsp(geometry, P=P, S=None, **kwargs)

    @staticmethod
    def read(sile, *args, **kwargs):
        """Reads Overlap from `Sile` using `read_overlap`.

        Parameters
        ----------
        sile : Sile, str or pathlib.Path
            a `Sile` object which will be used to read the Overlap
            and the overlap matrix (if any)
            if it is a string it will create a new sile using `get_sile`.
        * : args passed directly to ``read_overlap(,**)``
        """
        from sisl.io import BaseSile, get_sile

        if isinstance(sile, BaseSile):
            return sile.read_overlap(*args, **kwargs)
        else:
            with get_sile(sile, mode="r") as fh:
                return fh.read_overlap(*args, **kwargs)
