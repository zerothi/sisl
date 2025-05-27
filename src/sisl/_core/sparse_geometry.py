# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import warnings
from collections import namedtuple
from numbers import Integral
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from numpy import (
    allclose,
    argsort,
    concatenate,
    delete,
    diff,
    insert,
    int32,
    intersect1d,
    lexsort,
    repeat,
    searchsorted,
    tile,
    unique,
)
from numpy.lib.mixins import NDArrayOperatorsMixin
from scipy.sparse import csr_matrix

from sisl import _array as _a
from sisl._array import array_arange
from sisl._core import Atom, Geometry, Orbital
from sisl._internal import set_module
from sisl.messages import SislError, SislWarning, deprecate_argument, progressbar, warn
from sisl.typing import AtomsIndex, CellAxes, Coord, SeqOrScalarFloat
from sisl.typing._atom import AtomsLike
from sisl.utils.misc import direction
from sisl.utils.ranges import list2str

from .sparse import SparseCSR, _ncol_to_indptr, _to_coo, issparse, valid_index

__all__ = ["SparseAtom", "SparseOrbital"]


class _SparseGeometry(NDArrayOperatorsMixin):
    """Sparse object containing sparse elements for a given geometry.

    This is a base class intended to be sub-classed because the sparsity information
    needs to be extracted from the ``_size`` attribute.

    The sub-classed object _must_ implement the ``_size`` attribute.
    The sub-classed object may re-implement the ``_cls_kwargs`` routine
    to pass down keyword arguments when a new class is instantiated.

    This object contains information regarding the
     - geometry

    """

    def __init__(
        self,
        geometry: Geometry,
        dim: int = 1,
        dtype=None,
        nnzpr: Optional[int] = None,
        **kwargs,
    ):
        """Create sparse object with element between orbitals"""
        self._geometry = geometry

        # Initialize the sparsity pattern
        self.reset(dim, dtype, nnzpr)

    @property
    def geometry(self) -> Geometry:
        """Associated geometry"""
        return self._geometry

    @property
    def _size(self) -> int:
        """The size of the sparse object"""
        return self.geometry.na

    def __len__(self) -> int:
        """Number of rows in the basis"""
        return self._size

    def _cls_kwargs(self):
        """Custom keyword arguments when creating a new instance"""
        return {}

    def reset(
        self, dim: Optional[int] = None, dtype=np.float64, nnzpr: Optional[int] = None
    ) -> None:
        """The sparsity pattern has all elements removed and everything is reset.

        The object will be the same as if it had been
        initialized with the same geometry as it were
        created with.

        Parameters
        ----------
        dim :
           number of dimensions per element, default to the current number of
           elements per matrix element.
        dtype : numpy.dtype, optional
           the datatype of the sparse elements
        nnzpr :
           number of non-zero elements per row
        """
        # I know that this is not the most efficient way to
        # access a C-array, however, for constructing a
        # sparse pattern, it should be faster if memory elements
        # are closer...
        if dim is None:
            dim = self.dim

        # We check the first atom and its neighbors, we then
        # select max(5,len(nc) * 4)
        if nnzpr is None:
            nnzpr = self.geometry.close(0)
            if nnzpr is None:
                nnzpr = 8
            else:
                nnzpr = max(5, len(nnzpr) * 4)

        # query dimension of sparse matrix
        s = self._size
        self._csr = SparseCSR((s, s * self.geometry.n_s, dim), nnzpr=nnzpr, dtype=dtype)

        # Denote that one *must* specify all details of the elements
        self._def_dim = -1

    def empty(self, keep_nnz: bool = False) -> None:
        """See :meth:`~sparse.SparseCSR.empty` for details"""
        self._csr.empty(keep_nnz)

    @property
    def dim(self) -> int:
        """Number of components per element"""
        return self._csr.shape[-1]

    @property
    def shape(self) -> Tuple[int]:
        """Shape of sparse matrix"""
        return self._csr.shape

    @property
    def dtype(self):
        """Data type of sparse elements"""
        return self._csr.dtype

    @property
    def dkind(self):
        """Data type of sparse elements (in str)"""
        return self._csr.dkind

    @property
    def nnz(self) -> int:
        """Number of non-zero elements"""
        return self._csr.nnz

    def translate2uc(self, atoms: AtomsIndex = None, axes: Optional[CellAxes] = None):
        """Translates all primary atoms to the unit cell.

        With this, the coordinates of the geometry are translated to the unit cell
        and the supercell connections in the matrix are updated accordingly.

        Parameters
        ----------
        atoms :
            only translate the specified atoms. If not specified, all
            atoms will be translated.
        axes :
            only translate certain lattice directions, `None` specifies
            only the periodic directions

        Returns
        --------
        SparseOrbital or SparseAtom
            A new sparse matrix with the updated connections and a new associated geometry.
        """
        # Sanitize the axes argument
        if axes is None:
            axes = self.lattice.pbc.nonzero()[0]

        elif isinstance(axes, bool):
            if axes:
                axes = (0, 1, 2)
            else:
                raise ValueError(
                    "translate2uc with a bool argument can only be True to signal all axes"
                )
        else:
            axes = list(map(direction, axes))

        # Sanitize also the atoms argument
        if atoms is None:
            ats = slice(None)
        else:
            ats = self.geometry._sanitize_atoms(atoms).ravel()

        # Get the fractional coordinates of the associated geometry
        fxyz = self.geometry.fxyz

        # Get the cell where each atom resides. In fractional coordinates, atoms in the unit cell
        # are between 0 and 1. Anything else means that the atom resides in a periodic image. For
        # atoms and axes that the user doesn't desire the translation, we are going to set the supercell
        # offset as if it was 0, which will result in them not getting translated.
        current_sc = np.zeros([self.na, 3], dtype=np.int32)
        current_sc[ats, axes] = np.floor(fxyz[ats, axes]).astype(np.int32)

        # Simply translate the atoms to move all atoms to the unit cell. That is, all atoms
        # should be moved to supercell (0,0,0).
        return self._translate_atoms_sc(-current_sc)

    def _transpose_indices(
        self, indices: npt.ArrayLike, base: Optional[npt.ArrayLike] = None
    ) -> np.ndarray:
        """Converts from supercell indices to supercell transposed indices.

        Parameters
        ----------
        indices :
            the indices that contains the supercell indices.
        base :
            the resulting base orbitals that needs to translated.
            These defaults to the unit-cell indices of `indices`.
        """
        lattice = self.geometry.lattice

        # transposed offsets
        new_sc_off = lattice.sc_index(-lattice.sc_off)

        # Calculate the row-offsets in the new sparse geometry
        size = self.shape[0]
        if base is None:
            base = indices % size

        base = (
            base
            + new_sc_off[lattice.sc_index(lattice.sc_off[indices // size, :])] * size
        )
        return base

    def _translate_atoms_sc(self, sc_translations):
        """Translates atoms across supercells.

        This operation results in new coordinates of the associated geometry
        and new indices for the matrix elements.

        Parameters
        ----------
        sc_translations : array of int of shape (na, 3)
            For each atom, the displacement in number of supercells along each direction.

        Returns
        --------
        SparseOrbital or SparseAtom
            A new sparse matrix with the updated connections and a new associated geometry.
        """
        # Make sure that supercell translations is an array of integers
        sc_translations = np.asarray(sc_translations, dtype=int)

        # Get the row and column of every element in the matrix
        rows, cols = self.nonzero()

        n_rows = self.shape[0]
        is_atom = n_rows == self.na

        # Find out the unit cell indices for the columns, and the index of the supercell
        # where they are currently located. This is done by dividing into the number of
        # columns in the unit cell.
        # We will do the conversion back to supercell indices when we know their
        # new location after translation, and also the size of the new auxiliary supercell.
        sc_idx, uc_col = np.divmod(cols, n_rows)

        # We need the unit cell indices of the column atoms. If this is a SparseAtom object, then
        # we have already computed them in the previous line. Otherwise, compute them.
        if is_atom:
            # atomic indices
            at_row = rows
            at_col = uc_col
        else:
            # orbital indices
            at_row = self.o2a(rows)
            at_col = self.o2a(cols) % self.na

        # Get the supercell indices of the original positions.
        isc = self.sc_off[sc_idx]

        # We are going to now displace the supercell index of the connections
        # according to how the two orbitals involved have moved. We store the
        # result in the same array just to avoid using more memory.
        isc += sc_translations[at_row] - sc_translations[at_col]

        # It is possible that once we discover the new locations of the connections
        # we find out that we need a bigger or smaller auxiliary supercell. Find out
        # the size of the new auxiliary supercell.
        new_nsc = np.max(abs(isc), axis=0) * 2 + 1

        # Create a new geometry object with the new auxiliary supercell size.
        new_geometry = self.geometry.copy()
        new_geometry.set_nsc(new_nsc)

        # Update the coordinates of the geometry, according to the cell
        # displacements.
        new_geometry.xyz = new_geometry.xyz + sc_translations @ new_geometry.cell

        # Find out supercell indices in this new auxiliary supercell
        new_sc = new_geometry.isc_off[isc[:, 0], isc[:, 1], isc[:, 2]]

        # With this, we can compute the new columns
        new_cols = uc_col + new_sc * n_rows

        # Build the new csr matrix, which will just be a copy of the current one
        # but updating the column indices. It is possible that there are column
        # indices that are -1, which are the placeholders for new elements. We make sure
        # that we update only the indices that are not -1.
        # We also need to make sure that the shape of the matrix is appropiate
        # for the size of the new auxiliary cell.
        new_csr = self._csr.copy()
        new_csr.col[new_csr.col >= 0] = new_cols
        new_csr._shape = (n_rows, n_rows * new_geometry.n_s, new_csr.shape[-1])

        # Create the new SparseGeometry matrix and associate to it the csr matrix that we have built.
        new_matrix = self.__class__(new_geometry)
        new_matrix._csr = new_csr

        return new_matrix

    def _translate_cells(self, old, new):
        """Translates all columns in the `old` cell indices to the `new` cell indices

        Since the physical matrices are stored in a CSR form, with shape ``(no, no * n_s)`` each
        block of ``(no, no)`` refers to supercell matrices with an offset according to the internal
        supercell index.
        This routine may be used to translate from one sorting of the columns to another sorting of the columns.

        Parameters
        ----------
        old : list of int
           integer list of supercell indices (all smaller than `n_s`) that the current blocks of matrices
           belong to.
        new : list of int
           integer list of supercell indices (all smaller than `n_s`) that the current blocks of matrices
           are being transferred to. Must have same length as `old`.
        """
        old = _a.asarrayi(old).ravel()
        new = _a.asarrayi(new).ravel()

        if len(old) != len(new):
            raise ValueError(
                self.__class__.__name__
                + ".translate_cells requires input and output indices with "
                "equal length"
            )

        no = self.no
        # Number of elements per matrix
        n = _a.emptyi(len(old))
        n.fill(no)
        old = array_arange(old * no, n=n)
        new = array_arange(new * no, n=n)
        self._csr.translate_columns(old, new)

    def edges(self, atoms: AtomsIndex, exclude: AtomsIndex = None):
        """Retrieve edges (connections) for all `atoms`

        The returned edges are unique and sorted (see `numpy.unique`) and are returned
        in supercell indices (i.e. ``0 <= edge < self.geometry.na_s``).

        Parameters
        ----------
        atoms :
            the edges are returned only for the given atom
        exclude :
           remove edges which are in the `exclude` list.

        See Also
        --------
        SparseCSR.edges: the underlying routine used for extracting the edges
        """
        atoms = self.geometry._sanitize_atoms(atoms)
        if exclude is not None:
            exclude = self.geometry._sanitize_atoms(exclude)
        return self._csr.edges(atoms, exclude)

    def __str__(self) -> str:
        """Representation of the sparse model"""
        s = f"{self.__class__.__name__}{{dim: {self.dim}, non-zero: {self.nnz}, kind={self.dkind}\n "
        return s + str(self.geometry).replace("\n", "\n ") + "\n}"

    def __repr__(self) -> str:
        return f"<{self.__module__}.{self.__class__.__name__} shape={self._csr.shape[:-1]}, dim={self.dim}, nnz={self.nnz}, kind={self.dkind}>"

    def __getattr__(self, attr):
        """Overload attributes from the hosting geometry

        Any attribute not found in the sparse class will
        be looked up in the hosting geometry.
        """
        return getattr(self.geometry, attr)

    # Make the indicis behave on the contained sparse matrix
    def __delitem__(self, key):
        """Delete elements of the sparse elements"""
        del self._csr[key]

    def __contains__(self, key):
        """Check whether a sparse index is non-zero"""
        return key in self._csr

    def set_nsc(self, base_size, *args, **kwargs):
        """Reset the number of allowed supercells in the sparse geometry

        If one reduces the number of supercells, *any* sparse element
        that references the supercell will be deleted.

        See `Lattice.set_nsc` for allowed parameters.

        See Also
        --------
        Lattice.set_nsc : the underlying called method
        """
        lattice = self.lattice.copy()
        # Try first in the new one, then we figure out what to do
        lattice.set_nsc(*args, **kwargs)
        if allclose(lattice.nsc, self.lattice.nsc):
            return

        # Create an array of all things that should be translated
        old = []
        new = []
        deleted = np.empty(self.n_s, np.bool_)
        deleted[:] = True
        for i, sc_off in lattice:
            try:
                # Luckily there are *only* one time wrap-arounds
                j = self.lattice.sc_index(sc_off)
                # Now do translation
                old.append(j)
                new.append(i)
                deleted[j] = False
            except ValueError:
                # Not found, i.e. new, so no need to translate
                pass

        # 1. Ensure that any one of the *old* supercells that
        #    are now deleted are put in the end
        for i, j in enumerate(deleted.nonzero()[0]):
            # Old index (j)
            old.append(j)
            # Move to the end (*HAS* to be higher than the number of
            # cells in the new supercell structure)
            new.append(max(self.n_s, lattice.n_s) + i)

        # Check that we will translate all indices in the old
        # sparsity pattern to the new one
        if len(old) not in (self.n_s, lattice.n_s):
            raise SislError("Not all supercells are accounted for")

        old = _a.arrayi(old)
        new = _a.arrayi(new)

        # Assert that there are only unique values
        if len(unique(old)) != len(old):
            raise SislError("non-unique values in old set_nsc")
        if len(unique(new)) != len(new):
            raise SislError("non-unique values in new set_nsc")
        if self.n_s != len(old):
            raise SislError("non-valid size of in old set_nsc")

        # Figure out if we need to do any work
        keep = (old != new).nonzero()[0]
        if len(keep) > 0:
            # Reduce pivoting work
            old = old[keep]
            new = new[keep]

            # Create the translation tables
            n = tile([base_size], len(old))

            old = array_arange(old * base_size, n=n)
            new = array_arange(new * base_size, n=n)

            # Move data to new positions
            self._csr.translate_columns(old, new, clean=False)

            max_n = new.max() + 1
        else:
            max_n = 0
        # Make sure we delete all column values where we have put fake values
        delete = _a.arangei(lattice.n_s * base_size, max(max_n, self.shape[1]))
        if len(delete) > 0:
            self._csr.delete_columns(delete, keep_shape=True)

        # Ensure the shape is correct
        shape = list(self._csr.shape)
        shape[1] = base_size * lattice.n_s
        self._csr._shape = tuple(shape)
        self._csr._clean_columns()

        self.geometry.set_nsc(*args, **kwargs)

    def transpose(self, sort: bool = True) -> Self:
        """Create the transposed sparse geometry by interchanging supercell indices

        Sparse geometries are (typically) relying on symmetry in the supercell picture.
        Thus when one transposes a sparse geometry one should *ideally* get the same
        matrix. This is true for the Hamiltonian, density matrix, etc.

        This routine transposes all rows and columns such that any interaction between
        row, `r`, and column `c` in a given supercell `(i,j,k)` will be transposed
        into row `c`, column `r` in the supercell `(-i,-j,-k)`.

        Parameters
        ----------
        sort :
           the returned columns for the transposed structure will be sorted
           if this is true, default

        Notes
        -----
        The components for each sparse element are not changed in this method.

        Examples
        --------

        Force a sparse geometry to be Hermitian:

        >>> sp = SparseOrbital(...)
        >>> sp = (sp + sp.transpose()) / 2

        Returns
        -------
        object
            an equivalent sparse geometry with transposed matrix elements
        """
        # Create a temporary copy to put data into
        T = self.copy()
        # clean memory to not crowd memory too much
        T._csr.ptr = None
        T._csr.col = None
        T._csr.ncol = None
        T._csr._D = None

        # Create "DOK" format indices
        csr = self._csr
        # Number of rows (used for converting to supercell indices)
        # With this we don't need to figure out if we are dealing with
        # atoms or orbitals
        size = csr.shape[0]

        # First extract the sparse matrix in COO format
        row, col, D = _to_coo(csr)

        row = self._transpose_indices(col, base=row)

        # Now convert columns into unit-cell
        col %= size

        # Now we can re-create the sparse matrix
        # All we need is to count the number of non-zeros per column.
        rows, nrow = unique(col, return_counts=True)
        T._csr.ncol = _a.zerosi(size)
        T._csr.ncol[rows] = nrow
        del rows

        if sort:
            # also sort individual rows for each column
            idx = lexsort((row, col))
        else:
            # sort columns to get transposed values.
            # This will randomize the rows
            idx = argsort(col)

        # Our new data will then be
        T._csr.col = row[idx]
        del row
        T._csr._D = D[idx]
        del D
        T._csr.ptr = _ncol_to_indptr(T._csr.ncol)

        # If `sort` we have everything sorted, otherwise it
        # is not ensured
        T._csr._finalized = sort

        return T

    def spalign(self, other) -> None:
        """See :meth:`~sisl.sparse.SparseCSR.align` for details"""
        if isinstance(other, SparseCSR):
            self._csr.align(other)
        else:
            self._csr.align(other._csr)

    def eliminate_zeros(self, *args, **kwargs) -> None:
        """Removes all zero elements from the sparse matrix

        This is an *in-place* operation.

        See Also
        --------
        SparseCSR.eliminate_zeros : method called, see there for parameters
        """
        self._csr.eliminate_zeros(*args, **kwargs)

    # Create iterations on the non-zero elements
    def iter_nnz(self):
        """Iterations of the non-zero elements

        An iterator on the sparse matrix with, row and column

        Examples
        --------
        >>> for i, j in self.iter_nnz():
        ...    self[i, j] # is then the non-zero value
        """
        yield from self._csr

    __iter__ = iter_nnz

    def create_construct(self, R, params):
        """Create a simple function for passing to the `construct` function.

        This is simply to leviate the creation of simplistic
        functions needed for setting up the sparse elements.

        Basically this returns a function:

        >>> def func(self, ia, atoms, atoms_xyz=None):
        ...     idx = self.geometry.close(ia, R=R, atoms=atoms, atoms_xyz=atoms_xyz)
        ...     for ix, p in zip(idx, params):
        ...         self[ia, ix] = p

        Notes
        -----
        This function only works for geometry sparse matrices (i.e. one
        element per atom). If you have more than one element per atom
        you have to implement the function your-self.

        Parameters
        ----------
        R : array_like
           radii parameters for different shells.
           Must have same length as `params` or one less.
           If one less it will be extended with ``R[0]/100``
        params : array_like
           coupling constants corresponding to the `R`
           ranges. ``params[0, :]`` are the elements
           for the all atoms within ``R[0]`` of each atom.

        See Also
        --------
        construct : routine to create the sparse matrix from a generic function (as returned from `create_construct`)
        """
        if len(R) != len(params):
            raise ValueError(
                f"{self.__class__.__name__}.create_construct got different lengths of 'R' and 'params'"
            )

        def func(self, ia, atoms, atoms_xyz=None):
            idx = self.geometry.close(ia, R=R, atoms=atoms, atoms_xyz=atoms_xyz)
            for ix, p in zip(idx, params):
                self[ia, ix] = p

        func.R = R
        func.params = params

        return func

    def construct(self, func, na_iR: int = 1000, method: str = "rand", eta=None):
        """Automatically construct the sparse model based on a function that does the setting up of the elements

        This may be called in two variants.

        1. Pass a function (`func`), see e.g. ``create_construct``
           which does the setting up.
        2. Pass a tuple/list in `func` which consists of two
           elements, one is ``R`` the radii parameters for
           the corresponding parameters.
           The second is the parameters
           corresponding to the ``R[i]`` elements.
           In this second case all atoms must only have
           one orbital.

        Parameters
        ----------
        func : callable or array_like
           this function *must* take 4 arguments.
           1. Is this object (``self``)
           2. Is the currently examined atom (``ia``)
           3. Is the currently bounded indices (``idxs``)
           4. Is the currently bounded indices atomic coordinates (``idxs_xyz``)
           An example `func` could be:

           >>> def func(self, ia, atoms, atoms_xyz=None):
           ...     idx = self.geometry.close(ia, R=[0.1, 1.44], atoms=atoms, atoms_xyz=atoms_xyz)
           ...     self[ia, idx[0]] = 0
           ...     self[ia, idx[1]] = -2.7

        na_iR : int, optional
           number of atoms within the sphere for speeding
           up the `iter_block` loop.
        method : {'rand', str}
           method used in `Geometry.iter_block`, see there for details
        eta : bool, optional
           whether an ETA will be printed

        See Also
        --------
        create_construct : a generic function used to create a generic function which this routine requires
        tile : tiling *after* construct is much faster for very large systems
        repeat : repeating *after* construct is much faster for very large systems
        """
        if not callable(func):
            if not isinstance(func, (tuple, list)):
                raise ValueError(
                    "Passed `func` which is not a function, nor tuple/list of `R, param`"
                )

            if np.any(diff(self.geometry.lasto) > 1):
                raise ValueError(
                    "Automatically setting a sparse model "
                    "for systems with atoms having more than 1 "
                    "orbital *must* be done by your-self. You have to define a corresponding `func`."
                )

            # Convert to a proper function
            func = self.create_construct(func[0], func[1])

        try:
            # if the function was created through `create_construct`, then
            # we have access to the radii used.
            R = func.R
            try:
                if len(R) > 0:
                    R = R[-1]
            except TypeError:
                pass
        except AttributeError:
            R = None

        iR = self.geometry.iR(na_iR, R=R)

        # Create eta-object
        eta = progressbar(self.na, f"{self.__class__.__name__ }.construct", "atom", eta)

        # Do the loop
        for ias, idxs in self.geometry.iter_block(iR=iR, method=method, R=R):
            # Get all the indexed atoms...
            # This speeds up the searching for coordinates...
            idxs_xyz = self.geometry[idxs]

            # Loop the atoms inside
            for ia in ias:
                func(self, ia, idxs, idxs_xyz)

            eta.update(len(ias))

        eta.close()

    @property
    def finalized(self) -> bool:
        """Whether the contained data is finalized and non-used elements have been removed"""
        return self._csr.finalized

    def untile(
        self,
        prefix: str,
        reps: int,
        axis: int,
        segment: int = 0,
        *args,
        sym: bool = True,
        **kwargs,
    ) -> Self:
        """Untiles a sparse model into a minimum segment, reverse of `tile`

        Parameters
        ----------
        prefix : {a, o}
           which quantity to request for the size of the matrix
        reps :
           number of untiles that needs to be performed
        axis :
           which axis we need to untile (length with be ``1/reps`` along this axis)
        segment :
           which segment to return, default to the first segment. For a fully symmetric
           system there should not be a difference, requesting different segments can
           be used to assert this is the case.
        sym :
           whether to symmetrize before returning
        """
        # Create new geometry
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Create new untiled geometry
            geom = self.geometry.untile(reps, axis, segment, *args, **kwargs)
            # Check whether the warning exists
            if len(w) > 0:
                if issubclass(w[-1].category, SislWarning):
                    warn(
                        f"{str(w[-1].message)}\n---\n"
                        "The sparse matrix cannot be untiled as the structure "
                        "cannot be tiled accordingly. ANY use of the model has been "
                        "relieved from sisl."
                    )

        # Now we need to re-create number of supercells
        no = getattr(self, f"n{prefix}")
        geom_no = getattr(geom, f"n{prefix}")

        # orig-orbs
        orig_orbs = _a.arangei(segment * geom_no, (segment + 1) * geom_no)

        # create correct linear offset due to the segment.
        # Further below we will take out the linear indices by modulo and integer
        # division operations.
        lsc = tile(self.geometry.lattice.sc_off, (1, reps)).reshape(-1, reps, 3)
        lsc[:, :, axis] = lsc[:, :, axis] * reps + _a.arangei(reps) - segment
        lsc.shape = (-1, 3)

        # now we have the *correct* lsc that corresponds to the
        # sc_off in the cut structure.
        # We will later down correct the *wrong* indices
        # since we may have nsc == 1 and cut it X times.
        # In this case we may find [0, 1, 2, 3, ..., X-1]
        # which is clearly wrong. In stead we should determine
        # the correct nsc for the output geometry and convert it to
        #   [-X//2, ..., 0, ... , X//2] (or close to this)

        # First we need to figure out how long the interaction range is
        # in the cut-direction
        # We initialize to be the same as the parent direction
        nsc = self.nsc.copy()

        # get unique couplings for the orbitals we are cutting out
        # we ensure the *onsite* columns are also there
        # create sample sparse pattern so we can figure out the columns
        # they connect to
        S = self.tocsr(0)
        sub = np.union1d(unique(S[orig_orbs, :].indices), orig_orbs)
        del S

        if len(sub) == 0:
            raise ValueError(
                f"{self.__class__.__name__}.untile couples to no "
                "matrix elements, an empty sparse model cannot be split."
            )

        # Figure out the supercell indices of sub
        sub_sc = getattr(self.geometry, f"{prefix}2isc")(sub)

        # convert the sub_sc[axis] into the linear index to figure out the
        # actual number of required nsc
        sub_lsc = (sub % no + sub_sc[:, axis] * no) // geom_no - segment

        # calculate (sorted) unique linear cells
        # This is just to figure out how many we are connecting too
        sub_lsc = unique(sub_lsc)

        # determine the cut placement
        if nsc[axis] == 1:
            # no initial supercell, special handling
            dsub_lsc = np.diff(sub_lsc)
            if np.all(dsub_lsc == 1) and len(sub_lsc) == reps:
                # the full cut region is touched
                nsc[axis] = (reps // 2) * 2 + 1

                # here the couplings *touches* each others segments
                # We have to figure out if the couplings are *for* real
                # or whether we can easily cut them.
                if reps % 2 == 1:
                    # an un-even number of couplings in total
                    # this means that the same couplings will be duplicated
                    # to the right and left.
                    # Example:
                    #   [-1 0 1 2] or [0 1 2 3]
                    # [0] -> [2] positive direction
                    # [0] <- [2] negative direction
                    if sym:
                        msg = f"The elements connecting from the primary unit-cell to the {nsc[axis]//2} unit-cell will be halved, sym={sym}."
                    else:
                        msg = f"The returned matrix will not have symmetric couplings due to sym={sym} argument."

                    warn(
                        f"{self.__class__.__name__}.untile matrix has connections crossing "
                        "the entire unit cell. "
                        f"This may result in wrong behavior due to non-unique matrix elements. {msg}"
                    )

                else:
                    # even case
                    #   [-1 0 1]
                    # [0] -> [1] positive direction
                    # [0] <- [-1] negative direction
                    # or [0 1 2]
                    # [0] -> [1] positive direction
                    # [0] <- [2] negative direction
                    warn(
                        f"{self.__class__.__name__}.untile may have connections crossing "
                        "the entire unit cell. "
                        "This may result in wrong behavior due to non-unique matrix elements."
                    )

            else:
                # we have something like
                #  [0 1 - 3]
                # meaning that there is a gab in the couplings

                # remove duplicate neighboring values
                single_sel = np.ones(len(sub_lsc), dtype=bool)
                single_sel[1:] = sub_lsc[1:] != sub_lsc[:-1]

                single_sub_lsc = sub_lsc[single_sel]
                axis0 = (single_sub_lsc == 0).nonzero()[0][0]

                # initialize
                pos_nsc = 0
                found = True
                while found:
                    try:
                        if single_sub_lsc[axis0 + pos_nsc + 1] == pos_nsc + 1:
                            pos_nsc += 1
                        else:
                            found = False
                    except Exception:
                        found = False

                neg_nsc = 0
                found = True
                while found:
                    try:
                        if single_sub_lsc[axis0 + neg_nsc - 1] == neg_nsc - 1:
                            neg_nsc -= 1
                        else:
                            found = False
                    except Exception:
                        found = False

                nsc[axis] = max(pos_nsc, -neg_nsc) * 2 + 1

            # correct the linear indices that are *too* high
            hnsc = nsc[axis] // 2
            lsc[lsc[:, axis] > hnsc, axis] -= reps
            lsc[lsc[:, axis] < -hnsc, axis] += reps
            # this will still leave some supercell indices *wrong*
            # But the algorithm should detect that they are not coupled and
            # thus should not be queried.

        else:
            # *easy* case, we always have supercells so we can't cut too short
            # Simply track off the biggest one
            nsc[axis] = np.abs(sub_lsc).max() * 2 + 1

            # Create the to-columns
            if sub_lsc.max() != -sub_lsc.min():
                raise ValueError(
                    f"{self.__class__.__name__}.untile found inconsistent supercell matrix. "
                    f"The untiled sparse matrix couples to {sub_lsc} supercells but expected a symmetric set of couplings. "
                    "This may happen if doing multiple cuts along the same direction, or if the matrix is not correctly constructed."
                )

        # Update number of super-cells
        geom.set_nsc(nsc)

        # Now we have the following items:
        # 1. sub_sc, the supercell offsets for the connecting orbitals
        # 2. lsc, containing the linear indices of sub_sc that are directly related
        #    to the cut structure
        # 3. geom, which is the cut structure
        def conv(dim):
            nonlocal lsc
            csr = self.tocsr(dim)[orig_orbs, :]
            cols = csr.indices

            # now convert cols
            cols_lsc = lsc[cols // geom_no]

            cols = cols % geom_no + geom.sc_index(cols_lsc) * geom_no

            return csr_matrix(
                (csr.data, cols, csr.indptr),
                shape=(geom_no, geom_no * geom.n_s),
                dtype=self.dtype,
            )

        Ps = [conv(dim) for dim in range(self.dim)]
        S = self.fromsp(geom, Ps, **self._cls_kwargs())

        if sym:
            S *= 0.5
            return S + S.transpose()

        return S

    def unrepeat(
        self, reps: int, axis: int, segment: int = 0, *args, sym: bool = True, **kwargs
    ) -> Self:
        """Unrepeats the sparse model into different parts (retaining couplings)

        Please see `untile` for details, the algorithm and arguments are the same however,
        this is the opposite of `repeat`.
        """
        atoms = np.arange(self.geometry.na).reshape(-1, reps).T.ravel()
        return self.sub(atoms).untile(reps, axis, segment, *args, sym=sym, **kwargs)

    def finalize(self, *args, **kwargs) -> None:
        """Finalizes the model

        Finalizes the model so that all non-used elements are removed. I.e. this simply reduces the memory requirement for the sparse matrix.

        Notes
        -----
        Adding more elements to the sparse matrix is more time-consuming than for a non-finalized sparse matrix due to the
        internal data-representation.
        """
        self._csr.finalize(*args, **kwargs)

    def tocsr(self, dim: int = 0, isc=None, **kwargs):
        """Return a :class:`~scipy.sparse.csr_matrix` for the specified dimension

        Parameters
        ----------
        dim :
           the dimension in the sparse matrix (for non-orthogonal cases the last
           dimension is the overlap matrix)
        isc : int, optional
           the supercell index, or all (if ``isc=None``)
        """
        if isc is not None:
            raise NotImplementedError(
                "Requesting sub-sparse has not been implemented yet"
            )
        return self._csr.tocsr(dim, **kwargs)

    def spsame(self, other):
        """Compare two sparse objects and check whether they have the same entries.

        This does not necessarily mean that the elements are the same
        """
        return self._csr.spsame(other._csr)

    @classmethod
    def fromsp(cls, geometry: Geometry, P: OrSequence[SparseMatrix], **kwargs) -> Self:
        r"""Create a sparse model from a preset `Geometry` and a list of sparse matrices

        The passed sparse matrices are in one of `scipy.sparse` formats.

        Parameters
        ----------
        geometry :
           geometry to describe the new sparse geometry
        P :
           the new sparse matrices that are to be populated in the sparse
           matrix
        **kwargs :
           any arguments that are directly passed to the `__init__` method
           of the class.

        Returns
        -------
        SparseGeometry
             a new sparse matrix that holds the passed geometry and the elements of `P`
        """
        # Ensure list of * format (to get dimensions)
        if issparse(P):
            P = [P]
        if isinstance(P, tuple):
            P = list(P)

        p = cls(geometry, len(P), P[0].dtype, 1, **kwargs)
        p._csr = p._csr.fromsp(P, dtype=kwargs.get("dtype"))

        if p._size != P[0].shape[0]:
            raise ValueError(
                f"{cls.__name__}.fromsp cannot create a new class, the geometry "
                "and sparse matrices does not have coinciding dimensions size != P[0].shape[0]"
            )

        return p

    # numpy dispatch methods (same priority as SparseCSR!)
    __array_priority__ = 14

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # grab the inputs and convert to the respective csr matrices
        # such that we can defer the call to that function
        # while converting, also grab the first _SparseGeometry
        # object such that we may create the output matrix
        sp_inputs = []
        obj = None
        for inp in inputs:
            if isinstance(inp, _SparseGeometry):
                if obj is None:
                    # simply store a reference to the first argument that is a sparsegeometry
                    obj = inp
                sp_inputs.append(inp._csr)
            else:
                sp_inputs.append(inp)

        # determine if the user requested output into
        # a specific container
        out = kwargs.pop("out", None)
        if out is not None:
            (out,) = out
            # ensure the output returns in this field
            kwargs["out"] = (out._csr,)

        result = self._csr.__array_ufunc__(ufunc, method, *sp_inputs, **kwargs)

        if out is not None:
            # check that the resulting variable is indeed a sparsecsr
            assert isinstance(
                result, SparseCSR
            ), f"{self.__class__.__name__} ({ufunc.__name__}) requires out= to match the resulting operator"

        if isinstance(result, SparseCSR):
            # return a copy with the sparse result into the output sparse
            # matrix. If out was not None, the result should already
            # be stored in it.
            if out is None:
                out = obj.copy()
                out._csr = result
        else:
            # likely reductions etc.
            out = result
        return out

    def __getstate__(self):
        """Return dictionary with the current state"""
        return {
            "geometry": self.geometry.__getstate__(),
            "csr": self._csr.__getstate__(),
        }

    def __setstate__(self, state):
        """Return dictionary with the current state"""
        geom = Geometry([0] * 3, Atom(1))
        geom.__setstate__(state["geometry"])
        self._geometry = geom
        csr = SparseCSR((2, 2, 2))
        csr.__setstate__(state["csr"])
        self._csr = csr
        self._def_dim = -1


@set_module("sisl")
class SparseAtom(_SparseGeometry):
    """Sparse object with number of rows equal to the total number of atoms in the `Geometry`"""

    def __getitem__(self, key):
        """Elements for the index(s)"""
        dd = self._def_dim
        if len(key) > 2:
            # This may be a specification of supercell indices
            if isinstance(key[-1], tuple):
                # We guess it is the supercell index
                off = self.geometry.sc_index(key[-1]) * self.na
                key = [el for el in key[:-1]]
                key[1] = self.geometry.asc2uc(key[1]) + off
        if dd >= 0:
            key = tuple(key) + (dd,)
            self._def_dim = -1
        d = self._csr[key]
        return d

    def __setitem__(self, key, val):
        """Set or create elements in the sparse data

        Override set item for slicing operations and enables easy
        setting of parameters in a sparse matrix
        """
        if len(key) > 2:
            # This may be a specification of supercell indices
            if isinstance(key[-1], tuple):
                # We guess it is the supercell index
                off = self.geometry.sc_index(key[-1]) * self.na
                key = [el for el in key[:-1]]
                key[1] = self.geometry.asc2uc(key[1]) + off
        key = tuple(
            self.geometry._sanitize_atoms(k) if i < 2 else k for i, k in enumerate(key)
        )
        dd = self._def_dim
        if dd >= 0:
            key = tuple(key) + (dd,)
            self._def_dim = -1
        self._csr[key] = val

    @property
    def _size(self) -> int:
        return self.geometry.na

    def nonzero(self, atoms: AtomsIndex = None, only_cols: bool = False):
        """Indices row and column indices where non-zero elements exists

        Parameters
        ----------
        atoms :
           only return the tuples for the requested atoms, default is all atoms
        only_cols :
           only return the non-zero columns

        See Also
        --------
        SparseCSR.nonzero : the equivalent function call
        """
        atoms = self.geometry._sanitize_atoms(atoms)
        return self._csr.nonzero(rows=atoms, only_cols=only_cols)

    def iter_nnz(self, atoms: AtomsIndex = None):
        """Iterations of the non-zero elements

        An iterator on the sparse matrix with, row and column

        Examples
        --------
        >>> for i, j in self.iter_nnz():
        ...    self[i, j] # is then the non-zero value

        Parameters
        ----------
        atoms :
            only loop on the non-zero elements coinciding with the atoms
        """
        if atoms is None:
            yield from self._csr
        else:
            atoms = self.geometry._sanitize_atoms(atoms)
            yield from self._csr.iter_nnz(atoms)

    def set_nsc(self, *args, **kwargs):
        """Reset the number of allowed supercells in the sparse atom

        If one reduces the number of supercells *any* sparse element
        that references the supercell will be deleted.

        See `Lattice.set_nsc` for allowed parameters.

        See Also
        --------
        Lattice.set_nsc : the underlying called method
        """
        super().set_nsc(self.na, *args, **kwargs)

    def untile(
        self, reps: int, axis: int, segment: int = 0, *args, sym: bool = True, **kwargs
    ) -> Self:
        """Untiles the sparse model into different parts (retaining couplings)

        Recreates a new sparse object with only the cutted
        atoms in the structure. This will preserve matrix elements in the supercell.

        Parameters
        ----------
        reps :
           number of repetitions the tiling function created (opposite meaning as in `untile`)
        axis :
           which axis to untile along
        segment :
           which segment to retain. Generally each segment should be equivalent, however
           requesting individiual segments can help uncover inconsistencies in the sparse matrix
        *args :
           arguments passed directly to `Geometry.untile`
        sym :
           if True, the algorithm will ensure the returned matrix is symmetrized (i.e.
           return ``(M + M.transpose())/2``, else return data as is.
           False should generally only be used for debugging precision of the matrix elements,
           or if one wishes to check the warnings.
        **kwargs :
           keyword arguments passed directly to `Geometry.untile`

        Notes
        -----
        Untiling structures with ``nsc == 1`` along `axis` are assumed to have periodic boundary
        conditions.

        When untiling structures with ``nsc == 1`` along `axis` it is important to
        untile *as much as possible*. This is because otherwise the algorithm cannot determine
        the correct couplings. Therefore to create a geometry of 3 times a unit-cell, one should
        untile to the unit-cell, and subsequently tile 3 times.

        Consider for example a system of 4 atoms, each atom connects to its 2 neighbors.
        Due to the PBC atom 0 will connect to 1 and 3. Untiling this structure in 2 will
        group couplings of atoms 0 and 1. As it will only see one coupling to the right
        it will halve the coupling and use the same coupling to the left, which is clearly wrong.

        In the following the latter is the correct way to do it.

        >>> SPM.untile(2, 0) != SPM.untile(4, 0).tile(2, 0)

        Raises
        ------
        ValueError :
           in case the matrix elements are not conseuctive when determining the
           new supercell structure. This may often happen if untiling a matrix
           too few times, and then untiling it again.

        See Also
        --------
        tile : opposite of this method
        Geometry.untile : same as this method, see details about parameters here
        """
        return super().untile("a", reps, axis, segment, *args, sym=sym, **kwargs)

    def rij(self, dtype=np.float64):
        r"""Create a sparse matrix with the distance between atoms

        Parameters
        ----------
        dtype : numpy.dtype, optional
            the data-type of the sparse matrix.

        Notes
        -----
        The returned sparse matrix with distances are taken from the current sparse pattern.
        I.e. a subsequent addition of sparse elements will make them inequivalent.
        It is thus important to *only* create the sparse distance when the sparse
        structure is completed.
        """
        R = self.Rij(dtype)
        R._csr = np.sum(R._csr**2, axis=-1) ** 0.5
        return R

    def Rij(self, dtype=np.float64):
        r"""Create a sparse matrix with vectors between atoms

        Parameters
        ----------
        dtype : numpy.dtype, optional
            the data-type of the sparse matrix.

        Notes
        -----
        The returned sparse matrix with vectors are taken from the current sparse pattern.
        I.e. a subsequent addition of sparse elements will make them inequivalent.
        It is thus important to *only* create the sparse vector matrix when the sparse
        structure is completed.
        """
        geom = self.geometry
        Rij = geom.Rij

        # Pointers
        ncol = self._csr.ncol
        ptr = self._csr.ptr
        col = self._csr.col

        # Create the output class
        R = SparseAtom(geom, 3, dtype, nnzpr=1)

        # Re-create the sparse matrix data
        R._csr.ptr = ptr.copy()
        R._csr.ncol = ncol.copy()
        R._csr.col = col.copy()
        R._csr._nnz = self._csr.nnz
        R._csr._D = np.zeros([self._csr._D.shape[0], 3], dtype=dtype)
        R._csr._finalized = self.finalized
        for ia in range(self.shape[0]):
            sl = slice(ptr[ia], ptr[ia] + ncol[ia])
            R._csr._D[sl, :] = Rij(ia, col[sl])

        return R


@set_module("sisl")
class SparseOrbital(_SparseGeometry):
    """Sparse object with number of rows equal to the total number of orbitals in the `Geometry`"""

    def __getitem__(self, key):
        """Elements for the index(s)"""
        dd = self._def_dim
        if len(key) > 2:
            # This may be a specification of supercell indices
            if isinstance(key[-1], tuple):
                # We guess it is the supercell index
                off = self.geometry.sc_index(key[-1]) * self.no
                key = [el for el in key[:-1]]
                key[1] = self.geometry.osc2uc(key[1]) + off
        if dd >= 0:
            key = tuple(key) + (dd,)
            self._def_dim = -1
        d = self._csr[key]
        return d

    def __setitem__(self, key, val):
        """Set or create elements in the sparse data

        Override set item for slicing operations and enables easy
        setting of parameters in a sparse matrix
        """
        if len(key) > 2:
            # This may be a specification of supercell indices
            if isinstance(key[-1], tuple):
                # We guess it is the supercell index
                off = self.geometry.sc_index(key[-1]) * self.no
                key = [el for el in key[:-1]]
                key[1] = self.geometry.osc2uc(key[1]) + off
        key = tuple(
            self.geometry._sanitize_orbs(k) if i < 2 else k for i, k in enumerate(key)
        )
        dd = self._def_dim
        if dd >= 0:
            key = tuple(key) + (dd,)
            self._def_dim = -1
        self._csr[key] = val

    @property
    def _size(self) -> int:
        return self.geometry.no

    def edges(
        self,
        atoms: AtomsIndex = None,
        exclude: AtomsIndex = None,
        orbitals=None,
    ):
        """Retrieve edges (connections) for all `atoms`

        The returned edges are unique and sorted (see `numpy.unique`) and are returned
        in supercell indices (i.e. ``0 <= edge < self.geometry.no_s``).

        Parameters
        ----------
        atoms :
            the edges are returned only for the given atom (but by using  all orbitals of the
            requested atom). The returned edges are also atoms.
        exclude :
           remove edges which are in the `exclude` list, this list refers to orbitals.
        orbitals : int or list of int
            the edges are returned only for the given orbital. The returned edges are orbitals.

        Returns
        -------
        indices :
            If `orbitals` is None, the returned indices are atomic indices.
            Otherwise it will be orbital indices.

        See Also
        --------
        SparseCSR.edges: the underlying routine used for extracting the edges
        """
        if exclude is not None:
            exclude = self.geometry._sanitize_orbs(exclude)
        if atoms is None and orbitals is None:
            raise ValueError(
                f"{self.__class__.__name__}.edges must have either 'atoms' or 'orbitals' keyword defined."
            )
        if orbitals is None:
            orbs = self.geometry.a2o(atoms, all=True)
            return self.geometry.o2a(self._csr.edges(orbs, exclude), unique=True)
        orbitals = np.unique(self.geometry._sanitize_orbs(orbitals))
        return self._csr.edges(orbitals, exclude)

    def nonzero(self, atoms: AtomsIndex = None, only_cols: bool = False):
        """Indices row and column indices where non-zero elements exists

        Parameters
        ----------
        atoms :
           only return the tuples for the requested atoms, default is all atoms
           But for *all* orbitals.
        only_cols :
           only return then non-zero columns

        See Also
        --------
        SparseCSR.nonzero : the equivalent function call
        """
        if atoms is None:
            return self._csr.nonzero(only_cols=only_cols)
        rows = self.geometry.a2o(atoms, all=True)
        return self._csr.nonzero(rows=rows, only_cols=only_cols)

    def iter_nnz(self, atoms: AtomsIndex = None, orbitals=None):
        """Iterations of the non-zero elements

        An iterator on the sparse matrix with, row and column

        Examples
        --------
        >>> for i, j in self.iter_nnz():
        ...    self[i, j] # is then the non-zero value

        Parameters
        ----------
        atoms :
            only loop on the non-zero elements coinciding with the orbitals
            on these atoms (not compatible with the `orbitals` keyword)
        orbitals : int or array_like
            only loop on the non-zero elements coinciding with the orbital
            (not compatible with the `atoms` keyword)
        """
        if atoms is not None:
            orbitals = self.geometry.a2o(atoms, True)
        elif not orbitals is None:
            orbitals = _a.asarrayi(orbitals)
        if orbitals is None:
            yield from self._csr
        else:
            yield from self._csr.iter_nnz(orbitals)

    def set_nsc(self, *args, **kwargs):
        """Reset the number of allowed supercells in the sparse orbital

        If one reduces the number of supercells *any* sparse element
        that references the supercell will be deleted.

        See `Lattice.set_nsc` for allowed parameters.

        See Also
        --------
        Lattice.set_nsc : the underlying called method
        """
        super().set_nsc(self.no, *args, **kwargs)

    def remove_orbital(self, atoms: AtomsIndex, orbitals):
        """Remove a subset of orbitals on `atoms` according to `orbitals`

        For more detailed examples, please see the equivalent (but opposite) method
        `sub_orbital`.

        Parameters
        ----------
        atoms :
            indices of atoms or `Atom` that will be reduced in size according to `orbitals`
        orbitals : array_like of int or Orbital
            indices of the orbitals on `atoms` that are removed from the sparse matrix.

        See Also
        --------
        sub_orbital : retaining a set of orbitals (see here for examples)
        """
        # Get specie index of the atom (convert to list of indices)
        atoms = self.geometry._sanitize_atoms(atoms).ravel()

        # Figure out if all atoms have the same species
        species = self.geometry.atoms.species[atoms]
        uniq_species, indices = unique(species, return_inverse=True)
        if len(uniq_species) > 1:
            # In case there are multiple different species but one wishes to
            # retain the same orbital index, then we loop on the unique species
            new = self
            for i in range(uniq_species.size):
                idx = (indices == i).nonzero()[0]
                # now determine whether it is the whole atom
                # or only part of the geometry
                new = new.remove_orbital(atoms[idx], orbitals)
            return new

        # Get the atom object we wish to reduce
        # We know np.all(geom.atoms[atom] == old_atom)
        old_atom = self.geometry.atoms[atoms[0]]

        if isinstance(orbitals, (Orbital, Integral)):
            orbitals = [orbitals]
        if isinstance(orbitals[0], Orbital):
            orbitals = [old_atom.index(orb) for orb in orbitals]
        orbitals = delete(_a.arangei(len(old_atom)), np.asarray(orbitals).ravel())

        # now call sub_orbital
        return self.sub_orbital(atoms, orbitals)

    def sub_orbital(self, atoms: AtomsIndex, orbitals):
        r"""Retain only a subset of the orbitals on `atoms` according to `orbitals`

        This allows one to retain only a given subset of the sparse matrix elements.

        Parameters
        ----------
        atoms :
            indices of atoms or `Atom` that will be reduced in size according to `orbitals`
        orbitals : array_like of int or Orbital
            indices of the orbitals on `atoms` that are retained in the sparse matrix, the list of
            orbitals will be sorted. One cannot re-arrange matrix elements currently.

        Notes
        -----
        Future implementations may allow one to re-arange orbitals using this method.

        When using this method the internal species list will be populated by another species
        that is named after the orbitals removed. This is to distinguish different atoms.

        Examples
        --------

        >>> # a Carbon atom with 2 orbitals
        >>> C = sisl.Atom('C', [1., 2.])
        >>> # an oxygen atom with 3 orbitals
        >>> O = sisl.Atom('O', [1., 2., 2.4])
        >>> geometry = sisl.Geometry([[0] * 3, [1] * 3]], 2, [C, O])
        >>> obj = SparseOrbital(geometry).tile(3, 0)
        >>> # fill in obj data...

        Now ``obj`` is a sparse geometry with 2 different species and 6 atoms (3 of each).
        They are ordered ``[C, O, C, O, C, O]``. In the following we
        will note species that are different from the original by a ``'`` in the list.

        Retain 2nd orbital on the 2nd atom: ``[C, O', C, O, C, O]``

        >>> new_obj = obj.sub_orbital(1, 1)

        Retain 2nd orbital on 1st and 2nd atom: ``[C', O', C, O, C, O]``

        >>> new_obj = obj.sub_orbital([0, 1], 1)

        Retain 2nd orbital on the 1st atom and 3rd orbital on 4th atom: ``[C', O, C, O', C, O]``

        >>> new_obj = obj.sub_orbital(0, 1).sub_orbital(3, 2)

        Retain 2nd orbital on all atoms equivalent to the first atom: ``[C', O, C', O, C', O]``

        >>> new_obj = obj.sub_orbital(obj.geometry.atoms[0], 1)

        Retain 1st orbital on 1st atom, and 2nd orbital on 3rd and 5th atom: ``[C', O, C'', O, C'', O]``

        >>> new_obj = obj.sub_orbital(0, 0).sub_orbital([2, 4], 1)

        See Also
        --------
        remove_orbital : removing a set of orbitals (opposite of this)
        """
        atoms = self.geometry._sanitize_atoms(atoms).ravel()

        # Figure out if all atoms have the same species
        species = self.geometry.atoms.species[atoms]
        uniq_species, indices = unique(species, return_inverse=True)
        if len(uniq_species) > 1:
            # In case there are multiple different species but one wishes to
            # retain the same orbital index, then we loop on the unique species
            new = self
            for i in range(uniq_species.size):
                idx = (indices == i).nonzero()[0]
                # now determine whether it is the whole atom
                # or only part of the geometry
                new = new.sub_orbital(atoms[idx], orbitals)
            return new

        # Get the atom object we wish to reduce
        old_atom = self.geometry.atoms[atoms[0]]

        if isinstance(orbitals, (Orbital, Integral)):
            orbitals = [orbitals]
        if isinstance(orbitals[0], Orbital):
            orbitals = [old_atom.index(orb) for orb in orbitals]
        orbitals = np.sort(orbitals)

        # At this point we are sure that uniq_species is *only* one species!
        geom = self.geometry.sub_orbital(atoms, orbitals)

        # Now create the new sparse orbital class
        SG = self.__class__(geom, self.dim, self.dtype, 1, **self._cls_kwargs())

        rem_orbs = delete(_a.arangei(old_atom.no), orbitals)
        # Find orbitals to remove (note this HAS to be from the original array)
        rem_orbs = np.add.outer(self.geometry.a2o(atoms), rem_orbs).ravel()

        # Generate a list of orbitals to retain
        sub_idx = delete(_a.arangei(self.no), rem_orbs)

        # Generate full supercell indices
        n_s = self.geometry.n_s
        sc_off = _a.arangei(n_s) * self.no
        sub_idx = tile(sub_idx, n_s).reshape(n_s, -1) + sc_off.reshape(-1, 1)
        SG._csr = self._csr.sub(sub_idx)

        # just ensure we are doing the correct thing
        assert SG._csr.shape[0] == SG.geometry.no

        return SG

    def untile(
        self, reps: int, axis: int, segment: int = 0, *args, sym: bool = True, **kwargs
    ) -> Self:
        """Untiles the sparse model into different parts (retaining couplings)

        Recreates a new sparse object with only the cutted
        atoms in the structure. This will preserve matrix elements in the supercell.

        Parameters
        ----------
        reps :
           number of repetitions the tiling function created (opposite meaning as in `untile`)
        axis :
           which axis to untile along
        segment :
           which segment to retain. Generally each segment should be equivalent, however
           requesting individiual segments can help uncover inconsistencies in the sparse matrix
        *args :
           arguments passed directly to `Geometry.untile`
        sym :
           if True, the algorithm will ensure the returned matrix is symmetrized (i.e.
           return ``(M + M.transpose())/2``, else return data as is.
           False should generally only be used for debugging precision of the matrix elements,
           or if one wishes to check the warnings.
        **kwargs :
           keyword arguments passed directly to `Geometry.untile`

        Notes
        -----
        Untiling structures with ``nsc == 1`` along `axis` are assumed to have periodic boundary
        conditions.

        When untiling structures with ``nsc == 1`` along `axis` it is important to
        untile *as much as possible*. This is because otherwise the algorithm cannot determine
        the correct couplings. Therefore to create a geometry of 3 times a unit-cell, one should
        untile to the unit-cell, and subsequently tile 3 times.

        Consider for example a system of 4 atoms, each atom connects to its 2 neighbors.
        Due to the PBC atom 0 will connect to 1 and 3. Untiling this structure in 2 will
        group couplings of atoms 0 and 1. As it will only see one coupling to the right
        it will halve the coupling and use the same coupling to the left, which is clearly wrong.

        In the following the latter is the correct way to do it.

        >>> SPM.untile(2, 0) != SPM.untile(4, 0).tile(2, 0)

        Raises
        ------
        ValueError :
           in case the matrix elements are not conseuctive when determining the
           new supercell structure. This may often happen if untiling a matrix
           too few times, and then untiling it again.

        See Also
        --------
        tile : opposite of this method
        Geometry.untile : same as this method, see details about parameters here
        """
        return super().untile("o", reps, axis, segment, *args, sym=sym, **kwargs)

    def rij(self, what: str = "orbital", dtype=np.float64):
        r"""Create a sparse matrix with the distance between atoms/orbitals

        Parameters
        ----------
        what : {'orbital', 'atom'}
            which kind of sparse distance matrix to return, either an atomic distance matrix
            or an orbital distance matrix. The orbital matrix is equivalent to the atomic
            one with the same distance repeated for the same atomic orbitals.
            The default is the same type as the parent class.
        dtype : numpy.dtype, optional
            the data-type of the sparse matrix.

        Notes
        -----
        The returned sparse matrix with distances are taken from the current sparse pattern.
        I.e. a subsequent addition of sparse elements will make them inequivalent.
        It is thus important to *only* create the sparse distance when the sparse
        structure is completed.
        """
        R = self.Rij(what, dtype)
        R._csr = np.sum(R._csr**2, axis=-1) ** 0.5
        return R

    def Rij(self, what: str = "orbital", dtype=np.float64):
        r"""Create a sparse matrix with the vectors between atoms/orbitals

        Parameters
        ----------
        what : {'orbital', 'atom'}
            which kind of sparse vector matrix to return, either an atomic vector matrix
            or an orbital vector matrix. The orbital matrix is equivalent to the atomic
            one with the same vectors repeated for the same atomic orbitals.
            The default is the same type as the parent class.
        dtype : numpy.dtype, optional
            the data-type of the sparse matrix.

        Notes
        -----
        The returned sparse matrix with vectors are taken from the current sparse pattern.
        I.e. a subsequent addition of sparse elements will make them inequivalent.
        It is thus important to *only* create the sparse vector matrix when the sparse
        structure is completed.
        """
        geom = self.geometry

        # Pointers
        ncol = self._csr.ncol
        ptr = self._csr.ptr
        col = self._csr.col

        if what == "atom":
            R = SparseAtom(geom, 3, dtype, nnzpr=np.amax(ncol))
            Rij = geom.Rij
            o2a = geom.o2a

            # Orbitals
            orow = _a.arangei(self.shape[0])
            # Loop on orbitals and atoms
            for io, ia in zip(orow, o2a(orow)):
                coln = unique(o2a(col[ptr[io] : ptr[io] + ncol[io]]))
                R[ia, coln] = Rij(ia, coln)

        elif what in ("orbital", "orb"):
            # We create an *exact* copy of the Rij
            R = SparseOrbital(geom, 3, dtype, nnzpr=1)
            Rij = geom.oRij

            # Re-create the sparse matrix data
            R._csr.ptr = ptr.copy()
            R._csr.ncol = ncol.copy()
            R._csr.col = col.copy()
            R._csr._nnz = self._csr.nnz
            R._csr._D = np.zeros([self._csr._D.shape[0], 3], dtype=dtype)
            R._csr._finalized = self.finalized

            for io in range(self.shape[0]):
                sl = slice(ptr[io], ptr[io] + ncol[io])
                R._csr._D[sl, :] = Rij(io, col[sl])

        else:
            raise ValueError(
                self.__class__.__name__ + '.Rij "what" is not one of [atom, orbital].'
            )

        return R

    def add(self, other, axis: Optional[int] = None, offset: Coord = (0, 0, 0)):
        r"""Add two sparse matrices by adding the parameters to one set. The final matrix will have no couplings between `self` and `other`

        The final sparse matrix will not have any couplings between `self` and `other`. Not even if they
        have commensurate overlapping regions. If you want to create couplings you have to use `append` but that
        requires the structures are commensurate in the coupling region.

        Parameters
        ----------
        other : SparseGeometry
            the other sparse matrix to be added, all atoms will be appended
        axis :
            whether a specific axis of the cell will be added to the final geometry.
            For ``None`` the final cell will be that of `self`, otherwise the lattice
            vector corresponding to `axis` will be appended.
        offset :
            offset in geometry of `other` when adding the atoms.

        See Also
        --------
        append : append two matrices by also adding overlap couplings
        prepend : see `append`
        """
        # Check that the sparse matrices are compatible
        if not (type(self) is type(other)):
            raise ValueError(
                self.__class__.__name__
                + f".add requires other to be of same type: {other.__class__.__name__}"
            )

        if self.dtype != other.dtype:
            raise ValueError(
                self.__class__.__name__
                + ".add requires the same datatypes in the two matrices."
            )

        if self.dim != other.dim:
            raise ValueError(
                self.__class__.__name__
                + ".add requires the same number of dimensions in the matrix."
            )

        if axis is None:
            geom = self.geometry.add(other.geometry, offset=offset)
        else:
            # Same effect but also adds the lattice vectors
            geom = self.geometry.append(other.geometry, axis, offset=offset)

        # Now we have the correct geometry, then create the correct
        # class
        # New indices and data (the constructor for SparseCSR copies)
        full = self.__class__(geom, self.dim, self.dtype, 1, **self._cls_kwargs())
        full._csr.ptr = concatenate((self._csr.ptr[:-1], other._csr.ptr))
        full._csr.ptr[self.no :] += self._csr.ptr[-1]
        full._csr.ncol = concatenate((self._csr.ncol, other._csr.ncol))
        full._csr._D = concatenate((self._csr._D, other._csr._D))
        full._csr._nnz = full._csr.ncol.sum()
        full._csr._finalized = False

        # Retrieve the maximum number of orbitals (in the supercell)
        # This may be used to remove couplings
        full_no_s = geom.no_s

        # Now we have to transfer the indices to the new sparse pattern

        # First create a local copy of the columns, then we transfer, and then we collect.
        s_col = self._csr.col.copy()
        transfer_idx = _a.arangei(self.geometry.no_s).reshape(-1, self.geometry.no)
        transfer_idx += _a.arangei(self.geometry.n_s).reshape(-1, 1) * other.geometry.no
        # Remove couplings along axis
        if not axis is None:
            idx = (self.geometry.lattice.sc_off[:, axis] != 0).nonzero()[0]
            # Tell the routine to delete these indices
            transfer_idx[idx, :] = full_no_s + 1
        idx = array_arange(self._csr.ptr[:-1], n=self._csr.ncol)
        s_col[idx] = transfer_idx.ravel()[s_col[idx]]

        # Same for the other, but correct for deleted supercells and supercells along
        # disconnected auxiliary cells.
        o_col = other._csr.col.copy()
        transfer_idx = _a.arangei(other.geometry.no_s).reshape(-1, other.geometry.no)

        # Transfer the correct supercells
        o_idx = []
        s_idx = []
        idx_delete = []
        for isc, sc in enumerate(other.geometry.lattice.sc_off):
            try:
                s_idx.append(self.geometry.lattice.sc_index(sc))
                o_idx.append(isc)
            except ValueError:
                idx_delete.append(isc)
        # o_idx are transferred to s_idx
        transfer_idx[o_idx, :] += (
            _a.arangei(1, other.geometry.n_s + 1)[s_idx].reshape(-1, 1)
            * self.geometry.no
        )
        # Remove some columns
        transfer_idx[idx_delete, :] = full_no_s + 1
        # Clean-up to not confuse the rest of the algorithm
        del o_idx, s_idx, idx_delete

        # Now figure out if the supercells can be kept, at all...
        # find SC indices in other corresponding to self
        # o_idx_uc = other.geometry.lattice.sc_index([0] * 3)
        # o_idx_sc = _a.arangei(other.geometry.lattice.n_s)

        # Remove couplings along axis
        for i in range(3):
            if i == axis:
                idx = (other.geometry.lattice.sc_off[:, axis] != 0).nonzero()[0]
            elif not allclose(geom.cell[i, :], other.cell[i, :]):
                # This will happen in case `axis` is None
                idx = (other.geometry.lattice.sc_off[:, i] != 0).nonzero()[0]
            else:
                # When axis is not specified and cell parameters
                # are commensurate, then we will not change couplings
                continue
            # Tell the routine to delete these indices
            transfer_idx[idx, :] = full_no_s + 1

        idx = array_arange(other._csr.ptr[:-1], n=other._csr.ncol)
        o_col[idx] = transfer_idx.ravel()[o_col[idx]]

        # Now we need to decide whether the
        del transfer_idx, idx
        full._csr.col = concatenate([s_col, o_col])

        # Clean up (they could potentially be very large arrays)
        del s_col, o_col

        # Ensure we remove the elements
        full._csr._clean_columns()

        return full

    @deprecate_argument(
        "eps",
        "atol",
        "argument eps has been deprecated in favor of atol.",
        "0.15",
        "0.17",
    )
    def prepend(
        self, other, axis: int, atol: float = 0.005, scale: SeqOrScalarFloat = 1
    ) -> Self:
        r"""See `append` for details

        This is currently equivalent to:

        >>> other.append(self, axis, atol, scale)
        """
        return other.append(self, axis, atol, scale)

    @deprecate_argument(
        "eps",
        "atol",
        "argument eps has been deprecated in favor of atol.",
        "0.15",
        "0.17",
    )
    def append(
        self, other, axis: int, atol: float = 0.005, scale: SeqOrScalarFloat = 1
    ) -> Self:
        r"""Append `other` along `axis` to construct a new connected sparse matrix

        This method tries to append two sparse geometry objects together by
        the following these steps:

        1. Create the new extended geometry
        2. Use neighbor cell couplings from `self` as the couplings to `other`
           This *may* cause problems if the coupling atoms are not exactly equi-positioned.
           If the coupling coordinates and the coordinates in `other` differ by more than
           0.01 Ang, a warning will be issued.
           If this difference is above `atol` the couplings will be removed.

        When appending sparse matrices made up of atoms, this method assumes that
        the orbitals on the overlapping atoms have the same orbitals, as well as the
        same orbital ordering.

        Examples
        --------
        >>> sporb = SparseOrbital(....)
        >>> sporb2 = sporb.append(sporb, 0)
        >>> sporbt = sporb.tile(2, 0)
        >>> sporb2.spsame(sporbt)
        True

        To retain couplings only from the *left* sparse matrix, do:

        >>> sporb = left.append(right, 0, scale=(2, 0))
        >>> sporb = (sporb + sporb.transpose()) / 2

        To retain couplings only from the *right* sparse matrix, do:

        >>> sporb = left.append(right, 0, scale=(0, 2.))
        >>> sporb = (sporb + sporb.transpose()) / 2

        Notes
        -----
        The current implementation does not preserve the hermiticity of the matrix.
        If you want to preserve hermiticity of the matrix you have to do the
        following:

        >>> sm = (sm + sm.transpose()) / 2

        Parameters
        ----------
        other : object
            must be an object of the same type as `self`
        axis :
            axis to append the two sparse geometries along
        atol :
            tolerance that all coordinates *must* be within to allow an append.
            It is important that this value is smaller than half the distance between
            the two closests atoms such that there is no ambiguity in selecting
            equivalent atoms. An internal stricter tolerance is used as a baseline, see above.
        scale : float or array_like, optional
            the scale used for the overlapping region. For scalar values it corresponds
            to passing: ``(scale, scale)``.
            For array-like input ``scale[0]`` refers to the scale of the matrix elements
            coupling from `self`, while ``scale[1]`` is the scale of the matrix elements
            in `other`.

        See Also
        --------
        prepend : equivalent scheme as this method
        add : merge two matrices without considering overlap or commensurability
        transpose : ensure hermiticity by using this routine
        replace : replace a sub-set of atoms with another sparse matrix
        Geometry.append
        Geometry.prepend
        SparseCSR.scale_columns : method used to scale the two matrix elements values

        Raises
        ------
        ValueError
            if the two geometries are not compatible for either coordinate, orbital or supercell errors

        Returns
        -------
        object
            a new instance with two sparse matrices joined and appended together
        """
        if not (type(self) is type(other)):
            raise ValueError(
                f"{self.__class__.__name__}.append requires other to be of same type: {other.__class__.__name__}"
            )

        if self.geometry.nsc[axis] > 3 or other.geometry.nsc[axis] > 3:
            raise ValueError(
                f"{self.__class__.__name__}.append requires sparse-geometries to maximally "
                "have 3 supercells along appending axis."
            )

        if not allclose(self.geometry.nsc, other.geometry.nsc):
            raise ValueError(
                f"{self.__class__.__name__}.append requires sparse-geometries to have the same "
                "number of supercells along all directions."
            )

        if not allclose(
            self.geometry.lattice._isc_off, other.geometry.lattice._isc_off
        ):
            raise ValueError(
                f"{self.__class__.__name__}.append requires supercell offsets to be the same."
            )

        if self.dtype != other.dtype:
            raise ValueError(
                f"{self.__class__.__name__}.append requires the same datatypes in the two matrices."
            )

        if self.dim != other.dim:
            raise ValueError(
                f"{self.__class__.__name__}.append requires the same number of dimensions in the matrix."
            )

        if np.asarray(scale).size == 1:
            scale = np.array([scale, scale])
        scale = np.asarray(scale)

        # Our procedure will be to separate the sparsity patterns into separate chunks
        # First generate the full geometry
        geom = self.geometry.append(other.geometry, axis)

        # create the new sparsity patterns with offset

        # New indices and data (the constructor for SparseCSR copies)
        full = self.__class__(geom, self.dim, self.dtype, 1, **self._cls_kwargs())
        full._csr.ptr = concatenate((self._csr.ptr[:-1], other._csr.ptr))
        full._csr.ptr[self.no :] += self._csr.ptr[-1]
        full._csr.ncol = concatenate((self._csr.ncol, other._csr.ncol))
        full._csr._D = concatenate((self._csr._D, other._csr._D))
        full._csr._nnz = full._csr.ncol.sum()
        full._csr._finalized = False

        # First create a local copy of the columns, then we transfer, and then we collect.
        s_col = self._csr.col.copy()
        # transfer
        transfer_idx = _a.arangei(self.geometry.no_s).reshape(-1, self.geometry.no)
        transfer_idx += _a.arangei(self.geometry.n_s).reshape(-1, 1) * other.geometry.no
        idx = array_arange(self._csr.ptr[:-1], n=self._csr.ncol)
        s_col[idx] = transfer_idx.ravel()[s_col[idx]]

        o_col = other._csr.col.copy()
        # transfer
        transfer_idx = _a.arangei(other.geometry.no_s).reshape(-1, other.geometry.no)
        transfer_idx += (
            _a.arangei(1, other.geometry.n_s + 1).reshape(-1, 1) * self.geometry.no
        )
        idx = array_arange(other._csr.ptr[:-1], n=other._csr.ncol)
        o_col[idx] = transfer_idx.ravel()[o_col[idx]]

        # Store all column indices
        del transfer_idx, idx
        full._csr.col = concatenate((s_col, o_col))

        # Clean up (they could potentially be very large arrays)
        del s_col, o_col

        # Now everything is contained in 1 sparse matrix.
        # All matrix elements are as though they are in their own

        # What needs to be done is to find the overlapping atoms and transfer indices in
        # both these sparsity patterns to the correct elements.

        # 1. find overlapping atoms along axis
        idx_s_first, idx_o_first = self.geometry.overlap(other.geometry, atol=atol)
        idx_s_last, idx_o_last = self.geometry.overlap(
            other.geometry,
            atol=atol,
            offset=-self.geometry.lattice.cell[axis, :],
            offset_other=-other.geometry.lattice.cell[axis, :],
        )

        # IFF idx_s_* contains duplicates, then we have multiple overlapping atoms which is not
        # allowed
        def _test(diff):
            if diff.size != diff.nonzero()[0].size:
                raise ValueError(
                    f"{self.__class__.__name__}.append requires that there is maximally one "
                    "atom overlapping one other atom in the other structure."
                )

        _test(diff(idx_s_first))
        _test(diff(idx_s_last))
        # Also ensure that atoms have the same number of orbitals in the two cases
        if (
            not allclose(
                self.geometry.orbitals[idx_s_first],
                other.geometry.orbitals[idx_o_first],
            )
        ) or (
            not allclose(
                self.geometry.orbitals[idx_s_last], other.geometry.orbitals[idx_o_last]
            )
        ):
            raise ValueError(
                f"{self.__class__.__name__}.append requires the overlapping geometries "
                "to have the same number of orbitals per atom that is to be replaced."
            )

        def _check_edges_and_coordinates(spgeom, atoms, isc, err_help):
            # Figure out if we have found all couplings
            geom = spgeom.geometry
            # Find orbitals that we wish to exclude from the orbital connections
            # This ensures that we only find couplings crossing the supercell boundaries
            irrelevant_sc = delete(
                _a.arangei(geom.lattice.n_s), geom.lattice.sc_index(isc)
            )
            sc_orbitals = _a.arangei(geom.no_s).reshape(geom.lattice.n_s, -1)
            exclude = sc_orbitals[irrelevant_sc, :].ravel()
            # get connections and transfer them to the unit-cell
            edges_sc = geom.o2a(
                spgeom.edges(orbitals=_a.arangei(geom.no), exclude=exclude), True
            )
            edges_uc = geom.asc2uc(edges_sc, True)
            edges_valid = np.isin(edges_uc, atoms, assume_unique=True)
            if not np.all(edges_valid):
                edges_uc = edges_sc % geom.na
                # Reduce edges to those that are faulty
                edges_valid = np.isin(edges_uc, atoms, assume_unique=False)
                errors = edges_sc[~edges_valid]
                # Get supercell offset and unit-cell atom
                isc_off, uca = np.divmod(errors, geom.na)
                # group atoms for each supercell index
                # find unique supercell offsets
                sc_off_atoms = []
                # This will be much faster
                for isc in unique(isc_off):
                    idx = (isc_off == isc).nonzero()[0]
                    sc_off_atoms.append(
                        "{k}: {v}".format(
                            k=str(geom.lattice.sc_off[isc]),
                            v=list2str(np.sort(uca[idx])),
                        )
                    )
                sc_off_atoms = "\n   ".join(sc_off_atoms)
                raise ValueError(
                    f"{self.__class__.__name__}.append requires matching coupling elements.\n\n"
                    f"The following atoms in a {err_help[1]} connection of `{err_help[0]}` super-cell "
                    "are connected from the unit cell, but are not found in matches:\n\n"
                    f"[sc-offset]: atoms\n   {sc_off_atoms}"
                )

        # setup supercells to look up
        isc_inplace = [None] * 3
        isc_inplace[axis] = 0
        isc_forward = isc_inplace.copy()
        isc_forward[axis] = 1
        isc_back = isc_inplace.copy()
        isc_back[axis] = -1

        # Check that edges and overlapping atoms are the same (or at least that the
        # edges are all in the overlapping region)
        # [self|other]: self sc-connections forward must be on left-aligned matching atoms
        _check_edges_and_coordinates(
            self, idx_s_first, isc_forward, err_help=("self", "forward")
        )
        # [other|self]: other sc-connections forward must be on left-aligned matching atoms
        _check_edges_and_coordinates(
            other, idx_o_first, isc_forward, err_help=("other", "forward")
        )
        # [other|self]: self sc-connections backward must be on right-aligned matching atoms
        _check_edges_and_coordinates(
            self, idx_s_last, isc_back, err_help=("self", "backward")
        )
        # [self|other]: other sc-connections backward must be on right-aligned matching atoms
        _check_edges_and_coordinates(
            other, idx_o_last, isc_back, err_help=("other", "backward")
        )

        # Now we have ensured that the overlapping coordinates and the connectivity graph
        # co-incide and that we can actually perform the merge.
        idx = _a.arangei(geom.n_s).reshape(-1, 1) * geom.no

        def _sc_index_sort(isc):
            idx = geom.lattice.sc_index(isc)
            # Now sort so that all indices are corresponding one2one
            # This is important since two different supercell indices
            # need not be sorted in the same manner.
            # This ensures that there is a correspondance between
            # two different sparse elements
            off = delete(geom.lattice.sc_off[idx].T, axis, axis=0)
            return idx[np.lexsort(off)]

        idx_iscP = idx[_sc_index_sort(isc_forward)]
        idx_isc0 = idx[_sc_index_sort(isc_inplace)]
        idx_iscM = idx[_sc_index_sort(isc_back)]
        # Clean (for me to know what to do in this code)
        del idx, _sc_index_sort

        # First scale all values
        idx_s_first = self.geometry.a2o(idx_s_first, all=True).reshape(1, -1)
        idx_s_last = self.geometry.a2o(idx_s_last, all=True).reshape(1, -1)
        col = concatenate(
            ((idx_s_first + idx_iscP).ravel(), (idx_s_last + idx_iscM).ravel())
        )
        full._csr.scale_columns(col, scale[0])

        idx_o_first = (
            other.geometry.a2o(idx_o_first, all=True).reshape(1, -1) + self.geometry.no
        )
        idx_o_last = (
            other.geometry.a2o(idx_o_last, all=True).reshape(1, -1) + self.geometry.no
        )
        col = concatenate(
            ((idx_o_first + idx_iscP).ravel(), (idx_o_last + idx_iscM).ravel())
        )
        full._csr.scale_columns(col, scale[1])

        # Clean up (they may be very large)
        del col

        # Now we can easily build from->to arrays

        # other[0] -> other[1] changes to other[0] -> full_G[1] | self[1]
        # self[0] -> self[1] changes to self[0] -> full_G[0] | other[0]
        # self[0] -> self[-1] changes to self[0] -> full_G[-1] | other[-1]
        # other[0] -> other[-1] changes to other[0] -> full_G[0] | self[0]
        col_from = concatenate(
            (
                (idx_o_first + idx_iscP).ravel(),
                (idx_s_first + idx_iscP).ravel(),
                (idx_s_last + idx_iscM).ravel(),
                (idx_o_last + idx_iscM).ravel(),
            )
        )
        col_to = concatenate(
            (
                (idx_s_first + idx_iscP).ravel(),
                (idx_o_first + idx_isc0).ravel(),
                (idx_o_last + idx_iscM).ravel(),
                (idx_s_last + idx_isc0).ravel(),
            )
        )

        full._csr.translate_columns(col_from, col_to)
        return full

    @deprecate_argument(
        "eps",
        "atol",
        "argument eps has been deprecated in favor of atol",
        "0.15",
        "0.17",
    )
    def replace(
        self,
        atoms: AtomsIndex,
        other,
        other_atoms: AtomsIndex = None,
        atol: float = 0.005,
        scale: SeqOrScalarFloat = 1.0,
    ) -> Self:
        r"""Replace `atoms` in `self` with `other_atoms` in `other` and retain couplings between them

        This method replaces a subset of atoms in `self` with
        another sparse geometry retaining any couplings between them.
        The algorithm checks whether the coupling atoms have the same number of
        orbitals. Meaning that atoms in the overlapping region should have the same
        connections and number of orbitals per atom.
        It will _not_ check whether the orbitals or atoms _are_ the same, nor the order
        of the orbitals.

        The replacement algorithm takes the couplings from ``self -> other`` on atoms
        belonging to ``self`` and ``other -> self`` from ``other``. This will in some
        cases mean that the matrix becomes non-symmetric. See in Notes for details on
        symmetrizing the matrices.

        Examples
        --------
        >>> minimal = SparseOrbital(....)
        >>> big = minimal.tile(2, 0)
        >>> big2 = big.replace(np.arange(big.na), minimal)
        >>> big.spsame(big2)
        True

        To ensure hermiticity and using the average of the couplings from ``big`` and
        ``minimal`` one can do:

        >>> big2 = big.replace(np.arange(big.na), minimal)
        >>> big2 = (big2 + big2.transpose()) / 2

        To retain couplings only from the ``big`` sparse matrix, one should
        do the following (note the subsequent transposing which ensures hermiticy
        and is effectively copying couplings from ``big`` to the replaced region.

        >>> big2 = big.replace(np.arange(big.na), minimal, scale=(2, 0))
        >>> big2 = (big2 + big2.transpose()) / 2

        To only retain couplings from the ``minimal`` sparse matrix:

        >>> big2 = big.replace(np.arange(big.na), minimal, scale=(0, 2))
        >>> big2 = (big2 + big2.transpose()) / 2

        Notes
        -----
        The current implementation does not preserve the hermiticity of the matrix.
        If you want to preserve hermiticity of the matrix you have to do the
        following:

        >>> sm = (sm + sm.transpose()) / 2

        Also note that the ordering of the atoms will be ``range(atoms.min()), range(len(other_atoms)), <rest>``.

        Algorithms that utilizes atomic indices should be careful.

        When the tolerance `atol` is high, the elements may be more prone to differences in the
        symmetry elements. A good idea would be to check the difference between the couplings.
        The below variable ``diff`` will contain the difference ``(self -> other) - (other -> self)``

        >>> diff = sm - sm.transpose()

        Parameters
        ----------
        atoms :
            which atoms in `self` that are removed and replaced with ``other.sub(other_atoms)``
        other : object
            must be an object of the same type as `self`, a subset is taken from this
            sparse matrix and combined with `self` to create a new sparse matrix
        other_atoms :
            to select a subset of atoms in `other` that are taken out.
            Defaults to all atoms in `other`.
        atol :
            coordinate tolerance for allowing replacement.
            It is important that this value is at least smaller than half the distance between
            the two closests atoms such that there is no ambiguity in selecting
            equivalent atoms.
        scale :
            the scale used for the overlapping region. For scalar values it corresponds
            to passing: ``(scale, scale)``.
            For array-like input ``scale[0]`` refers to the scale of the matrix elements
            coupling from `self`, while ``scale[1]`` is the scale of the matrix elements
            in `other`.

        See Also
        --------
        prepend : prepend two sparse matrices, see `append` for details
        add : merge two matrices without considering overlap or commensurability
        transpose : may be used to ensure hermiticity (symmetrization of the matrix elements)
        append : append two sparse matrices
        Geometry.append
        Geometry.prepend
        SparseCSR.scale_columns : method used to scale the two matrix elements values

        Raises
        ------
        ValueError
           if the two geometries are not compatible for either coordinate, orbital or supercell errors
        AssertionError
           if the two geometries are not compatible for either coordinate, orbital or supercell errors

        Warns
        -----
        SislWarning
           in case the overlapping atoms are not comprising the same atomic specie.
           In some cases this may not be a problem.
           However, care must be taken by the user if this warning is issued.

        Returns
        -------
        object
            a new instance with two sparse matrices merged together by replacing some atoms
        """
        scale = np.asarray(scale)
        if scale.size == 1:
            scale = np.repeat(scale, 2)

        # here our connection is defined as what is connected to "in"
        # and what is connected to "out"
        # Say 0 -> 1
        # And `atoms` is [0].
        # Then in = [0], out = [1]
        # since atoms connect out to [1]
        # In certain cases, when `atoms` does not connect *into* itself,
        # it may be smaller than `atoms`.

        # figure out the atoms that needs replacement
        def get_reduced_system(sp, atoms):
            """convert the geometry in `sp` to only atoms `atoms` and return the following:

            1. atoms (sanitized and no order change)
            2. orbitals (ordered as `atoms`)
            3. the atoms that are connected to OUT and IN
            4. the orbitals that are connected to OUT and IN
            """
            geom = sp.geometry
            atoms = _a.asarrayi(geom._sanitize_atoms(atoms)).ravel()
            if unique(atoms).size != atoms.size:
                raise ValueError(
                    f"{self.__class__.__name__}.replace requires a unique set of atoms"
                )
            orbs = geom.a2o(atoms, all=True)
            # other_orbs = geom.ouc2sc(np.delete(_a.arangei(geom.no), orbs))

            # Find the orbitals that these atoms connect to such that we can compare
            # atomic coordinates
            out_connect_orb_sc = sp.edges(orbitals=orbs, exclude=orbs)
            out_connect_orb = geom.osc2uc(out_connect_orb_sc, unique=True)
            out_connect_atom_sc = geom.o2a(out_connect_orb_sc, unique=True)
            out_connect_atom = geom.asc2uc(out_connect_atom_sc, unique=True)

            # figure out connecting back
            atoms_orbs = list(
                map(_a.arangei, geom.firsto[atoms], geom.firsto[atoms + 1])
            )
            in_connect_atom = []
            in_connect_orb = []

            for atom, atom_orbs in zip(atoms, atoms_orbs):
                edges = sp.edges(orbitals=atom_orbs, exclude=orbs)
                if len(intersect1d(edges, out_connect_orb_sc)) > 0:
                    in_connect_atom.append(atom)
                    in_connect_orb.append(atom_orbs)

            in_connect_atom = _a.arrayi(in_connect_atom)
            in_connect_orb = concatenate(in_connect_orb)

            # create the connection tables
            atom_uc = Connect(in_connect_atom, out_connect_atom)
            atom_sc = Connect(in_connect_atom, out_connect_atom_sc)
            orb_uc = Connect(in_connect_orb, out_connect_orb)
            orb_sc = Connect(in_connect_orb, out_connect_orb_sc)
            atom_connect = UCSC(atom_uc, atom_sc)
            orb_connect = UCSC(orb_uc, orb_sc)

            return Info(atoms, orbs, atom_connect, orb_connect)

        UCSC = namedtuple("UCSC", ["uc", "sc"])
        Connect = namedtuple("Connect", ["IN", "OUT"])
        Info = namedtuple("Info", ["atoms", "orbitals", "atom_connect", "orb_connect"])

        sgeom = self.geometry
        s_info = get_reduced_system(self, atoms)
        atoms = s_info.atoms  # sanitized (no order change)

        ogeom = other.geometry
        o_info = get_reduced_system(other, other_atoms)
        other_atoms = o_info.atoms  # sanitized (no order change)

        # Get overlapping atoms by their offset
        # We need to get a 1-1 correspondence between the two connecting geometries
        # For instance `self` may be ordered differently than `other`.
        # So we need to figure out how the atoms are arranged in *both* regions.
        # This is where `atol` comes into play since we have to ensure that the
        # connecting regions are within some given tolerance.

        def create_geometry(geom, atoms):
            """Create the supercell geometry with coordinates as given"""
            xyz = geom.axyz(atoms)
            uc_atoms = geom.asc2uc(atoms)
            return Geometry(xyz, atoms=geom.atoms[uc_atoms])

        # We know that the *IN* connections are in the primary unit-cell
        # so we don't need to handle supercell information
        # Atoms *inside* the replacement region that couples out
        sgeom_in = sgeom.sub(s_info.atom_connect.uc.IN)
        ogeom_in = ogeom.sub(o_info.atom_connect.uc.IN)
        soverlap_in, ooverlap_in = sgeom_in.overlap(
            ogeom_in,
            atol=atol,
            offset=-sgeom_in.xyz.min(0),
            offset_other=-ogeom_in.xyz.min(0),
        )

        # Not replacement region, i.e. the IN (above) atoms are connecting to
        # these atoms:
        sgeom_out = create_geometry(sgeom, s_info.atom_connect.sc.OUT)
        ogeom_out = create_geometry(ogeom, o_info.atom_connect.sc.OUT)
        soverlap_out, ooverlap_out = sgeom_out.overlap(
            ogeom_out,
            atol=atol,
            offset=-sgeom_out.xyz.min(0),
            offset_other=-ogeom_out.xyz.min(0),
        )

        # trigger for errors
        msg = ""

        # Now we have the different geometries around to handle how the merging
        # process.
        # Before proceeding we will check whether the dimensions match.
        if len(sgeom_in) != len(soverlap_in) or len(ogeom_in) != len(ooverlap_in):
            # We check that the couplings INTO the replaced atoms
            # are equivalent.
            # I.e. # of atoms

            # figure out which atoms are not connecting
            s_diff = np.setdiff1d(
                np.arange(s_info.atom_connect.uc.IN.size), soverlap_in
            )
            o_diff = np.setdiff1d(
                np.arange(o_info.atom_connect.uc.IN.size), ooverlap_in
            )
            if len(s_diff) > 0 or len(o_diff) > 0:
                msg = f"""{msg}

The number of atoms in the replacement region that connects to the surrounding
atoms are not the same in 'self' and 'other'.
This means that the number of connections is not the same."""

            if len(s_diff) > 0:
                tmp = list2str(np.sort(s_info.atom_connect.uc.IN[s_diff]))
                msg = f"""{msg}

self: atoms not matched in 'other': {tmp}."""
            if len(o_diff) > 0:
                tmp = list2str(np.sort(o_info.atom_connect.uc.IN[o_diff]))
                msg = f"""{msg}

other: atoms not matched in 'self': {tmp}."""

        elif not np.allclose(
            sgeom_in.orbitals[soverlap_in], ogeom_in.orbitals[ooverlap_in]
        ):
            tmp_self = list2str(np.sort(sgeom_in.orbitals[soverlap_in]))
            tmp_other = list2str(np.sort(ogeom_in.orbitals[ooverlap_in]))
            msg = f"""{msg}

Atoms in the replacement region have different number of orbitals on the atoms
that lie at the border.

self orbitals:
   {tmp_self}
other orbitals:
   {tmp_other}"""

        # print("out:")
        # print(s_info.atom_connect.uc.OUT)
        # print(soverlap_out)
        # print(o_info.atom_connect.uc.OUT)
        # print(ooverlap_out)

        # [so]overlap_out are now in the order of [so]_info.atom_connect.out
        # so we still have to convert them to proper indices if used
        # We cannot really check the soverlap_out == len(sgeom_out)
        # in case we have a replaced sparse matrix in the middle of another bigger
        # sparse matrix.
        if len(sgeom_out) != len(soverlap_out) or len(ogeom_out) != len(ooverlap_out):
            # figure out which atoms are not connecting
            s_diff = np.setdiff1d(
                np.arange(s_info.atom_connect.sc.OUT.size), soverlap_out
            )
            o_diff = np.setdiff1d(
                np.arange(o_info.atom_connect.sc.OUT.size), ooverlap_out
            )
            if len(s_diff) > 0 or len(o_diff) > 0:
                msg = f"""{msg}

Number of atoms connecting to the replacement region are not the same in 'self' and 'other'.
Please ensure this."""

            if len(s_diff) > 0:
                tmp = list2str(np.sort(s_info.atom_connect.sc.OUT[s_diff]))
                msg = f"""{msg}

self: atoms (in supercell) connecting to 'atoms' not matched in 'other': {tmp}."""
            if len(o_diff) > 0:
                tmp = list2str(np.sort(o_info.atom_connect.sc.OUT[o_diff]))
                msg = f"""{msg}

other: atoms (in supercell) connecting to 'other_atoms' not matched in 'self': {tmp}."""

        elif not np.allclose(
            sgeom_out.orbitals[soverlap_out], ogeom_out.orbitals[ooverlap_out]
        ):
            tmp_self = list2str(np.sort(sgeom_out.orbitals[soverlap_out]))
            tmp_other = list2str(np.sort(ogeom_out.orbitals[ooverlap_out]))
            msg = f"""{msg}

Atoms in the connection region have different number of orbitals on the atoms.

self orbitals:
   {tmp_self}
other orbitals:
   {tmp_other}"""

        # we can only ensure the orbitals that connect *out* have the same count
        # For supercell connections hopping *IN* might be different due to the supercell
        if len(s_info.orb_connect.sc.OUT) != len(o_info.orb_connect.sc.OUT) and not msg:
            msg = f"""{msg}

Number of orbitals connecting to replacement region is not consistent
between 'self' and 'other'."""

        if msg:
            raise ValueError(msg[1:])

        warn_msg = ""
        S_ = s_info.atom_connect.uc.IN
        O_ = o_info.atom_connect.uc.IN
        for s_, o_ in zip(soverlap_in, ooverlap_in):
            if sgeom_in.atoms[s_] != ogeom_in.atoms[o_]:
                warn_msg = f"""{warn_msg}
Atom 'self[{S_[s_]}]' is not equivalent to 'other[{O_[o_]}]':
  {sgeom_in.atoms[s_]}  !=  {ogeom_in.atoms[o_]}"""

        if warn_msg:
            warn(
                f"""Inequivalent atoms found in replacement region, this may or may not be a problem
depending on your use case. Please be careful though.{warn_msg}"""
            )

        warn_msg = ""
        S_ = s_info.atom_connect.sc.OUT
        O_ = o_info.atom_connect.sc.OUT
        checked1d = _a.zerosi([self.geometry.na])
        for s_, o_ in zip(soverlap_out, ooverlap_out):
            uc_s_ = S_[s_] % self.geometry.na
            if sgeom_out.atoms[s_] != ogeom_out.atoms[o_] and checked1d[uc_s_] == 0:
                checked1d[uc_s_] = 1
                warn_msg = f"""{warn_msg}
Atom 'self[{S_[s_]}]' is not equivalent to 'other[{O_[o_]}]':
  {sgeom_out.atoms[s_]}  !=  {ogeom_out.atoms[o_]}"""

        if warn_msg:
            warn(
                f"""Inequivalent atoms found in connection region, this may or may not be a problem
depending on your use case. Note indices in the following are supercell indices. Please be careful though.{warn_msg}"""
            )

        # clean-up to make it clear that we are not going to use them.
        del sgeom_out, ogeom_out

        # this is where other.sub(other_atoms) gets inserted
        ainsert_idx = atoms.min()
        oinsert_idx = sgeom.a2o(ainsert_idx)
        # this is the indices of the new atoms in the new geometry
        # self_other_atoms = _a.arangei(ainsert_idx, ainsert_idx + len(other_atoms))

        # We need to do the replacement in two steps
        # A. the geometry
        #    This will insert other at ainsert_idx
        #    Note that sub(other_atoms) re-arranges the atoms correctly
        idx = np.argmin((sgeom_in.xyz[soverlap_in] ** 2).sum(1))
        offset = sgeom_in.xyz[soverlap_in[idx]] - ogeom_in.xyz[ooverlap_in[idx]]
        # this will perhaps re-order atoms from other_atoms
        geom = sgeom.replace(atoms, other.geometry.sub(other_atoms), offset=offset)
        del sgeom_in, ogeom_in
        # A. DONE

        # B. Merge the two sparse patterns
        scsr = self._csr
        ncol = scsr.ncol
        col = scsr.col
        D = scsr._D
        # helper function

        def a2o(geom, atoms, sc=True):
            if sc:
                return geom.ouc2sc(geom.a2o(atoms, all=True))
            return geom.a2o(atoms, all=True)

        # Our first task is to merge the two sparse patterns.
        # Delete the *old* values
        # To ensure that inserting will not leave *empty* values
        # we first reduce arrays so that the ptr array is not needed
        ncol = delete(ncol, s_info.orbitals)
        ptr = delete(scsr.ptr, s_info.orbitals)
        idx = array_arange(ptr[:-1], n=ncol)
        col = col[idx]
        D = D[idx]

        # Do the same reduction for the inserted values
        ocsr = other._csr
        idx = array_arange(ocsr.ptr[o_info.orbitals], n=ocsr.ncol[o_info.orbitals])
        # we offset the new columns by self.shape[1], in this way we know
        # which couplings belong to the inserted and the original csr
        col = insert(col, ncol[:oinsert_idx].sum(), ocsr.col[idx] + self.shape[1])
        D = insert(D, ncol[:oinsert_idx].sum(), ocsr._D[idx], axis=0)
        ncol = insert(ncol, oinsert_idx, ocsr.ncol[o_info.orbitals])

        # Create the sparse pattern
        csr = SparseCSR(
            (D, col, _ncol_to_indptr(ncol)),
            shape=(geom.no, sgeom.no_s + ogeom.no_s, D.shape[1]),
        )
        del D, col, ncol

        # Now we have merged the two sparse patterns
        # But we need to correct the orbital couplings
        # : *outside* refers to the original sparse pattern (without `atoms`)
        # : *inside* refers to the inserted sparse pattern (other.sub(other_atoms))
        # We have to do 1 and 2 simultaneously.
        # We have to do 3 and 4 simultaneously.
        # This is because they may have overlapping columns

        # 1: couplings from *outside* to *outside* (no scale)
        # 2: couplings from *outside* to *inside* (scaled)
        # 3: couplings from *inside* to *inside* (no scale)
        # 4: couplings from *inside* to *outside* (scaled)
        convert = [[], []]

        def assert_unique(old, new):
            old = concatenate(old)
            new = concatenate(new)
            assert len(unique(old)) == len(old)
            assert len(unique(new)) == len(new)
            return old, new

        # 1:
        # print("1:")
        old = delete(_a.arangei(len(sgeom)), atoms)
        new = _a.arangei(len(old))
        new[ainsert_idx:] += len(other_atoms)
        old = a2o(sgeom, old)
        convert[0].append(old)
        new = a2o(geom, new)
        convert[1].append(new)
        rows = geom.osc2uc(new, unique=True)

        # 2:
        # print("2:")
        old = s_info.atom_connect.uc.IN[soverlap_in]
        # algorithm to get indices in other_atoms
        new = o_info.atom_connect.uc.IN[ooverlap_in]
        tmp = argsort(other_atoms)
        new = tmp[searchsorted(other_atoms, new, sorter=tmp)] + ainsert_idx
        old = a2o(sgeom, old)
        convert[0].append(old)
        new = a2o(geom, new)
        convert[1].append(new)

        # translate columns
        csr.translate_columns(*assert_unique(convert[0], convert[1]), rows=rows)
        # scale columns that connects inside
        csr.scale_columns(convert[1][1], scale=scale[0], rows=rows)

        # on to the *inside* 3, 4
        convert = [[], []]

        # 3:
        # print("3:")
        # we have all the *inside* column indices offset by self.shape[1]
        old = a2o(ogeom, other_atoms, False) + self.shape[1]
        new = ainsert_idx + _a.arangei(len(other_atoms))
        # print("old: ", old)
        # print("new: ", new)
        new = a2o(geom, new, False)
        convert[0].append(old)
        convert[1].append(new)
        rows = geom.osc2uc(new, unique=True)

        # 4:
        # print("4:")
        old = o_info.atom_connect.sc.OUT
        new = _a.emptyi(len(old))
        for i, atom in enumerate(old):
            idx = geom.close(ogeom.axyz(atom) + offset, R=atol)
            assert (
                len(idx) == 1
            ), f"More than 1 atom {idx} for atom {atom} = {ogeom.axyz(atom)}, {geom.axyz(idx)}"
            new[i] = idx[0]
        # print("old: ", old)
        # print("new: ", new)
        old = a2o(ogeom, old, False) + self.shape[1]
        new = a2o(geom, new, False)

        convert[0].append(old)
        convert[1].append(new)

        # translate columns
        csr.translate_columns(*assert_unique(convert[0], convert[1]), rows=rows)
        # scale columns that connects inside
        csr.scale_columns(convert[1][1], scale=scale[1], rows=rows)

        # ensure we have translated all columns correctly
        assert valid_index(csr.col, geom.no_s).all()
        # correct shape of column matrix
        csr._shape = (csr.shape[0], geom.no_s, csr.shape[2])
        out = self.copy()
        out._csr = csr
        out._geometry = geom
        return out

    def prune_range(
        self,
        *,
        R: float,
        atoms: Optional[AtomsLike] = None,
        atoms_to: Optional[AtomsLike] = None,
    ) -> "Self":
        r"""Prune elements coupling further than `R`.

        Search for connections from `atoms` to `atoms_to` where the distances are
        further than `R`. If such couplings exists, remove them.

        Parameters
        ----------
        R :
            the distance at which the couplings are cut
        atoms :
            atoms that will be considered in the check.
            The rows of the couplings that is taken into account.
        atoms_to :
            The columns of the couplings that is taken into account.

        Notes
        -----
        The transposed couplings are *also* deleted to ensure a symmetric
        matrix.

        Currently, one cannot select subset of atoms.
        """
        geom = self.geometry
        atoms = np.unique(geom._sanitize_atoms(atoms))
        atoms_to = geom.auc2sc(geom._sanitize_atoms(atoms_to))
        atoms_all = atoms.size == geom.na and atoms_to.size == geom.na_s
        if not atoms_all:
            raise NotImplementedError(
                f"{self.__class__.__name__}.prune_range "
                "does not work with subsets of atoms."
            )

        # change to *not* selected atoms
        atoms_to_exclude = np.delete(np.arange(geom.na_s), atoms_to)
        orbs_to_exclude = geom.a2o(atoms_to_exclude, all=True)

        # Now search the couplings
        edges = self.edges(atoms, exclude=orbs_to_exclude)

        out = self.copy()

        # Now we have all edges that couples between atoms and atoms_to.
        # Determine the distances between the atoms
        for ia in atoms:
            iorbs = geom.a2o(ia, all=True)

            dist = geom.rij(ia, edges)
            # Select atoms with distances larger than R
            idx = edges[dist > R]
            if len(idx) == 0:
                continue

            # find orbitals that idx represents
            jorbs = geom.a2o(idx, all=True)
            uc_jorbs = np.unique(jorbs % self.shape[0])
            for io in iorbs:
                del out[io, jorbs]
                if not atoms_all:
                    # this ends up deleting everything...
                    io_sc = np.unique(self._transpose_indices(jorbs, base=io))
                    del out[uc_jorbs, io_sc]

        return out

    def toSparseAtom(self, dim: Optional[int] = None, dtype=None):
        """Convert the sparse object (without data) to a new sparse object with equivalent but reduced sparse pattern

        This converts the orbital sparse pattern to an atomic sparse pattern.

        Parameters
        ----------
        dim :
           number of dimensions allocated in the SparseAtom object, default to the same
        dtype : numpy.dtype, optional
           used data-type for the sparse object. Defaults to the same.
        """
        if dim is None:
            dim = self.shape[-1]
        if dtype is None:
            dtype = self.dtype

        geom = self.geometry

        # Create a conversion vector
        orb2atom = tile(geom.o2a(_a.arangei(geom.no)), geom.n_s)
        orb2atom.shape = (-1, geom.no)
        orb2atom += _a.arangei(geom.n_s).reshape(-1, 1) * geom.na
        orb2atom.shape = (-1,)

        # First convert all rows to the same
        csr = self._csr

        # Now build the new sparse pattern
        ptr = _a.emptyi(geom.na + 1)
        ptr[0] = 0
        col = [None] * geom.na
        for ia in range(geom.na):
            o1, o2 = geom.a2o([ia, ia + 1])
            # Get current atomic elements
            idx = array_arange(csr.ptr[o1:o2], n=csr.ncol[o1:o2])

            # These are now the atomic columns
            # Immediately reduce to unique elements
            acol = unique(orb2atom[csr.col[idx]])

            # Step counters
            col[ia] = acol
            ptr[ia + 1] = ptr[ia] + len(acol)

        # Now we can create the sparse atomic
        col = concatenate(col, axis=0).astype(int32, copy=False)
        spAtom = SparseAtom(geom, dim=dim, dtype=dtype, nnzpr=0)
        spAtom._csr.ptr[:] = ptr[:]
        spAtom._csr.ncol[:] = diff(ptr)
        spAtom._csr.col = col
        spAtom._csr._D = np.zeros([len(col), dim], dtype=dtype)
        spAtom._csr._nnz = len(col)
        spAtom._csr._finalized = True  # unique returns sorted elements
        return spAtom
