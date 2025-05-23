# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sisl._array import array_arange
from sisl._internal import set_module

__all__ = ["BlockMatrixIndexer", "BlockMatrix"]


@set_module("sisl_toolbox.btd")
class BlockMatrixIndexer:
    def __init__(self, bm: BlockMatrix):
        self._bm = bm

    def __len__(self) -> int:
        """Number of blocks in this block matrix"""
        return len(self._bm.blocks)

    def __iter__(self):
        """Loop contained indices in the BlockMatrix"""
        yield from self._bm._M.keys()

    def _assert_key(self, key) -> None:
        if not isinstance(key, tuple):
            raise ValueError(
                f"{self.__class__.__name__} index deletion must be done with a tuple."
            )

    def __delitem__(self, key) -> None:
        self._assert_key(key)
        del self._bm._M[key]

    def __contains__(self, key) -> bool:
        self._assert_key(key)
        return key in self._bm._M

    def __getitem__(self, key):
        self._assert_key(key)
        M = self._bm._M.get(key)
        if M is None:
            i, j = key
            # the data-type is probably incorrect.. :(
            return np.zeros([self._bm.blocks[i], self._bm.blocks[j]])
        return M

    def __setitem__(self, key, M) -> None:
        self._assert_key(key)
        # Check that the shapes coincide for the block specified
        s = (self._bm.blocks[key[0]], self._bm.blocks[key[1]])
        assert (
            M.shape == s
        ), f"Could not assign matrix of shape {M.shape} into matrix of shape {s}"
        self._bm._M[key] = M

    def __getattr__(self, attr):
        return getattr(self._bm, attr)


@set_module("sisl_toolbox.btd")
class BlockMatrix:
    """Container class that holds a block matrix.

    A block matrix will always be a square matrix.
    """

    def __init__(self, blocks: npt.ArrayLike):
        self._M = {}
        self._blocks = np.asarray(blocks, dtype=int)
        self._blocks_cum0 = np.empty([len(self.blocks) + 1], dtype=self.blocks.dtype)
        self._blocks_cum0[0] = 0
        self._blocks_cum0[1:] = np.cumsum(self.blocks)
        self._index_blocks = np.arange(len(self))
        # Correct the blocks
        min_index = 0
        for block, max_index in enumerate(self._blocks_cum0[1:]):
            self._index_blocks[min_index:max_index] = block
            min_index = max_index

    @property
    def blocks(self) -> np.ndarray:
        """Size of the blocks that define this block-matrix"""
        return self._blocks

    def __len__(self) -> int:
        """The length of the first dimension of the matrix"""
        return np.sum(self._blocks)

    @property
    def shape(self) -> tuple:
        n = len(self)
        return (n, n)

    @property
    def dtype(self):
        """Retrieve the data-type of the first element of the matrix dictionary"""
        # Retrieve the dtype for the first element of the dictionary
        values = self._M.values()
        if values:
            return values[0].dtype
        return np.float64  # the default data-type of numpy arrays

    def _get_blocks(self, indices, in_range: bool = False):
        """Returns the blocks that an index belongs to"""
        if in_range:
            block1 = (indices.min() < self._blocks_cum0[1:]).nonzero()[0][0]
            block2 = (indices.max() < self._blocks_cum0[1:]).nonzero()[0][0]
            if block1 == block2:
                blocks = [block1]
            else:
                blocks = np.arange(block1, block2 + 1)
        else:
            blocks = np.unique(self._index_blocks[indices])
        return blocks

    def _get_block_indices(self, blocks):
        """Convert a list of blocks into a list of indices for each block

        If a block is supplied multiple times, its indices will be repeated.
        """
        b_cum0 = self._blocks_cum0
        indices = array_arange(b_cum0[blocks], b_cum0[1:][blocks])
        # Split into separate lists
        return np.split(indices, np.cumsum(self.blocks[blocks]))

    def _get_block_slices(self, blocks):
        """Convert a list of blocks into a list of indices for each block

        If a block is supplied multiple times, its indices will be repeated.
        """
        bc0 = self._blocks_cum0
        return [slice(bc0[b], bc0[b + 1]) for b in blocks]

    def copy(self) -> Self:
        """Create a copy of this block matrix"""
        new = self.__class__(self.blocks)
        # Copy values
        for k, v in self._M.items():
            new._M[k] = np.copy(v)
        return new

    def asformat(self, format):
        """Convert to a particular format"""
        return getattr(self, f"to{format}")()

    def toarray(self) -> np.ndarray:
        """Convert to a dense matrix, where non-defined blocks will be zero"""
        BI = self.block_indexer
        nb = len(BI)
        # stack stuff together
        return np.concatenate(
            [np.concatenate([BI[i, j] for i in range(nb)], axis=0) for j in range(nb)],
            axis=1,
        )

    def tobtd(self) -> BlockMatrix:
        """Return only the block tridiagonal part of the matrix"""
        ret = self.__class__(self.blocks)
        sBI = self.block_indexer
        rBI = ret.block_indexer
        nb = len(sBI)
        for j in range(nb):
            for i in range(max(0, j - 1), min(j + 2, nb)):
                rBI[i, j] = sBI[i, j]
        return ret

    def tobd(self) -> BlockMatrix:
        """Return only the block diagonal part of the matrix"""
        ret = self.__class__(self.blocks)
        sBI = self.block_indexer
        rBI = ret.block_indexer
        nb = len(sBI)
        for i in range(nb):
            rBI[i, i] = sBI[i, i]
        return ret

    def diagonal(self) -> np.ndarray:
        """Returns the diagonal of the matrix"""
        BI = self.block_indexer
        return np.concatenate([BI[b, b].diagonal() for b in range(len(BI))])

    @property
    def block_indexer(self) -> BlockMatrixIndexer:
        """Get the indexer for this block matrix

        The indexer allows manipulating the matrix using indices of the blocks,
        rather than indices of the matrix elements.
        """
        return BlockMatrixIndexer(self)
