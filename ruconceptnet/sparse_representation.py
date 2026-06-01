from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix


class Sparse3DTensor:
    """
    A sparse 3D tensor backed by SciPy sparse matrices.

    The depth axis is unrolled into the columns, so a ``(rows, cols, depth)``
    tensor is stored as a ``(rows, cols * depth)`` matrix.  Each stored cell
    holds the edge ``weight`` (a positive float); when no weights are supplied
    the value defaults to ``1.0``, which reproduces the original binary
    presence behaviour and keeps tensors pickled by older versions working.

    Row look-ups use a CSR matrix.  Column look-ups use a CSC matrix that is
    built lazily on first use and cached, since slicing columns out of a CSR
    matrix is very slow.  The CSC view is derived from the CSR matrix at
    runtime, so it is never pickled.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        triples,
        weights: np.ndarray | None = None,
    ) -> None:
        assert len(shape) == 3
        matrix_shape = (shape[0], shape[1] * shape[2])
        self.step = shape[2]

        (rows_idx, cols_idx), vals = triples
        new_cols_idx = cols_idx * self.step + vals
        data = np.ones(rows_idx.shape[0]) if weights is None else np.asarray(weights, dtype=float)
        self.t = coo_matrix(
            (data, (rows_idx, new_cols_idx)),
            shape=matrix_shape,
        ).tocsr()
        self._csc = None  # lazily built CSC view for column look-ups

    def _as_csc(self):
        # Derived on demand from the CSR matrix; works for tensors that were
        # pickled before this attribute existed (hence getattr with a default).
        csc = getattr(self, "_csc", None)
        if csc is None:
            csc = self.t.tocsc()
            self._csc = csc
        return csc

    def __getitem__(self, indices: tuple[int, int, int]) -> float:
        row, col, dep = indices
        return self.t[row, col * self.step + dep]

    def __setitem__(self, indices: tuple[int, int, int], value: float) -> None:
        row, col, dep = indices
        self.t[row, col * self.step + dep] = value
        self._csc = None  # invalidate cached CSC view

    def __str__(self) -> str:
        return str(self.t.todense())

    def row_nz(self, ix: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(column_indices, depth_indices, weights)`` for a row."""
        block = self.t[ix].tocoo()
        col_indices = block.col // self.step
        depth_indices = block.col % self.step
        return col_indices, depth_indices, block.data

    def col_nz(self, ix: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(row_indices, depth_indices, weights)`` for a column."""
        block = self._as_csc()[:, ix * self.step : (ix + 1) * self.step].tocoo()
        return block.row, block.col, block.data

    def rowcol_nz(self, row: int, col: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(depth_indices, weights)`` for a ``(row, col)`` pair."""
        block = self.t[row, col * self.step : (col + 1) * self.step].tocoo()
        return block.col, block.data
