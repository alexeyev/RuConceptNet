# -*- coding: utf-8 -*-

from scipy.sparse import coo_matrix


class Sparse3DTensor(object):
    """
        Unrolling depths as rows using scipy's sparse matrix;
        could be compressed even more since our case is a binary matrix one,
        but I don't care
    """

    def __init__(self, shape, triples):
        assert len(shape) == 3
        matrix_shape = (shape[0], shape[1] * shape[2])
        self.step = shape[2]

        (rows_idx, cols_idx), vals = triples
        new_cols_idx = cols_idx * self.step + vals
        self.t = coo_matrix(([1.] * rows_idx.shape[0], (rows_idx, new_cols_idx)), shape=matrix_shape).tocsr()

    def __getitem__(self, indices):
        row, col, dep = indices
        return self.t[row, col * self.step + dep]

    def __setitem__(self, indices, value):
        row, col, dep = indices
        self.t[row, col * self.step + dep] = value

    def __str__(self) -> str:
        return str(self.t.todense())

    def row_nz(self, ix):
        """
            Returns a list if indices of nonzero values along the row
        """
        _, nz = self.t[ix].nonzero()
        row_indices = nz // self.step
        depth_indices = nz % self.step
        return row_indices, depth_indices

    def col_nz(self, ix):
        """
            Returs a list of coordinates of nonzero values in a column
        """
        nz = self.t[:, ix * self.step:(ix + 1) * self.step].nonzero()
        return nz

    def rowcol_nz(self, row, col):
        """
            Returns a list of nonzero values along the depth axis
        """
        return self.t[row, col * self.step:(col + 1) * self.step].nonzero()[1]
