"""Tests for ruconceptnet.sparse_representation.Sparse3DTensor."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ruconceptnet.sparse_representation import Sparse3DTensor


def _simple_tensor() -> Sparse3DTensor:
    """
    3 x 3 x 2 tensor with entries:
        (0, 1, 0) = 1
        (0, 2, 1) = 1
        (2, 0, 0) = 1
    """
    rows = np.array([0, 0, 2])
    cols = np.array([1, 2, 0])
    vals = np.array([0, 1, 0])  # depth indices
    return Sparse3DTensor((3, 3, 2), ((rows, cols), vals))


class TestSparse3DTensorConstruction:
    def test_shape_stored(self):
        t = _simple_tensor()
        assert t.t.shape == (3, 6)

    def test_step_attribute(self):
        t = _simple_tensor()
        assert t.step == 2

    def test_invalid_shape_raises(self):
        rows = np.array([0])
        cols = np.array([0])
        vals = np.array([0])
        with pytest.raises(AssertionError):
            Sparse3DTensor((3, 3), ((rows, cols), vals))

    def test_backing_matrix_is_csr(self):
        t = _simple_tensor()
        assert isinstance(t.t, csr_matrix)


class TestSparse3DTensorGetItem:
    def test_existing_entry(self):
        t = _simple_tensor()
        assert t[0, 1, 0] != 0

    def test_nonexistent_entry_is_zero(self):
        t = _simple_tensor()
        assert t[0, 0, 0] == 0

    def test_depth_second_channel(self):
        t = _simple_tensor()
        assert t[0, 2, 1] != 0

    def test_out_of_plane_is_zero(self):
        t = _simple_tensor()
        assert t[0, 2, 0] == 0


class TestSparse3DTensorSetItem:
    def test_setitem_updates_value(self):
        t = _simple_tensor()
        t[1, 1, 0] = 99
        assert t[1, 1, 0] == 99

    def test_setitem_does_not_corrupt_other_entries(self):
        t = _simple_tensor()
        t[1, 1, 0] = 5
        assert t[0, 1, 0] != 0


class TestSparse3DTensorRowNz:
    def test_row_nz_returns_nonempty_for_active_row(self):
        t = _simple_tensor()
        cols, depths = t.row_nz(0)
        assert len(cols) > 0
        assert len(depths) > 0

    def test_row_nz_correct_columns(self):
        t = _simple_tensor()
        cols, _ = t.row_nz(0)
        assert set(cols.tolist()) == {1, 2}

    def test_row_nz_empty_for_inactive_row(self):
        t = _simple_tensor()
        cols, depths = t.row_nz(1)
        assert len(cols) == 0

    def test_row_nz_depth_values_in_range(self):
        t = _simple_tensor()
        _, depths = t.row_nz(0)
        assert all(0 <= d < 2 for d in depths)


class TestSparse3DTensorColNz:
    def test_col_nz_returns_nonempty_for_active_col(self):
        t = _simple_tensor()
        rows, _ = t.col_nz(1)
        assert len(rows) > 0

    def test_col_nz_correct_rows(self):
        t = _simple_tensor()
        rows, _ = t.col_nz(1)
        assert 0 in rows

    def test_col_nz_col2_has_row0(self):
        t = _simple_tensor()
        rows, _ = t.col_nz(2)
        assert 0 in rows


class TestSparse3DTensorRowColNz:
    def test_rowcol_nz_returns_depths(self):
        t = _simple_tensor()
        depths = t.rowcol_nz(0, 1)
        assert 0 in depths

    def test_rowcol_nz_empty_for_absent_pair(self):
        t = _simple_tensor()
        depths = t.rowcol_nz(1, 2)
        assert len(depths) == 0

    def test_rowcol_nz_depth_second_channel(self):
        t = _simple_tensor()
        depths = t.rowcol_nz(0, 2)
        assert 1 in depths


class TestSparse3DTensorStr:
    def test_str_returns_string(self):
        t = _simple_tensor()
        assert isinstance(str(t), str)
