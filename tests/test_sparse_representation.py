"""Tests for ruconceptnet.sparse_representation.Sparse3DTensor."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ruconceptnet.sparse_representation import Sparse3DTensor


def _simple_tensor(weights=None) -> Sparse3DTensor:
    """
    3 x 3 x 2 tensor with entries:
        (0, 1, 0)
        (0, 2, 1)
        (2, 0, 0)
    """
    rows = np.array([0, 0, 2])
    cols = np.array([1, 2, 0])
    vals = np.array([0, 1, 0])  # depth indices
    return Sparse3DTensor((3, 3, 2), ((rows, cols), vals), weights=weights)


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

    def test_default_value_is_one(self):
        t = _simple_tensor()
        assert t[0, 1, 0] == 1.0


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
        cols, depths, weights = t.row_nz(0)
        assert len(cols) > 0
        assert len(depths) > 0
        assert len(weights) == len(cols)

    def test_row_nz_correct_columns(self):
        t = _simple_tensor()
        cols, _, _ = t.row_nz(0)
        assert set(cols.tolist()) == {1, 2}

    def test_row_nz_empty_for_inactive_row(self):
        t = _simple_tensor()
        cols, depths, weights = t.row_nz(1)
        assert len(cols) == 0
        assert len(weights) == 0

    def test_row_nz_depth_values_in_range(self):
        t = _simple_tensor()
        _, depths, _ = t.row_nz(0)
        assert all(0 <= d < 2 for d in depths)


class TestSparse3DTensorColNz:
    def test_col_nz_returns_nonempty_for_active_col(self):
        t = _simple_tensor()
        rows, _, _ = t.col_nz(1)
        assert len(rows) > 0

    def test_col_nz_correct_rows(self):
        t = _simple_tensor()
        rows, _, _ = t.col_nz(1)
        assert 0 in rows

    def test_col_nz_col2_has_row0(self):
        t = _simple_tensor()
        rows, _, _ = t.col_nz(2)
        assert 0 in rows


class TestSparse3DTensorRowColNz:
    def test_rowcol_nz_returns_depths(self):
        t = _simple_tensor()
        depths, _ = t.rowcol_nz(0, 1)
        assert 0 in depths

    def test_rowcol_nz_empty_for_absent_pair(self):
        t = _simple_tensor()
        depths, weights = t.rowcol_nz(1, 2)
        assert len(depths) == 0
        assert len(weights) == 0

    def test_rowcol_nz_depth_second_channel(self):
        t = _simple_tensor()
        depths, _ = t.rowcol_nz(0, 2)
        assert 1 in depths


class TestSparse3DTensorWeights:
    def test_row_nz_returns_weights(self):
        t = _simple_tensor(weights=np.array([2.5, 3.0, 1.0]))
        cols, depths, weights = t.row_nz(0)
        by_col = dict(zip(cols.tolist(), weights.tolist(), strict=True))
        assert by_col[1] == 2.5  # entry (0, 1, 0)
        assert by_col[2] == 3.0  # entry (0, 2, 1)

    def test_col_nz_returns_weights(self):
        t = _simple_tensor(weights=np.array([2.5, 3.0, 1.0]))
        rows, _, weights = t.col_nz(1)
        by_row = dict(zip(rows.tolist(), weights.tolist(), strict=True))
        assert by_row[0] == 2.5

    def test_rowcol_nz_returns_weight(self):
        t = _simple_tensor(weights=np.array([2.5, 3.0, 1.0]))
        depths, weights = t.rowcol_nz(0, 2)
        assert weights[0] == 3.0

    def test_default_weight_is_one(self):
        t = _simple_tensor()
        _, _, weights = t.row_nz(0)
        assert all(w == 1.0 for w in weights)


class TestSparse3DTensorStr:
    def test_str_returns_string(self):
        t = _simple_tensor()
        assert isinstance(str(t), str)


class TestSparse3DTensorCscView:
    def test_csc_is_lazy_and_cached(self):
        t = _simple_tensor()
        assert t._csc is None  # not built until first column lookup
        t.col_nz(1)
        first = t._csc
        assert first is not None
        t.col_nz(2)
        assert t._csc is first  # reused, not rebuilt

    def test_col_nz_works_without_csc_attribute(self):
        """Tensors unpickled from older versions have no ``_csc`` attribute."""
        t = _simple_tensor()
        del t._csc
        assert not hasattr(t, "_csc")
        rows, _, _ = t.col_nz(1)
        assert 0 in rows  # must not raise and must return correct data

    def test_setitem_invalidates_csc_cache(self):
        t = _simple_tensor()
        t.col_nz(1)  # build cache
        assert t._csc is not None
        t[2, 1, 0] = 1  # mutate -> cache must be dropped
        assert t._csc is None
        rows, _, _ = t.col_nz(1)  # rebuilt; reflects the new entry
        assert 2 in rows
