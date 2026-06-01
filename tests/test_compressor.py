"""
Tests for compressor.py — the offline data-prep utilities.

We do not exercise assertions2triples() in a file I/O sense because it reads
a raw TSV that is not committed to the repo.  Instead we test triples2bundle()
directly with a hand-crafted DataFrame, and verify that the resulting Bundle
round-trips through pickle correctly.
"""

import bz2
import pickle

import pandas as pd
import pytest

from compressor import triples2bundle
from ruconceptnet.conceptnet import Bundle
from ruconceptnet.sparse_representation import Sparse3DTensor


def _minimal_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "relation": ["Synonym", "RelatedTo", "FormOf"],
            "source": ["алкоголь", "алкоголь", "йога"],
            "target": ["спирт", "алкоголизм", "йоги"],
        }
    )


class TestTriples2Bundle:
    def test_returns_bundle(self):
        bundle = triples2bundle(_minimal_df())
        assert isinstance(bundle, Bundle)

    def test_vocab_contains_all_words(self):
        bundle = triples2bundle(_minimal_df())
        for word in ["алкоголь", "спирт", "алкоголизм", "йога", "йоги"]:
            assert word in bundle.v

    def test_rel_vocab_contains_all_relations(self):
        bundle = triples2bundle(_minimal_df())
        for rel in ["Synonym", "RelatedTo", "FormOf"]:
            assert rel in bundle.rv

    def test_tensor_is_sparse3dtensor(self):
        bundle = triples2bundle(_minimal_df())
        assert isinstance(bundle.t, Sparse3DTensor)

    def test_vocab_indices_are_unique(self):
        bundle = triples2bundle(_minimal_df())
        indices = list(bundle.v.values())
        assert len(indices) == len(set(indices))

    def test_rel_vocab_indices_are_unique(self):
        bundle = triples2bundle(_minimal_df())
        indices = list(bundle.rv.values())
        assert len(indices) == len(set(indices))

    def test_single_row_df(self):
        df = pd.DataFrame({"relation": ["Synonym"], "source": ["кошка"], "target": ["кот"]})
        bundle = triples2bundle(df)
        assert "кошка" in bundle.v
        assert "кот" in bundle.v
        assert "Synonym" in bundle.rv

    def test_duplicate_relations_deduplicated_in_vocab(self):
        df = pd.DataFrame(
            {
                "relation": ["Synonym", "Synonym"],
                "source": ["а", "б"],
                "target": ["в", "г"],
            }
        )
        bundle = triples2bundle(df)
        assert len(bundle.rv) == 1

    def test_source_equals_target_allowed(self):
        df = pd.DataFrame({"relation": ["RelatedTo"], "source": ["слово"], "target": ["слово"]})
        bundle = triples2bundle(df)
        assert "слово" in bundle.v


class TestBundlePickleRoundtrip:
    def test_roundtrip_bz2(self, tmp_path):
        bundle = triples2bundle(_minimal_df())
        path = tmp_path / "out.pickle.bz2"
        with bz2.open(path, "wb") as fh:
            pickle.dump(bundle, fh)
        with bz2.open(path, "rb") as fh:
            loaded = pickle.load(fh)
        assert isinstance(loaded, Bundle)
        assert loaded.v == bundle.v
        assert loaded.rv == bundle.rv

    def test_roundtrip_preserves_tensor_shape(self, tmp_path):
        bundle = triples2bundle(_minimal_df())
        path = tmp_path / "out.pickle.bz2"
        with bz2.open(path, "wb") as fh:
            pickle.dump(bundle, fh)
        with bz2.open(path, "rb") as fh:
            loaded = pickle.load(fh)
        assert loaded.t.t.shape == bundle.t.t.shape


class TestTriples2BundleWeights:
    def test_missing_weight_column_defaults_to_one(self):
        # _minimal_df has no weight column -> should not raise
        bundle = triples2bundle(_minimal_df())
        assert isinstance(bundle, Bundle)

    def test_weight_propagated_into_tensor(self):
        df = pd.DataFrame(
            {
                "relation": ["Synonym"],
                "source": ["кошка"],
                "target": ["кот"],
                "weight": [4.2],
            }
        )
        bundle = triples2bundle(df)
        s, t, r = bundle.v["кошка"], bundle.v["кот"], bundle.rv["Synonym"]
        assert bundle.t[s, t, r] == pytest.approx(4.2)

    def test_duplicate_edges_keep_max_weight(self):
        df = pd.DataFrame(
            {
                "relation": ["Synonym", "Synonym"],
                "source": ["а", "а"],
                "target": ["б", "б"],
                "weight": [1.0, 5.0],
            }
        )
        bundle = triples2bundle(df)
        s, t, r = bundle.v["а"], bundle.v["б"], bundle.rv["Synonym"]
        assert bundle.t[s, t, r] == pytest.approx(5.0)
