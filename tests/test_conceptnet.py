"""
Tests for ruconceptnet.conceptnet.ConceptNet.

All tests use the 'cn' fixture from conftest.py which loads a tiny in-memory
bundle, so no network access or real data file is required.
"""

import pytest

# ---------------------------------------------------------------------------
# get_targets
# ---------------------------------------------------------------------------


class TestGetTargets:
    def test_known_word_returns_list(self, cn):
        result = cn.get_targets("алкоголь")
        assert isinstance(result, list)

    def test_known_word_returns_tuples(self, cn):
        result = cn.get_targets("алкоголь")
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_known_word_target_is_str(self, cn):
        for target, _rels in cn.get_targets("алкоголь"):
            assert isinstance(target, str)

    def test_known_word_relations_is_set(self, cn):
        for _, rels in cn.get_targets("алкоголь"):
            assert isinstance(rels, set)

    def test_synonym_relation_present(self, cn):
        targets = dict(cn.get_targets("алкоголь"))
        assert "спирт" in targets
        assert "Synonym" in targets["спирт"]

    def test_relatedto_relation_present(self, cn):
        targets = dict(cn.get_targets("алкоголь"))
        assert "алкоголизм" in targets
        assert "RelatedTo" in targets["алкоголизм"]

    def test_unknown_word_returns_empty_list(self, cn):
        assert cn.get_targets("несуществующееслово") == []

    def test_empty_string_returns_empty_list(self, cn):
        assert cn.get_targets("") == []

    def test_yoga_targets_formof(self, cn):
        targets = dict(cn.get_targets("йога"))
        assert "йоги" in targets
        assert "FormOf" in targets["йоги"]


# ---------------------------------------------------------------------------
# get_sources
# ---------------------------------------------------------------------------


class TestGetSources:
    def test_known_target_returns_list(self, cn):
        result = cn.get_sources("спирт")
        assert isinstance(result, list)

    def test_known_target_source_word_present(self, cn):
        sources = dict(cn.get_sources("спирт"))
        assert "алкоголь" in sources

    def test_known_target_relation_present(self, cn):
        sources = dict(cn.get_sources("спирт"))
        assert "Synonym" in sources["алкоголь"]

    def test_unknown_word_returns_empty_list(self, cn):
        assert cn.get_sources("никогданебыло") == []

    def test_sources_are_tuples_with_sets(self, cn):
        for src, rels in cn.get_sources("спирт"):
            assert isinstance(src, str)
            assert isinstance(rels, set)

    def test_yogi_has_yoga_as_source(self, cn):
        sources = dict(cn.get_sources("йоги"))
        assert "йога" in sources


# ---------------------------------------------------------------------------
# check_pair
# ---------------------------------------------------------------------------


class TestCheckPair:
    def test_returns_two_lists(self, cn):
        result = cn.check_pair("человек", "зверь")
        assert isinstance(result, (list, tuple))
        assert len(result) == 2

    def test_direct_relation_found(self, cn):
        direct, reverse = cn.check_pair("человек", "зверь")
        assert "DistinctFrom" in direct

    def test_reverse_is_empty_for_asymmetric_pair(self, cn):
        _, reverse = cn.check_pair("человек", "зверь")
        assert reverse == []

    def test_flipped_pair_gives_empty_direct(self, cn):
        direct, reverse = cn.check_pair("зверь", "человек")
        assert direct == []
        assert "DistinctFrom" in reverse

    def test_unknown_source_returns_empty_lists(self, cn):
        direct, reverse = cn.check_pair("фантом", "зверь")
        assert direct == [] and reverse == []

    def test_unknown_target_returns_empty_lists(self, cn):
        direct, reverse = cn.check_pair("человек", "робот")
        assert direct == [] and reverse == []

    def test_both_unknown_returns_empty_lists(self, cn):
        direct, reverse = cn.check_pair("нет", "такого")
        assert direct == [] and reverse == []

    def test_self_pair_known_word(self, cn):
        direct, reverse = cn.check_pair("алкоголь", "алкоголь")
        assert isinstance(direct, list) and isinstance(reverse, list)


# ---------------------------------------------------------------------------
# ConceptNet construction
# ---------------------------------------------------------------------------


class TestConceptNetConstruction:
    def test_custom_filepath_loads(self, bundle_file):
        from ruconceptnet.conceptnet import ConceptNet

        cn = ConceptNet(filepath=str(bundle_file))
        assert cn is not None

    def test_vocab_is_dict(self, cn):
        assert isinstance(cn.v, dict)

    def test_inverse_vocab_is_consistent(self, cn):
        for word, idx in cn.v.items():
            assert cn.iv[idx] == word

    def test_rel_vocab_is_dict(self, cn):
        assert isinstance(cn.rv, dict)

    def test_inverse_rel_vocab_is_consistent(self, cn):
        for rel, idx in cn.rv.items():
            assert cn.irv[idx] == rel

    def test_bad_filepath_raises(self):
        from ruconceptnet.conceptnet import ConceptNet

        with pytest.raises(FileNotFoundError):
            ConceptNet(filepath="/nonexistent/path/file.bz2")


# ---------------------------------------------------------------------------
# weights (with_weights=True)
# ---------------------------------------------------------------------------


class TestGetTargetsWithWeights:
    def test_returns_dict_per_target(self, cn):
        targets = dict(cn.get_targets("алкоголь", with_weights=True))
        assert isinstance(targets["спирт"], dict)

    def test_correct_weight(self, cn):
        targets = dict(cn.get_targets("алкоголь", with_weights=True))
        assert targets["спирт"]["Synonym"] == 2.5
        assert targets["алкоголизм"]["RelatedTo"] == 1.0

    def test_default_still_returns_sets(self, cn):
        targets = dict(cn.get_targets("алкоголь"))
        assert isinstance(targets["спирт"], set)

    def test_unknown_word_empty(self, cn):
        assert cn.get_targets("несуществующее", with_weights=True) == []


class TestGetSourcesWithWeights:
    def test_correct_weight(self, cn):
        sources = dict(cn.get_sources("спирт", with_weights=True))
        assert sources["алкоголь"]["Synonym"] == 2.5

    def test_formof_weight(self, cn):
        sources = dict(cn.get_sources("йоги", with_weights=True))
        assert sources["йога"]["FormOf"] == 3.0


class TestCheckPairWithWeights:
    def test_direct_weight(self, cn):
        direct, reverse = cn.check_pair("человек", "зверь", with_weights=True)
        assert direct["DistinctFrom"] == 0.5
        assert reverse == {}

    def test_reverse_weight(self, cn):
        direct, reverse = cn.check_pair("зверь", "человек", with_weights=True)
        assert direct == {}
        assert reverse["DistinctFrom"] == 0.5

    def test_unknown_returns_empty_dicts(self, cn):
        direct, reverse = cn.check_pair("нет", "такого", with_weights=True)
        assert direct == {} and reverse == {}
