"""
Shared pytest fixtures that build a minimal in-memory ConceptNet bundle
so tests can run without the bundled data file.
"""

import bz2
import pickle
from pathlib import Path

import numpy as np
import pytest

from ruconceptnet.conceptnet import Bundle
from ruconceptnet.sparse_representation import Sparse3DTensor


def _make_minimal_bundle() -> Bundle:
    """
    Build a tiny ConceptNet bundle with four Russian concepts and three
    relation types.  The relations encoded are:

        алкоголь  --Synonym-->    спирт
        алкоголь  --RelatedTo-->  алкоголизм
        йога      --FormOf-->     йоги
        человек   --DistinctFrom--> зверь
    """
    words = ["алкоголь", "спирт", "алкоголизм", "йога", "йоги", "человек", "зверь"]
    rels = ["Synonym", "RelatedTo", "FormOf", "DistinctFrom"]

    v = {w: i for i, w in enumerate(words)}
    rv = {r: i for i, r in enumerate(rels)}

    rows = np.array([v["алкоголь"], v["алкоголь"], v["йога"], v["человек"]])
    cols = np.array([v["спирт"], v["алкоголизм"], v["йоги"], v["зверь"]])
    vals = np.array([rv["Synonym"], rv["RelatedTo"], rv["FormOf"], rv["DistinctFrom"]])

    tensor = Sparse3DTensor(
        (len(v), len(v), len(rv)),
        ((rows, cols), vals),
    )
    return Bundle(tensor, v, rv)


@pytest.fixture(scope="session")
def bundle_file(tmp_path_factory) -> Path:
    """Write the minimal bundle to a temp bz2 file and return its path."""
    bundle = _make_minimal_bundle()
    p = tmp_path_factory.mktemp("data") / "test.pickle.bz2"
    with bz2.open(p, "wb") as fh:
        pickle.dump(bundle, fh)
    return p


@pytest.fixture(scope="session")
def cn(bundle_file):
    """Return a ConceptNet instance loaded from the minimal fixture bundle."""
    from ruconceptnet.conceptnet import ConceptNet

    return ConceptNet(filepath=str(bundle_file))
