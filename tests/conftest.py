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

# Edge weights, aligned with the edges created in _make_minimal_bundle.
EDGE_WEIGHTS = {
    ("алкоголь", "спирт", "Synonym"): 2.5,
    ("алкоголь", "алкоголизм", "RelatedTo"): 1.0,
    ("йога", "йоги", "FormOf"): 3.0,
    ("человек", "зверь", "DistinctFrom"): 0.5,
}


def _make_minimal_bundle() -> Bundle:
    """
    Build a tiny ConceptNet bundle with four Russian concepts and three
    relation types.  The relations encoded are:

        алкоголь  --Synonym-->    спирт          (w=2.5)
        алкоголь  --RelatedTo-->  алкоголизм     (w=1.0)
        йога      --FormOf-->     йоги           (w=3.0)
        человек   --DistinctFrom--> зверь        (w=0.5)
    """
    words = ["алкоголь", "спирт", "алкоголизм", "йога", "йоги", "человек", "зверь"]
    rels = ["Synonym", "RelatedTo", "FormOf", "DistinctFrom"]

    v = {w: i for i, w in enumerate(words)}
    rv = {r: i for i, r in enumerate(rels)}

    rows = np.array([v["алкоголь"], v["алкоголь"], v["йога"], v["человек"]])
    cols = np.array([v["спирт"], v["алкоголизм"], v["йоги"], v["зверь"]])
    vals = np.array([rv["Synonym"], rv["RelatedTo"], rv["FormOf"], rv["DistinctFrom"]])
    weights = np.array([2.5, 1.0, 3.0, 0.5])

    tensor = Sparse3DTensor(
        (len(v), len(v), len(rv)),
        ((rows, cols), vals),
        weights=weights,
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
