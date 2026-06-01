from __future__ import annotations

import bz2
import logging
import pickle
from collections import defaultdict
from importlib.resources import as_file, files

from ruconceptnet.sparse_representation import Sparse3DTensor

logger = logging.getLogger(__name__)


class Bundle:
    """Minimal container for the ConceptNet data we provide access to."""

    def __init__(self, triplets: Sparse3DTensor, vocab: dict, rel_vocab: dict) -> None:
        self.t = triplets
        self.v = vocab
        self.rv = rel_vocab


class ConceptNet:
    def __init__(self, filepath: str | None = None) -> None:
        logger.debug("Reading archive...")

        if filepath is None:
            from . import data

            resource = files(data).joinpath("russian-conceptnet.pickle.bz2")
            with as_file(resource) as path, bz2.open(path, "rb") as rf:
                bundle = pickle.load(rf)
        else:
            with bz2.open(filepath, "rb") as rf:
                bundle = pickle.load(rf)

        self.v: dict = bundle.v
        self.rv: dict = bundle.rv
        self.iv = {v: k for k, v in self.v.items()}
        self.irv = {v: k for k, v in self.rv.items()}
        logger.debug("Vocabularies constructed.")

        self.tensor = bundle.t

        logger.debug("All set up.")

    def _get_related(
        self, method, s: str, with_weights: bool
    ) -> list[tuple[str, set[str] | dict[str, float]]]:
        if s not in self.v:
            return []

        nodes, relations, weights = method(self.v[s])

        if with_weights:
            grouped_w: defaultdict[str, dict[str, float]] = defaultdict(dict)
            for node, rel, weight in zip(nodes, relations, weights, strict=True):
                grouped_w[self.iv[node]][self.irv[rel]] = float(weight)
            return list(grouped_w.items())

        grouped: defaultdict[str, set[str]] = defaultdict(set)
        for node, rel in zip(nodes, relations, strict=True):
            grouped[self.iv[node]].add(self.irv[rel])
        return list(grouped.items())

    def get_targets(
        self, s: str, with_weights: bool = False
    ) -> list[tuple[str, set[str] | dict[str, float]]]:
        """Relations going *out* of ``s``.

        By default each item is ``(target, {relation, ...})``.  With
        ``with_weights=True`` each item is ``(target, {relation: weight, ...})``.
        """
        return self._get_related(self.tensor.row_nz, s, with_weights)

    def get_sources(
        self, s: str, with_weights: bool = False
    ) -> list[tuple[str, set[str] | dict[str, float]]]:
        """Relations pointing *into* ``s`` (see :meth:`get_targets` for the shape)."""
        return self._get_related(self.tensor.col_nz, s, with_weights)

    def check_pair(
        self, s: str, t: str, with_weights: bool = False
    ) -> tuple[list[str], list[str]] | tuple[dict[str, float], dict[str, float]]:
        """Relations between ``s`` and ``t`` as ``(direct, reverse)``.

        By default each side is a list of relations.  With ``with_weights=True``
        each side is a ``{relation: weight}`` mapping.
        """
        if s not in self.v or t not in self.v:
            return ({}, {}) if with_weights else ([], [])

        d_depths, d_weights = self.tensor.rowcol_nz(self.v[s], self.v[t])
        r_depths, r_weights = self.tensor.rowcol_nz(self.v[t], self.v[s])

        if with_weights:
            direct = {self.irv[d]: float(w) for d, w in zip(d_depths, d_weights, strict=True)}
            reverse = {self.irv[d]: float(w) for d, w in zip(r_depths, r_weights, strict=True)}
            return direct, reverse

        return [self.irv[d] for d in d_depths], [self.irv[d] for d in r_depths]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    cn = ConceptNet()
    print(cn.get_targets("алкоголь"))
    print(cn.get_targets("алкоголь", with_weights=True))
    print(cn.get_sources("йога"))
    print(cn.check_pair("человек", "зверь"))
    print(cn.check_pair("зверь", "человек", with_weights=True))
