# -*- coding: utf-8 -*-

import bz2
import logging
import pickle
from collections import defaultdict
from typing import List, Set, Tuple

from ruconceptnet.sparse_representation import Sparse3DTensor


class Bundle(object):
    """ Minimal object storing ConceptNet data we provide access to """

    def __init__(self, triplets: Sparse3DTensor, vocab: dict, rel_vocab: dict):
        self.t = triplets
        self.v = vocab
        self.rv = rel_vocab


class ConceptNet(object):

    def __init__(self, filepath: str = None):

        logging.debug("Reading archive...")

        if filepath is None:

            try:
                import importlib.resources as pkg_resources
            except ImportError:
                # Trying backported to PY<37 `importlib_resources`.
                import importlib_resources as pkg_resources

            from . import data

            with pkg_resources.path(data, "russian-conceptnet.pickle.bz2") as filepath:
                with bz2.open(filepath, "rb") as rf:
                    bundle = pickle.load(rf)
        else:
            with bz2.open(filepath, "rb") as rf:
                bundle = pickle.load(rf)

        self.v: dict = bundle.v
        self.rv: dict = bundle.rv
        self.iv = {v: k for k, v in self.v.items()}
        self.irv = {v: k for k, v in self.rv.items()}
        logging.debug("Vocabularies constructed.")

        self.tensor = bundle.t

        logging.debug("All set up.")

    def __get_stuff__(self, method, s: str) -> List[Tuple[str, Set[str]]]:

        if s in self.v:
            targets, relations = method(self.v[s])
            grouped = defaultdict(lambda: [])

            for t, r in zip(targets, relations):
                grouped[self.iv[t]].append(self.irv[r])

            return [(t, set(r)) for t, r in grouped.items()]
        else:
            return []

    def get_targets(self, s: str) -> List[Tuple[str, Set[str]]]:
        return self.__get_stuff__(self.tensor.row_nz, s)

    def get_sources(self, s: str) -> List[Tuple[str, Set[str]]]:
        return self.__get_stuff__(self.tensor.col_nz, s)

    def check_pair(self, s: str, t: str) -> [List[str], List[str]]:

        if not s in self.v or not t in self.v:
            return [], []

        direct = self.tensor.rowcol_nz(self.v[s], self.v[t])
        direct = [self.irv[d] for d in direct]
        reverse = self.tensor.rowcol_nz(self.v[t], self.v[s])
        reverse = [self.irv[r] for r in reverse]

        return direct, reverse


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    cn = ConceptNet()
    print(cn.get_targets("алкоголь"))
    print(cn.get_sources("йога"))
    print(cn.check_pair("человек", "зверь"))
    print(cn.check_pair("зверь", "человек"))
