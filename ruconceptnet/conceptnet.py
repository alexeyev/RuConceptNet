import bz2
import logging
import pickle
from collections import defaultdict

from ruconceptnet.sparse_representation import Sparse3DTensor


class Bundle:
    """Minimal object storing ConceptNet data we provide access to"""

    def __init__(self, triplets: Sparse3DTensor, vocab: dict, rel_vocab: dict):
        self.t = triplets
        self.v = vocab
        self.rv = rel_vocab


class ConceptNet:
    def __init__(self, filepath: str = None):

        logging.debug("Reading archive...")

        if filepath is None:
            try:
                import importlib.resources as pkg_resources
            except ImportError:
                # Trying backported to PY<37 `importlib_resources`.
                import importlib_resources as pkg_resources

            from . import data

            with (
                pkg_resources.path(data, "russian-conceptnet.pickle.bz2") as filepath,
                bz2.open(filepath, "rb") as rf,
            ):
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

    def __get_stuff__(self, method, s: str) -> list[tuple[str, set[str]]]:

        if s in self.v:
            targets, relations = method(self.v[s])
            grouped = defaultdict(lambda: [])

            for t, r in zip(targets, relations):
                grouped[self.iv[t]].append(self.irv[r])

            return [(t, set(r)) for t, r in grouped.items()]
        else:
            return []

    def get_targets(self, s: str) -> list[tuple[str, set[str]]]:
        return self.__get_stuff__(self.tensor.row_nz, s)

    def get_sources(self, s: str) -> list[tuple[str, set[str]]]:
        return self.__get_stuff__(self.tensor.col_nz, s)

    def check_pair(self, s: str, t: str) -> [list[str], list[str]]:

        if s not in self.v or t not in self.v:
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
