# coding: utf-8
import pickle
import bz2
from collections import defaultdict

from compressor import Bundle
from compressor import Sparse3DTensor


class ConceptNet(object):

    def __init__(self, filepath="data/russian-conceptnet.pickle.bz2"):

        print("Reading archive...")

        with bz2.open(filepath, "rb") as rf:
            bundle = pickle.load(rf)  # type: Bundle

        self.v = bundle.v # type: dict
        self.rv = bundle.rv # type: dict
        self.iv = {v: k for k, v in self.v.items()}
        self.irv = {v: k for k, v in self.rv.items()}
        print("Vocabularies constructed.")

        self.tensor = bundle.t  # type: Sparse3DTensor

        print("All set up.")

    def __get_stuff__(self, method, s):
        if s in self.v:
            targets, relations = method(self.v[s])
            grouped = defaultdict(lambda: [])

            for t, r in zip(targets, relations):
                grouped[self.iv[t]].append(self.irv[r])

            return [(t, set(r)) for t, r in grouped.items()]
        else:
            return []

    def get_targets(self, s):
        return self.__get_stuff__(self.tensor.row_nz, s)

    def get_sources(self, s):
        return self.__get_stuff__(self.tensor.col_nz, s)

    def check_pair(self, s, t):

        if not s in self.v or not t in self.v:
            return []

        direct = self.tensor.rowcol_nz(self.v[s], self.v[t])
        direct = [self.irv[d] for d in direct]
        reverse = self.tensor.rowcol_nz(self.v[t], self.v[s])
        reverse = [self.irv[r] for r in reverse]

        return direct, reverse


if __name__ == "__main__":
    cn = ConceptNet()
    print(cn.get_targets("красота"))
    print(cn.get_sources("красота"))
    print(cn.check_pair("человек", "зверь"))
    print(cn.check_pair("зверь", "человек"))