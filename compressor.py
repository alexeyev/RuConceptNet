# coding: utf-8

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from tqdm import tqdm


class Bundle(object):
    def __init__(self, triplets, vocab, rel_vocab):
        self.t = triplets
        self.v = vocab
        self.rv = rel_vocab


class Sparse3DTensor(object):
    """
        Unrolling depths as rows using scipy's sparse matrix;
        could be compressed even more since our case is a binary matrix one,
        but I don't care
    """

    def __init__(self, shape, triples):
        assert len(shape) == 3
        matrix_shape = (shape[0], shape[1] * shape[2])
        self.step = shape[2]
        (rows_idx, cols_idx), vals = triples
        new_cols_idx = cols_idx * self.step + vals
        self.t = coo_matrix(([1.] * rows_idx.shape[0], (rows_idx, new_cols_idx)), shape=matrix_shape).tocsr()

    def __getitem__(self, indices):
        row, col, dep = indices
        return self.t[row, col * self.step + dep]

    def __setitem__(self, indices, value):
        row, col, dep = indices
        self.t[row, col * self.step + dep] = value

    def __str__(self):
        return str(self.t.todense())

    def row_nz(self, ix):
        _, nz = self.t[ix].nonzero()
        row_indices = nz // self.step
        depth_indices = nz % self.step
        return row_indices, depth_indices

    def col_nz(self, ix):
        nz = self.t[:, ix * self.step:(ix + 1) * self.step].nonzero()
        return nz

    def rowcol_nz(self, row, col):
        return self.t[row, col * self.step:(col + 1) * self.step].nonzero()[1]


# rows = np.array([1, 1, 1, 1])
# cols = np.array([0, 1, 0, 0])
# vals = np.array([2, 2, 1, 0])
#
# st = Sparse3DTensor((2, 2, 3), ((rows, cols), vals))
#
# print(st.row_nz(0))
# print(st.row_nz(1))
# print(st.col_nz(0))
# print(st.col_nz(1))
# print(st.rowcol_nz(1, 0))


def assertions2triples(filepath="data/russian-conceptnet.tsv"):
    df_data = {"relation": [], "source": [], "target": []}

    @lru_cache(100000000)
    def name(x):
        s = x.split("/")
        return s[2], s[3]

    for line in tqdm(open(filepath, "r+", encoding="utf-8"), "Reading raw ConceptNet triples"):
        s = line.strip().split("\t")
        relation = s[1].split("/")[-1]
        lang_s, source = name(s[2])
        lang_t, target = name(s[3])

        df_data["relation"].append(relation)
        df_data["source"].append(source)
        df_data["target"].append(target)

    return pd.DataFrame(df_data)


def triples2bundle(data: pd.DataFrame):

    print("Triplets read.", data.shape)

    v = {k: i for i, k in enumerate(set(pd.concat([data["source"], data["target"]])))}
    rv = {rel: i for i, rel in enumerate(set(data["relation"]))}
    print("Vocabularies constructed. Words:", len(v), "Relations types:", len(rv))

    rows_idx = data["source"].map(lambda x: v[x]).values
    cols_idx = data["target"].map(lambda x: v[x]).values
    vals = data["relation"].map(lambda x: rv[x]).values

    print(rows_idx.shape, cols_idx.shape, vals.shape)

    tensor = Sparse3DTensor((len(v), len(v), len(rv)), ((rows_idx, cols_idx), vals))

    return Bundle(tensor, v, rv)


if __name__ == "__main__":
    import pickle

    df_nice_triples = assertions2triples(filepath="data/russian-conceptnet.tsv")
    bundle = triples2bundle(df_nice_triples)

    with open("data/russian-conceptnet.pickle", "wb") as wf:
        pickle.dump(bundle, wf)

    print("All set up.")
