import logging
from functools import lru_cache

import pandas as pd
from tqdm import tqdm

from ruconceptnet.conceptnet import Bundle
from ruconceptnet.sparse_representation import Sparse3DTensor


def assertions2triples(filepath="data/russian-conceptnet.tsv"):
    df_data = {"relation": [], "source": [], "target": []}

    @lru_cache(100000000)
    def name(x: str) -> tuple[str, str]:
        s = x.split("/")
        return s[2], s[3]

    with open(filepath, encoding="utf-8") as fh:
        for line in tqdm(fh, "Reading raw ConceptNet triples"):
            s = line.strip().split("\t")
            relation = s[1].split("/")[-1]
            lang_s, source = name(s[2])
            lang_t, target = name(s[3])

            df_data["relation"].append(relation)
            df_data["source"].append(source)
            df_data["target"].append(target)

    return pd.DataFrame(df_data)


def triples2bundle(data: pd.DataFrame):
    logging.debug("Triplets read. " + str(data.shape))

    v = {k: i for i, k in enumerate(set(pd.concat([data["source"], data["target"]])))}
    rv = {rel: i for i, rel in enumerate(set(data["relation"]))}
    logging.debug(f"Vocabularies constructed. Words: {len(v)} Relations types: {len(rv)}")

    rows_idx = data["source"].map(lambda x: v[x]).values
    cols_idx = data["target"].map(lambda x: v[x]).values
    vals = data["relation"].map(lambda x: rv[x]).values

    tensor = Sparse3DTensor((len(v), len(v), len(rv)), ((rows_idx, cols_idx), vals))

    return Bundle(tensor, v, rv)


if __name__ == "__main__":
    import bz2
    import pickle

    df_nice_triples = assertions2triples(filepath="ruconceptnet/data/russian-conceptnet.tsv")
    bundle = triples2bundle(df_nice_triples)

    with bz2.open("ruconceptnet/data/russian-conceptnet.pickle.bz2", "wb") as wf:
        pickle.dump(bundle, wf)

    print("All set up.")
