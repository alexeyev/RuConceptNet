from __future__ import annotations

import json
import logging
from functools import cache

import pandas as pd
from tqdm import tqdm

from ruconceptnet.conceptnet import Bundle
from ruconceptnet.sparse_representation import Sparse3DTensor

logger = logging.getLogger(__name__)


def assertions2triples(filepath: str = "data/russian-conceptnet.tsv") -> pd.DataFrame:
    df_data: dict[str, list] = {"relation": [], "source": [], "target": [], "weight": []}

    @cache
    def name(uri: str) -> tuple[str, str]:
        parts = uri.split("/")
        return parts[2], parts[3]

    with open(filepath, encoding="utf-8") as fh:
        for line in tqdm(fh, "Reading raw ConceptNet triples"):
            row = line.strip().split("\t")
            relation = row[1].split("/")[-1]
            _, source = name(row[2])
            _, target = name(row[3])

            weight = 1.0
            if len(row) > 4 and row[4]:
                try:
                    weight = float(json.loads(row[4]).get("weight", 1.0))
                except (ValueError, TypeError, json.JSONDecodeError):
                    weight = 1.0

            df_data["relation"].append(relation)
            df_data["source"].append(source)
            df_data["target"].append(target)
            df_data["weight"].append(weight)

    return pd.DataFrame(df_data)


def triples2bundle(data: pd.DataFrame) -> Bundle:
    logger.debug("Triplets read. Shape: %s", data.shape)

    if "weight" not in data.columns:
        data = data.assign(weight=1.0)

    # Collapse identical (source, target, relation) assertions, keeping the
    # strongest weight, so every tensor cell holds exactly one value.
    data = data.groupby(["source", "target", "relation"], as_index=False)["weight"].max()

    v = {k: i for i, k in enumerate(set(pd.concat([data["source"], data["target"]])))}
    rv = {rel: i for i, rel in enumerate(set(data["relation"]))}
    logger.debug("Vocabularies constructed. Words: %d Relation types: %d", len(v), len(rv))

    rows_idx = data["source"].map(lambda x: v[x]).to_numpy()
    cols_idx = data["target"].map(lambda x: v[x]).to_numpy()
    depth = data["relation"].map(lambda x: rv[x]).to_numpy()
    weights = data["weight"].to_numpy()

    tensor = Sparse3DTensor(
        (len(v), len(v), len(rv)), ((rows_idx, cols_idx), depth), weights=weights
    )

    return Bundle(tensor, v, rv)


if __name__ == "__main__":
    import bz2
    import pickle

    logging.basicConfig(level=logging.DEBUG)

    triples = assertions2triples(filepath="ruconceptnet/data/russian-conceptnet.tsv")
    bundle = triples2bundle(triples)

    with bz2.open("ruconceptnet/data/russian-conceptnet.pickle.bz2", "wb") as wf:
        pickle.dump(bundle, wf)

    print("All set up.")
