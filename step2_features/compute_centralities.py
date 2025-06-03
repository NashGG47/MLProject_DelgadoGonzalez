"""Calculate 7 centralities
 Auto-generated template. Fill in with the step logic.
"""
"""
step2_features/compute_centralities.py
--------------------------------------------------
Calculate 7 centralities per node:
deg, harm, betw, pager, evec, close, clust
and create ➜ train_node_features.csv
➜ test_node_features.csv
"""
# --- Paths ---------------------------------
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
# ---------------------------------------------------------------

import pathlib, pickle, pandas as pd, networkx as nx
from tqdm import tqdm
from common.io_utils import detect_columns  # utility to detect columns

ROOT    = pathlib.Path(__file__).resolve().parents[1]
PRE_DIR = ROOT / "step1_preprocess"
OUT_DIR = ROOT / "step2_features"; OUT_DIR.mkdir(exist_ok=True)

# ------------- Aux functions --------------------------------
def graph_to_df(G: nx.Graph, label_dict=None):
    sent = G.graph["sentence_id"]

    # --- centralities (per grafo) -------------
    deg   = {v: G.degree[v] / (len(G)-1)           for v in G}
    harm  = nx.harmonic_centrality(G)
    betw  = nx.betweenness_centrality(G)
    pager = nx.pagerank(G, max_iter=500)
    evec  = nx.eigenvector_centrality_numpy(G)      # spectral
    close = nx.closeness_centrality(G)              # conventional
    clust = nx.clustering(G)                        # coef. clustering

    rows = []
    for v in G.nodes():
        rows.append({
            "sentence_id": sent,
            "node_id": v,
            "deg":   deg[v],
            "harm":  harm[v],
            "betw":  betw[v],
            "pager": pager[v],
            "evec":  evec[v],
            "close": close[v],
            "clust": clust[v],
            "is_root": int(v == label_dict[sent]) if label_dict else None,
        })
    return pd.DataFrame(rows)

# ------------- bucle train / test -------------------------------
for split in ["train", "test"]:
    graphs = pickle.load(open(PRE_DIR / f"{split}_graphs.pkl", "rb"))

    label_dict = None
    if split == "train":
        # assign is_root in training
        train_df = pd.read_csv(ROOT / "data" / "train.csv")
        _, root_col = detect_columns(train_df)
        label_dict = train_df[root_col].to_dict()

    df = pd.concat([graph_to_df(G, label_dict) for G in tqdm(graphs)],
                   ignore_index=True)

    df.to_csv(OUT_DIR / f"{split}_node_features.csv", index=False)
    print(f"{split:5s}: {len(df)} filas guardadas con 7 centralidades.")
