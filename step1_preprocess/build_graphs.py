"""Randomization + ND graphs
Auto-generated template. Fill in with the step logic.
""
""
step1_preprocess/build_graphs.py
--------------------------------------------------
• Read train.csv / test.csv
• Apply anti-leakage randomization and create undirected graphs
• Save pickles train_graphs.pkl and test_graphs.pkl
"""

# --- Paths ---------------------------------
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
# ---------------------------------------------------------------

import pathlib, pickle, pandas as pd
from common.io_utils import detect_columns, parse_graph

ROOT  = pathlib.Path(__file__).resolve().parents[1]
DATA  = ROOT / "data"
OUT   = ROOT / "step1_preprocess"; OUT.mkdir(exist_ok=True)

for split in ["train", "test"]:
    df = pd.read_csv(DATA / f"{split}.csv")
    edge_col, _ = detect_columns(df)

    graphs = []
    for sent_id, row in df.iterrows():
        G = parse_graph(row[edge_col])            # randomize + graph ND
        G.graph["sentence_id"] = sent_id
        graphs.append(G)

    with open(OUT / f"{split}_graphs.pkl", "wb") as f:
        pickle.dump(graphs, f)

    print(f"{split}: {len(graphs)} graphs saved.")
