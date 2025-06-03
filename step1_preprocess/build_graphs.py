"""Randomización + grafos ND
🏗️  Plantilla autogenerada. Rellena con la lógica del paso.
"""
"""
step1_preprocess/build_graphs.py
---------------------------------------------------
• Lee train.csv / test.csv
• Aplica randomización anti-fuga y crea grafos NO dirigidos
• Guarda pickles  train_graphs.pkl  y  test_graphs.pkl
"""

# --- 2 líneas mágicas de ruta ---------------------------------
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
        G = parse_graph(row[edge_col])            # aleatoriza + grafo ND
        G.graph["sentence_id"] = sent_id
        graphs.append(G)

    with open(OUT / f"{split}_graphs.pkl", "wb") as f:
        pickle.dump(graphs, f)

    print(f"{split}: {len(graphs)} grafos guardados → sin fuga.")
