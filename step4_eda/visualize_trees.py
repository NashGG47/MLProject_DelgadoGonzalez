"""Tree visualisation - article-ready (panel labels A/B, no embedded titles)
step4_eda/visualize_trees.py
-------------------------------------------------
Creates a two-panel PNG for a given sentence:
  • Panel A = unrooted graph (IDs permuted)
  • Panel B = same graph with root highlighted in red
Figures saved at 500-dpi. Captions belong to the manuscript.
Usage:
    python step4_eda/visualize_trees.py 62 ar   # sentence 62, language 'ar'
If arguments are omitted, defaults to sentence 0.
"""

# --- path setup ----------------------------------------------------
import sys, pathlib
sys.path.append(str(pathlib.Path(_file_).resolve().parents[1]))
# ------------------------------------------------------------------

import pathlib, pickle, argparse
import matplotlib.pyplot as plt, networkx as nx, pandas as pd

ROOT = pathlib.Path(_file_).resolve().parents[1]
PRE  = ROOT / "step1_preprocess"
DATA = ROOT / "data"

# ---------- CLI arguments -----------------------------------------
argp = argparse.ArgumentParser()
argp.add_argument("sentence_id", nargs="?", type=int, default=0,
                  help="Row index of the sentence in train.csv (default=0)")
argp.add_argument("language", nargs="?", type=str, default=None,
                  help="ISO language code to double-check the row")
args = argp.parse_args()

# ---------- load graph and metadata ------------------------------
with open(PRE / "train_graphs.pkl", "rb") as f:
    graphs = pickle.load(f)
G = graphs[args.sentence_id]

train_df = pd.read_csv(DATA / "train.csv")
row = train_df.iloc[args.sentence_id]

if args.language and row["language"] != args.language:
    print("⚠ requested language does not match this row →", row["language"])

root_id = row["root"]

# ---------- layout & drawing -------------------------------------
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(8,4))

# Panel A: unrooted graph
axA = plt.subplot(1,2,1)
nx.draw_networkx(G, pos, with_labels=True, node_size=300)
axA.axis("off")
axA.text(-0.05, 1.05, "A", transform=axA.transAxes, fontweight="bold", fontsize=14)

# Panel B: rooted graph (root in red)
axB = plt.subplot(1,2,2)
colors = ["red" if n == root_id else "skyblue" for n in G.nodes()]
nx.draw_networkx(G, pos, node_color=colors, with_labels=True, node_size=300)
axB.axis("off")
axB.text(-0.05, 1.05, "B", transform=axB.transAxes, fontweight="bold", fontsize=14)

plt.tight_layout()
OUT_DIR = ROOT / "step4_eda" / "trees"; OUT_DIR.mkdir(exist_ok=True)
outfile = OUT_DIR / f"tree_sent{args.sentence_id}.png"
plt.savefig(outfile, dpi=500)
plt.close()
print(" saved", outfile)