"""Descriptive statistics of raw data
🏗️ Auto-generated template – English, article-ready (no embedded figure titles).

step0_setup/descriptive_raw.py
---------------------------------
Generates:
  • raw_stats.csv          – basic statistics
  • lang_counts.png        – # sentences per language
  • tree_size_hist.png     – histogram of tree size
  • centrality_raw_box.png – preliminary box‑plots of 4 centralities
Run from project root:
    python step0_setup/descriptive_raw.py
"""

import pathlib, ast, pandas as pd, matplotlib.pyplot as plt, networkx as nx

ROOT = pathlib.Path(__file__).resolve().parents[1]   # ← ahora sí
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "step0_setup"; OUT_DIR.mkdir(exist_ok=True)

# ---------- load ----------------------------------------------------
train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")
df    = pd.concat([train.assign(split="train"), test.assign(split="test")])

# ---------- tree size ----------------------------------------------
df["tree_nodes"] = df["edgelist"].apply(lambda s: len(ast.literal_eval(s)) + 1)

# ---------- numeric stats ------------------------------------------
stats = df[["n", "tree_nodes"]].describe()
stats.to_csv(OUT_DIR / "raw_stats.csv")
print("✓ raw_stats.csv")

# ---------- language bar‑chart -------------------------------------
lang_counts = df["language"].value_counts().sort_index()
plt.figure(figsize=(10,4))
lang_counts.plot(kind="bar")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(OUT_DIR / "lang_counts.png", dpi=500)
plt.close()
print("✓ lang_counts.png")

# ---------- tree size histogram ------------------------------------
plt.figure()
df["tree_nodes"].hist(bins=30)
plt.xlabel("nodes per sentence")
plt.ylabel("frequency")
plt.tight_layout()
plt.savefig(OUT_DIR / "tree_size_hist.png", dpi=500)
plt.close()
print("✓ tree_size_hist.png")

# ---------- quick centralities (undirected graphs) -----------------
def quick_centralities(edges):
    G = nx.Graph(); G.add_edges_from(edges)
    harm = nx.harmonic_centrality(G)
    betw = nx.betweenness_centrality(G)
    pager = nx.pagerank(G, max_iter=500)
    deg = {v: G.degree[v]/(len(G)-1) for v in G}
    return pd.DataFrame({
        "deg":   pd.Series(deg),
        "harm":  pd.Series(harm),
        "betw":  pd.Series(betw),
        "pager": pd.Series(pager)
    })

sample = df.sample(5000, random_state=42)              # 5000‑sentence subsample
box_df = pd.concat(
    [quick_centralities(ast.literal_eval(s)).assign(sent=i)
     for i, s in zip(sample.index, sample["edgelist"])]
)
plt.figure(figsize=(8,4))
box_df.melt(id_vars="sent").boxplot(by="variable", column="value")
plt.suptitle("")  # remove overall title for article style
plt.tight_layout()
plt.savefig(OUT_DIR / "centrality_raw_box.png", dpi=500)
plt.close()
print("✓ centrality_raw_box.png")