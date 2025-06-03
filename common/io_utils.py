"""I/O Utilities and Randomization
Auto-generated template. Fill in the step logic.

common/io_utils.py
---------------------------------------------------
Shared Input-Output Utilities
• detect_columns(df) → detects 'edgelist*' and 'root'
• parse_graph(str) → randomizes list and pairs, returns nx.Graph
"""

import ast, random, numpy as np, networkx as nx

RANDOM_STATE = 42  # global reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ---------- flexible column detection --------------------
def detect_columns(df):
    edge_col = next(c for c in df.columns if c.startswith("edgelist"))
    root_col = "root" if "root" in df.columns else None
    return edge_col, root_col

# ---------- randomization + graph construction -------------
def parse_graph(row_edges, shuffle=True, flip_prob=0.5):
    """
    1. Convert the string '[ (a,b), ... ]' → list of tuples
    2. Shuffle the list (A-1 anti-leakage)
    3. Flip each pair (a,b) → (b,a) with prob flip_prob (A-2)
    4. Return an undirected nx.Graph() (G3)
    """
    edges = list(ast.literal_eval(row_edges))

    if shuffle:
        random.shuffle(edges)                     # A-1

    edges = [(b, a) if np.random.rand() < flip_prob else (a, b)
             for a, b in edges]                   # A-2

    G = nx.Graph()
    G.add_edges_from(edges)
    return G
