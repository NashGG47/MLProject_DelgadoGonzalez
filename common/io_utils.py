"""Utilidades de I/O y randomización
🏗️  Plantilla autogenerada. Rellena con la lógica del paso.
"""
"""
common/io_utils.py
---------------------------------------------------
Utilidades de entrada-salida compartidas
• detect_columns(df)   → detecta 'edgelist*' y 'root'
• parse_graph(str)     → randomiza lista y pares, devuelve nx.Graph
"""

import ast, random, numpy as np, networkx as nx

RANDOM_STATE = 42  # reproducibilidad global
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ---------- detección flexible de columnas --------------------
def detect_columns(df):
    edge_col = next(c for c in df.columns if c.startswith("edgelist"))
    root_col = "root" if "root" in df.columns else None
    return edge_col, root_col

# ---------- randomización + construcción de grafo -------------
def parse_graph(row_edges, shuffle=True, flip_prob=0.5):
    """
    1. Convierte la cadena '[ (a,b), ... ]' → lista de tuplas
    2. Baraja la lista (anti-fuga A-1)
    3. Con prob flip_prob invierte cada par (a,b)→(b,a) (A-2)
    4. Devuelve nx.Graph()  NO dirigido   (G3)
    """
    edges = list(ast.literal_eval(row_edges))

    if shuffle:
        random.shuffle(edges)                     # A-1

    edges = [(b, a) if np.random.rand() < flip_prob else (a, b)
             for a, b in edges]                   # A-2

    G = nx.Graph()
    G.add_edges_from(edges)
    return G
