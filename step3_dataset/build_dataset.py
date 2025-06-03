"""Intra-sentence normalization
Auto-generated template. Fill in with the step logic.

step3_dataset/build_dataset.py
-------------------------------------------------
Normalizes each centrality within its sentence
and saves train_scaled.csv / test_scaled.csv."""""

# -------------Paths------------------
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
# ------------------------------------

import pathlib, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
FEA  = ROOT / "step2_features"
OUT  = ROOT / "step3_dataset"; OUT.mkdir(exist_ok=True)

train = pd.read_csv(FEA / "train_node_features.csv")
test  = pd.read_csv(FEA / "test_node_features.csv")

# --------- features columns ------------------------------
XCOLS = ["deg", "harm", "betw", "pager", "evec", "close", "clust"]

def scale_group(g):
    scaler = StandardScaler()
    g[XCOLS] = scaler.fit_transform(g[XCOLS])
    return g

train = train.groupby("sentence_id", group_keys=False).apply(scale_group)
test  = test.groupby("sentence_id", group_keys=False).apply(scale_group)

train.to_csv(OUT / "train_scaled.csv", index=False)
test.to_csv(OUT / "test_scaled.csv",  index=False)
joblib.dump(XCOLS, OUT / "feature_cols.pkl")

print(" Scaled datasets and save with features.")
