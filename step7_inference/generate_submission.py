"""
Generar submission.csv (corregido explícitamente para Top1)
"""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd, numpy as np, joblib, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
XCOLS = joblib.load(ROOT / "step3_dataset" / "feature_cols.pkl")
TEST = pd.read_csv(ROOT / "step3_dataset" / "test_scaled.csv")

# ---------- cargar modelo top1 + umbral explícitamente --------------------
top1_pkl = next(ROOT.glob("step5_modeling/top1_*.pkl"))
name = top1_pkl.stem.replace("top1_", "")
MODEL = joblib.load(top1_pkl)
best_thr = joblib.load(ROOT / "step5_modeling" / "thresholds.pkl")[name]

# Scores y predicciones
if hasattr(MODEL, "predict_proba"):
    scores = MODEL.predict_proba(TEST[XCOLS])[:, 1]
else:
    scores = MODEL.decision_function(TEST[XCOLS])
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

TEST["score"] = scores

def choose_root(g):
    above = g[g["score"] >= best_thr]
    winner = above if not above.empty else g
    return winner.sort_values("score", ascending=False).iloc[0]

roots = TEST.groupby("sentence_id", group_keys=False).apply(choose_root).reset_index(drop=True)[["sentence_id", "node_id"]]

# Mapear IDs reales
TEST_RAW = pd.read_csv(ROOT / "data" / "test.csv", usecols=["id"])
id_map = TEST_RAW.reset_index().rename(columns={"index": "sentence_id"})
submission = roots.merge(id_map, on="sentence_id", how="left").rename(columns={"node_id": "root"})[["id", "root"]]

OUT = ROOT / "step7_inference"; OUT.mkdir(exist_ok=True)
submission.to_csv(OUT / "submission.csv", index=False)

print(f"✓ submission.csv generado con modelo {name} y umbral {best_thr:.2f}")
