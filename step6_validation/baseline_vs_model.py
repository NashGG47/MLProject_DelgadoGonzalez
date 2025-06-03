"""
Baseline PageRank vs. Model
Auto-generated template. Fill in the step logic.

step6_validation/baseline_vs_model.py
------------------------------------------------
Compare: baseline PageRank + top1 + top2
Save:
• comparison.csv - predictions from all three approaches
• metrics.txt - table with F1 and Balanced Accuracy
"""""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pathlib, joblib, pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score

ROOT  = pathlib.Path(__file__).resolve().parents[1]
OOF_DATA  = pd.read_csv(ROOT / "step5_modeling" / "train_oof_predictions.csv")
thr_dict = joblib.load(ROOT / "step5_modeling" / "all_thresholds.pkl")
DATA = pd.read_csv(ROOT / "step3_dataset" / "train_scaled.csv")

# ---------- baseline (max-PageRank) ------------
def baseline_preds(df):
    out = pd.Series(0, index=df.index)
    for _, g in df.groupby("sentence_id"):
        out.loc[g["pager"].idxmax()] = 1
    return out
DATA["pred_base"] = baseline_preds(DATA)

# Honest predictions top 1 and top 2
for model_name in thr_dict.keys():
    thr = thr_dict[model_name]
    DATA[f"pred_{model_name}"] = (OOF_DATA[f"{model_name}_oof"] >= thr).astype(int)

# ---------- metrics ---------------------------
rows = []
for col in ["pred_base"] + [f"pred_{name}" for name in thr_dict.keys()]:
    f1  = f1_score(DATA["is_root"], DATA[col])
    bal = balanced_accuracy_score(DATA["is_root"], DATA[col])
    rows.append([col.replace("pred_",""), f1, bal])

MET = pd.DataFrame(rows, columns=["model","f1","bal_acc"])
MET.to_csv(ROOT / "step6_validation" / "metrics.txt", sep="\t", index=False)

# ---------- save comparison ----------------
OUT = ROOT / "step6_validation"; OUT.mkdir(exist_ok=True)
keep_cols = ["sentence_id","node_id","is_root","pred_base"] + [f"pred_{name}" for name in thr_dict.keys()]
DATA[keep_cols].to_csv(OUT / "comparison.csv", index=False)

print(MET.to_string(index=False))
print(" comparison.csv saved using honest predictions")
