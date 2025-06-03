"""
PR curve and threshold-vs-F1
Adapted to use honest predictions (OOF).
"""

import sys, pathlib, joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from sklearn.metrics import precision_recall_curve

ROOT   = pathlib.Path(__file__).resolve().parents[1]
OOF_DATA = pd.read_csv(ROOT / "step5_modeling" / "train_oof_predictions.csv")
thr_dict = joblib.load(ROOT / "step5_modeling" / "all_thresholds.pkl")

for col in OOF_DATA.columns:
    if col in ["sentence_id", "is_root"]:
        continue

    model_name = col.replace("_oof", "")
    s = OOF_DATA[col]

    prec, rec, thr = precision_recall_curve(OOF_DATA["is_root"], s)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_thr = thr_dict.get(model_name)

    plt.figure()
    plt.plot(rec, prec)
    idx = np.argmin(np.abs(thr - best_thr))
    plt.scatter(rec[idx], prec[idx], color="red")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR-curve {model_name}")
    plt.tight_layout()
    plt.savefig(ROOT / "step5_modeling" / f"pr_curve_{model_name}.png")
    plt.close()

    plt.figure()
    plt.plot(thr, f1[:-1])
    plt.axvline(best_thr, color="red")
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.title(f"F1 vs threshold {model_name}")
    plt.tight_layout()
    plt.savefig(ROOT / "step5_modeling" / f"thr_vs_f1_{model_name}.png")
    plt.close()

print(" PR and thr-F1 curves generated for models with OOF predictions.")
