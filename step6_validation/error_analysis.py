"""Error analysis with OOF predictions - article style
step6_validation/error_analysis.py
-------------------------------------------------
For each model present in *train_oof_predictions.csv* (those that have a
threshold in *thresholds.pkl*) this script outputs:
  • confusion_matrix_<model>.png      - heat-map, no title
  • score_distribution_<model>.png    - histograms by class, threshold marked
Additionally, for the best model (first in `thresholds.pkl`):
  • error_cases.csv                   - 20 sentences with highest error rate
  • f1_vs_size.png                    - F1 vs average degree
  • accuracy_vs_size.png              - Precision vs average degree
  • error_heatmap_top20.png           - Top-20 sentences by error rate
All figures at 500-dpi, no embedded titles. Compatible with pandas ≤-2.x
"""

# --- path setup ----------------------------------------------------
import sys, pathlib, warnings
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
# ------------------------------------------------------------------

import pathlib, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT   = pathlib.Path(__file__).resolve().parents[1]
OOF    = pd.read_csv(ROOT / "step5_modeling" / "train_oof_predictions.csv")
DATA   = pd.read_csv(ROOT / "step3_dataset" / "train_scaled.csv")
FEAT   = pd.read_csv(ROOT / "step2_features" / "train_node_features.csv")
THR = joblib.load(ROOT / "step5_modeling" / "all_thresholds.pkl")
OUT_DIR = ROOT / "step6_validation"; OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------
for m_idx, (model_name, thr) in enumerate(THR.items(), start=1):
    col = f"{model_name}_oof"
    if col not in OOF.columns:
        continue  # skip if OOF column missing

    scores = OOF[col]
    DATA["pred"] = (scores >= thr).astype(int)

    # 1. Confusion matrix (heat‑map)
    cm = confusion_matrix(DATA["is_root"], DATA["pred"])
    plt.figure(); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(OUT_DIR / f"confusion_matrix_{model_name}.png", dpi=500); plt.close()

    # 2. Score distribution by class
    plt.figure(figsize=(6,4))
    sns.histplot(x=scores, hue=DATA["is_root"], bins=50, kde=True,
                 palette=["#d95f02", "#1b9e77"])
    plt.axvline(thr, color="black", linestyle="--")
    plt.xlabel("Model score"); plt.ylabel("Frequency")
    plt.tight_layout(); plt.savefig(OUT_DIR / f"score_distribution_{model_name}.png", dpi=500); plt.close()

    # Extended analysis only for the first (best) model
    if m_idx == 1:
        errors = DATA[DATA["pred"] != DATA["is_root"]].copy()
        errors.head(20).to_csv(OUT_DIR / "error_cases.csv", index=False)

        mean_deg = FEAT.groupby("sentence_id")["deg"].mean().reset_index(name="mean_deg")
        tmp = DATA.merge(mean_deg, on="sentence_id", how="left")

        f1_sent = tmp.groupby("sentence_id").apply(lambda g: f1_score(g["is_root"], g["pred"], zero_division=0))
        df_sz = pd.DataFrame({"size": tmp.groupby("sentence_id").first()["mean_deg"], "f1": f1_sent})
        bins = df_sz.groupby(pd.cut(df_sz["size"], bins=10, include_lowest=True)).mean()
        plt.figure(); plt.plot(bins.index.map(lambda x: x.mid), bins["f1"], marker="o")
        plt.xlabel("Average node degree"); plt.ylabel("F1")
        plt.tight_layout(); plt.savefig(OUT_DIR / "f1_vs_size.png", dpi=500); plt.close()

        prec_sent = tmp.groupby("sentence_id").apply(lambda g: precision_score(g["is_root"], g["pred"], zero_division=0))
        df_sz_acc = pd.DataFrame({"size": tmp.groupby("sentence_id").first()["mean_deg"], "precision": prec_sent})
        bins_acc = df_sz_acc.groupby(pd.cut(df_sz_acc["size"], bins=10, include_lowest=True)).mean()
        plt.figure(); plt.plot(bins_acc.index.map(lambda x: x.mid), bins_acc["precision"], marker="s", color="green")
        plt.xlabel("Average node degree"); plt.ylabel("Precision")
        plt.tight_layout(); plt.savefig(OUT_DIR / "accuracy_vs_size.png", dpi=500); plt.close()

        err_rate = tmp.groupby("sentence_id").apply(lambda g: (g["pred"] != g["is_root"]).mean())
        err_mat = err_rate.to_frame("error").sort_values("error", ascending=False).head(20)
        plt.figure(figsize=(10,2.5))
        sns.heatmap(err_mat.T, cmap="Reds", annot=True, fmt=".2f",
                    cbar_kws={"label": "Error rate"})
        plt.xlabel("sentence_id"); plt.yticks([])
        plt.tight_layout(); plt.savefig(OUT_DIR / "error_heatmap_top20.png", dpi=500); plt.close()

print(" Error analysis completed for models:", ", ".join(THR.keys()))
