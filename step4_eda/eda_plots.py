"""EDA figures – article style (percentages shown where appropriate)
step4_eda/eda_plots.py
---------------------------------
Generates:
  • deg_box.png … clust_box.png  (7 box-plots)
  • class_balance.png             (class distribution in %)
  • corr_heatmap.png              (feature correlation)
Figures saved at 500 dpi, no embedded titles.
Run from project root:
    python step4_eda/eda_plots.py
"""

# --- path magic ----------------------------------------------------
import sys, pathlib
sys.path.append(str(pathlib.Path(_file_).resolve().parents[1]))
# ------------------------------------------------------------------

import pathlib, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

ROOT = pathlib.Path(_file_).resolve().parents[1]
OUT  = ROOT / "step4_eda"; OUT.mkdir(exist_ok=True)

# ---------- read data ---------------------------------------------
data = pd.read_csv(ROOT / "step3_dataset" / "train_scaled.csv")
COLS = ["deg", "harm", "betw", "pager", "evec", "close", "clust"]

# ---------- 1. centrality box-plots -------------------------------
for col in COLS:
    plt.figure(figsize=(5,5))
    sns.boxplot(data=data, x="is_root", y=col)
    plt.xlabel("Node type (0 = non-root, 1 = root)")
    plt.ylabel(f"{col} (z-score)")
    plt.tight_layout()
    plt.savefig(OUT / f"{col}_box.png", dpi=500)
    plt.close()

print("✓ 7 box-plots saved (article style)")

# ---------- 2. class balance – percentage bar --------------------
plt.figure(figsize=(4,5))
counts = data["is_root"].value_counts().sort_index()
percent = counts / counts.sum() * 100
colors = ["#56B4E9", "#E69F00"]  # skyblue, orange
bars = plt.bar([0,1], percent, color=colors)
plt.xticks([0,1], ["Non-root", "Root"], rotation=0)
plt.ylabel("Percentage of nodes (%)")
plt.xlabel("Class")
# annotate percentage values
for bar, pct in zip(bars, percent):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1, f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(OUT / "class_balance.png", dpi=500)
plt.close()
print("✓ class_balance.png saved (percentages)")

# ---------- 3. correlation heat-map ------------------------------
plt.figure(figsize=(8,6))
corr = data[COLS].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", cbar_kws={"label": "Pearson r"})
plt.tight_layout()
plt.savefig(OUT / "corr_heatmap.png", dpi=500)
plt.close()
print("✓ corr_heatmap.png saved at 500 dpi")