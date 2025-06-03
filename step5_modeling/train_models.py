"""
Multi-model training
Auto-generated template corrected with explicit OOF predictions.
"""
"""
step5_modeling/train_models.py
------------------------------------------------------------------
• Runs GridSearchCV with GroupKFold (15 folds) on 8 classical models.
• Uses the F1 metric as the primary score in cross-validation.
• Calculates out-of-fold predictions to honestly estimate:
- F1 score
- Balanced Accuracy
- Precision
- Recall
- Matthews Correlation Coefficient (MCC)
- Area under the precision-recall curve (PR-AUC)
• Estimates the optimal threshold (max-F1) using the precision-recall curve.
• Saves cross-validation results to cv_results.csv.
• Exports the top 2 models (based on F1) + their optimal thresholds.
• Additionally, saves ALL models, ALL thresholds, and predictions OOF.
"""

# ---------------------- Training ----------------------

import sys, pathlib, warnings
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import joblib, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, precision_score, recall_score,
    matthews_corrcoef, average_precision_score, precision_recall_curve, make_scorer
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pd.read_csv(ROOT / "step3_dataset" / "train_scaled.csv")
XCOLS = joblib.load(ROOT / "step3_dataset" / "feature_cols.pkl")

X, y = DATA[XCOLS].values, DATA["is_root"].values
groups = DATA["sentence_id"].values

def best_threshold(y_true, scores):
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2*p*r / (p+r + 1e-9)
    return t[f1.argmax()]

models_grids = {
    "log_reg": (
        LogisticRegression(class_weight="balanced", max_iter=5000),
        {"C": [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000], "penalty": ["l2"], "solver": ["lbfgs"]}
    ),
    "ridge": (
        RidgeClassifier(class_weight={0:1,1:5}),
        {"alpha": [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000]}
    ),
    "gnb": (
        GaussianNB(),
        {"var_smoothing": [1e-11,3e-11,1e-10,3e-10,1e-9,3e-9,1e-8,3e-8,1e-7,3e-7,1e-6]}
    ),
    "lda": (
        LDA(),
        {"shrinkage": [None,"auto",0.001,0.01,0.03,0.05,0.1,0.2,0.3], "solver": ["lsqr"]}
    ),
    "qda": (
        QDA(),
        {"reg_param": [0.01,0.03,0.05,0.1,0.2,0.5,0.7,0.9]}
    ),
    "knn": (
        make_pipeline(StandardScaler(), KNeighborsClassifier()),
        {"kneighborsclassifier__n_neighbors": list(range(1,31)), "kneighborsclassifier__weights": ["uniform","distance"], "kneighborsclassifier__p": [1,2,3], "kneighborsclassifier__leaf_size": [10,20,30,40,50]}
    ),
    "tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        {"max_depth": [3,5,10,15,20,25,30,None], "min_samples_leaf": [1,2,5,10], "min_samples_split": [2,5,10,20], "criterion": ["gini","entropy"]}
    ),
    "svm": (
        LinearSVC(class_weight="balanced", random_state=42, max_iter=5000),
        {"C": [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100], "dual": [False], "loss": ["squared_hinge"]}
    ),
}

gkf = GroupKFold(n_splits=15)
f1_scorer = make_scorer(f1_score)

results, best_estimators, thresholds = [], {}, {}
oof_predictions = pd.DataFrame({"sentence_id": DATA["sentence_id"], "is_root": y})

for name, (base_model, param_grid) in models_grids.items():
    print(f"→ GridSearch {name} ...")
    grid = GridSearchCV(base_model, param_grid, cv=gkf, scoring=f1_scorer, n_jobs=-1, refit=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid.fit(X, y, groups=groups)

    best = grid.best_estimator_
    method = "predict_proba" if hasattr(best,"predict_proba") else "decision_function"
    s_cv = cross_val_predict(best, X, y, groups=groups, cv=gkf, method=method)
    s_cv = s_cv[:,1] if method=="predict_proba" else (s_cv-s_cv.min())/(s_cv.max()-s_cv.min()+1e-9)

    thr = best_threshold(y, s_cv)
    y_pred = (s_cv >= thr).astype(int)

    results.append([
        name, f1_score(y,y_pred), balanced_accuracy_score(y,y_pred),
        precision_score(y,y_pred), recall_score(y,y_pred),
        matthews_corrcoef(y,y_pred), average_precision_score(y,s_cv), thr,
        grid.best_params_
    ])

    best_estimators[name] = best
    thresholds[name] = thr

    oof_predictions[f"{name}_oof"] = s_cv
    joblib.dump(best, ROOT / "step5_modeling" / f"model_{name}.pkl")

cv_df = pd.DataFrame(results,columns=["model","f1_full","bal_full","precision","recall","mcc","pr_auc","thr","best_params"])
cv_df.sort_values("f1_full",ascending=False,inplace=True)
cv_df.to_csv(ROOT/"step5_modeling"/"cv_results.csv",index=False)

for rank,model_name in enumerate(cv_df.head(2)["model"],start=1):
    joblib.dump(best_estimators[model_name],ROOT/"step5_modeling"/f"top{rank}_{model_name}.pkl")

joblib.dump({k: thresholds[k] for k in cv_df.head(2)["model"]},ROOT/"step5_modeling"/"thresholds.pkl")
joblib.dump(thresholds,ROOT/"step5_modeling"/"all_thresholds.pkl")

oof_predictions.to_csv(ROOT/"step5_modeling"/"train_oof_predictions.csv",index=False)

print(cv_df[["model","f1_full","bal_full","precision","recall","mcc","pr_auc"]].to_string(index=False))