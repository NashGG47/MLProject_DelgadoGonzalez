#!/usr/bin/env python3
"""
bootstrap_project.py
-------------------------------------------------
Create the complete folder structure and (empty) template scripts for the Root-Detection project, including ALL agreed-upon tasks.
-------------------------------------------------
    python bootstrap_project.py
"""

import pathlib, textwrap

ROOT = pathlib.Path(__file__).resolve().parent

# ---------- PATHS -------------------------------------------------
DIRS = [
    "data",
    "common",
    "step0_setup",
    "step1_preprocess",
    "step2_features",
    "step3_dataset",
    "step4_eda",
    "step5_modeling",
    "step6_validation",
    "step7_inference",
]

# ---------- EMPTY scripts  -------------------------------------------
TEMPLATES = {
    "common/io_utils.py":                    "Utilidades de I/O y randomización",
    "step0_setup/descriptive_raw.py":        "Estadística descriptiva de datos crudos",
    "step1_preprocess/build_graphs.py":      "Randomización + grafos ND",
    "step2_features/compute_centralities.py":"Cálculo de 7 centralidades",
    "step3_dataset/build_dataset.py":        "Normalización intrafrase",
    "step4_eda/eda_plots.py":                "Boxplots + heat-map",
    "step4_eda/visualize_trees.py":          "Dibujo de árboles por idioma",
    "step5_modeling/train_models.py":        "Entrenamiento de múltiples modelos",
    "step5_modeling/add_pr_curve.py":        "Curva PR y threshold-vs-F1",
    "step6_validation/baseline_vs_model.py": "Baseline PageRank vs modelo",
    "step6_validation/error_analysis.py":    "Matriz de confusión y análisis de error",
    "step7_inference/generate_submission.py":"Generar submission.csv (ids reales)",
}

HEADER = """\
\"\"\"{comment}
 Auto-generated template. Fill in the step logic.
\"\"\"\n"""

# ---------- CREATE FOLDERS -------------------------------------------
for d in DIRS:
    (ROOT / d).mkdir(parents=True, exist_ok=True)

# ---------- Create scripts templates ----------------------------------
for path, comment in TEMPLATES.items():
    file_path = ROOT / path
    if not file_path.exists():
        file_path.write_text(textwrap.dedent(HEADER.format(comment=comment)), encoding="utf-8")
        print("✓ creado", file_path.relative_to(ROOT))

print("\n scripts ready.")
print("   Copy train.csv and test.csv on /data and fill each script.")
