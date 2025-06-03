# **Root‑Detection in Free‑Growing Trees** Project

A complete guide to reproducing the pipeline — from the initial structure to `submission.csv` — using the provided scripts. It is recommended to run the scripts from an IDE terminal (e.g., POSITRON IDE) or directly from a Linux terminal.

---

## 0. Bootstrap (initial structure setup)

```bash
python bootstrap_project.py
```

This will generate the folder hierarchy: `data/`, `step0_*`, … `step7_inference/`, and 12 empty template scripts with explanatory docstrings.

**Important**

The bootstrap script only creates the folder structure and empty script templates. To run the pipeline, you must replace each template with the full script provided in this repository.

If you cloned this repository, the scripts are already complete, and you can skip the bootstrap step.

Make sure to also copy `run_all.py`, `requirements.txt`, and this `README.md` file if you are setting up the project in a different directory.

Then, place `train.csv` and `test.csv` inside the `data/` folder.

---

## 1. Requirements

| Resource | Version |
| -------- | ------- |
| Python   | ≥ 3.9   |
| pip      | ≥ 22    |

To install dependencies:

```bash
python -m venv .venv
# activate the virtual environment
pip install -r requirements.txt
```

---

## 2. Minimum Project Structure

```
MLProject_DelgadoGonzalez/
├─bootstrap_project.py
├─run_all.py
├─requirements.txt
├─README.md
│
├─data/
│   ├─train.csv
│   └─test.csv
│
├─common/
│   └─io_utils.py
│        • detect_columns()
│        • parse_graph()
│        • … (other utilities)
│   *Do not run directly*
│
├─step0_setup/
│   └─descriptive_raw.py
├─step1_preprocess/
│   └─build_graphs.py
├─step2_features/
│   └─compute_centralities.py
├─step3_dataset/
│   └─build_dataset.py
├─step4_eda/
│   ├─eda_plots.py
│   └─visualize_trees.py
├─step5_modeling/
│   ├─train_models.py
│   └─add_pr_curve.py
├─step6_validation/
│   ├─baseline_vs_model.py
│   └─error_analysis.py
└─step7_inference/
    └─generate_submission.py
```

---

## 3. Step-by-Step Execution

| Step | Script                                          | Main Output                                                                 |
| ---- | ----------------------------------------------- | ---------------------------------------------------------------------------- |
| 0    | `python step0_setup/descriptive_raw.py`         | Initial EDA stats and figures                                                |
| 1    | `python step1_preprocess/build_graphs.py`       | `train_graphs.pkl`, `test_graphs.pkl`                                       |
| 2    | `python step2_features/compute_centralities.py` | `*_node_features.csv` (7 centralities)                                      |
| 3    | `python step3_dataset/build_dataset.py`         | `train_scaled.csv`, `test_scaled.csv`                                       |
| 4a   | `python step4_eda/eda_plots.py`                 | Box plots + heat map (500 dpi)                                              |
| 4b   | `python step4_eda/visualize_trees.py [id] [lang]` | PNG tree image (panels A/B)                                                 |
| 5    | `python step5_modeling/train_models.py`         | **Takes ~4 h** (GridSearch 8 models) → models, OOF, `all_thresholds.pkl`, PR curves |
| 5b   | `python step5_modeling/add_pr_curve.py`         | PR curve and F1-vs-threshold (Top-2)                                        |
| 6a   | `python step6_validation/baseline_vs_model.py`  | `metrics.txt`, `comparison.csv`                                             |
| 6b   | `python step6_validation/error_analysis.py`     | Confusion matrices + error analysis                                         |
| 7    | `python step7_inference/generate_submission.py` | `submission.csv` ready for Kaggle                                           |

---

## 4. Full Pipeline Automation

### 4.1 You can run the full pipeline with `run_all.py`

```python
#!/usr/bin/env python3
import subprocess, sys
steps = [
    "step0_setup/descriptive_raw.py",
    "step1_preprocess/build_graphs.py",
    "step2_features/compute_centralities.py",
    "step3_dataset/build_dataset.py",
    "step4_eda/eda_plots.py",
    "step4_eda/visualize_trees.py",
    "step5_modeling/train_models.py",
    "step5_modeling/add_pr_curve.py",
    "step6_validation/baseline_vs_model.py",
    "step6_validation/error_analysis.py",
    "step7_inference/generate_submission.py",
]
for s in steps:
    print(f">>> python {s}")
    res = subprocess.run([sys.executable, s])
    if res.returncode != 0:
        sys.exit(f"Error in {s}")
print("Pipeline complete")
```

Run with:

```bash
python run_all.py
```

---

## 5. Key Parameters & Notes

* **Global seed**: 42
* **Thresholds**: stored in `all_thresholds.pkl`
* **Training ≈** 4 h on an 8-thread CPU
* **Suppress warnings with:**

```python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
```

---

## 6. Quick Customization

| Task                            | File                                  | What to change                        |
| ------------------------------- | ------------------------------------- | ------------------------------------- |
| Use a different threshold       | `train_models.py` → `best_threshold`  | Metric or fixed value                 |
| Add more features               | `compute_centralities.py`             | Additional centralities, POS, etc.    |
| Reduce GridSearch combinations  | `train_models.py`                     | Shorten lists in `param_grid`         |

---

