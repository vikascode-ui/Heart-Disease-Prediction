# Heart Disease Prediction using Machine Learning

A complete educational AIA project that cleans the Cleveland Heart Disease dataset, explores features, compares classifiers, evaluates the selected model, and provides a Streamlit prediction interface.

> **Medical disclaimer:** This is a classroom ML demonstration only. It is not a clinical device and must not be used to diagnose, treat, or make health decisions.

## Dataset

- Dataset: Cleveland subset of the UCI Heart Disease dataset (303 records, 13 input attributes + diagnosis).
- Original source: UCI Machine Learning Repository, Heart Disease, https://doi.org/10.24432/C52P4X
- Kaggle option: https://www.kaggle.com/datasets/ronitf/heart-disease-uci
- If you do not already have `dataset/heart.csv`, run `python download_dataset.py`.
- Diagnosis target is converted to binary: original `0` = no disease; original `1–4` = disease present.

The project keeps the original feature codes (for example `cp`, `restecg`, `slope`, `ca`, `thal`) and uses a preprocessing pipeline. `?` values become missing values and are imputed within the training pipeline.

## Features / workflow

1. Load dataset with Pandas.
2. Remove duplicate rows; convert values to numeric and handle missing values.
3. Exploratory analysis: target distribution and correlation matrix.
4. Stratified 80/20 train-test split (fixed random seed 42).
5. Preprocess numeric fields with median imputation + standard scaling.
6. Preprocess categorical-coded fields with most-frequent imputation + one-hot encoding.
7. Compare Logistic Regression, Random Forest, and Extra Trees using 5-fold CV F1 on training data.
8. Evaluate the best CV-F1 model once on the held-out test set.
9. Save metrics, classification report, confusion matrix, ROC curve, feature importance, and model.
10. Run the Streamlit form to demonstrate a prediction.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python download_dataset.py
python train.py
streamlit run app.py
```

## Output files

Training generates:

- `model.pkl` — trained preprocessing + classifier pipeline
- `outputs/metrics.json` — held-out test metrics
- `outputs/model_comparison.csv` — 5-fold CV F1 comparison
- `outputs/classification_report.txt`
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `outputs/correlation_matrix.png`
- `outputs/target_distribution.png`
- `outputs/feature_importance.csv` and `.png`

**Do not write a fixed accuracy in your report before running the code.** Use the metrics produced by your own run. The test set is held out from model selection; cross-validation is used for selecting the model.

## Expected CSV columns

`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`

The target may be named `num`, `condition`, or `diagnosis` in a downloaded CSV; `train.py` supports those aliases. Values `?` are treated as missing.

## GitHub upload

1. Create a public/private GitHub repository named `Heart-Disease-Prediction`.
2. Upload this folder (include `dataset/heart.csv` after downloading it, if your instructor requires the dataset in the repo).
3. Commit the notebook/code, README, requirements, and generated evaluation plots.
4. Do not commit personal/medical records. This project uses a public educational dataset.

## Dataset reference

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Heart Disease. https://doi.org/10.24432/C52P4X
