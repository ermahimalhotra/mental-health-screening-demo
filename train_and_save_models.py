"""
mental_health_screening_pipeline_v3.py
Same leakage-controlled design as v2, plus persistence of the trained models
and everything app.py needs to reproduce the exact feature contract.

Run:  python train_and_save_models.py
Expects the 5 CSVs in ./data/ (demographic, pss, isi, gad7, phq9).
Writes:  ./models/*.joblib  and  ./models/metadata.json
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

DATA_DIR = "data"
MODEL_DIR = "models"
RANDOM_STATE = 42
os.makedirs(MODEL_DIR, exist_ok=True)

PSS_ITEMS = [f"pss_q{i}" for i in range(1, 15)]
ISI_ITEMS = [f"isi_q{i}" for i in range(1, 8)]
DEMO_COLS = ["gender", "age", "edu", "smoke", "drink"]
FEATURES = PSS_ITEMS + ISI_ITEMS + DEMO_COLS

# --------------------------------------------------------------------------
# 1. Load and merge
# --------------------------------------------------------------------------

def load_scale(name, n_items, prefix):
    """Load one scale CSV, keep only export_id + item columns (+ total), rename items."""
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))
    item_cols = [f"question{i}" for i in range(1, n_items + 1)]
    keep = ["export_id", "score"] + item_cols
    df = df[keep].copy()
    rename = {f"question{i}": f"{prefix}_q{i}" for i in range(1, n_items + 1)}
    rename["score"] = f"{prefix}_total"
    # time1..timeN are never selected -> response-time leakage impossible
    return df.rename(columns=rename)


print("Loading CSVs...")
demo = pd.read_csv(os.path.join(DATA_DIR, "demographic.csv"))[["export_id"] + DEMO_COLS]
pss = load_scale("pss", 14, "pss")
isi = load_scale("isi", 7, "isi")
gad = load_scale("gad7", 7, "gad7")
phq = load_scale("phq9", 9, "phq9")

df = demo.merge(pss, on="export_id").merge(isi, on="export_id") \
         .merge(gad, on="export_id").merge(phq, on="export_id")
df = df.dropna(subset=FEATURES + ["gad7_total", "phq9_total"]).reset_index(drop=True)
print(f"Merged sample: {len(df)} students, {len(FEATURES)} predictors")

# --------------------------------------------------------------------------
# 1b. Encode categorical demographic columns (they arrive as text, e.g. "male")
#     Models need numbers; we save the code->label mapping for the app's UI.
# --------------------------------------------------------------------------

CATEGORICAL_DEMOS = [c for c in DEMO_COLS if not pd.api.types.is_numeric_dtype(df[c])]
label_maps = {}  # column -> {code: original_label}
for col in CATEGORICAL_DEMOS:
    df[col] = df[col].astype(str).str.strip()
    codes, uniques = pd.factorize(df[col], sort=True)
    df[col] = codes
    label_maps[col] = {int(i): str(v) for i, v in enumerate(uniques)}
    print(f"Encoded '{col}':", label_maps[col])

# --------------------------------------------------------------------------
# 2. Targets + leakage guard
# --------------------------------------------------------------------------

df["y_gad7"] = (df["gad7_total"] >= 10).astype(int)
df["y_phq9"] = (df["phq9_total"] >= 10).astype(int)

banned = [c for c in FEATURES if c.startswith(("gad7", "phq9")) or "time" in c]
assert not banned, f"LEAKAGE: {banned} must not be in X"

X = df[FEATURES]
prevalence = {
    "gad7": float(df["y_gad7"].mean()),
    "phq9": float(df["y_phq9"].mean()),
}
print(f"Prevalence  GAD-7>=10: {prevalence['gad7']:.2%}   PHQ-9>=10: {prevalence['phq9']:.2%}")

# --------------------------------------------------------------------------
# 3. Train / evaluate / persist, per target
# --------------------------------------------------------------------------

RF_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [8, 10, 12],
    "min_samples_leaf": [20, 30, 50],
}

summary_rows = []
metadata = {
    "feature_columns": FEATURES,
    "pss_items": PSS_ITEMS,
    "isi_items": ISI_ITEMS,
    "demo_columns": DEMO_COLS,
    "prevalence": prevalence,
    "n_total": int(len(df)),
    "targets": {},
}

for target, label in [("gad7", "GAD-7 (anxiety)"), ("phq9", "PHQ-9 (depression)")]:
    y = df[f"y_{target}"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {}

    print(f"\n=== {label} : RandomForest grid search ===")
    rf = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        RF_GRID, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0,
    ).fit(X_tr, y_tr)
    print("best params:", rf.best_params_)
    models["RandomForest"] = rf.best_estimator_

    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ]).fit(X_tr, y_tr)

    models["SVM_RBF"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced",
                    random_state=RANDOM_STATE)),
    ]).fit(X_tr, y_tr)

    target_meta = {"label": label, "models": {}}

    for name, model in models.items():
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_te, pred),
            "balanced_accuracy": balanced_accuracy_score(y_te, pred),
            "f1_macro": f1_score(y_te, pred, average="macro"),
            "roc_auc": roc_auc_score(y_te, proba),
        }

        # Operating threshold chosen on TRAINING data only (out-of-fold),
        # so the reported test metrics stay honest.
        oof = cross_val_predict(model, X_tr, y_tr, cv=3, method="predict_proba", n_jobs=-1)[:, 1]
        fpr, tpr, thr = roc_curve(y_tr, oof)
        tuned_threshold = float(thr[np.argmax(tpr - fpr)])  # Youden's J

        pred_t = (proba >= tuned_threshold).astype(int)
        metrics["balanced_accuracy_tuned"] = balanced_accuracy_score(y_te, pred_t)
        metrics["f1_macro_tuned"] = f1_score(y_te, pred_t, average="macro")

        print(f"\n--- {label} / {name} ---")
        for k, v in metrics.items():
            print(f"{k:26s} {v:.3f}")
        print("threshold (Youden, from train OOF):", round(tuned_threshold, 3))
        print("confusion matrix @0.5:\n", confusion_matrix(y_te, pred))
        print(classification_report(y_te, pred, digits=3))

        fname = f"{target}_{name}.joblib"
        joblib.dump(model, os.path.join(MODEL_DIR, fname), compress=3)
        size_mb = os.path.getsize(os.path.join(MODEL_DIR, fname)) / 1e6
        print(f"saved -> models/{fname} ({size_mb:.1f} MB)")
        if size_mb > 90:
            print("  WARNING: >90 MB, too big for a normal GitHub push. "
                  "Re-train this model with n_estimators=300 or max_depth=8.")

        target_meta["models"][name] = {
            "file": fname,
            "threshold": tuned_threshold,
            "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "size_mb": round(size_mb, 2),
        }
        summary_rows.append({"target": target, "model": name, **metrics})

    # feature importances from RF, for the app's explanation panel
    rf_best = models["RandomForest"]
    target_meta["feature_importances"] = {
        f: round(float(i), 5)
        for f, i in sorted(zip(FEATURES, rf_best.feature_importances_),
                           key=lambda t: -t[1])
    }
    target_meta["default_model"] = "RandomForest"
    metadata["targets"][target] = target_meta

# --------------------------------------------------------------------------
# 4. Input contract for the app: exact value ranges seen in training
# --------------------------------------------------------------------------

metadata["feature_ranges"] = {
    c: {"min": float(X[c].min()), "max": float(X[c].max()),
        "median": float(X[c].median())}
    for c in FEATURES
}
metadata["categorical_levels"] = {
    c: sorted(int(v) for v in df[c].dropna().unique())
    for c in ["gender", "edu", "smoke", "drink"]
    if df[c].dropna().nunique() <= 12
}
metadata["categorical_labels"] = label_maps  # column -> {code: original text label}

with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

pd.DataFrame(summary_rows).to_csv(os.path.join(MODEL_DIR, "model_summary.csv"), index=False)
import sklearn
metadata["sklearn_version"] = sklearn.__version__
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved models/metadata.json and models/model_summary.csv")
print(f"IMPORTANT: pin scikit-learn=={sklearn.__version__} in requirements.txt, "
      "or the saved models may fail to load on Streamlit Cloud.")
print("Done. Copy the whole models/ folder into your GitHub repo.")
