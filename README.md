# Predicting anxiety and depression risk from stress and sleep

A machine-learning screening demo built on Su et al. (2024), *Scientific Data*
(Zenodo 10423537, CC-BY 4.0) — 24,292 Chinese university students.

The model takes **only** perceived-stress items (PSS-14), insomnia items (ISI-7)
and five demographic fields, and predicts whether a student would screen positive
on GAD-7 (≥10) or PHQ-9 (≥10). No item or total from the target questionnaires is
ever used as an input, and all response-time columns are dropped, so the model is
making a genuine inference rather than re-deriving a scoring formula.

**Not a diagnostic tool.** Research demonstration only.

## Results (held-out 20% test set, 4,859 students)

| Target | Model | Accuracy | Balanced acc. | F1-macro | ROC-AUC |
|---|---|---|---|---|---|
| GAD-7 | Random forest | 0.946 | 0.905 | 0.695 | 0.969 |
| GAD-7 | Logistic regression | 0.921 | 0.915 | 0.650 | 0.973 |
| GAD-7 | SVM (RBF) | 0.965 | 0.848 | 0.733 | 0.922 |
| PHQ-9 | Random forest | 0.898 | 0.876 | 0.709 | 0.947 |
| PHQ-9 | Logistic regression | 0.884 | 0.879 | 0.692 | 0.949 |
| PHQ-9 | SVM (RBF) | 0.903 | 0.868 | 0.714 | 0.935 |

Only 2.25% of the sample screened positive on GAD-7 and 5.4% on PHQ-9, so plain
accuracy is inflated — a model that always answers "no" scores 97.7% and 94.6%
for free. Balanced accuracy, macro-F1 and ROC-AUC are the metrics that mean
something here.

## Repo layout

```
.
├── app.py                      # Streamlit demo
├── train_and_save_models.py    # training + persistence
├── requirements.txt
├── models/                     # produced by the training script
│   ├── gad7_RandomForest.joblib
│   ├── gad7_LogisticRegression.joblib
│   ├── gad7_SVM_RBF.joblib
│   ├── phq9_*.joblib
│   ├── metadata.json
│   └── model_summary.csv
└── data/                       # NOT committed — download from Zenodo
```

## 1. Train and save

In Colab, upload `train_and_save_models.py` through the file browser (do not
paste it into a cell — the editor re-indents pasted code), then:

```bash
!python train_and_save_models.py
```

Download the whole `models/` folder afterwards.

Check the printed file sizes. If any `.joblib` is over ~90 MB, GitHub will
reject it — re-run with `n_estimators` fixed at 300 and `max_depth` at 8, which
costs very little accuracy here.

## 2. Push to GitHub

```bash
git init
git add app.py train_and_save_models.py requirements.txt README.md models/
git commit -m "Screening demo: models + Streamlit app"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

Add a `.gitignore` containing `data/` — the raw CSVs are large and don't belong
in the repo. Link to the Zenodo record instead (CC-BY requires attribution, which
the README and the app both give).

**Pin the scikit-learn version.** The training script prints the version it ran
under. Put exactly that version in `requirements.txt`, otherwise the `.joblib`
files may fail to unpickle on Streamlit Cloud and the app will crash on startup.

## 3. Deploy

1. Go to share.streamlit.io and sign in with GitHub.
2. "New app" → pick the repo, branch `main`, main file `app.py`.
3. Deploy. First build takes 2–4 minutes.
4. Open the public URL and test it before you present.

## 4. Test inputs before presenting

- **Low risk:** all stress items 0–1, all sleep items 0. Both probabilities
  should land near or below the sample base rate.
- **High risk:** stress items 3–4, sleep items 3–4. Both probabilities should
  jump well above the cut-off.
- **Mixed:** high stress, no sleep problems. Useful for showing the model is
  weighing two independent signals rather than one.

If low-risk inputs produce a *high* probability, your answer coding is probably
reversed relative to the CSVs — check whether the source data stores raw
responses or reverse-scored values for PSS items 4, 5, 6, 7, 9, 10 and 13, and
adjust the app's input handling to match.

## Attribution

Su, et al. (2024). Temporal dynamics in psychological assessments.
*Scientific Data*. Zenodo record 10423537. CC-BY 4.0.
