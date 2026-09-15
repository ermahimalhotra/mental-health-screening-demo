"""
Streamlit demo for the anxiety / depression screening models.
Run locally:  streamlit run app.py
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = "models"

st.set_page_config(page_title="Stress & Sleep Screening Demo",
                   page_icon="🧠", layout="wide")

# --------------------------------------------------------------------------
# Item labels. These are SHORT DESCRIPTIVE LABELS, not the licensed item text.
# Replace with the official wording from the dataset documentation if your
# rubric requires the exact questionnaire.
# --------------------------------------------------------------------------

PSS_LABELS = [
    "Upset by something that happened unexpectedly",
    "Unable to control the important things in your life",
    "Felt nervous and stressed",
    "Handled day-to-day annoyances well",
    "Coped well with important changes",
    "Felt confident handling personal problems",
    "Felt things were going your way",
    "Could not keep up with everything you had to do",
    "Able to manage things that irritate you",
    "Felt on top of things",
    "Angered by things outside your control",
    "Preoccupied with things you still had to finish",
    "In control of how you spend your time",
    "Felt problems piling up beyond what you could handle",
]

ISI_LABELS = [
    "Trouble falling asleep",
    "Trouble staying asleep",
    "Waking up too early",
    "Dissatisfied with your current sleep pattern",
    "Sleep problem is noticeable to other people",
    "Worried or distressed about your sleep",
    "Sleep problem interferes with daily functioning",
]

PSS_SCALE = {0: "0 — Never", 1: "1 — Almost never", 2: "2 — Sometimes",
             3: "3 — Fairly often", 4: "4 — Very often"}
ISI_SCALE = {0: "0 — None", 1: "1 — Mild", 2: "2 — Moderate",
             3: "3 — Severe", 4: "4 — Very severe"}

# Populated from metadata["categorical_labels"] once it's loaded below;
# falls back to "code N" for any column the training run left numeric.


@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    models = {}
    for target, tmeta in meta["targets"].items():
        for name, minfo in tmeta["models"].items():
            path = os.path.join(MODEL_DIR, minfo["file"])
            if os.path.exists(path):
                models[(target, name)] = joblib.load(path)
    return meta, models


try:
    meta, MODELS = load_artifacts()
except FileNotFoundError:
    st.error("No trained models found. Run train_and_save_models.py first, "
             "then put the models/ folder next to app.py.")
    st.stop()

FEATURES = meta["feature_columns"]
DEMO_LABELS = {
    col: {int(k): v for k, v in mapping.items()}
    for col, mapping in meta.get("categorical_labels", {}).items()
}

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("Predicting anxiety and depression risk from stress and sleep")
st.write(
    "This demo takes 14 perceived-stress answers, 7 insomnia answers and five "
    f"background details, and estimates the chance that a student would screen "
    f"positive on GAD-7 and PHQ-9. It was trained on {meta['n_total']:,} Chinese "
    "university students (Su et al., 2024). No anxiety or depression item ever "
    "enters the model's input, so the prediction is a real inference rather than "
    "a recalculated questionnaire score."
)
st.warning(
    "Research demonstration only. This is not a diagnosis, not a medical device, "
    "and must not be used to make decisions about a real person. If you are "
    "struggling, talk to a doctor, a campus counsellor, or someone you trust."
)

# --------------------------------------------------------------------------
# Sidebar: model choice
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    available = sorted({name for (_, name) in MODELS})
    model_name = st.selectbox("Algorithm", available,
                              index=available.index("RandomForest")
                              if "RandomForest" in available else 0)
    use_tuned = st.checkbox(
        "Use the imbalance-tuned cut-off", value=True,
        help="Only about 2–5% of this sample screened positive, so a plain 0.5 "
             "cut-off almost never flags anyone. The tuned cut-off was chosen on "
             "training data only.",
    )
    st.divider()
    st.caption(
        "Answer values must follow the same coding as the source CSVs — raw "
        "responses, not reverse-scored. The model learned whatever convention "
        "the dataset uses."
    )

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------

with st.form("screening"):
    tab_stress, tab_sleep, tab_about_you = st.tabs(
        ["Stress (14 items)", "Sleep (7 items)", "About you"]
    )
    answers = {}

    with tab_stress:
        st.caption("In the last month, how often have you felt this way?")
        cols = st.columns(2)
        for i, label in enumerate(PSS_LABELS, start=1):
            with cols[(i - 1) % 2]:
                answers[f"pss_q{i}"] = st.select_slider(
                    f"{i}. {label}", options=list(PSS_SCALE),
                    value=2, format_func=lambda v: PSS_SCALE[v],
                    key=f"pss_q{i}",
                )

    with tab_sleep:
        st.caption("Over the last two weeks, how severe has this been?")
        cols = st.columns(2)
        for i, label in enumerate(ISI_LABELS, start=1):
            with cols[(i - 1) % 2]:
                answers[f"isi_q{i}"] = st.select_slider(
                    f"{i}. {label}", options=list(ISI_SCALE),
                    value=1, format_func=lambda v: ISI_SCALE[v],
                    key=f"isi_q{i}",
                )

    with tab_about_you:
        c1, c2 = st.columns(2)
        rng = meta["feature_ranges"]["age"]
        with c1:
            answers["age"] = st.number_input(
                "Age", min_value=int(rng["min"]), max_value=int(rng["max"]),
                value=int(rng["median"]), step=1,
            )
        levels = meta.get("categorical_levels", {})
        slots = [c1, c2, c1, c2]
        for col, field in zip(slots, ["gender", "edu", "smoke", "drink"]):
            opts = levels.get(field, [0, 1])
            with col:
                answers[field] = st.selectbox(
                    field.capitalize(), opts,
                    format_func=lambda v, f=field: DEMO_LABELS.get(f, {}).get(v, f"code {v}"),
                )

    submitted = st.form_submit_button("Estimate risk", type="primary")

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

if submitted:
    row = pd.DataFrame([[answers[f] for f in FEATURES]], columns=FEATURES)

    st.subheader("Estimated screening risk")
    cols = st.columns(2)

    for col, target in zip(cols, ["gad7", "phq9"]):
        tmeta = meta["targets"][target]
        model = MODELS.get((target, model_name))
        if model is None:
            col.error(f"{model_name} not saved for {target}.")
            continue

        proba = float(model.predict_proba(row)[0, 1])
        thr = tmeta["models"][model_name]["threshold"] if use_tuned else 0.5
        flagged = proba >= thr
        base = meta["prevalence"][target]

        with col:
            st.markdown(f"**{tmeta['label']} ≥ 10**")
            st.metric("Probability", f"{proba:.1%}",
                      delta=f"{(proba - base) * 100:+.1f} pts vs. sample average")
            st.progress(min(proba, 1.0))
            if flagged:
                st.error("Above the screening cut-off — would be referred for "
                         "a full assessment in a real screening workflow.")
            else:
                st.success("Below the screening cut-off.")
            st.caption(f"Cut-off in use: {thr:.3f} · "
                       f"test ROC-AUC {tmeta['models'][model_name]['metrics']['roc_auc']:.3f}")

    st.subheader("What the model is paying attention to")
    imp = meta["targets"]["gad7"]["feature_importances"]
    top = pd.DataFrame(list(imp.items())[:10], columns=["feature", "importance"])
    top["your answer"] = top["feature"].map(answers)
    st.dataframe(top, hide_index=True, use_container_width=True)
    st.caption("Importances are from the anxiety random forest, computed at "
               "training time across all students — not an explanation of this "
               "one prediction.")

# --------------------------------------------------------------------------
# Performance table
# --------------------------------------------------------------------------

with st.expander("How well does it actually perform?"):
    st.write(
        "Roughly 2% of this sample screened positive for anxiety and 5% for "
        "depression, so plain accuracy is inflated — always predicting 'no' "
        "would already score above 94%. Balanced accuracy, macro-F1 and ROC-AUC "
        "are the honest numbers."
    )
    rows = []
    for target, tmeta in meta["targets"].items():
        for name, minfo in tmeta["models"].items():
            rows.append({"Target": tmeta["label"], "Model": name,
                         **{k.replace("_", " ").title(): v
                            for k, v in minfo["metrics"].items()}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Held-out test set, 20% of students, never seen during training "
               "or threshold selection.")
