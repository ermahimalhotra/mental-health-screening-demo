"""
Streamlit demo for the anxiety / depression screening models (Su + UK merged version).
Run locally:  streamlit run app.py
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = "models"

st.set_page_config(page_title="Stress & Sleep Screening Demo", page_icon="🧠", layout="wide")

# --------------------------------------------------------------------------
# Questionnaire items. These are SHORT DESCRIPTIVE LABELS, not the licensed
# item text. Replace with the official wording if your rubric requires it.
#
# The model uses the 10-item Perceived Stress Scale (PSS-10). Positively worded
# items are marked positive=True: a high answer there means LOW stress, so the
# app reverses it (4 - answer) before it goes to the model. The training data
# was already coded "higher = more stress", so this makes the app match it.
# --------------------------------------------------------------------------

PSS10_ITEMS = [
    ("Upset by something that happened unexpectedly", False),
    ("Unable to control the important things in your life", False),
    ("Felt nervous and stressed", False),
    ("Felt confident handling personal problems", True),
    ("Felt things were going your way", True),
    ("Could not keep up with everything you had to do", False),
    ("Able to manage things that irritate you", True),
    ("Felt on top of things", True),
    ("Angered by things outside your control", False),
    ("Felt problems piling up beyond what you could handle", False),
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
    st.error("No trained models found. Run train_final_models.py first, "
             "then put the models/ folder next to app.py.")
    st.stop()

FEATURES = meta["feature_columns"]
SOURCES = {int(k): v for k, v in meta["sources"].items()}
TARGET_ORDER = ["gad", "phq"]

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

n0, n1 = meta["n_by_source"]["0"], meta["n_by_source"]["1"]
st.title("Predicting anxiety and depression risk from stress and sleep")
st.write(
    "This demo takes 10 perceived-stress answers, 7 insomnia answers, age and sex, and "
    "estimates the chance that a student would screen positive on GAD-7 (anxiety) and "
    f"PHQ-9 (depression). It was trained on {n0:,} Chinese university students "
    f"(Su et al., 2024) and {n1:,} UK university students (Akram et al., 2023). "
    "No anxiety or depression item ever enters the model's input, so the prediction is a "
    "real inference rather than a recalculated questionnaire score."
)
st.warning(
    "Research demonstration only. This is not a diagnosis, not a medical device, "
    "and must not be used to make decisions about a real person. If you are "
    "struggling, talk to a doctor, a campus counsellor, or someone you trust."
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    available = sorted({name for (_, name) in MODELS})
    model_name = st.selectbox("Algorithm", available,
                              index=available.index("RandomForest") if "RandomForest" in available else 0)
    source = st.selectbox(
        "Which student group is this most like?", list(SOURCES), format_func=lambda k: SOURCES[k],
        help="The two datasets have very different rates of anxiety and depression, so each "
             "group has its own screening cut-off.")
    use_tuned = st.checkbox(
        "Use the group's tuned cut-off", value=True,
        help="Chosen on training data only, separately for each group. Unticked, a plain 0.5 "
             "cut-off is used.")
    st.divider()
    st.caption(
        "Answer each question as it is worded. The app reverses the positively worded "
        "stress questions for you."
    )

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------

with st.form("screening"):
    tab_stress, tab_sleep, tab_about_you = st.tabs(["Stress (10 items)", "Sleep (7 items)", "About you"])
    answers = {}

    with tab_stress:
        st.caption("In the last month, how often have you felt this way?")
        cols = st.columns(2)
        for i, (label, _) in enumerate(PSS10_ITEMS):
            with cols[i % 2]:
                answers[f"pss_{i}"] = st.select_slider(
                    f"{i + 1}. {label}", options=list(PSS_SCALE), value=2,
                    format_func=lambda v: PSS_SCALE[v], key=f"pss_{i}")

    with tab_sleep:
        st.caption("Over the last two weeks, how severe has this been?")
        cols = st.columns(2)
        for i, label in enumerate(ISI_LABELS):
            with cols[i % 2]:
                answers[f"isi_{i}"] = st.select_slider(
                    f"{i + 1}. {label}", options=list(ISI_SCALE), value=1,
                    format_func=lambda v: ISI_SCALE[v], key=f"isi_{i}")

    with tab_about_you:
        c1, c2 = st.columns(2)
        with c1:
            answers["age"] = st.number_input("Age", min_value=16, max_value=35, value=20, step=1)
        with c2:
            answers["sex"] = st.selectbox("Sex", ["Female", "Male"])

    submitted = st.form_submit_button("Estimate risk", type="primary")


def build_features(ans, src):
    """Turn the form answers into the model's input row (same scaling as training)."""
    stress_total = 0
    for i, (_, positive) in enumerate(PSS10_ITEMS):
        v = ans[f"pss_{i}"]
        stress_total += (4 - v) if positive else v
    sleep_total = sum(ans[f"isi_{i}"] for i in range(len(ISI_LABELS)))
    row = {
        "pss": stress_total / 40,
        "sleep": sleep_total / 28,
        "age": min(max(ans["age"], 16), 35),
        "male": 1 if ans["sex"] == "Male" else 0,
        "source": src,
    }
    return pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES), stress_total, sleep_total


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

if submitted:
    row, stress_total, sleep_total = build_features(answers, source)
    st.subheader("Estimated screening risk")
    st.caption(f"Stress score {stress_total}/40 · Insomnia score {sleep_total}/28 · "
               f"Group: {SOURCES[source]}")
    cols = st.columns(2)

    for col, target in zip(cols, TARGET_ORDER):
        tmeta = meta["targets"][target]
        model = MODELS.get((target, model_name))
        if model is None:
            col.error(f"{model_name} not saved for {target}.")
            continue
        minfo = tmeta["models"][model_name]
        proba = float(model.predict_proba(row)[0, 1])
        thr = minfo["thresholds"][str(source)] if use_tuned else 0.5
        flagged = proba >= thr
        base = meta["prevalence"][target][str(source)]

        with col:
            st.markdown(f"**{tmeta['label']}**")
            st.metric("Probability", f"{proba:.1%}",
                      delta=f"{(proba - base) * 100:+.1f} pts vs. this group's average")
            st.progress(min(proba, 1.0))
            if flagged:
                st.error("Above the screening cut-off — would be referred for a full "
                         "assessment in a real screening workflow.")
            else:
                st.success("Below the screening cut-off.")
            st.caption(f"Cut-off in use: {thr:.3f} · test ROC-AUC for this group "
                       f"{minfo['metrics'][str(source)]['roc_auc']:.3f}")

    st.subheader("What the model is paying attention to")
    imp = meta["targets"]["gad"]["feature_importances"]
    names = {"pss": "Stress score", "sleep": "Insomnia score", "age": "Age",
             "male": "Sex", "source": "Student group"}
    top = pd.DataFrame([(names.get(k, k), v) for k, v in imp.items()],
                       columns=["feature", "importance"])
    st.dataframe(top, hide_index=True, use_container_width=True)
    st.caption("Importances are from the anxiety random forest, computed at training time "
               "across all students — not an explanation of this one prediction.")

# --------------------------------------------------------------------------
# Performance table
# --------------------------------------------------------------------------

with st.expander("How well does it actually perform?"):
    st.write(
        "Anxiety and depression are rare in the Chinese sample (about 2% and 5%) and common in "
        "the UK sample (about 44% and 47%), so plain accuracy is misleading. Balanced accuracy "
        "and ROC-AUC are the honest numbers. Each group is scored separately, using its own cut-off."
    )
    rows = []
    for target in TARGET_ORDER:
        tmeta = meta["targets"][target]
        for name, minfo in tmeta["models"].items():
            for k, m in minfo["metrics"].items():
                rows.append({
                    "Target": tmeta["label"], "Model": name, "Group": SOURCES[int(k)],
                    "Test students": m["n_test"], "ROC-AUC": m["roc_auc"],
                    "Balanced accuracy": m["balanced_accuracy"], "F1 macro": m["f1_macro"],
                    "Sensitivity": m["sensitivity"], "Specificity": m["specificity"],
                })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Held-out test set, 20% of each dataset, never seen during training or cut-off selection.")

with st.expander("Limitations"):
    st.markdown(
        "- The two datasets use different sleep questionnaires (ISI in China, Sleep Condition "
        "Indicator in the UK), so the sleep score is only approximately comparable.\n"
        "- The UK sample is small (about 1,400 students) and mostly young, female and white.\n"
        "- Both samples are volunteers who answered online, not random samples of students.\n"
        "- A screening score is not a diagnosis."
    )
