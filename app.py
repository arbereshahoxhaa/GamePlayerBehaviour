"""Streamlit dashboard for the thesis demo.

Run:  streamlit run app.py
Prereq:  python scripts/01_train.py   (trains + persists models + figures)
Optional:  python scripts/03_faithfulness_eval.py   (fills the study tab)
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from gamexai import agents, config as C, data, models  # noqa: E402
from gamexai.llm import LLMClient  # noqa: E402
from gamexai.xai import LocalExplainer  # noqa: E402

FIG = C.FIGURES
CLASSES = C.CLASSES

st.set_page_config(page_title="Player Behavior XAI", layout="wide")
st.title("Interpreting ML Models for Video Game Player Behavior (XAI)")
st.caption("ML classifiers + Explainable AI + a multi-agent LLM pipeline that "
           "turns SHAP into plain language — and audits itself for faithfulness.")

if not (C.RESULTS / "metrics.json").exists():
    st.error("Run `python scripts/01_train.py` first.")
    st.stop()
res = json.loads((C.RESULTS / "metrics.json").read_text())
faith = (json.loads((C.RESULTS / "faithfulness.json").read_text())
         if (C.RESULTS / "faithfulness.json").exists() else None)


@st.cache_resource
def get_dataset():
    return data.make_dataset()


@st.cache_resource
def get_rf_and_explainer():
    ds = get_dataset()
    rf = models.load("Random Forest")
    return rf, LocalExplainer(rf, ds.feature_names)


@st.cache_resource
def get_client():
    return LLMClient()


ds = get_dataset()
medians = res["dataset"].get("medians", {})

t1, t2, t3, t4, t5 = st.tabs(
    ["Dataset", "Models", "Explainability (XAI)", "Explain a player",
     "Faithfulness study"])

# ------------------------------------------------------------------ Dataset
with t1:
    d = res["dataset"]
    c = st.columns(4)
    c[0].metric("Players", f"{d['rows_used']:,}")
    c[1].metric("Features", d["n_features"])
    c[2].metric("Train / Test", f"{d['train']:,} / {d['test']:,}")
    c[3].metric("Missing values", 0)
    st.bar_chart(pd.Series(d["class_counts"]).reindex(CLASSES),
                 x_label="Engagement", y_label="Players")
    st.image(str(FIG / "eda_distributions.png"))
    st.dataframe(ds.raw.head(50), use_container_width=True)

# ------------------------------------------------------------------ Models
with t2:
    rows = [{"Model": k, "Accuracy": v["accuracy"], "F1-macro (test)": v["f1_macro"],
             "F1-macro (5-fold CV)": f"{v['cv_f1_macro_mean']:.4f} ± {v['cv_f1_macro_std']:.4f}"}
            for k, v in res["models"].items()]
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {"Accuracy": "{:.4f}", "F1-macro (test)": "{:.4f}"}),
        use_container_width=True, hide_index=True)
    cols = st.columns(3)
    for col, name in zip(cols, res["models"]):
        col.image(str(FIG / f"cm_{name.replace(' ', '_')}.png"),
                  caption=f"Confusion matrix – {name}")

# ------------------------------------------------------------------ Global XAI
with t3:
    a, b = st.columns(2)
    a.image(str(FIG / "xai_rf_importance.png"), caption="Impurity-based importance (RF)")
    b.image(str(FIG / "xai_rf_permutation.png"),
            caption="Permutation importance (RF, test set)")
    a, b = st.columns(2)
    a.image(str(FIG / "shap_global.png"), caption="Mean |SHAP|")
    b.image(str(FIG / "xai_tree.png"), caption="Decision Tree (top 3 levels)")
    st.subheader("SHAP per engagement class")
    c = st.columns(3)
    for col, cl in zip(c, CLASSES):
        col.image(str(FIG / f"shap_{cl}.png"))

# ------------------------------------------------------------------ Explain a player
with t4:
    rf, local = get_rf_and_explainer()
    client = get_client()
    st.caption(f"LLM backend: **{client.describe()}**  ·  build a player, predict, "
               "then generate a SHAP-grounded explanation.")
    c = st.columns(3)
    sessions = c[0].slider("Sessions per week", 0, 19, 9)
    dur = c[1].slider("Avg session duration (min)", 10, 179, 95)
    ach = c[2].slider("Achievements unlocked", 0, 49, 25)
    c = st.columns(3)
    level = c[0].slider("Player level", 1, 99, 50)
    hours = c[1].slider("Play time (hours)", 0.0, 24.0, 12.0)
    age = c[2].slider("Age", 15, 49, 32)

    row = dict.fromkeys(ds.feature_names, 0)
    row.update(Age=age, PlayTimeHours=hours, SessionsPerWeek=sessions,
               AvgSessionDurationMinutes=dur, PlayerLevel=level,
               AchievementsUnlocked=ach, InGamePurchases=0)
    for col_, default in [("Gender_Male", 1), ("Location_USA", 1),
                          ("GameGenre_Strategy", 1), ("GameDifficulty_Medium", 1)]:
        if col_ in row:
            row[col_] = default
    ev = local.explain_row(pd.Series(row))

    left, right = st.columns([1, 1.3])
    with left:
        st.metric("Predicted engagement", ev.predicted_class,
                  f"{ev.probabilities[ev.predicted_class]:.0%} confidence")
        st.bar_chart(pd.Series(ev.probabilities).reindex(CLASSES), y_label="Probability")
    with right:
        st.markdown("**Local SHAP contributions (why this prediction)**")
        st.dataframe(pd.DataFrame(ev.contributions)[["label", "value", "shap", "effect"]],
                     use_container_width=True, hide_index=True)

    if st.button("Generate explanation", type="primary"):
        with st.spinner("Running Explainer + Critic agents..."):
            out = agents.explain_player(ev, client, median=medians)
        for name in out["narratives"]:
            s = out["scores"][name]
            st.markdown(f"#### {name}")
            st.info(out["narratives"][name])
            m = st.columns(3)
            m[0].metric("Faithfulness", f"{s['faithfulness']:.2f}")
            m[1].metric("Coverage", f"{s['coverage']:.2f}")
            m[2].metric("Hallucination", f"{s['hallucination_rate']:.2f}")
            if s["hallucinated"]:
                st.caption("Hallucinated / unsupported: " + ", ".join(s["hallucinated"]))
        if out["critic"].get("issues"):
            st.caption(f"Critic flagged: {out['critic']['issues']}")

# ------------------------------------------------------------------ Faithfulness study
with t5:
    st.subheader("Does the Critic agent make explanations more faithful to SHAP?")
    if faith is None:
        st.warning("Run `python scripts/03_faithfulness_eval.py --n 60` to populate "
                   "this tab.")
    else:
        st.caption(f"Backend: {faith['backend']}  ·  {faith['n_players']} players  ·  "
                   "higher faithfulness / coverage is better, lower hallucination is better.")
        tbl = pd.DataFrame({
            s: {m: faith["strategies"][s][m]["mean"]
                for m in ["faithfulness", "coverage", "precision", "hallucination_rate"]}
            for s in faith["strategies"]}).T
        st.dataframe(tbl.style.format("{:.3f}"), use_container_width=True)
        if (FIG / "faithfulness_comparison.png").exists():
            st.image(str(FIG / "faithfulness_comparison.png"))
        st.markdown("**Example explanations**")
        for ex in faith.get("examples", [])[:3]:
            with st.expander(f"Player {ex['player']} — predicted {ex['predicted']}"):
                for s, text in ex["narratives"].items():
                    sc = ex["scores"][s]
                    st.markdown(f"*{s}* — faithfulness {sc['faithfulness']:.2f}")
                    st.write(text)
