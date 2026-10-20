"""Streamlit dashboard for the thesis demo.  Run:  streamlit run app.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).parent
FIG = ROOT / "results" / "figures"
CLASSES = ["Low", "Medium", "High"]

st.set_page_config(page_title="Player Behavior XAI", layout="wide")
st.title("Interpreting ML Models for Video Game Player Behavior (XAI)")

if not (ROOT / "results" / "metrics.json").exists():
    st.error("Run `python run_experiments.py` first.")
    st.stop()
res = json.loads((ROOT / "results" / "metrics.json").read_text())


@st.cache_data
def load():
    d = pd.read_csv(ROOT / "data" / "online_gaming_behavior_dataset.csv")
    return d


@st.cache_resource
def train_rf():
    d = load().drop(columns="PlayerID")
    y = d.pop("EngagementLevel").map({c: i for i, c in enumerate(CLASSES)})
    X = pd.get_dummies(d, dtype=int)
    m = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42).fit(X, y)
    return m, list(X.columns)


df = load()
t1, t2, t3, t4 = st.tabs(["Dataset", "Models", "Explainability (XAI)", "Try a player"])

with t1:
    d = res["dataset"]
    c = st.columns(4)
    c[0].metric("Players", f"{d['rows_used']:,}")
    c[1].metric("Features", d["n_features"])
    c[2].metric("Train / Test", f"{d['train']:,} / {d['test']:,}")
    c[3].metric("Missing values", 0)
    st.bar_chart(pd.Series(d["class_counts"]).reindex(CLASSES), x_label="Engagement", y_label="Players")
    st.image(str(FIG / "eda_distributions.png"))
    st.dataframe(df.head(50), use_container_width=True)

with t2:
    rows = [{"Model": k, "Accuracy": v["accuracy"], "F1-macro (test)": v["f1_macro"],
             "F1-macro (5-fold CV)": f"{v['cv_f1_macro_mean']:.4f} ± {v['cv_f1_macro_std']:.4f}"}
            for k, v in res["models"].items()]
    st.dataframe(pd.DataFrame(rows).style.format({"Accuracy": "{:.4f}", "F1-macro (test)": "{:.4f}"}),
                 use_container_width=True, hide_index=True)
    cols = st.columns(3)
    for col, name in zip(cols, res["models"]):
        col.image(str(FIG / f"cm_{name.replace(' ', '_')}.png"), caption=f"Confusion matrix – {name}")

with t3:
    a, b = st.columns(2)
    a.image(str(FIG / "xai_rf_importance.png"), caption="Impurity-based importance (RF)")
    b.image(str(FIG / "xai_rf_permutation.png"), caption="Permutation importance (RF, test set)")
    a, b = st.columns(2)
    a.image(str(FIG / "shap_global.png"), caption="Mean |SHAP|")
    b.image(str(FIG / "xai_tree.png"), caption="Decision Tree (top 3 levels)")
    st.subheader("SHAP per engagement class")
    c = st.columns(3)
    for col, cl in zip(c, CLASSES):
        col.image(str(FIG / f"shap_{cl}.png"))

with t4:
    st.caption("Adjust a player profile; the Random Forest predicts the engagement level.")
    rf, cols_ = train_rf()
    c = st.columns(3)
    sessions = c[0].slider("Sessions per week", 0, 19, 9)
    dur = c[1].slider("Avg session duration (min)", 10, 179, 95)
    ach = c[2].slider("Achievements unlocked", 0, 49, 25)
    c = st.columns(3)
    level = c[0].slider("Player level", 1, 99, 50)
    hours = c[1].slider("Play time (hours)", 0.0, 24.0, 12.0)
    age = c[2].slider("Age", 15, 49, 32)
    row = dict.fromkeys(cols_, 0)
    row.update(Age=age, PlayTimeHours=hours, SessionsPerWeek=sessions,
               AvgSessionDurationMinutes=dur, PlayerLevel=level, AchievementsUnlocked=ach,
               InGamePurchases=0, Gender_Male=1, Location_USA=1,
               GameGenre_Strategy=1, GameDifficulty_Medium=1)
    p = rf.predict_proba(pd.DataFrame([row])[cols_])[0]
    st.metric("Predicted engagement", CLASSES[int(p.argmax())], f"{p.max():.0%} confidence")
    st.bar_chart(pd.Series(p, index=CLASSES), y_label="Probability")
