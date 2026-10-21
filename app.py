"""Interactive dashboard for the thesis demo.

Run:  streamlit run app.py   (after: python scripts/01_train.py)
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix
from sqlalchemy import inspect as sa_inspect

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from gamexai import agents, config as C, data, db, models, viz  # noqa: E402
from gamexai.llm import LLMClient  # noqa: E402
from gamexai.xai import LocalExplainer  # noqa: E402

CLASSES = C.CLASSES
st.set_page_config(page_title="Player Behavior XAI", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1150px;}
h1 {font-size: 1.7rem !important;}
.stTabs [data-baseweb="tab"] {font-size: 0.95rem; padding: 6px 14px;}
[data-testid="stMetric"] {background: #1f1f1e; border: 1px solid #2c2c2a;
  border-radius: 12px; padding: 14px 16px;}
[data-testid="stMetricLabel"] p {font-size: 0.8rem; color: #c3c2b7;}
</style>
""", unsafe_allow_html=True)

if not (C.RESULTS / "metrics.json").exists():
    st.error("Run `python scripts/01_train.py` first.")
    st.stop()
res = json.loads((C.RESULTS / "metrics.json").read_text())
faith = (json.loads((C.RESULTS / "faithfulness.json").read_text())
         if (C.RESULTS / "faithfulness.json").exists() else None)
# Per-backend study files (e.g. faithfulness_openai.json) for the model comparison.
faith_by_backend = {p.stem.split("_", 1)[1]: json.loads(p.read_text())
                    for p in sorted(C.RESULTS.glob("faithfulness_*.json"))}
extval = (json.loads((C.RESULTS / "external_validation.json").read_text())
          if (C.RESULTS / "external_validation.json").exists() else None)


@st.cache_resource
def get_dataset():
    return data.make_dataset()


@st.cache_resource
def get_model(name):
    return models.load(name)


@st.cache_resource
def get_explainer():
    ds = get_dataset()
    return LocalExplainer(get_model("Random Forest"), ds.feature_names)


@st.cache_resource
def get_client():
    return LLMClient()


@st.cache_resource
def get_db_engine():
    return db.get_engine()


@st.cache_data
def get_cm(name):
    ds = get_dataset()
    pred = get_model(name).predict(ds.X_test)
    return confusion_matrix(ds.y_test, pred)


ds = get_dataset()
medians = res["dataset"].get("medians", {})

# --------------------------------------------------------------- header + KPIs
st.title("Interpreting Player Behavior with Explainable AI")
st.caption("Machine-learning models predict player engagement; SHAP explains them; "
           "a multi-agent LLM pipeline turns that into plain language — and audits itself.")

best = max(res["models"], key=lambda m: res["models"][m]["f1_macro"])
top_feat, top_val = next(iter(res["xai"]["shap_mean_abs"].items()))
k = st.columns(4)
k[0].metric("Players analysed", f"{res['dataset']['rows_used']:,}")
k[1].metric("Best model (F1)", f"{res['models'][best]['f1_macro']:.2f}", best)
k[2].metric("Top driver", C.readable(top_feat),
            f"{top_val:.2f} mean |SHAP|", delta_color="off")
if faith:
    g = faith["strategies"].get("llm+guardrail", {})
    k[3].metric("Guardrail hallucination", f"{g['hallucination_rate']['mean']:.0%}",
                "eliminated", delta_color="off")
else:
    k[3].metric("Guardrail", "run study")

t1, t2, t3, t4, t5, t6 = st.tabs(
    ["  Overview  ", "  Models  ", "  What drives engagement  ",
     "  Explain a player  ", "  Faithfulness study  ", "  Database  "])

# --------------------------------------------------------------- Overview
with t1:
    c1, c2 = st.columns([1, 1.3], gap="large")
    with c1:
        st.subheader("Engagement mix")
        counts = pd.Series(res["dataset"]["class_counts"]).reindex(CLASSES)
        fig = go.Figure(go.Pie(
            labels=CLASSES, values=counts.values, hole=.58, sort=False,
            marker=dict(colors=[viz.CLASS_COLORS[c] for c in CLASSES],
                        line=dict(color="#141413", width=2)),
            textfont=dict(color=viz.INK),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:,} players<extra></extra>"))
        viz.style(fig, height=330, legend=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("How a behaviour differs by engagement")
        feat = st.selectbox("Feature", ["SessionsPerWeek", "AvgSessionDurationMinutes",
                                         "PlayTimeHours", "PlayerLevel",
                                         "AchievementsUnlocked", "Age"])
        fig = go.Figure()
        for cl in CLASSES:
            fig.add_histogram(x=ds.raw.loc[ds.raw[C.TARGET] == cl, feat], name=cl,
                              marker_color=viz.CLASS_COLORS[cl], opacity=.75, nbinsx=30)
        fig.update_layout(barmode="overlay")
        viz.style(fig, height=330)
        fig.update_layout(xaxis_title=C.readable(feat), yaxis_title="players")
        st.plotly_chart(fig, width="stretch")
    st.caption("Sessions per week and average session duration separate the classes "
               "cleanly; age and play-time barely move the needle.")

    st.subheader("Engagement by player category")
    cat = st.selectbox("Category", ["GameGenre", "GameDifficulty", "Gender", "Location"],
                       format_func=C.readable)
    share = (pd.crosstab(ds.raw[cat], ds.raw[C.TARGET], normalize="index")
             .reindex(columns=CLASSES) * 100)
    fig = go.Figure()
    for cl in CLASSES:
        fig.add_bar(name=cl, x=share.index.tolist(), y=share[cl].round(1).tolist(),
                    marker_color=viz.CLASS_COLORS[cl],
                    hovertemplate="%{x} · " + cl + ": %{y:.1f}%<extra></extra>")
    fig.update_layout(barmode="stack")
    viz.style(fig, height=320)
    fig.update_layout(yaxis_title="% of players", yaxis_range=[0, 100])
    st.plotly_chart(fig, width="stretch")
    st.caption("The engagement split stays almost identical across genres, difficulty, "
               "gender and location — these attributes carry little signal, which matches "
               "the SHAP ranking.")

    with st.expander("Browse raw data"):
        st.dataframe(ds.raw.head(100), width="stretch")

# --------------------------------------------------------------- Models
with t2:
    st.subheader("Model comparison")
    names = list(res["models"])
    fig = go.Figure()
    for metric, color in [("accuracy", viz.BLUE), ("f1_macro", viz.ORANGE)]:
        fig.add_bar(name="Accuracy" if metric == "accuracy" else "F1-macro",
                    x=names, y=[res["models"][n][metric] for n in names],
                    marker_color=color,
                    text=[f"{res['models'][n][metric]:.3f}" for n in names],
                    textposition="outside", textfont=dict(size=12, color=viz.INK2))
    viz.style(fig, height=360)
    fig.update_layout(barmode="group", yaxis_title="score", yaxis_range=[0, 1])
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Random Forest leads on both metrics. 5-fold CV confirms it is stable "
               f"({res['models'][best]['cv_f1_macro_mean']:.3f} ± "
               f"{res['models'][best]['cv_f1_macro_std']:.3f} F1).")

    st.subheader("Where does it get confused?")
    pick = st.radio("Model", names, index=names.index(best), horizontal=True)
    cm = get_cm(pick)
    fig = go.Figure(go.Heatmap(
        z=cm, x=CLASSES, y=CLASSES, colorscale=viz.SEQ_BLUE, showscale=False,
        text=cm, texttemplate="%{text}", textfont=dict(size=15, color=viz.INK),
        hovertemplate="true %{y} → predicted %{x}: %{z}<extra></extra>"))
    viz.style(fig, height=360, legend=False)
    fig.update_layout(xaxis_title="predicted", yaxis_title="actual",
                      yaxis_autorange="reversed")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, width="stretch")

    if extval:
        st.subheader("Does it generalise? (external dataset)")
        em = extval["models"]
        names2 = list(em)
        fig = go.Figure()
        fig.add_bar(name="Dataset 1 (test)", x=names2,
                    y=[em[n]["internal_f1_macro"] for n in names2], marker_color=viz.BLUE,
                    text=[f"{em[n]['internal_f1_macro']:.2f}" for n in names2],
                    textposition="outside")
        fig.add_bar(name="Dataset 2 (external)", x=names2,
                    y=[em[n]["external_f1_macro"] for n in names2], marker_color=viz.ORANGE,
                    text=[f"{em[n]['external_f1_macro']:.2f}" for n in names2],
                    textposition="outside")
        viz.style(fig, height=360)
        fig.update_layout(barmode="group", yaxis_title="F1-macro", yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")
        cl = extval["cleaning"]
        st.caption(
            f"Tested on an **independent, differently-generated** dataset "
            f"({cl['rows_clean']:,} rows after cleaning {cl['missing_cells_before']:,} "
            f"missing cells and {cl['implausible_age_fixed']} impossible ages). "
            "F1 drops from ~0.90 to ~0.49: the high in-sample score is partly an "
            "artefact of dataset 1's synthetic label. Random Forest stays the most "
            "robust, and still beats the majority-class baseline — the sessions/duration "
            "signal partially generalises, the exact decision boundary does not.")

# --------------------------------------------------------------- Drivers / XAI
with t3:
    st.subheader("Which features drive engagement?")
    method = st.radio("Explainability method", ["SHAP", "Permutation", "Impurity"],
                      horizontal=True, help="Three independent XAI methods — they agree.")
    key = {"SHAP": "shap_mean_abs", "Permutation": "rf_permutation",
           "Impurity": "rf_impurity"}[method]
    s = res["xai"][key]
    labels = [C.readable(f) for f in s]
    fig = viz.hbar(labels, list(s.values()),
                   color={"SHAP": viz.AQUA, "Permutation": viz.ORANGE,
                          "Impurity": viz.BLUE}[method], height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption("All three methods rank **sessions per week** and **average session "
               "duration** far above everything else — the result is robust, not an "
               "artefact of one technique.")
    with st.expander("Advanced: SHAP detail & decision tree (static)"):
        a, b = st.columns(2)
        a.image(str(C.FIGURES / "shap_global.png"), caption="Mean |SHAP|")
        b.image(str(C.FIGURES / "xai_tree.png"), caption="Decision tree (top levels)")
        cols = st.columns(3)
        for col, cl in zip(cols, CLASSES):
            col.image(str(C.FIGURES / f"shap_{cl}.png"), caption=f"SHAP · {cl}")

# --------------------------------------------------------------- Explain a player
with t4:
    client = get_client()
    local = get_explainer()
    st.caption(f"Backend: **{client.describe()}** — build a player, see why the model "
               "predicts what it does, then generate a plain-language explanation.")
    cats = {b: sorted(c.split("_", 1)[1] for c in ds.feature_names
                      if c.startswith(b + "_"))
            for b in ["GameGenre", "GameDifficulty", "Gender", "Location"]}
    with st.container():
        c = st.columns(3)
        sessions = c[0].slider("Sessions per week", 0, 19, 9)
        dur = c[1].slider("Avg session duration (min)", 10, 179, 95)
        ach = c[2].slider("Achievements unlocked", 0, 49, 25)
        c = st.columns(3)
        level = c[0].slider("Player level", 1, 99, 50)
        hours = c[1].slider("Play time (hours)", 0.0, 24.0, 12.0)
        age = c[2].slider("Age", 15, 49, 32)
        c = st.columns(4)
        genre = c[0].selectbox("Game genre", cats["GameGenre"])
        diff = c[1].selectbox("Difficulty", cats["GameDifficulty"])
        gender = c[2].selectbox("Gender", cats["Gender"])
        loc = c[3].selectbox("Location", cats["Location"])
        purch = c[0].checkbox("Made in-game purchases")

    row = dict.fromkeys(ds.feature_names, 0)
    row.update(Age=age, PlayTimeHours=hours, SessionsPerWeek=sessions,
               AvgSessionDurationMinutes=dur, PlayerLevel=level,
               AchievementsUnlocked=ach, InGamePurchases=int(purch))
    for col_ in [f"GameGenre_{genre}", f"GameDifficulty_{diff}",
                 f"Gender_{gender}", f"Location_{loc}"]:
        if col_ in row:
            row[col_] = 1
    ev = local.explain_row(pd.Series(row))

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.metric("Predicted engagement", ev.predicted_class,
                  f"{ev.probabilities[ev.predicted_class]:.0%} confidence",
                  delta_color="off")
        probs = pd.Series(ev.probabilities).reindex(CLASSES)
        fig = go.Figure(go.Bar(
            x=probs.values, y=CLASSES, orientation="h",
            marker_color=[viz.CLASS_COLORS[c] for c in CLASSES],
            text=[f"{v:.0%}" for v in probs.values], textposition="outside",
            hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
        viz.style(fig, height=230, legend=False)
        fig.update_layout(xaxis_range=[0, 1], title="Class probability")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("**Why — SHAP contributions**")
        contrib = ev.contributions[::-1]
        fig = go.Figure(go.Bar(
            x=[c["shap"] for c in contrib], y=[c["label"] for c in contrib],
            orientation="h",
            marker_color=[viz.DIVERGING["pos"] if c["shap"] >= 0 else viz.DIVERGING["neg"]
                          for c in contrib],
            customdata=[c["value"] for c in contrib],
            hovertemplate="%{y} = %{customdata}<br>SHAP %{x:+.3f}<extra></extra>"))
        viz.style(fig, height=300, legend=False)
        fig.update_layout(title="← pushes away · supports the prediction →",
                          xaxis_title="SHAP value")
        fig.add_vline(x=0, line_color=viz.AXIS)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch")

    if st.button("Generate explanation", type="primary"):
        with st.spinner("Explainer + Critic + guardrail…"):
            out = agents.explain_player(ev, client, median=medians)
        order = [n for n in ["template", "llm", "llm+critic", "llm+guardrail"]
                 if n in out["narratives"]]
        for name in order:
            sc = out["scores"][name]
            st.markdown(f"**{name}**")
            st.info(out["narratives"][name])
            m = st.columns(3)
            m[0].metric("Faithfulness", f"{sc['faithfulness']:.2f}")
            m[1].metric("Coverage", f"{sc['coverage']:.2f}")
            m[2].metric("Hallucination", f"{sc['hallucination_rate']:.2f}")

# --------------------------------------------------------------- Faithfulness study
with t5:
    st.subheader("Can we trust the AI's explanations?")
    if faith is None and not faith_by_backend:
        st.warning("Run `python scripts/03_faithfulness_eval.py --n 60` to populate this.")
    else:
        if len(faith_by_backend) > 1:
            which = st.radio("LLM model", list(faith_by_backend), horizontal=True,
                             help="Compare how explanation quality depends on model capacity.")
            faith = faith_by_backend[which]
        strat = list(faith["strategies"])
        labels = {"faithfulness": "Faithfulness (higher better)",
                  "coverage": "Coverage (higher better)",
                  "hallucination_rate": "Hallucination (lower better)",
                  "avg_words": "Length (words)"}
        choice = st.radio("Metric", list(labels), format_func=lambda m: labels[m],
                          horizontal=True)
        if choice == "avg_words":
            vals = [faith["strategies"][s]["avg_words"] for s in strat]
        else:
            vals = [faith["strategies"][s][choice]["mean"] for s in strat]
        winner = (min if choice == "hallucination_rate" else max)(range(len(vals)),
                                                                   key=lambda i: vals[i])
        colors = [viz.GOOD if i == winner else "#c9d6e5" for i in range(len(strat))]
        fmt = "{:.0f}" if choice == "avg_words" else "{:.3f}"
        fig = go.Figure(go.Bar(
            x=strat, y=vals, marker_color=colors,
            text=[fmt.format(v) for v in vals], textposition="outside",
            textfont=dict(size=13, color=viz.INK2),
            hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
        viz.style(fig, height=380, legend=False)
        fig.update_layout(yaxis_title=labels[choice].split(" (")[0])
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Study on {faith['n_players']} players · backend "
                   f"{faith['backend']}. The single LLM is fluent but invents features; "
                   "the LLM critic helps only a little; the **deterministic guardrail** "
                   "removes every hallucination while keeping the LLM's coverage and fluency.")

        inv = faith["strategies"].get("llm", {}).get("hallucinated_features", {})
        if inv:
            with st.expander("What does the LLM invent?"):
                f2 = viz.hbar([C.readable(k_) for k_ in inv], list(inv.values()),
                              color=viz.CRITICAL, height=220, suffix="")
                f2.update_traces(hovertemplate="%{y}: %{x} times<extra></extra>")
                st.plotly_chart(f2, width="stretch")
        with st.expander("Example explanations"):
            for ex in faith.get("examples", [])[:3]:
                st.markdown(f"**Player {ex['player']} — predicted {ex['predicted']}**")
                for s, txt in ex["narratives"].items():
                    st.markdown(f"*{s}* · faithfulness "
                                f"{ex['scores'][s]['faithfulness']:.2f}")
                    st.write(txt)
                st.divider()

# --------------------------------------------------------------- Database
with t6:
    st.subheader("SQL database")
    try:
        engine, label = get_db_engine()
    except Exception as e:
        engine, label = None, None
        st.error(f"Could not open a database: {e}")
    if engine is not None:
        st.caption(f"Connected to **{label}**. Every pipeline output is stored here "
                   "and can be browsed in pgAdmin / DBeaver / SSMS. Set `DATABASE_URL` "
                   "in `.env` for PostgreSQL; otherwise a local SQLite file is used.")
        tables = sorted(sa_inspect(engine).get_table_names())
        if not tables:
            st.warning("No tables yet — run `python scripts/05_load_db.py` to populate "
                       "the database.")
        else:
            counts = []
            for t in tables:
                try:
                    n = pd.read_sql(f'SELECT COUNT(*) AS n FROM "{t}"', engine)["n"][0]
                    counts.append({"table": t, "rows": int(n)})
                except Exception:
                    pass
            st.dataframe(pd.DataFrame(counts), width="stretch", hide_index=True)

            st.markdown("**Tables & columns** — pick a table to see its schema")
            insp = sa_inspect(engine)
            tsel = st.selectbox("Table", tables, key="schema_table")
            schema = pd.DataFrame([{"column": c["name"], "type": str(c["type"])}
                                   for c in insp.get_columns(tsel)])
            c_schema, c_prev = st.columns([1, 1.5], gap="large")
            c_schema.dataframe(schema, width="stretch", hide_index=True,
                               height=min(360, 40 + 35 * len(schema)))
            try:
                prev = pd.read_sql(f'SELECT * FROM "{tsel}" LIMIT 5', engine)
                c_prev.caption("First 5 rows")
                c_prev.dataframe(prev, width="stretch", hide_index=True)
            except Exception:
                pass

            # Mixed-case columns from the dataset must be double-quoted so they
            # work on PostgreSQL (which lowercases unquoted identifiers) as well
            # as SQLite; 2-arg ROUND needs a numeric cast on PostgreSQL.
            queries = {
                "Players by dataset":
                    "SELECT dataset, COUNT(*) AS players FROM players GROUP BY dataset",
                "Avg behaviour by engagement (dataset 1)":
                    'SELECT "EngagementLevel",'
                    ' ROUND(CAST(AVG("SessionsPerWeek") AS numeric),1) AS avg_sessions,'
                    ' ROUND(CAST(AVG("AvgSessionDurationMinutes") AS numeric),1) AS avg_duration'
                    " FROM players WHERE dataset='kaggle_40k' GROUP BY \"EngagementLevel\"",
                "Model metrics": "SELECT * FROM model_metrics",
                "Faithfulness by backend":
                    "SELECT backend, strategy, faithfulness, hallucination_rate"
                    " FROM faithfulness ORDER BY backend, strategy",
                "External validation": "SELECT * FROM external_validation",
            }
            avail = {k: q for k, q in queries.items()
                     if q.split(" FROM ")[1].split()[0].strip('"') in tables}
            st.markdown("**Query the database** — pick an example to start, then "
                        "edit freely and run. Read-only `SELECT` / `WITH`; quote "
                        "mixed-case columns, e.g. `\"EngagementLevel\"`.")
            if avail:
                choice = st.selectbox("Start from an example", list(avail))
                if st.session_state.get("_ex_choice") != choice:   # load on change
                    st.session_state["sql_editor"] = avail[choice]
                    st.session_state["_ex_choice"] = choice
            st.session_state.setdefault("sql_editor", "SELECT * FROM model_metrics")
            user_sql = st.text_area("SQL", key="sql_editor", height=120,
                                    label_visibility="collapsed")
            if st.button("Run query", type="primary"):
                low = user_sql.strip().rstrip(";").lower()
                starts = low.startswith("select") or low.startswith("with")
                writes = re.search(r"\b(insert|update|delete|drop|alter|create|"
                                   r"truncate|grant|revoke|call|merge|copy)\b", low)
                if not starts or ";" in user_sql.strip().rstrip(";") or writes:
                    st.error("Only a single read-only SELECT / WITH query is allowed.")
                else:
                    try:
                        df = pd.read_sql(user_sql, engine)
                        st.caption(f"{len(df):,} rows")
                        st.dataframe(df, width="stretch", hide_index=True)
                    except Exception as e:
                        st.warning(f"Query failed: {e}")
