"""Stage 5 - load everything the pipeline produced into a SQL database.

Tables created (replaced on each run):
  players              - both datasets, shared schema + dataset tag + game name
  predictions          - Random Forest predictions on dataset 1's test set
  shap_contributions   - per-player local SHAP for a sample of players
  model_metrics        - accuracy / F1 / CV per model
  feature_importance   - importance per feature per XAI method
  faithfulness         - the faithfulness study, per backend/strategy/metric
  external_validation  - internal vs external F1 per model

Configure the target with DATABASE_URL in .env (PostgreSQL); falls back to a
local SQLite file if unset/unreachable.  Run:  python scripts/05_load_db.py
"""
import json

import _bootstrap  # noqa: F401
import pandas as pd

from gamexai import config as C, data, db, models
from gamexai.xai import LocalExplainer

SHARED = ["Age", "Gender", "Location", "GameGenre", "GameDifficulty", "PlayTimeHours",
          "InGamePurchases", "SessionsPerWeek", "AvgSessionDurationMinutes",
          "PlayerLevel", "AchievementsUnlocked", "EngagementLevel"]


def players_frame(ds):
    d1 = ds.raw.copy()
    d1["dataset"], d1["GameName"] = "kaggle_40k", None
    d1 = d1[SHARED + ["dataset", "GameName"]]

    ext = pd.read_csv(C.ROOT / "data" / "external" / "generated_gaming_dataset.csv")
    ext = ext.dropna(subset=["EngagementLevel"]).copy()
    ext["InGamePurchases"] = (pd.to_numeric(ext.get("PurchaseCount"), errors="coerce")
                              .fillna(0) > 0).astype(int)
    ext["dataset"] = "generated_10k"
    d2 = ext[SHARED + ["dataset", "GameName"]]
    return pd.concat([d1, d2], ignore_index=True)


def main():
    engine, label = db.get_engine()
    print(f"Target database: {label}")
    ds = data.make_dataset()

    def write(df, name):
        df.to_sql(name, engine, if_exists="replace", index=False)
        print(f"  {name:20s} {len(df):>6,} rows")

    # 1) players (both datasets)
    write(players_frame(ds), "players")

    # 2) predictions on dataset 1 test set
    rf = models.load("Random Forest")
    proba = rf.predict_proba(ds.X_test)
    pred = pd.DataFrame({
        "row_id": ds.X_test.index,
        "actual": ds.y_test.map(dict(enumerate(C.CLASSES))).values,
        "predicted": [C.CLASSES[i] for i in proba.argmax(1)],
        "p_low": proba[:, 0], "p_medium": proba[:, 1], "p_high": proba[:, 2],
    })
    write(pred, "predictions")

    # 3) local SHAP contributions for a sample of players
    local = LocalExplainer(rf, ds.feature_names)
    rows = []
    for idx in ds.X_test.sample(min(200, len(ds.X_test)), random_state=C.SEED).index:
        ev = local.explain_row(ds.X_test.loc[idx])
        for c in ev.contributions:
            rows.append({"row_id": idx, "predicted": ev.predicted_class,
                         "feature": c["label"], "value": c["value"],
                         "shap": c["shap"], "effect": c["effect"]})
    write(pd.DataFrame(rows), "shap_contributions")

    res = json.loads((C.RESULTS / "metrics.json").read_text())
    # 4) model metrics
    mm = pd.DataFrame([{"model": k, "accuracy": v["accuracy"],
                        "f1_macro": v["f1_macro"],
                        "cv_f1_mean": v["cv_f1_macro_mean"],
                        "cv_f1_std": v["cv_f1_macro_std"]}
                       for k, v in res["models"].items()])
    write(mm, "model_metrics")

    # 5) feature importance (per method)
    fi = [{"method": m, "feature": C.readable(f), "importance": v}
          for m, d in res["xai"].items() for f, v in d.items()]
    write(pd.DataFrame(fi), "feature_importance")

    # 6) faithfulness study (per backend)
    frows = []
    for p in sorted(C.RESULTS.glob("faithfulness_*.json")):
        rep = json.loads(p.read_text())
        backend = p.stem.split("_", 1)[1]
        for strat, d in rep["strategies"].items():
            frows.append({
                "backend": backend, "strategy": strat,
                "faithfulness": d["faithfulness"]["mean"],
                "coverage": d["coverage"]["mean"],
                "hallucination_rate": d["hallucination_rate"]["mean"],
                "avg_words": d["avg_words"]})
    if frows:
        write(pd.DataFrame(frows), "faithfulness")

    # 7) external validation
    ev_path = C.RESULTS / "external_validation.json"
    if ev_path.exists():
        e = json.loads(ev_path.read_text())["models"]
        ext = pd.DataFrame([{"model": k, **v} for k, v in e.items()])
        write(ext, "external_validation")

    print("Done.")


if __name__ == "__main__":
    main()
