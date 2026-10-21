"""Stage 4 - external validation / transfer test.

Take the models trained on dataset 1 (the 40k Kaggle set) and test them,
unchanged, on an INDEPENDENT second dataset (v3ngen's generated set, 10k rows,
different generation process, different class balance, real missing values).

This answers: does the "sessions + duration drive engagement" model generalise
to a differently-generated population, or did it just memorise dataset 1?

Honest caveats (reported, not hidden):
  * Both datasets are synthetic, so this tests robustness to distribution shift,
    not real-world validity.
  * Some shared column names differ in scale/meaning (e.g. PlayTimeHours ranges
    0-24 in dataset 1 but up to ~2400 here); those features have ~0 importance,
    so the transfer leans on the comparable, dominant features.

Run:  python scripts/04_external_validation.py
"""
import json

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from gamexai import config as C, data, models

EXT = C.ROOT / "data" / "external" / "generated_gaming_dataset.csv"
NUM = ["Age", "PlayTimeHours", "SessionsPerWeek", "AvgSessionDurationMinutes",
       "PlayerLevel", "AchievementsUnlocked"]
CATS = ["Gender", "Location", "GameGenre", "GameDifficulty"]


def clean(df):
    """Clean dataset 2 and report what was done."""
    log = {"rows_raw": len(df)}
    df = df.dropna(subset=["EngagementLevel"]).copy()
    log["dropped_missing_target"] = log["rows_raw"] - len(df)

    # Impossible ages -> treat as missing (dataset has values up to 199).
    bad_age = (df["Age"] > 99) | (df["Age"] < 10)
    log["implausible_age_fixed"] = int(bad_age.sum())
    df.loc[bad_age, "Age"] = np.nan

    log["missing_cells_before"] = int(df[NUM + CATS].isna().sum().sum())
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
    for c in CATS:
        df[c] = df[c].fillna(df[c].mode(dropna=True)[0])
    # InGamePurchases isn't a column here; derive it from PurchaseCount.
    df["InGamePurchases"] = (pd.to_numeric(df.get("PurchaseCount"), errors="coerce")
                             .fillna(0) > 0).astype(int)
    log["rows_clean"] = len(df)
    return df, log


def to_model_space(df, feature_names):
    """Map cleaned dataset 2 to the exact feature columns the model expects."""
    X = pd.get_dummies(df[NUM + CATS + ["InGamePurchases"]], columns=CATS, dtype=int)
    return X.reindex(columns=feature_names, fill_value=0)


def main():
    ds = data.make_dataset()                       # dataset 1 (for feature schema)
    raw = pd.read_csv(EXT)
    clean_df, log = clean(raw)
    y = clean_df["EngagementLevel"].map({c: i for i, c in enumerate(C.CLASSES)})
    X = to_model_space(clean_df, ds.feature_names)

    print(f"Dataset 2: {log['rows_raw']} -> {log['rows_clean']} rows after cleaning "
          f"({log['implausible_age_fixed']} implausible ages, "
          f"{log['missing_cells_before']} missing cells imputed)")
    print("Class balance (dataset 2):", dict(clean_df["EngagementLevel"].value_counts()))

    res = json.loads((C.RESULTS / "metrics.json").read_text())
    report = {"cleaning": log,
              "ext_class_counts": clean_df["EngagementLevel"].value_counts().to_dict(),
              "models": {}}
    for name in res["models"]:
        m = models.load(name)
        pred = m.predict(X)
        report["models"][name] = {
            "internal_f1_macro": res["models"][name]["f1_macro"],
            "external_accuracy": accuracy_score(y, pred),
            "external_f1_macro": f1_score(y, pred, average="macro"),
        }
        r = report["models"][name]
        print(f"{name:20s} internal F1={r['internal_f1_macro']:.3f}  ->  "
              f"external F1={r['external_f1_macro']:.3f}  acc={r['external_accuracy']:.3f}")

    # figure: internal vs external F1 per model
    names = list(report["models"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    ax.bar(x - .2, [report["models"][n]["internal_f1_macro"] for n in names], .4,
           label="Dataset 1 (test)", color="#2a78d6")
    ax.bar(x + .2, [report["models"][n]["external_f1_macro"] for n in names], .4,
           label="Dataset 2 (external)", color="#eb6834")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1)
    ax.set_ylabel("F1-macro"); ax.set_title("Internal vs external (transfer) performance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(C.FIGURES / "external_validation.png", dpi=130)
    plt.close()

    (C.RESULTS / "external_validation.json").write_text(json.dumps(report, indent=2,
                                                                   default=float))
    print("Saved results/external_validation.json + figure")


if __name__ == "__main__":
    main()
