"""Stage 1: clean data, train the 3 classifiers, evaluate, build global XAI,
persist models + results/metrics.json.  Run:  python scripts/01_train.py"""
import json

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gamexai import config as C
from gamexai import data, evaluate, models, xai


def top(series, k=10):
    return {k_: float(v) for k_, v in series.sort_values(ascending=False).head(k).items()}


def main():
    ds = data.make_dataset()
    print(f"Players: {len(ds.raw):,} | features: {len(ds.feature_names)} | "
          f"train/test: {len(ds.X_train):,}/{len(ds.X_test):,}")

    # EDA distribution figure
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for col, a in zip(["SessionsPerWeek", "AvgSessionDurationMinutes",
                       "PlayTimeHours"], ax):
        for c in C.CLASSES:
            a.hist(ds.raw.loc[ds.raw[C.TARGET] == c, col], bins=25, alpha=.55, label=c)
        a.set_title(col); a.legend(title="Engagement")
    plt.tight_layout(); plt.savefig(C.FIGURES / "eda_distributions.png", dpi=130)
    plt.close()

    metrics = {}
    fitted = {}
    for name, m in models.build_models().items():
        metrics[name] = evaluate.evaluate(m, name, ds)
        fitted[name] = m
        models.save(m, name)
        print(f"{name:20s} acc={metrics[name]['accuracy']:.4f} "
              f"F1={metrics[name]['f1_macro']:.4f} "
              f"CV={metrics[name]['cv_f1_macro_mean']:.4f}"
              f"+/-{metrics[name]['cv_f1_macro_std']:.4f}")

    imps = xai.global_importances(fitted["Random Forest"], fitted["Decision Tree"],
                                  fitted["Logistic Regression"], ds)

    # medians (for direction checking later) + feature names
    med = ds.raw[["SessionsPerWeek", "AvgSessionDurationMinutes", "PlayTimeHours",
                  "PlayerLevel", "AchievementsUnlocked", "Age"]].median().to_dict()

    summary = {
        "dataset": {
            "rows_raw": ds.n_raw, "rows_used": len(ds.raw),
            "n_features": len(ds.feature_names),
            "class_counts": ds.raw[C.TARGET].value_counts().to_dict(),
            "train": len(ds.X_train), "test": len(ds.X_test),
            "medians": med, "feature_names": ds.feature_names,
        },
        "models": metrics,
        "xai": {k: top(v) for k, v in imps.items()},
    }
    (C.RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print("Saved models + results/metrics.json")


if __name__ == "__main__":
    main()
