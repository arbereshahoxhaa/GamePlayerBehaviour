"""Stage 3 - THE EXPERIMENT.

Sample N players, generate explanations with each strategy (template / llm /
llm+critic), score every explanation for faithfulness-to-SHAP, and report the
aggregate. Produces results/faithfulness.json and a comparison figure.

Research question: does adding a Critic agent make LLM explanations measurably
more faithful to the model's actual SHAP attributions?

Run:  python scripts/03_faithfulness_eval.py --n 60
"""
import argparse
import json
import time

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gamexai import agents, config as C, data, models
from gamexai.llm import LLMClient
from gamexai.xai import LocalExplainer

METRICS = ["faithfulness", "coverage", "precision", "hallucination_rate"]


def main(n):
    ds = data.make_dataset()
    rf = models.load("Random Forest")
    local = LocalExplainer(rf, ds.feature_names)
    client = LLMClient()
    med = ds.raw[["SessionsPerWeek", "AvgSessionDurationMinutes", "PlayTimeHours",
                  "PlayerLevel", "AchievementsUnlocked", "Age"]].median().to_dict()
    print(f"LLM backend: {client.describe()} | players: {n}")

    strategies = ["template", "llm", "llm+critic"] if client.available else ["template"]
    agg = {s: {m: [] for m in METRICS} for s in strategies}
    examples = []

    t0 = time.time()
    for i, idx in enumerate(ds.X_test.sample(n, random_state=C.SEED).index, 1):
        ev = local.explain_row(ds.X_test.loc[idx])
        out = agents.explain_player(ev, client, median=med)
        for s in strategies:
            for m in METRICS:
                agg[s][m].append(out["scores"][s][m])
        if len(examples) < 4:
            examples.append({"player": int(idx), "predicted": ev.predicted_class,
                             "narratives": out["narratives"],
                             "scores": {s: out["scores"][s] for s in strategies},
                             "critic_issues": out["critic"].get("issues")})
        if i % 10 == 0 or i == n:
            print(f"  {i}/{n}  ({time.time()-t0:.0f}s)")

    report = {"backend": client.describe(), "n_players": n, "strategies": {}}
    for s in strategies:
        report["strategies"][s] = {m: {"mean": float(np.mean(agg[s][m])),
                                       "std": float(np.std(agg[s][m]))}
                                   for m in METRICS}
    report["examples"] = examples
    (C.RESULTS / "faithfulness.json").write_text(json.dumps(report, indent=2))

    # figure: mean faithfulness per strategy
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = strategies
    means = [report["strategies"][s]["faithfulness"]["mean"] for s in labels]
    stds = [report["strategies"][s]["faithfulness"]["std"] for s in labels]
    bars = ax[0].bar(labels, means, yerr=stds, capsize=6,
                     color=["#9ecae1", "#f58518", "#54a24b"][:len(labels)])
    ax[0].set_ylim(0, 1.05); ax[0].set_title("Mean faithfulness-to-SHAP")
    ax[0].bar_label(bars, fmt="%.2f", padding=3)
    for m, mk, c in zip(["coverage", "hallucination_rate"], "os", ["#1b7837", "#b2182b"]):
        ax[1].plot(labels, [report["strategies"][s][m]["mean"] for s in labels],
                   marker=mk, label=m, color=c)
    ax[1].set_ylim(0, 1.05); ax[1].legend(); ax[1].set_title("Coverage vs hallucination")
    plt.tight_layout(); plt.savefig(C.FIGURES / "faithfulness_comparison.png", dpi=130)
    plt.close()

    print("\n=== FAITHFULNESS RESULTS ===")
    for s in strategies:
        r = report["strategies"][s]
        print(f"{s:12s} faithfulness={r['faithfulness']['mean']:.3f} "
              f"coverage={r['coverage']['mean']:.3f} "
              f"hallucination={r['hallucination_rate']['mean']:.3f}")
    print("Saved results/faithfulness.json + figure")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    main(ap.parse_args().n)
