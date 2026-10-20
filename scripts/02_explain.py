"""Stage 2: generate per-player explanations (template / LLM / LLM+critic) for a
few example players and print them.  Run:  python scripts/02_explain.py"""
import _bootstrap  # noqa: F401

from gamexai import agents, data, models
from gamexai.llm import LLMClient
from gamexai.xai import LocalExplainer


def main(n=3):
    ds = data.make_dataset()
    rf = models.load("Random Forest")
    local = LocalExplainer(rf, ds.feature_names)
    client = LLMClient()
    med = ds.raw[["SessionsPerWeek", "AvgSessionDurationMinutes", "PlayTimeHours",
                  "PlayerLevel", "AchievementsUnlocked", "Age"]].median().to_dict()
    print(f"LLM backend: {client.describe()}\n" + "=" * 70)

    for idx in ds.X_test.sample(n, random_state=7).index:
        ev = local.explain_row(ds.X_test.loc[idx])
        out = agents.explain_player(ev, client, median=med)
        print(f"\nPLAYER {idx} -> predicted {ev.predicted_class}")
        print(ev.as_prompt_block())
        for name, text in out["narratives"].items():
            s = out["scores"][name]
            print(f"\n[{name}] faithfulness={s['faithfulness']} "
                  f"(coverage={s['coverage']}, halluc={s['hallucination_rate']})")
            print("  " + text.replace("\n", "\n  "))
        print("-" * 70)


if __name__ == "__main__":
    main()
