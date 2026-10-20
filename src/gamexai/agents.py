"""The multi-agent explanation pipeline.

    PlayerEvidence  --[Explainer]-->  narrative
                    --[Critic]----->  faithfulness check + revised narrative

Three explanation strategies are supported so the thesis can compare them:
  * "template"  - deterministic, 100% faithful by construction (baseline)
  * "llm"       - single Explainer agent only
  * "llm+critic"- Explainer followed by the Critic agent (the proposed method)

The Critic is a *separate* agent with a different job: it is given the narrative
AND the ground-truth SHAP evidence and must flag/repair any claim not supported
by that evidence. This is why the pipeline is multi-agent rather than one prompt.
"""
from . import faithfulness as F

EXPLAINER_SYS = (
    "You are a data analyst explaining a machine-learning prediction about a "
    "video-game player's engagement level to a non-technical game designer. "
    "You MUST base every statement ONLY on the SHAP evidence provided. Do not "
    "invent features or numbers. Mention the top contributing features by name. "
    "Write 3-4 short sentences, then one concrete retention recommendation. "
    "Be plain and specific."
)

CRITIC_SYS = (
    "You are a strict fact-checker auditing an explanation of an ML prediction. "
    "You are given the explanation and the ONLY admissible evidence (SHAP "
    "contributions). Identify any sentence that cites a feature or effect NOT "
    "present in the evidence, or that contradicts it. Then rewrite the "
    "explanation so it is fully supported by the evidence, keeping it concise "
    "(3-4 sentences + one recommendation). "
    "Reply EXACTLY in this format:\n"
    "ISSUES: <comma-separated problems, or 'none'>\n"
    "REVISED: <the corrected explanation>"
)


# --------------------------------------------------------------- template baseline
def template_narrative(ev) -> str:
    drivers = [c for c in ev.contributions if c["effect"] == "supports"][:2]
    againsts = [c for c in ev.contributions if c["effect"] == "opposes"][:1]
    parts = [f"This player is predicted to have {ev.predicted_class} engagement "
             f"({ev.probabilities[ev.predicted_class]:.0%} confidence)."]
    if drivers:
        d = " and ".join(f"{c['label']} ({c['value']})" for c in drivers)
        parts.append(f"The main drivers of this prediction are {d}.")
    if againsts:
        c = againsts[0]
        parts.append(f"In contrast, {c['label']} ({c['value']}) pushes slightly "
                     f"the other way.")
    top = ev.contributions[0]
    parts.append(f"Recommendation: focus on {top['label']} to influence this "
                 f"player's engagement.")
    return " ".join(parts)


# --------------------------------------------------------------- agents
class ExplainerAgent:
    def __init__(self, client):
        self.client = client

    def explain(self, ev) -> str:
        if not self.client.available:
            return template_narrative(ev)
        user = (ev.as_prompt_block() + "\n\nWrite the explanation for a game "
                "designer now.")
        try:
            return self.client.chat(EXPLAINER_SYS, user)
        except Exception:
            return template_narrative(ev)


class CriticAgent:
    def __init__(self, client):
        self.client = client

    def review(self, narrative: str, ev) -> dict:
        if not self.client.available:
            return {"issues": "none (no LLM)", "revised": narrative}
        user = (f"EVIDENCE:\n{ev.as_prompt_block()}\n\n"
                f"EXPLANATION TO AUDIT:\n{narrative}")
        try:
            raw = self.client.chat(CRITIC_SYS, user, temperature=0.0)
        except Exception:
            return {"issues": "critic unavailable", "revised": narrative}
        issues, revised = _parse_critic(raw, fallback=narrative)
        return {"issues": issues, "revised": revised, "raw": raw}


def _parse_critic(raw, fallback):
    issues, revised = "none", fallback
    for line in raw.splitlines():
        s = line.strip()
        if s.upper().startswith("ISSUES:"):
            issues = s.split(":", 1)[1].strip()
        elif s.upper().startswith("REVISED:"):
            revised = s.split(":", 1)[1].strip()
    # If the model put the revision on following lines, grab everything after tag.
    if "REVISED:" in raw:
        revised = raw.split("REVISED:", 1)[1].strip() or fallback
    return issues, revised


# --------------------------------------------------------------- orchestrator
def explain_player(ev, client, median=None, k=4) -> dict:
    """Run all three strategies and score each for faithfulness."""
    explainer = ExplainerAgent(client)
    critic = CriticAgent(client)

    narratives = {"template": template_narrative(ev)}
    if client.available:
        base = explainer.explain(ev)
        narratives["llm"] = base
        rev = critic.review(base, ev)
        narratives["llm+critic"] = rev["revised"]
        critic_info = {"issues": rev.get("issues")}
    else:
        critic_info = {"issues": "n/a (template-only mode)"}

    scores = {name: F.score(text, ev, k=k, median=median)
              for name, text in narratives.items()}
    return {"evidence": ev, "narratives": narratives, "scores": scores,
            "critic": critic_info, "backend": client.describe()}
