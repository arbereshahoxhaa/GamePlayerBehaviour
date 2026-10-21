"""Measuring whether a natural-language explanation is *faithful* to the SHAP
attributions it claims to describe.

This is the thesis's measurable contribution. A fluent LLM narrative is useless
if it cites features the model never used. Given a generated text and the
`PlayerEvidence` (ground-truth SHAP contributions), we compute:

  * coverage   - of the top-k SHAP features, how many are mentioned      (recall)
  * precision  - of the features mentioned, how many are truly influential
  * halluc.    - mentions of non-influential / absent features           (1-precision'ish)
  * direction  - of mentioned top features, how many have the right sign/magnitude
  * faithfulness - composite in [0,1]

Feature detection is alias-based keyword matching (config.FEATURE_ALIASES); it is
a deliberately simple, transparent heuristic, which is appropriate to state as a
limitation in the thesis.
"""
import re

from . import config as C


def canonical(feature: str) -> str:
    """Map a model column (possibly a one-hot dummy) to its narrative concept,
    e.g. 'GameGenre_RPG' -> 'GameGenre', 'SessionsPerWeek' -> 'SessionsPerWeek'."""
    for base in C.FEATURE_ALIASES:
        if feature == base or feature.startswith(base + "_"):
            return base
    return feature


def guardrail_filter(narrative: str, evidence) -> str:
    """Deterministic SHAP-grounded guardrail: drop any sentence that cites a
    feature outside the evidence, keeping the rest of the LLM's text untouched.

    Unlike an LLM critic that rewrites wholesale (and tends to over-prune), this
    removes only the offending sentences, so fluent, faithful content survives.
    """
    admissible = {canonical(c["feature"]) for c in evidence.contributions}
    sentences = re.split(r"(?<=[.!?])\s+", narrative.strip())
    kept = [s for s in sentences
            if all(f in admissible for f in detect_mentions(s))]
    return " ".join(kept).strip() or narrative


def detect_mentions(text: str) -> dict:
    """Map each feature mentioned in `text` -> inferred value polarity ('high' /
    'low' / None), using the nearest polarity word in the same sentence."""
    low = text.lower()
    found = {}
    for feature, aliases in C.FEATURE_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            for m in re.finditer(r"\b" + re.escape(alias) + r"\b", low):
                window = low[max(0, m.start() - 60): m.end() + 60]
                found.setdefault(feature, _polarity(window))
                break
            if feature in found:
                break
    return found


def _polarity(window: str):
    hi = any(re.search(r"\b" + re.escape(w) + r"\b", window) for w in C.POLARITY_HIGH)
    lo = any(re.search(r"\b" + re.escape(w) + r"\b", window) for w in C.POLARITY_LOW)
    if hi and not lo:
        return "high"
    if lo and not hi:
        return "low"
    return None


def score(text: str, evidence, k: int = 3, median=None) -> dict:
    """Score `text` against the ground-truth SHAP `evidence`.

    An explanation is *faithful* if it discusses only features the model actually
    attributed the prediction to. The admissible set is therefore every feature
    shown in the evidence (`evidence.contributions`); a mention outside it is a
    hallucination. Coverage (recall) is measured against the top-`k` drivers.

    `median`: optional {feature: median_value} to judge whether a stated value
    polarity ('high'/'low') matches the player's actual value.
    """
    # Work in canonical concept-space so one-hot dummies match narrative words.
    top = list(dict.fromkeys(canonical(f) for f in evidence.top_features[:k]))
    # Admissible = every feature we actually showed the model.
    admissible = {canonical(c["feature"]) for c in evidence.contributions}
    mentioned = detect_mentions(text)          # already canonical keys
    m_set = set(mentioned)

    hits = [f for f in top if f in m_set]
    coverage = len(hits) / len(top) if top else 0.0
    precision = (len([f for f in m_set if f in admissible]) / len(m_set)
                 if m_set else 0.0)
    hallucinated = sorted(f for f in m_set if f not in admissible)
    halluc_rate = len(hallucinated) / len(m_set) if m_set else 0.0

    # Direction: does the stated polarity match the player's real value?
    dir_checked, dir_ok = 0, 0
    if median is not None:
        vals = {canonical(c["feature"]): c["value"] for c in evidence.contributions}
        for f in hits:
            pol = mentioned.get(f)
            if pol is None or f not in median or not isinstance(vals.get(f), (int, float)):
                continue
            dir_checked += 1
            actual = "high" if vals[f] >= median[f] else "low"
            dir_ok += int(pol == actual)
    direction = dir_ok / dir_checked if dir_checked else None

    parts = [coverage, precision, 1 - halluc_rate]
    if direction is not None:
        parts.append(direction)
    faithfulness = sum(parts) / len(parts)

    return {
        "coverage": round(coverage, 3),
        "precision": round(precision, 3),
        "hallucination_rate": round(halluc_rate, 3),
        "hallucinated": hallucinated,
        "direction_accuracy": round(direction, 3) if direction is not None else None,
        "faithfulness": round(faithfulness, 3),
        "mentioned": sorted(m_set),
        "top_shap_features": top,
    }
