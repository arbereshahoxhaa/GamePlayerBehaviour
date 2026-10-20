"""Explainable-AI layer: global importances and per-player local attributions.

Global views (impurity, permutation, SHAP summary) answer "what drives
engagement overall?".  The local view (`local_explanation`) answers "why THIS
player?" and produces the structured *evidence* that the LLM agents turn into
natural language and that `faithfulness.py` audits them against.
"""
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree

from . import config as C


# --------------------------------------------------------------- global XAI
def global_importances(rf, dt, lr, ds):
    """Return (dict of pandas Series) and write the global figures."""
    names = ds.feature_names
    imp = pd.Series(rf.feature_importances_, index=names).sort_values()
    _barh(imp, "Random Forest - Feature Importance (impurity)",
          "xai_rf_importance.png", "#4c78a8")

    perm = permutation_importance(rf, ds.X_test, ds.y_test, n_repeats=10,
                                  random_state=C.SEED, scoring="f1_macro", n_jobs=-1)
    perm_s = pd.Series(perm.importances_mean, index=names).sort_values()
    fig, a = plt.subplots(figsize=(7, 6))
    a.barh(perm_s.index, perm_s.values,
           xerr=pd.Series(perm.importances_std, index=names)[perm_s.index].values,
           color="#f58518")
    a.set_title("Random Forest - Permutation Importance (drop in F1-macro)")
    plt.tight_layout(); plt.savefig(C.FIGURES / "xai_rf_permutation.png", dpi=130); plt.close()

    # SHAP summary (global) on a sample, per class + overall.
    sample = ds.X_test.sample(min(600, len(ds.X_test)), random_state=C.SEED)
    sv = _shap_values(rf, sample)
    for ci, cname in enumerate(C.CLASSES):
        plt.figure()
        shap.summary_plot(sv[:, :, ci], sample, show=False, max_display=10)
        plt.title(f"SHAP - class '{cname}'")
        plt.tight_layout()
        plt.savefig(C.FIGURES / f"shap_{cname}.png", dpi=130, bbox_inches="tight")
        plt.close()
    shap_global = pd.Series(np.abs(sv).mean(axis=(0, 2)), index=names).sort_values()
    _barh(shap_global, "Mean |SHAP| (all classes)", "shap_global.png", "#54a24b")

    fig, a = plt.subplots(figsize=(18, 8))
    plot_tree(dt, feature_names=names, class_names=C.CLASSES, filled=True,
              max_depth=3, fontsize=8, ax=a)
    plt.savefig(C.FIGURES / "xai_tree.png", dpi=110, bbox_inches="tight"); plt.close()

    return {"rf_impurity": imp, "rf_permutation": perm_s, "shap_mean_abs": shap_global,
            "dt_importance": pd.Series(dt.feature_importances_, index=names)}


def _barh(series, title, fname, color):
    fig, a = plt.subplots(figsize=(7, 6))
    series.plot.barh(ax=a, color=color)
    a.set_title(title)
    plt.tight_layout(); plt.savefig(C.FIGURES / fname, dpi=130); plt.close()


def _shap_values(rf, X):
    """Return SHAP values shaped (n_samples, n_features, n_classes)."""
    sv = shap.TreeExplainer(rf).shap_values(X)
    return np.stack(sv, axis=-1) if isinstance(sv, list) else sv


# --------------------------------------------------------------- local XAI
@dataclass
class PlayerEvidence:
    """Structured, model-grounded facts about one prediction."""
    predicted_class: str
    probabilities: dict
    raw_features: dict                 # human-readable feature -> value
    contributions: list = field(default_factory=list)  # [{feature, value, shap, effect}]

    @property
    def top_features(self):
        return [c["feature"] for c in self.contributions]

    def as_prompt_block(self, k=5) -> str:
        lines = [f"Predicted engagement: {self.predicted_class} "
                 f"(confidence {self.probabilities[self.predicted_class]:.0%}).",
                 "Top SHAP contributions toward this prediction "
                 "(these are the ONLY features you may cite):"]
        for c in self.contributions[:k]:
            lines.append(f"  - {c['label']} = {c['value']} "
                         f"({c['effect']} this class, SHAP {c['shap']:+.3f})")
        return "\n".join(lines)


class LocalExplainer:
    """Wraps a TreeExplainer to produce `PlayerEvidence` for single players."""

    def __init__(self, rf, feature_names):
        self.rf = rf
        self.names = feature_names
        self.explainer = shap.TreeExplainer(rf)

    def explain_row(self, row: pd.Series, k=6) -> PlayerEvidence:
        x = row[self.names].to_frame().T
        proba = self.rf.predict_proba(x)[0]
        ci = int(proba.argmax())
        sv = self.explainer.shap_values(x)
        sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv  # (1, feat, class)
        contrib = sv[0, :, ci]
        order = np.argsort(np.abs(contrib))[::-1][:k]
        contributions = [{
            "feature": self.names[j],
            "label": C.readable(self.names[j]),
            "value": _fmt(row[self.names[j]]),
            "shap": float(contrib[j]),
            "effect": "supports" if contrib[j] >= 0 else "opposes",
        } for j in order]
        return PlayerEvidence(
            predicted_class=C.CLASSES[ci],
            probabilities={C.CLASSES[i]: float(p) for i, p in enumerate(proba)},
            raw_features={n: _fmt(row[n]) for n in self.names if row[n] != 0 or
                          n in ("Age", "SessionsPerWeek", "AvgSessionDurationMinutes")},
            contributions=contributions,
        )


def _fmt(v):
    if isinstance(v, float):
        return round(v, 2)
    return int(v) if isinstance(v, (int, np.integer)) else v
