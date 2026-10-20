"""The three classifiers from the thesis proposal, plus persistence."""
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config as C


def build_models() -> dict:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=C.SEED)),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=C.SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=C.SEED),
    }


def save(model, name: str):
    joblib.dump(model, C.MODELS / f"{name.replace(' ', '_')}.joblib")


def load(name: str):
    return joblib.load(C.MODELS / f"{name.replace(' ', '_')}.joblib")
