"""Model evaluation: cross-validation, held-out metrics, confusion matrices."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import config as C


def evaluate(model, name, ds):
    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    cv_f1 = cross_val_score(model, ds.X_train, ds.y_train, cv=cv, scoring="f1_macro")
    model.fit(ds.X_train, ds.y_train)
    pred = model.predict(ds.X_test)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        ds.y_test, pred, display_labels=C.CLASSES, ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
    plt.tight_layout()
    plt.savefig(C.FIGURES / f"cm_{name.replace(' ', '_')}.png", dpi=130)
    plt.close()

    return {
        "accuracy": accuracy_score(ds.y_test, pred),
        "f1_macro": f1_score(ds.y_test, pred, average="macro"),
        "cv_f1_macro_mean": cv_f1.mean(),
        "cv_f1_macro_std": cv_f1.std(),
        "per_class": classification_report(
            ds.y_test, pred, target_names=C.CLASSES, output_dict=True),
    }
