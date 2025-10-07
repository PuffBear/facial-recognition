# src/metrics.py
from typing import Dict, List, Tuple, Callable
import json, itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate(predict_fn: Callable[[np.ndarray], str], X: List[np.ndarray], y: List[str]) -> Tuple[Dict, List[str]]:
    yhat = [predict_fn(x) for x in X]
    acc = accuracy_score(y, yhat)
    p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="macro", zero_division=0)
    return {"acc": acc, "macro_f1": f1, "precision": p, "recall": r}, yhat

def plot_confusion(y_true: List[str], y_pred: List[str], classes: List[str], outpath: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig = plt.figure(figsize=(max(6, len(classes) * 0.4),) * 2)
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=90)
    plt.yticks(ticks, classes)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] > 0:
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def save_json(obj: Dict, outpath: str) -> None:
    with open(outpath, "w") as f:
        json.dump(obj, f, indent=2)
