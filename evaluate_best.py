# evaluate_best.py
# Re-run the saved model on the same 20% holdout (using the stored seed) and report counts + CM.

import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

FEATURE_BASE = ["blink_rate_bpm","incomplete_blink_ratio","avg_ibi_sec","redness_index"]
ENGINEERED = ["ibr_x_red","blink_per_sec","ibi_inv","ibi_lt6","red_gt0_3","ibr_gt0_2"]
ALL_FEATURES = FEATURE_BASE + ENGINEERED
TARGET_COL = "risk_label"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ibr_x_red"] = out["incomplete_blink_ratio"] * out["redness_index"]
    out["blink_per_sec"] = out["blink_rate_bpm"] / 60.0
    out["ibi_inv"] = 1.0 / np.clip(out["avg_ibi_sec"].astype(float), 1e-6, None)
    out["ibi_lt6"] = (out["avg_ibi_sec"] < 6.0).astype(int)
    out["red_gt0_3"] = (out["redness_index"] > 0.30).astype(int)
    out["ibr_gt0_2"] = (out["incomplete_blink_ratio"] > 0.20).astype(int)
    return out

def cm_plot(cm, labels, out_path="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(5.6, 5.1), dpi=160)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v>cm.max()/2 else "black")
    fig.colorbar(im, ax=ax).ax.set_ylabel("Count")
    plt.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)

def main(dataset="dataset.csv", model="best_model.joblib", labeler="label_encoder.joblib", metrics_path="metrics.json"):
    df = pd.read_csv(dataset)
    df = add_features(df)

    # Load saved items
    clf = joblib.load(model)
    le  = joblib.load(labeler)
    metrics = json.loads(Path(metrics_path).read_text())
    seed = int(metrics["seed"])

    X = df[ALL_FEATURES].astype(float).values
    y = le.transform(df[TARGET_COL].astype(str).values)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)

    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    cm  = confusion_matrix(yte, ypred)
    report = classification_report(yte, ypred, target_names=list(le.classes_))

    # Counts
    print("\n=== EVALUATION (reproduced) ===")
    print(f"Seed: {seed}")
    print(f"Total: {len(y)} | Train: {len(ytr)} | Test: {len(yte)}")
    for i, cls in enumerate(le.classes_):
        print(f"  - {cls:>6s}: total={np.sum(y==i)}, train={np.sum(ytr==i)}, test={np.sum(yte==i)}")
    print(f"\nAccuracy (test): {acc:.3f}\n")
    print(report)

    cm_plot(cm, list(le.classes_), "confusion_matrix.png")
    print("Saved confusion_matrix.png")

if __name__ == "__main__":
    main()
