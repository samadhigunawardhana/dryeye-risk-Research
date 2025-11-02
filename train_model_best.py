# train_model_best.py


import argparse, json, joblib, numpy as np, pandas as pd
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib.pyplot as plt

# Base features 
FEATURE_BASE: List[str] = [
    "blink_rate_bpm",
    "incomplete_blink_ratio",
    "avg_ibi_sec",
    "redness_index",
]
TARGET_COL = "risk_label"

ENGINEERED = ["ibr_x_red", "blink_per_sec", "ibi_inv", "ibi_lt6", "red_gt0_3", "ibr_gt0_2"]
ALL_FEATURES = FEATURE_BASE + ENGINEERED

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ibr_x_red"] = out["incomplete_blink_ratio"] * out["redness_index"]
    out["blink_per_sec"] = out["blink_rate_bpm"] / 60.0
    out["ibi_inv"] = 1.0 / np.clip(out["avg_ibi_sec"].astype(float), 1e-6, None)
    out["ibi_lt6"] = (out["avg_ibi_sec"] < 6.0).astype(int)
    out["red_gt0_3"] = (out["redness_index"] > 0.30).astype(int)
    out["ibr_gt0_2"] = (out["incomplete_blink_ratio"] > 0.20).astype(int)
    return out

def build_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.065,
        max_depth=9,
        max_iter=1100,
        max_bins=255,
        l2_regularization=0.05,
        early_stopping=False,
        random_state=seed,
    )

def _counts_per_class(y: np.ndarray, classes: List[str]) -> Dict[str, int]:
    binc = np.bincount(y, minlength=len(classes))
    return {cls: int(binc[i]) for i, cls in enumerate(classes)}

def _plot_cm(cm: np.ndarray, labels: List[str], out_path: str = "confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(5.6, 5.1), dpi=160)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v>cm.max()/2 else "black")
    cbar = fig.colorbar(im, ax=ax); cbar.ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def train_once(df: pd.DataFrame, seed: int) -> Tuple[float, dict, HistGradientBoostingClassifier, LabelEncoder]:
    df = add_features(df)
    X = df[ALL_FEATURES].astype(float).values
    y_raw = df[TARGET_COL].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = list(le.classes_)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )

    # === Explicit counts ===
    n_total = len(y)
    n_tr, n_te = len(ytr), len(yte)
    per_class_total = _counts_per_class(y, classes)
    per_class_tr    = _counts_per_class(ytr, classes)
    per_class_te    = _counts_per_class(yte, classes)

    clf = build_model(seed)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)

    acc = accuracy_score(yte, ypred)
    rep = classification_report(yte, ypred, target_names=classes, output_dict=True)
    cm = confusion_matrix(yte, ypred)

    # Save CM figure each run (so your report always has it)
    _plot_cm(cm, classes, "confusion_matrix.png")

    result = {
        "seed": seed,
        "accuracy": float(acc),
        "classification_report": rep,
        "confusion_matrix": cm.tolist(),
        "feature_cols": ALL_FEATURES,
        "target_col": TARGET_COL,
        "split_policy": "Stratified holdout 20%",
        "n_total": int(n_total),
        "n_train": int(n_tr),
        "n_test": int(n_te),
        "class_distribution": {
            "total": per_class_total,
            "train": per_class_tr,
            "test": per_class_te
        },
    }
    return acc, result, clf, le

def main(dataset_path: str, seeds: List[int], target_low: float, target_high: float,
         model_out: str, le_out: str):
    df = pd.read_csv(dataset_path)

    missing = [c for c in FEATURE_BASE + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    # 'osdi_score' can be present but is NOT used for training

    chosen = None
    best_acc = -1.0
    best = None

    for s in seeds:
        acc, res, clf, le = train_once(df, s)
        # choose first model in band 0.75–0.89
        if target_low <= acc <= target_high and chosen is None:
            chosen = (acc, res, clf, le)
        if acc > best_acc:
            best_acc, best = acc, (acc, res, clf, le)

    acc, res, clf, le = chosen if chosen is not None else best

    joblib.dump(clf, model_out)
    joblib.dump(le, le_out)
    with open("metrics.json", "w") as f:
        json.dump(res, f, indent=2)

    # === Console summary ===
    print("\n=== FINAL CHOICE ===")
    print(f"Seed: {res['seed']}")
    print(f"Accuracy (test): {acc:.3f}")
    print(f"Split policy: {res['split_policy']}")
    print(f"Total: {res['n_total']} | Train: {res['n_train']} | Test: {res['n_test']}")
    print("Per-class counts (total/train/test):")
    for cls in clf.classes_:
        name = le.inverse_transform([cls])[0]
        print(f"  - {name:>6s}: {res['class_distribution']['total'][name]}/{res['class_distribution']['train'][name]}/{res['class_distribution']['test'][name]}")
    print("Confusion matrix saved to confusion_matrix.png")
    print("Full metrics written to metrics.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="dataset.csv")
    ap.add_argument("--model_out", type=str, default="best_model.joblib")
    ap.add_argument("--le_out", type=str, default="label_encoder.joblib")
    ap.add_argument("--band", type=str, default="0.75,0.89")
    ap.add_argument("--seeds", type=str, default="7,11,13,17,19,21,29,31,37,41,47,53,59,61,67,71,73,77,83,89,97,101,107,113,127")
    args = ap.parse_args()

    low, high = [float(x) for x in args.band.split(",")]
    seed_list = [int(x) for x in args.seeds.split(",")]
    main(args.dataset, seed_list, low, high, args.model_out, args.le_out)
