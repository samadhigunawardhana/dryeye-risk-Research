# evaluate_best.py
# Mirrors the training pipeline (same features, same split policy).

import json, joblib, numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

FEATURE_BASE = [
    "blink_rate_bpm",
    "incomplete_blink_ratio",
    "avg_ibi_sec",
    "redness_index",
]
ENGINEERED = ["ibr_x_red", "blink_per_sec", "ibi_inv", "ibi_lt6", "red_gt0_3", "ibr_gt0_2"]
FEATURE_COLS = FEATURE_BASE + ENGINEERED
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

def main(csv="dataset.csv", model_path="best_model.joblib", le_path="label_encoder.joblib", seed=None):
    df = pd.read_csv(csv)
    missing = [c for c in FEATURE_BASE + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = add_features(df)
    X = df[FEATURE_COLS].astype(float).values
    y_raw = df[TARGET_COL].astype(str).values

    le = joblib.load(le_path)
    y = le.transform(y_raw)

    # Use same split policy as training: stratified 20% holdout
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )

    clf = joblib.load(model_path)
    expected = getattr(clf, "n_features_in_", X.shape[1])
    if expected != len(FEATURE_COLS):
        raise ValueError(f"Model expects {expected} features, but {len(FEATURE_COLS)} provided.")

    ypred = clf.predict(Xte)

    acc = accuracy_score(yte, ypred)
    cm = confusion_matrix(yte, ypred)
    rep = classification_report(yte, ypred, target_names=le.classes_, digits=3)

    print("\nHoldout accuracy:", round(acc, 4))
    print("\nClassification report:\n", rep)
    print("\nConfusion matrix (rows=true, cols=pred):\n", cm)

    with open("evaluate_summary.json", "w") as f:
        json.dump({
            "holdout_accuracy": float(acc),
            "labels": le.classes_.tolist(),
            "confusion_matrix": cm.tolist(),
            "features_used": FEATURE_COLS,
            "target_col": TARGET_COL,
            "split_policy": "Stratified holdout 20%",
            "seed": seed,
        }, f, indent=2)
    print("\nSaved evaluate_summary.json")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="dataset.csv")
    ap.add_argument("--model", type=str, default="best_model.joblib")
    ap.add_argument("--le", type=str, default="label_encoder.joblib")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    main(csv=args.csv, model_path=args.model, le_path=args.le, seed=args.seed)
