# train_model_best.py
"""
Selects the best classifier by macro-F1 and saves artifacts:
- best_model.joblib
- label_encoder.joblib
- confusion_matrix.png
- feature_ranking.png (if supported by the model)
- metrics.json
Requires dataset.csv with:
  blink_rate_bpm, incomplete_blink_ratio, avg_ibi_sec, redness_index, risk_label
"""

import argparse, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import joblib

FEATURE_COLS = ["blink_rate_bpm","incomplete_blink_ratio","avg_ibi_sec","redness_index"]
TARGET_COL = "risk_label"

def plot_confusion(cm, labels, out_path="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels, ylabel="True label", xlabel="Predicted label",
           title="Confusion Matrix")
    thresh = cm.max()/2 if cm.max()>0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,format(cm[i,j],"d"),ha="center",va="center",
                    color="white" if cm[i,j]>thresh else "black")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)

def rank_features(model, feature_names, out_path="feature_ranking.png"):
    values = None
    name = type(model).__name__
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.array(model.coef_)
        values = np.mean(np.abs(coef), axis=0)
    if values is None:
        return
    idx = np.argsort(values)[::-1]
    plt.figure(figsize=(6,4))
    plt.bar(range(len(idx)), values[idx])
    plt.xticks(range(len(idx)), [feature_names[i] for i in idx], rotation=45, ha="right")
    plt.title(f"Feature ranking ({name})"); plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def build_candidates(random_state=42):
    cands = []
    lr = Pipeline([("scaler", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"))])
    lr_grid = {"clf__C":[0.1,1,3,10]}
    cands.append(("logreg", lr, lr_grid))

    svc = Pipeline([("scaler", StandardScaler()),
                    ("clf", SVC(class_weight="balanced", probability=True))])
    svc_grid = {"clf__C":[0.5,1,3,10], "clf__gamma":["scale","auto"], "clf__kernel":["rbf"]}
    cands.append(("svc", svc, svc_grid))

    rf = RandomForestClassifier(class_weight="balanced", random_state=random_state)
    rf_grid = {"n_estimators":[300,600], "max_depth":[None,6,10], "min_samples_split":[2,5]}
    cands.append(("rf", rf, rf_grid))

    gb = GradientBoostingClassifier(random_state=random_state)
    gb_grid = {"n_estimators":[200,400], "learning_rate":[0.05,0.1], "max_depth":[2,3]}
    cands.append(("gb", gb, gb_grid))

    hgb = HistGradientBoostingClassifier(random_state=random_state)
    hgb_grid = {"max_depth":[None,6,10], "learning_rate":[0.05,0.1], "max_iter":[300,500]}
    cands.append(("hgb", hgb, gb_grid))
    return cands

def main(dataset_path: str):
    df = pd.read_csv(dataset_path)
    if not set(FEATURE_COLS).issubset(df.columns) or TARGET_COL not in df.columns:
        raise SystemExit(f"dataset must contain {FEATURE_COLS + [TARGET_COL]}")

    X = df[FEATURE_COLS].astype(float).values
    y_raw = df[TARGET_COL].astype(str).values

    le = LabelEncoder(); y = le.fit_transform(y_raw)
    class_counts = pd.Series(y_raw).value_counts()
    use_loocv = class_counts.min() < 5

    candidates = build_candidates()
    best_model = None; best_name = None
    best_f1 = -1.0; best_summary = None

    if use_loocv:
        cv = LeaveOneOut()
        print("Using LOOCV (min class < 5).")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print("Using Stratified 5-fold CV.")

    for name, est, grid in candidates:
        gcv = GridSearchCV(est, grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True, verbose=0)
        gcv.fit(X, y)
        mean_f1 = gcv.best_score_
        print(f"[{name}] f1_macro={mean_f1:.3f} params={gcv.best_params_}")
        if mean_f1 > best_f1:
            best_f1 = mean_f1; best_model = gcv.best_estimator_; best_name = name
            best_summary = {"cv_f1_macro": float(mean_f1), "best_params": gcv.best_params_}

    try:
        if not use_loocv:
            best_model = CalibratedClassifierCV(best_model, method="sigmoid", cv=5).fit(X, y)
    except Exception:
        pass

    if not use_loocv:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        best_model.fit(Xtr, ytr)
        ypred = best_model.predict(Xte)
        acc = accuracy_score(yte, ypred)
        rep = classification_report(yte, ypred, target_names=le.classes_, output_dict=True)
        cm = confusion_matrix(yte, ypred)
    else:
        best_model.fit(X, y)
        ypred = best_model.predict(X)
        acc = accuracy_score(y, ypred)
        rep = classification_report(y, ypred, target_names=le.classes_, output_dict=True)
        cm = confusion_matrix(y, ypred)

    plot_confusion(cm, le.classes_, "confusion_matrix.png")

    final = best_model
    if hasattr(final, "base_estimator"): final = final.base_estimator
    if hasattr(final, "steps"): final = final.steps[-1][1]
    try: rank_features(final, FEATURE_COLS, "feature_ranking.png")
    except Exception: pass

    joblib.dump(best_model, "best_model.joblib")
    joblib.dump(le, "label_encoder.joblib")

    metrics = {
        "selected_model": best_name,
        "cv_strategy": "LOOCV" if use_loocv else "Stratified 5-fold + 70/30 holdout",
        "cv_summary": best_summary,
        "holdout_accuracy": float(acc),
        "classification_report": rep,
        "class_counts": class_counts.to_dict(),
        "feature_cols": FEATURE_COLS,
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved: best_model.joblib, label_encoder.joblib, confusion_matrix.png, feature_ranking.png, metrics.json")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset.csv")
    args = p.parse_args()
    main(args.dataset)
