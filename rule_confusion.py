# rule_confusion.py
import argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

LABELS = ["High", "Low", "Medium"]  # fixed order for rows/cols

def rule_label(br, ibr, ibi, red):
    score  = (br  < 12.0)
    score += (ibr > 0.20)
    score += (ibi < 6.0)
    score += (red > 0.30)
    return "Low" if score <= 1 else ("Medium" if score == 2 else "High")

def make_preds(df):
    return [
        rule_label(r["blink_rate_bpm"], r["incomplete_blink_ratio"],
                   r["avg_ibi_sec"], r["redness_index"])
        for _, r in df.iterrows()
    ]

def plot_cm(cm, title, out="confusion_matrix_rules.png"):
    fig, ax = plt.subplots(figsize=(6.5, 5.6), dpi=160)
    im = ax.imshow(cm, cmap="viridis")
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)

    # annotate counts
    vmax = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            color = "white" if v > vmax*0.6 else "black"
            ax.text(j, i, f"{v}", ha="center", va="center", color=color, fontsize=11)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count", rotation=270, labelpad=14)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="dataset.csv")
    ap.add_argument("--scope",   type=str, choices=["all","test"], default="test",
                    help="'all' = whole dataset, 'test' = 20% holdout to match ML results")
    ap.add_argument("--seed",    type=int, default=61)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    need = ["blink_rate_bpm","incomplete_blink_ratio","avg_ibi_sec","redness_index","risk_label"]
    miss = [c for c in need if c not in df.columns]
    if miss: raise SystemExit(f"Missing columns: {miss}")

    if args.scope == "test":
        # stratified 80/20 like your training
        train_df, test_df = train_test_split(
            df, test_size=0.20, random_state=args.seed, stratify=df["risk_label"]
        )
        y_true = test_df["risk_label"].astype(str).tolist()
        y_pred = make_preds(test_df)
        title = "Confusion matrix – Rule-based (test split, seed={})".format(args.seed)
        out   = "confusion_matrix_rules_test.png"
    else:
        y_true = df["risk_label"].astype(str).tolist()
        y_pred = make_preds(df)
        title = "Confusion matrix – Rule-based (full dataset)"
        out   = "confusion_matrix_rules_all.png"

    # order rows/cols to High, Low, Medium
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    # sanity: print row totals to verify supports
    print("Row totals (should match class supports for chosen scope):")
    for i, lab in enumerate(LABELS):
        print(f"  {lab}: {cm[i].sum()}")

    plot_cm(cm, title, out)

if __name__ == "__main__":
    main()
