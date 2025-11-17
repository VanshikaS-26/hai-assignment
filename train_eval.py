import argparse, re, sys, time, platform, random
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, precision_recall_fscore_support,
    ConfusionMatrixDisplay, confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# reproducibility
random.seed(42)
np.random.seed(42)

def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2)),
        ("clf", LogisticRegression(max_iter=300))
    ])

def load_csv(path, expected_cols=None):
    df = pd.read_csv(path)
    if expected_cols:
        for c in expected_cols:
            if c not in df.columns:
                raise ValueError(f"missing column: {c} in {path}")
    return df

def save_metrics(path, y_true, y_pred, target_names, extra_lines=None):
    with open(path, "w") as f:
        f.write(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"F1: {f1_score(y_true, y_pred):.4f}\n")
        f.write(f"Precision: {precision_score(y_true, y_pred):.4f}\n")
        f.write(f"Recall: {recall_score(y_true, y_pred):.4f}\n\n")
        f.write(classification_report(y_true, y_pred, target_names=target_names))
        if extra_lines:
            f.write("\n")
            for line in extra_lines:
                f.write(line + "\n")

def dataset_summary(df, path):
    lens = df["input_text"].astype(str).str.split().apply(len)
    pos = int((df["ground_truth_label"]==1).sum())
    neg = int((df["ground_truth_label"]==0).sum())
    with open(path,"w") as f:
        f.write(f"rows: {len(df)}\n")
        f.write(f"consensual: {pos}  too_forward: {neg}\n")
        f.write(f"avg_words: {lens.mean():.2f}  median_words: {lens.median():.0f}\n")
        f.write(f"min_words: {lens.min()}  max_words: {lens.max()}\n")

def error_buckets(text):
    t = str(text).lower()
    pats = {
        "pressure": [
            r"\bprove it\b", r"\bsay yes\b", r"\bjust say yes\b", r"\bstop being shy\b",
            r"\bdo not make me wait\b", r"\bkeep asking\b"
        ],
        "deadline_pressure": [
            r"\bdecide now\b", r"\btonight\b", r"\bmidnight\b", r"\bright now\b", r"\bno small talk\b"
        ],
        "photo_pressure": [
            r"\bsend a pic\b", r"\bpicture\b"
        ],
        "guilt_push": [
            r"\byou never reply\b", r"\byou do not care\b", r"\bat least\b", r"\balready\b", r"\bagain\b"
        ],
        "boundary_language": [
            r"\bcomfortable\b", r"\bokay if\b", r"\bok if\b", r"\bstop me\b", r"\bboundar", r"\bconsent\b",
            r"\bpace\b", r"\bslow down\b", r"\btone it down\b"
        ],
        "hedges_uncertainty": [
            r"\bmaybe\b", r"\bnot sure\b", r"\bi guess\b", r"\bi think\b", r"\bunless\b",
            r"\bif you want\b", r"\bif you really want\b", r"\bcould\b", r"\bmight\b"
        ],
        "teasing_push": [
            r"\bcome on\b", r"\bhang out\b", r"\bjust come\b", r"\bjust hang\b", r"\byou know you want to\b"
        ],
    }
    hits = []
    for name, regs in pats.items():
        if any(re.search(rg, t) for rg in regs):
            hits.append(name)
    return hits or ["other"]

def write_error_table(study_df, preds, path):
    err = study_df.loc[study_df["ground_truth_label"] != preds].copy()
    if err.empty:
        pd.DataFrame([{"bucket":"none", "count":0, "percent":0.0}]).to_csv(path, index=False)
        return
    from collections import Counter
    ctr = Counter()
    for txt in err["input_text"]:
        for b in error_buckets(txt):
            ctr[b] += 1
    total = sum(ctr.values())
    rows = [{"bucket": b, "count": c, "percent": round(100*c/total,2)} for b, c in ctr.most_common()]
    pd.DataFrame(rows).to_csv(path, index=False)

def ai_rationale(text):
    t = str(text).lower()
    checks = [
        ("pressure", [r"\bprove it\b", r"\bsay yes\b", r"\bjust say yes\b", r"\bstop being shy\b"]),
        ("deadline", [r"\bright now\b", r"\bdecide now\b", r"\btonight\b", r"\bmidnight\b"]),
        ("photo", [r"\bsend a pic\b", r"\bpicture\b"]),
        ("guilt", [r"\byou never reply\b", r"\byou do not care\b", r"\bat least\b", r"\balready\b", r"\bagain\b"]),
        ("boundary", [r"\bcomfortable\b", r"\bokay if\b", r"\bstop me\b", r"\bboundar\b", r"\bpace\b", r"\bslow down\b"]),
        ("hedge", [r"\bmaybe\b", r"\bnot sure\b", r"\bi guess\b", r"\bi think\b", r"\bunless\b", r"\bif you want\b", r"\bmight\b", r"\bcould\b"]),
        ("teasing", [r"\bcome on\b", r"\bhang out\b", r"\byou know you want to\b"]),
    ]
    for name, regs in checks:
        if any(re.search(p, t) for p in regs):
            return {
                "pressure": "flagged pressure phrase",
                "deadline": "deadline or time pressure",
                "photo": "photo request pressure",
                "guilt": "guilt framing detected",
                "boundary": "boundary or consent wording",
                "hedge": "hedge or uncertainty phrasing",
                "teasing": "teasing push phrasing",
            }[name]
    return "lexical pattern features"

def save_plots(y_true, probs, preds):
    # confusion matrix
    cm = confusion_matrix(y_true, preds, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["TOO_FORWARD","CONSENSUAL"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d", colorbar=False)
    fig.tight_layout()
    fig.savefig("a2_confusion_matrix.png", dpi=200)
    plt.close(fig)

    # calibration curve
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0,1],[0,1])
    plt.xlabel("predicted confidence")
    plt.ylabel("empirical accuracy")
    plt.title("reliability curve")
    plt.tight_layout()
    plt.savefig("a2_calibration.png", dpi=200)
    plt.close()

def main(args):
    # load data
    train_df = load_csv(args.train, ["input_text","ground_truth_label"])
    study_df = load_csv(args.study, ["input_text","ground_truth_label","model_output_label","model_score","human_rating_placeholder"])

    # split train for threshold tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        train_df["input_text"], train_df["ground_truth_label"],
        test_size=0.1, random_state=42, stratify=train_df["ground_truth_label"]
    )

    # train
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)

    # choose threshold on validation
    val_probs = pipe.predict_proba(X_val)[:,1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 33):
        f1_t = f1_score(y_val, (val_probs >= t).astype(int))
        if f1_t > best_f1:
            best_f1, best_t = f1_t, float(t)

    # evaluate on study set
    probs = pipe.predict_proba(study_df["input_text"])[:,1]
    preds = (probs >= best_t).astype(int)

    # metrics
    extra = [f"Chosen threshold: {best_t:.3f}  val_f1: {best_f1:.4f}"]
    save_metrics(args.metrics, study_df["ground_truth_label"], preds, ["TOO_FORWARD","CONSENSUAL_FLIRT"], extra_lines=extra)

    # per class stats
    p, r, f, s = precision_recall_fscore_support(study_df["ground_truth_label"], preds, labels=[0,1])
    with open(args.metrics, "a") as ftxt:
        ftxt.write("\nPer class:\n")
        ftxt.write(f"TOO_FORWARD  P={p[0]:.3f} R={r[0]:.3f} F1={f[0]:.3f} n={s[0]}\n")
        ftxt.write(f"CONSENSUAL   P={p[1]:.3f} R={r[1]:.3f} F1={f[1]:.3f} n={s[1]}\n")

    # save study with outputs and rationale
    label_name = ["TOO_FORWARD","CONSENSUAL_FLIRT"]
    study_out = study_df.copy()
    study_out["model_output_label"] = preds
    study_out["model_score"] = probs
    study_out["ai_readable_label"] = [label_name[i] for i in preds]
    study_out["ai_rationale"] = study_out["input_text"].apply(ai_rationale)
    study_out.to_csv(args.out, index=False)

    # save model
    joblib.dump(pipe, args.model)

    # plots
    save_plots(study_df["ground_truth_label"].values, probs, preds)

    # error table
    write_error_table(study_out, preds, args.errors)

    # dataset summary and run info
    dataset_summary(study_df, "a2_dataset_summary.txt")
    with open("a2_runinfo.txt","w") as f:
        f.write(f"python: {sys.version.split()[0]}\n")
        f.write(f"os: {platform.platform()}\n")
        f.write(f"time_s: {time.time()}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="a2_training_dataset.csv")
    p.add_argument("--study", default="a2_study_dataset.csv")
    p.add_argument("--out", default="a2_study_with_outputs.csv")
    p.add_argument("--model", default="a2_model.pkl")
    p.add_argument("--metrics", default="a2_metrics.txt")
    p.add_argument("--errors", default="a2_error_analysis.csv")
    args = p.parse_args()
    main(args)
