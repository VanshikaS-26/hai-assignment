import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import ttest_ind

base = Path(".")
logs_path = base / "logs" / "a3_logs.csv"
sess_path = base / "logs" / "a3_sessions.csv"
study_path = base / "a2_study_with_outputs.csv"

print("Loading files...")
logs = pd.read_csv(logs_path)
sess = pd.read_csv(sess_path)
study = pd.read_csv(study_path)

print("logs shape:", logs.shape)
print("sessions shape:", sess.shape)
print("study shape:", study.shape)

# merge logs with ground truth and AI outputs
merged = logs.merge(
    study.reset_index().rename(columns={"index": "message_index_study"}),
    left_on="message_index",
    right_on="message_index_study",
    how="left"
)

# sanity check
print("Merged shape:", merged.shape)

# helper to map labels
def gt_label_str(row):
    # 1 means consensual, 0 means too forward
    if row["ground_truth_label"] == 1:
        return "CONSENSUAL_FLIRT"
    else:
        return "TOO_FORWARD"

merged["gt_label_str"] = merged.apply(gt_label_str, axis=1)

# model label as string
merged["model_label_str"] = merged["ai_label"].fillna(merged["ai_readable_label"])

# human labels

def human_label_noai(ans):
    if ans == "consensual":
        return "CONSENSUAL_FLIRT"
    if ans == "too_forward":
        return "TOO_FORWARD"
    return None   # not sure

def human_label_with_ai(ans, ai_label):
    if ans == "agree":
        return ai_label
    if ans == "disagree_too_forward":
        return "TOO_FORWARD"
    return None

merged["human_label"] = None

# no AI condition
mask_noai = merged["cond"] == "noai"
merged.loc[mask_noai, "human_label"] = merged.loc[mask_noai, "answer"].apply(human_label_noai)

# with AI condition
mask_ai = merged["cond"] == "ai"
merged.loc[mask_ai, "human_label"] = merged.loc[mask_ai].apply(
    lambda r: human_label_with_ai(r["answer"], r["model_label_str"]),
    axis=1
)

# correctness flags
def label_to_int(lbl):
    if lbl == "CONSENSUAL_FLIRT":
        return 1
    if lbl == "TOO_FORWARD":
        return 0
    return np.nan

merged["gt_int"] = merged["gt_label_str"].apply(label_to_int)
merged["human_int"] = merged["human_label"].apply(label_to_int)
merged["model_int"] = merged["model_label_str"].apply(label_to_int)

# drop trials where human answered Not sure
valid_human = merged[~merged["human_int"].isna()].copy()

# splits
human_noai = valid_human[valid_human["cond"] == "noai"]
human_ai = valid_human[valid_human["cond"] == "ai"]

# model evaluated on all trials that have ground truth
valid_model = merged[~merged["gt_int"].isna()].copy()

def compute_metrics(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    print(f"\n=== {name} ===")
    print(f"n = {len(y_true)}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision (CONSENSUAL): {prec:.3f}")
    print(f"Recall (CONSENSUAL): {rec:.3f}")
    print(f"F1 (CONSENSUAL): {f1:.3f}")
    return acc

acc_model = compute_metrics(valid_model["gt_int"], valid_model["model_int"], "AI only, model from A2")

if len(human_noai):
    acc_h_noai = compute_metrics(human_noai["gt_int"], human_noai["human_int"], "Human only, no AI")
else:
    acc_h_noai = np.nan
    print("\nNo trials for human no AI")

if len(human_ai):
    acc_h_ai = compute_metrics(human_ai["gt_int"], human_ai["human_int"], "Human with AI support")
else:
    acc_h_ai = np.nan
    print("\nNo trials for human with AI")

# response time comparison
def show_rt(df, name):
    if len(df) == 0:
        print(f"\nNo RT data for {name}")
        return
    ms = df["response_time_ms"]
    print(f"\nResponse time {name}")
    print(f"n = {len(ms)}, mean = {ms.mean():.1f} ms, median = {ms.median():.1f} ms")

show_rt(human_noai, "Human no AI")
show_rt(human_ai, "Human with AI")

# simple t test on correctness at trial level between human no AI and human with AI
if len(human_noai) > 0 and len(human_ai) > 0:
    c1 = (human_noai["human_int"] == human_noai["gt_int"]).astype(float)
    c2 = (human_ai["human_int"] == human_ai["gt_int"]).astype(float)
    t_stat, p_val = ttest_ind(c1, c2, equal_var=False)
    print("\n=== Statistical test, trial correctness human no AI vs human with AI ===")
    print(f"mean no AI = {c1.mean():.3f}, mean with AI = {c2.mean():.3f}")
    print(f"t = {t_stat:.3f}, p = {p_val:.4f}")
else:
    print("\nNot enough data for t test on correctness")

# confidence by condition
if "confidence_1to5" in sess.columns:
    print("\n=== Confidence summary from sessions ===")
    for cond_name in ["noai", "ai"]:
        sub = sess[sess["cond"] == cond_name]
        if len(sub) == 0:
            continue
        vals = sub["confidence_1to5"]
        print(f"{cond_name}: n = {len(vals)}, mean = {vals.mean():.2f}, std = {vals.std():.2f}")
else:
    print("\nNo confidence_1to5 column in sessions")
