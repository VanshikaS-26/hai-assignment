import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
base = Path(".")
logs_path = base / "logs" / "a3_logs.csv"
study_path = base / "a2_study_with_outputs.csv"

logs = pd.read_csv(logs_path)
study = pd.read_csv(study_path)

# Convert ground truth + labels for accuracy
def gt_label_str(row):
    return "CONSENSUAL_FLIRT" if row["ground_truth_label"] == 1 else "TOO_FORWARD"

study["gt_label_str"] = study.apply(gt_label_str, axis=1)

# Merge logs with ground truth + AI predictions
merged = logs.merge(
    study.reset_index().rename(columns={"index": "message_index_study"}),
    left_on="message_index",
    right_on="message_index_study",
    how="left"
)

# Label mapping
merged["model_label_str"] = merged["ai_label"].fillna(merged["ai_readable_label"])

def label_to_int(lbl):
    if lbl == "CONSENSUAL_FLIRT": return 1
    if lbl == "TOO_FORWARD": return 0
    return None

merged["gt_int"] = merged["gt_label_str"].apply(label_to_int)
merged["model_int"] = merged["model_label_str"].apply(label_to_int)

# Human labels
def human_label_noai(ans):
    if ans == "consensual": return "CONSENSUAL_FLIRT"
    if ans == "too_forward": return "TOO_FORWARD"
    return None

def human_label_with_ai(ans, ai_label):
    if ans == "agree": return ai_label
    if ans == "disagree_too_forward": return "TOO_FORWARD"
    return None

merged["human_int"] = None
mask_noai = merged["cond"] == "noai"
mask_ai = merged["cond"] == "ai"

merged.loc[mask_noai, "human_int"] = merged.loc[mask_noai, "answer"].apply(
    lambda x: 1 if human_label_noai(x) == "CONSENSUAL_FLIRT" else 0 if human_label_noai(x) == "TOO_FORWARD" else None
)

merged.loc[mask_ai, "human_int"] = merged.loc[mask_ai].apply(
    lambda r: 1 if human_label_with_ai(r["answer"], r["model_label_str"]) == "CONSENSUAL_FLIRT"
    else 0 if human_label_with_ai(r["answer"], r["model_label_str"]) == "TOO_FORWARD"
    else None,
    axis=1
)

# Drop trials where human said "Not sure"
valid_human = merged.dropna(subset=["human_int"]).copy()

# Compute accuracy
def compute_acc(df, label_col):
    return (df[label_col] == df["gt_int"]).mean()

acc_ai = compute_acc(merged, "model_int")
acc_h_noai = compute_acc(valid_human[valid_human["cond"] == "noai"], "human_int")
acc_h_ai = compute_acc(valid_human[valid_human["cond"] == "ai"], "human_int")

# --------------------------
# FIGURE 1: Accuracy bar chart
# --------------------------
conditions = ["Model (AI only)", "Human (No AI)", "Human (With AI)"]
acc_values = [acc_ai, acc_h_noai, acc_h_ai]

plt.figure(figsize=(6,4))
plt.bar(conditions, acc_values, color=["#4C8BF5", "#999999", "#FFB347"])
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Accuracy Comparison Across Conditions")

for i, v in enumerate(acc_values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=300)
print("Saved accuracy_comparison.png")

# --------------------------
# FIGURE 2: Response time boxplot
# --------------------------
noai_times = valid_human[valid_human["cond"] == "noai"]["response_time_ms"]
ai_times = valid_human[valid_human["cond"] == "ai"]["response_time_ms"]

plt.figure(figsize=(6,4))
plt.boxplot([noai_times, ai_times], labels=["No AI", "With AI"])
plt.ylabel("Response Time (ms)")
plt.title("Response Time by Condition")
plt.tight_layout()
plt.savefig("response_time_boxplot.png", dpi=300)
print("Saved response_time_boxplot.png")
