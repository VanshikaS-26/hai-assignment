import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load sessions file
base = Path(".")
sess_path = base / "logs" / "a3_sessions.csv"

sess = pd.read_csv(sess_path)

# Filter confidence values
if "confidence_1to5" not in sess.columns:
    raise ValueError("confidence_1to5 column not found in a3_sessions.csv")

# Group by condition
means = sess.groupby("cond")["confidence_1to5"].mean()
stds = sess.groupby("cond")["confidence_1to5"].std()
counts = sess.groupby("cond")["confidence_1to5"].count()

# Prepare data for plot
conditions = ["noai", "ai"]
mean_values = [means.get("noai", 0), means.get("ai", 0)]

plt.figure(figsize=(6,4))
bars = plt.bar(
    ["No AI", "With AI"],
    mean_values,
    color=["#999999", "#FFB347"]
)

plt.ylabel("Mean Confidence (1 to 5)")
plt.ylim(0, 5)
plt.title("Confidence Comparison Across Conditions")

# Add value labels on top of bars
for i, v in enumerate(mean_values):
    plt.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("confidence_comparison.png", dpi=300)
print("Saved confidence_comparison.png")
