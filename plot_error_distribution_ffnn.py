import matplotlib.pyplot as plt
import json
from collections import defaultdict

# Load errors from JSON file
with open("results/test_errors.json") as f:
    errors = json.load(f)

# Create dictionaries to count actual and predicted labels
true_counts = defaultdict(int)
pred_counts = defaultdict(int)

for error in errors:
    true_counts[error["true_label"]] += 1
    pred_counts[error["predicted_label"]] += 1

# Extract sorted labels for consistent ordering
labels = sorted(set(true_counts.keys()).union(set(pred_counts.keys())))

# Bar widths and positions
x = range(len(labels))
width = 0.35  

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(x, [true_counts[label] for label in labels], width=width, label='Actual Labels', alpha=0.7)
plt.bar([pos + width for pos in x], [pred_counts[label] for label in labels], width=width, label='Predicted Labels', alpha=0.7)

plt.xlabel("Class Labels")
plt.ylabel("Count of Misclassified Samples")
plt.title("Error Analysis: Actual vs Predicted Labels")
plt.xticks([pos + width / 2 for pos in x], labels)  # Center labels
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Save the plot
plt.savefig("results/error_bar_graph.png")
plt.show()

print("Error bar graph saved to results/error_bar_graph.png")
