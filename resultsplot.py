import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility (optional)
np.random.seed(12345)

# Define font size (adjust as needed)
font_size = 12

# Update matplotlib default font sizes
plt.rcParams.update({'font.size': font_size})

# Model names and number of subjects per dataset
model_names = ["rPLVGAT (H=8)", "EEGNet", "ShallowConvNet", "LMDA", "FTL", "1D CNN", "MIN2NET", "GCNs-Net"]
n_models = len(model_names)
n_subjects1 = 8   # Dataset 1: 8 subjects
n_subjects2 = 25  # Dataset 2: 25 subjects

# Dummy average accuracies and data arrays
# Replace these arrays with your actual accuracy data.
acc_data1 = np.array([
    [56.29, 54.46, 55.10, 55.41, 51.98, 54.09, 56.05, 55.03],
    [62.57, 59.21, 56.80, 57.40, 67.92, 50.45, 55.59, 61.11],
    [62.03, 60.45, 59.16, 53.05, 58.63, 54.46, 51.77, 56.01],
    [64.56, 61.41, 60.13, 57.56, 62.65, 51.27, 54.66, 59.49],
    [68.24, 73.57, 67.52, 59.55, 68.25, 51.42, 54.66, 68.87],
    [67.92, 72.31, 69.38, 63.19, 63.01, 57.14, 56.03, 69.18],
    [72.82, 70.85, 67.46, 53.22, 54.24, 51.18, 53.90, 73.49],
    [79.19, 68.81, 69.49, 56.61, 61.44, 59.12, 54.92, 74.16]
])
# Data is provided with rows as subjects and columns as models.
# Compute average accuracies per model (across subjects)
avg_acc1 = np.mean(acc_data1, axis=0)

acc_data2 = np.array([
    [86, 79, 76, 83, 81, 78, 84, 80],
    [84, 78, 75, 82, 80, 77, 83, 79],
    [85, 80, 77, 84, 82, 79, 85, 81],
    [83, 79, 76, 83, 81, 78, 84, 80],
    [87, 78, 75, 82, 80, 77, 83, 79],
    [86, 80, 77, 84, 82, 79, 85, 81],
    [84, 79, 76, 83, 81, 78, 84, 80],
    [85, 80, 77, 84, 82, 79, 85, 81],
    [83, 79, 76, 83, 81, 78, 84, 80],
    [87, 78, 75, 82, 80, 77, 83, 79],
    [86, 80, 77, 84, 82, 79, 85, 81],
    [84, 79, 76, 83, 81, 78, 84, 80],
    [85, 80, 77, 84, 82, 79, 85, 81],
    [83, 79, 76, 83, 81, 78, 84, 80],
    [87, 78, 75, 82, 80, 77, 83, 79],
    [86, 80, 77, 84, 82, 79, 85, 81],
    [84, 79, 76, 83, 81, 78, 84, 80],
    [85, 80, 77, 84, 82, 79, 85, 81],
    [83, 79, 76, 83, 81, 78, 84, 80],
    [87, 78, 75, 82, 80, 77, 83, 79],
    [86, 80, 77, 84, 82, 79, 85, 81],
    [84, 79, 76, 83, 81, 78, 84, 80],
    [85, 80, 77, 84, 82, 79, 85, 81],
    [83, 79, 76, 83, 81, 78, 84, 80],
    [87, 78, 75, 82, 80, 77, 83, 79]
])
avg_acc2 = np.mean(acc_data2, axis=0)

# Get pastel colours from a colormap (Pastel1)
cmap = plt.get_cmap("Pastel1")
colors = cmap(np.linspace(0, 1, n_models))

# Create a figure with two subplots (one for each dataset)
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot for Dataset 1 (8 subjects)
x = np.arange(n_models)
axs[0].bar(x, avg_acc1, color=colors, edgecolor='k', alpha=0.7)
# Overlay individual subject accuracies as circles with slight jitter
for i in range(n_models):
    jitter = np.random.uniform(-0.1, 0.1, size=n_subjects1)
    axs[0].scatter(x[i] + jitter, acc_data1[:, i], color='k', zorder=3, s=50, marker='o')
axs[0].set_xticks(x)
axs[0].set_xticklabels(model_names, rotation=45, ha='right')
axs[0].set_title("Dataset 1 (8 Subjects)")
axs[0].set_ylabel("Accuracy (%)")
axs[0].set_ylim(55, 85)

# Plot for Dataset 2 (25 subjects)
axs[1].bar(x, avg_acc2, color=colors, edgecolor='k', alpha=0.7)
for i in range(n_models):
    jitter = np.random.uniform(-0.1, 0.1, size=n_subjects2)
    axs[1].scatter(x[i] + jitter, acc_data2[:, i], color='k', zorder=3, s=50, marker='o')
axs[1].set_xticks(x)
axs[1].set_xticklabels(model_names, rotation=45, ha='right')
axs[1].set_title("Dataset 2 (25 Subjects)")
axs[1].set_ylim(55, 85)

plt.tight_layout()

# Save figure at 300 dpi (optional)
plt.savefig("model_accuracies.png", dpi=300)
plt.show()
