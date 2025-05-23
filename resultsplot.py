import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(12345)

# Define font size
font_size = 12
plt.rcParams.update({'font.size': font_size})

# Model names and subject counts
model_names = ["rPLVGAT", "EEGNet", "ShallowConvNet", "LMDA", "FTL", "1D CNN", "MIN2NET", "GCNs-Net"]
n_models = len(model_names)
n_subjects1 = 8
n_subjects2 = 20

# Accuracy data
acc_data1 = np.array([
    [52.52, 50.00, 53.50, 50.64, 48.41, 50.63, 46.50, 50.31],
    [54.39, 53.17, 48.04, 47.43, 60.38, 53.13, 55.29, 50.29],
    [56.01, 60.77, 53.05, 51.45, 50.60, 48.09, 45.98, 50.00],
    [60.44, 56.27, 54.02, 51.77, 51.81, 50.95, 50.80, 55.06],
    [67.30, 69.11, 57.32, 54.46, 51.59, 53.31, 50.00, 68.24],
    [62.26, 65.80, 65.80, 49.51, 43.09, 54.29, 53.75, 64.47],
    [73.49, 62.71, 61.36, 48.14, 45.34, 51.52, 50.85, 55.70],
    [74.83, 54.92, 64.07, 50.17, 53.81, 46.28, 48.81, 53.69]
])
avg_acc1 = np.mean(acc_data1, axis=0)

acc_data2 = np.array([
    [57.82, 51.94, 0.00, 00.00, 49.79, 50.17, 00.00, 59.49],
    [55.16, 54.83, 00.00, 00.00, 50.76, 50.17, 00.00, 55.05],
    [64.71, 56.38, 00.00, 00.00, 50.07, 50.06, 00.00, 67.70],
    [58.47, 60.91, 00.00, 00.00, 59.20, 50.06, 00.00, 60.69],
    [59.82, 55.72, 00.00, 00.00, 52.43, 49.94, 00.00, 56.60],
    [56.10, 50.55, 00.00, 00.00, 52.77, 49.89, 00.00, 54.55],
    [61.73, 60.40, 00.00, 00.00, 49.17, 50.22, 00.00, 64.05],
    [71.62, 50.00, 00.00, 00.00, 53.46, 50.00, 00.00, 72.73],
    [65.15, 65.70, 00.00, 00.00, 49.93, 50.06, 00.00, 60.16],
    [56.40, 51.39, 00.00, 00.00, 59.58, 49.83, 00.00, 54.95],
    [57.51, 49.94, 00.00, 00.00, 53.47, 49.94, 00.00, 59.96],
    [66.04, 49.94, 00.00, 00.00, 47.16, 50.06, 00.00, 57.60],
    [59.73, 50.72, 00.00, 00.00, 49.17, 49.61, 00.00, 56.40],
    [54.84, 50.39, 00.00, 00.00, 52.50, 50.39, 00.00, 54.84],
    [53.16, 50.06, 00.00, 00.00, 53.54, 49.94, 00.00, 51.17],
    [54.73, 50.06, 00.00, 00.00, 53.61, 50.06, 00.00, 53.95],
    [64.26, 52.50, 00.00, 00.00, 50.90, 49.50, 00.00, 60.16],
    [50.83, 49.94, 00.00, 00.00, 50.97, 49.94, 00.00, 50.39],
    [58.69, 50.06, 00.00, 00.00, 49.93, 49.94, 00.00, 54.71],
    [57.40, 50.28, 00.00, 00.00, 47.36, 49.72, 00.00, 58.84]
])
avg_acc2 = np.mean(acc_data2, axis=0)

# Colors
cmap = plt.get_cmap("Pastel1")
colors = cmap(np.linspace(0, 1, n_models))

# Create figure
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

x = np.arange(n_models)

# --- Dataset 1 plot ---
bars1 = axs[0].bar(x, avg_acc1, color=colors, edgecolor='k', alpha=0.7)
for i in range(n_models):
    jitter = np.random.uniform(-0.1, 0.1, size=n_subjects1)
    axs[0].scatter(x[i] + jitter, acc_data1[:, i], color='k', zorder=3, s=50, marker='o')
    # Place labels around y=79
    axs[0].text(x[i], 79, f"{avg_acc1[i]:.2f}%", ha='center', va='top', fontsize=10, fontweight='bold')

axs[0].set_xticks(x)
axs[0].set_xticklabels(model_names, rotation=45, ha='right')
axs[0].set_title("Dataset 1 (8 Subjects)")
axs[0].set_ylabel("Accuracy (%)")
axs[0].set_ylim(40, 80)

# --- Dataset 2 plot ---
bars2 = axs[1].bar(x, avg_acc2, color=colors, edgecolor='k', alpha=0.7)
for i in range(n_models):
    jitter = np.random.uniform(-0.1, 0.1, size=n_subjects2)
    axs[1].scatter(x[i] + jitter, acc_data2[:, i], color='k', zorder=3, s=50, marker='o')
    axs[1].text(x[i], 79, f"{avg_acc2[i]:.2f}%", ha='center', va='top', fontsize=10, fontweight='bold')

axs[1].set_xticks(x)
axs[1].set_xticklabels(model_names, rotation=45, ha='right')
axs[1].set_title("Dataset 2 (20 Subjects)")
axs[1].set_ylim(40, 80)

plt.tight_layout()
plt.savefig("model_accuracies_with_floating_averages_top80.png", dpi=300)
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ttest_rel

# Set seed for reproducibility
np.random.seed(12345)

# Font settings
font_size = 12
plt.rcParams.update({'font.size': font_size})

# Model names
model_names = ["rPLVGAT", "EEGNet", "ShallowConvNet", "LMDA", "FTL", "1D CNN", "MIN2NET", "GCNs-Net"]
n_models = len(model_names)
n_subjects1 = 8
n_subjects2 = 20

# Accuracy data for both datasets
acc_data1 = np.array([
    [52.52, 50.00, 53.50, 50.64, 48.41, 50.63, 46.50, 50.31],
    [54.39, 53.17, 48.04, 47.43, 60.38, 53.13, 55.29, 50.29],
    [56.01, 60.77, 53.05, 51.45, 50.60, 48.09, 45.98, 50.00],
    [60.44, 56.27, 54.02, 51.77, 51.81, 50.95, 50.80, 55.06],
    [67.30, 69.11, 57.32, 54.46, 51.59, 53.31, 50.00, 68.24],
    [62.26, 65.80, 65.80, 49.51, 43.09, 54.29, 53.75, 64.47],
    [73.49, 62.71, 61.36, 48.14, 45.34, 51.52, 50.85, 55.70],
    [74.83, 54.92, 64.07, 50.17, 53.81, 46.28, 48.81, 53.69]
])
avg_acc1 = np.mean(acc_data1, axis=0)

acc_data2 = np.array([
    [57.82, 51.94, 0.00, 0.00, 49.79, 50.17, 0.00, 59.49],
    [55.16, 54.83, 0.00, 0.00, 50.76, 50.17, 0.00, 55.05],
    [64.71, 56.38, 0.00, 0.00, 50.07, 50.06, 0.00, 67.70],
    [58.47, 60.91, 0.00, 0.00, 59.20, 50.06, 0.00, 60.69],
    [59.82, 55.72, 0.00, 0.00, 52.43, 49.94, 0.00, 56.60],
    [56.10, 50.55, 0.00, 0.00, 52.77, 49.89, 0.00, 54.55],
    [61.73, 60.40, 0.00, 0.00, 49.17, 50.22, 0.00, 64.05],
    [71.62, 50.00, 0.00, 0.00, 53.46, 50.00, 0.00, 72.73],
    [65.15, 65.70, 0.00, 0.00, 49.93, 50.06, 0.00, 60.16],
    [56.40, 51.39, 0.00, 0.00, 59.58, 49.83, 0.00, 54.95],
    [57.51, 49.94, 0.00, 0.00, 53.47, 49.94, 0.00, 59.96],
    [66.04, 49.94, 0.00, 0.00, 47.16, 50.06, 0.00, 57.60],
    [59.73, 50.72, 0.00, 0.00, 49.17, 49.61, 0.00, 56.40],
    [54.84, 50.39, 0.00, 0.00, 52.50, 50.39, 0.00, 54.84],
    [53.16, 50.06, 0.00, 0.00, 53.54, 49.94, 0.00, 51.17],
    [54.73, 50.06, 0.00, 0.00, 53.61, 50.06, 0.00, 53.95],
    [64.26, 52.50, 0.00, 0.00, 50.90, 49.50, 0.00, 60.16],
    [50.83, 49.94, 0.00, 0.00, 50.97, 49.94, 0.00, 50.39],
    [58.69, 50.06, 0.00, 0.00, 49.93, 49.94, 0.00, 54.71],
    [57.40, 50.28, 0.00, 0.00, 47.36, 49.72, 0.00, 58.84]
])
avg_acc2 = np.mean(acc_data2, axis=0)

# Colors
cmap = plt.get_cmap("Pastel1")
colors = cmap(np.linspace(0, 1, n_models))
x = np.arange(n_models)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Dataset 1
bars1 = axs[0].bar(x, avg_acc1, color=colors, edgecolor='k', alpha=0.7)
for i in range(n_models):
    jitter = np.random.uniform(-0.1, 0.1, size=n_subjects1)
    axs[0].scatter(x[i] + jitter, acc_data1[:, i], color='k', zorder=3, s=50)
    axs[0].text(x[i], 79, f"{avg_acc1[i]:.2f}%", ha='center', va='top', fontsize=10, fontweight='bold')
axs[0].set_xticks(x)
axs[0].set_xticklabels(model_names, rotation=45, ha='right')
axs[0].set_title("Dataset 1 (8 Subjects)")
axs[0].set_ylabel("Accuracy (%)")
axs[0].set_ylim(40, 100)

# Dataset 2
bars2 = axs[1].bar(x, avg_acc2, color=colors, edgecolor='k', alpha=0.7)
for i in range(n_models):
    jitter = np.random.uniform(-0.1, 0.1, size=n_subjects2)
    axs[1].scatter(x[i] + jitter, acc_data2[:, i], color='k', zorder=3, s=50)
    axs[1].text(x[i], 79, f"{avg_acc2[i]:.2f}%", ha='center', va='top', fontsize=10, fontweight='bold')
axs[1].set_xticks(x)
axs[1].set_xticklabels(model_names, rotation=45, ha='right')
axs[1].set_title("Dataset 2 (20 Subjects)")
axs[1].set_ylim(40, 100)

plt.tight_layout()
plt.savefig("model_accuracies_with_stats.png", dpi=300)
plt.show()

# ======================
# STATISTICAL COMPARISONS
# ======================

print("\n=== Dataset 1: Statistical Comparison vs rPLVGAT ===")
for i in range(1, n_models):
    try:
        p_wilcox = wilcoxon(acc_data1[:, 0], acc_data1[:, i]).pvalue
        p_ttest = ttest_rel(acc_data1[:, 0], acc_data1[:, i]).pvalue
        print(f"{model_names[i]:15s} | Wilcoxon p = {p_wilcox:.4f}, t-test p = {p_ttest:.4f}")
    except Exception as e:
        print(f"{model_names[i]:15s} | Error: {e}")

print("\n=== Dataset 2: Statistical Comparison vs rPLVGAT ===")
valid_idxs = [1, 4, 5, 7]  # Exclude models with all-zero columns (3, 6)
for i in valid_idxs:
    try:
        p_wilcox = wilcoxon(acc_data2[:, 0], acc_data2[:, i]).pvalue
        p_ttest = ttest_rel(acc_data2[:, 0], acc_data2[:, i]).pvalue
        print(f"{model_names[i]:15s} | Wilcoxon p = {p_wilcox:.4f}, t-test p = {p_ttest:.4f}")
    except Exception as e:
        print(f"{model_names[i]:15s} | Error: {e}")
