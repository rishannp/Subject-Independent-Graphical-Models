# --- Import libraries ---
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# --- Manually create the dataset based on the table you provided ---
data = {
    'Subject': list(range(1, 21)),
    'MSBR': [1,1,1,1,1,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0],
    'Meditation': [25.04,18,16,25.25,8,np.nan,28,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,19.7833,np.nan,np.nan,np.nan,23.75,np.nan],
    'Handedness': ['R', np.nan, 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'R', 'R', 'R'],
    'Age': [42,59,19,60,18,30,62,47,31,53,52,48,42,34,53,32,43,34,58,24],
    'Gender': ['M','F','F','F','M','M','F','F','F','F','F','F','F','F','F','M','M','M','F','F'],
    'rPLVGAT': [57.82,55.16,64.71,58.47,59.82,56.10,61.73,71.62,65.15,56.40,57.51,66.04,59.73,54.84,53.16,54.73,64.26,50.83,58.69,57.40],
    'EEGNet': [51.94,54.83,56.38,60.91,55.72,50.55,60.40,50.00,65.70,51.39,49.94,49.94,50.72,50.39,50.06,50.06,52.50,49.94,50.06,50.28],
    'ShallowConvNet': [40.00]*20,
    'LMDA': [40.00]*20,
    'FTL': [49.79,50.76,50.07,59.20,52.43,52.77,49.17,53.46,49.93,59.58,53.47,47.16,49.17,52.50,53.54,53.61,50.90,50.97,49.93,47.36],
    '1D CNN': [50.17,50.17,50.06,50.06,49.94,49.89,50.22,50.00,50.06,49.83,49.94,50.06,49.61,50.39,49.94,50.06,49.50,49.94,49.94,49.72],
    'MIN2NET': [40.00]*20,
    'GCNs-Net': [59.49,55.05,67.70,60.69,56.60,54.55,64.05,72.73,60.16,54.95,59.96,57.60,56.40,54.84,51.17,53.95,60.16,50.39,54.71,58.84],
}

# Create the DataFrame
df = pd.DataFrame(data)

# --- Preprocessing ---
# 1. Fill NaN Meditation values with 0 (interpreting missing as "no meditation practice")
df['Meditation'] = df['Meditation'].fillna(0)

# 2. Meditation Group: 0 mins vs >0 mins
df['Meditation_group'] = df['Meditation'].apply(lambda x: 'Meditator' if x > 0 else 'Non-meditator')

# --- List of all model columns (for plotting) ---
performance_cols = ['rPLVGAT', 'EEGNet', 'GCNs-Net', 'FTL', '1D CNN', 'MIN2NET']

# --- Statistical Analysis: ONLY on rPLVGAT ---

print("\n--- MSBR (1 vs 0) Group Comparison for rPLVGAT ---\n")
group1 = df[df['MSBR'] == 1]['rPLVGAT']
group0 = df[df['MSBR'] == 0]['rPLVGAT']
stat, p = mannwhitneyu(group1, group0, alternative='two-sided')
print(f"rPLVGAT MSBR: U-statistic = {stat:.2f}, p-value = {p:.4f}")

print("\n--- Meditation (0 vs >0) Group Comparison for rPLVGAT ---\n")
group_meditators = df[df['Meditation_group'] == 'Meditator']['rPLVGAT']
group_nonmeditators = df[df['Meditation_group'] == 'Non-meditator']['rPLVGAT']
stat, p = mannwhitneyu(group_meditators, group_nonmeditators, alternative='two-sided')
print(f"rPLVGAT Meditation: U-statistic = {stat:.2f}, p-value = {p:.4f}")

print("\n--- Meditation Minutes vs rPLVGAT Performance (Spearman Correlation) ---\n")
corr, p = spearmanr(df['Meditation'], df['rPLVGAT'])
print(f"rPLVGAT Meditation correlation: Spearman rho = {corr:.2f}, p-value = {p:.4f}")

# --- Visualization Section (Optional) ---

# 1. Boxplots: MSBR vs Performance
for model in performance_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='MSBR', y=model, data=df)
    plt.title(f'{model} Performance by MSBR (0 = No, 1 = Yes)')
    plt.xlabel('MSBR Participation')
    plt.ylabel('Performance (%)')
    plt.tight_layout()
    plt.show()

# 2. Boxplots: Meditation (0 vs >0) vs Performance
for model in performance_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Meditation_group', y=model, data=df)
    plt.title(f'{model} Performance by Meditation Group')
    plt.xlabel('Meditation Group')
    plt.ylabel('Performance (%)')
    plt.tight_layout()
    plt.show()

# 3. Scatterplots: Meditation minutes vs Performance
for model in performance_cols:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='Meditation', y=model, data=df)
    sns.regplot(x='Meditation', y=model, data=df, scatter=False, ci=None, color='red')
    plt.title(f'{model} vs Daily Meditation Minutes')
    plt.xlabel('Meditation Minutes/Day')
    plt.ylabel('Performance (%)')
    plt.tight_layout()
    plt.show()
