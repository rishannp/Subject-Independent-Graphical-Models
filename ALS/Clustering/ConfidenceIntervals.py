
import numpy as np
import scipy.stats as st

# Each model is keyed by its name; the value is a list of the 8 accuracies (in %).
models = {
    "rPLVGAT (H=8)":   [56.29, 62.57, 62.03, 64.56, 68.24, 67.92, 72.82, 79.19],
    "EEGNet":          [54.46, 59.21, 60.45, 61.41, 73.57, 72.31, 70.85, 68.81],
    "ShallowConvNet":  [55.10, 56.80, 59.16, 60.13, 67.52, 69.38, 67.46, 69.49],
    "LMDA":            [55.41, 57.40, 53.05, 57.56, 59.55, 63.19, 53.22, 56.61],
    "FTL":             [51.98, 67.92, 58.63, 62.65, 68.25, 63.01, 54.24, 61.44],
    "1D CNN":          [54.09, 50.45, 54.46, 51.27, 51.42, 57.14, 51.18, 59.12],
    "MIN2NET":         [56.05, 55.59, 51.77, 54.66, 54.66, 56.03, 53.90, 54.92],
    "GCNs-Net":        [55.03, 61.11, 56.01, 59.49, 68.87, 69.18, 73.49, 74.16]
}

def compute_confidence_interval(acc_list, confidence=0.95):
    """
    Given a list of accuracies (in %), returns the mean accuracy and the
    lower and upper bounds of the confidence interval at the specified confidence level.
    Uses the t-distribution for small sample sizes (n=8).
    """
    # Convert to a NumPy array
    arr = np.array(acc_list, dtype=float)
    
    # Sample size
    n = len(arr)
    
    # Sample mean and sample standard deviation
    mean_acc = np.mean(arr)
    std_acc = np.std(arr, ddof=1)  # ddof=1 for sample standard deviation
    
    # Compute standard error of the mean
    sem = std_acc / np.sqrt(n)
    
    # Degrees of freedom for t-distribution
    df = n - 1
    
    # Critical t-value for two-tailed confidence
    alpha = 1 - confidence
    t_crit = st.t.ppf(1 - alpha/2, df)
    
    # Margin of error
    margin_of_error = t_crit * sem
    
    # Lower and upper confidence bounds
    ci_lower = mean_acc - margin_of_error
    ci_upper = mean_acc + margin_of_error
    
    return mean_acc, ci_lower, ci_upper

# Print header
print("Model\t\t\tMean Accuracy (%)\t95% CI Lower\t95% CI Upper")

# Compute and print CI for each model
for model_name, accs in models.items():
    mean_acc, ci_lower, ci_upper = compute_confidence_interval(accs, confidence=0.95)
    print(f"{model_name:16s}\t{mean_acc:6.2f}\t\t\t{ci_lower:6.2f}\t\t{ci_upper:6.2f}")
