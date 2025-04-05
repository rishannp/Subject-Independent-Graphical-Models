import os
import pickle
import numpy as np
import scipy.signal as sig
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Subject-Independent Graphical Models\Subject-Independent-Graphical-Models\He\He_Dataset'
subset_labels = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7",
                 "C3", "CZ", "C4", "T8", "P7", "P3", "PZ", "P4",
                 "P8", "O1", "O2"]

# --- PLV Calculation Function ---
def plvfcn(eegData):
    n_channels, n_times = eegData.shape
    plv = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase1 = np.angle(sig.hilbert(eegData[i]))
            phase2 = np.angle(sig.hilbert(eegData[j]))
            diff = phase2 - phase1
            val = np.abs(np.sum(np.exp(1j * diff)) / n_times)
            plv[i, j] = val
            plv[j, i] = val
    return plv

# --- Subset Extractor ---
def extract_subset(plv_matrix, labels, subset_labels):
    indices = [i for i, lbl in enumerate(labels) if lbl in subset_labels]
    return plv_matrix[np.ix_(indices, indices)], indices

# --- Compute CV Map ---
def compute_cv(plv_list):
    stack = np.stack(plv_list, axis=2)
    return np.std(stack, axis=2) / (np.mean(stack, axis=2) + 1e-8)

# --- Session Processor (Parallelized) ---
def process_session(subj, sess, data_dir, subset_labels):
    session_path = os.path.join(data_dir, f"{subj}_Session_{sess}.pkl")
    if not os.path.exists(session_path):
        return [], [], [], []

    with open(session_path, 'rb') as f:
        bci = pickle.load(f)

    eeg_trials = bci['data']
    trial_meta = bci['TrialData']
    chan_labels = [str(lbl) for lbl in bci['chaninfo']['label']]

    left_full, right_full = [], []
    left_subset, right_subset = [], []

    for trial, meta in zip(eeg_trials, trial_meta):
        label = meta.get('targetnumber')
        if label not in [1, 2]:
            continue

        plv_full = plvfcn(trial)
        plv_sub, _ = extract_subset(plv_full, chan_labels, subset_labels)

        if label == 1:
            left_full.append(plv_full)
            left_subset.append(plv_sub)
        elif label == 2:
            right_full.append(plv_full)
            right_subset.append(plv_sub)

    return left_full, right_full, left_subset, right_subset

# --- Initialize storage ---
subject_cv_maps = {}

# --- Get subjects ---
all_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
subjects = sorted(set(f.split('_')[0] for f in all_files))

# --- Process Each Subject with Parallelized Sessions ---
for subj in tqdm(subjects[:20], desc="Processing subjects"):
    left_full, right_full = [], []
    left_subset, right_subset = [], []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_session, subj, sess, data_dir, subset_labels)
                   for sess in range(1, 5)]

        for future in tqdm(as_completed(futures), total=4, desc=f"  {subj} sessions", leave=False):
            lf, rf, lsub, rsub = future.result()
            left_full.extend(lf)
            right_full.extend(rf)
            left_subset.extend(lsub)
            right_subset.extend(rsub)

    # --- Compute and store CV maps ---
    if left_full:
        subject_cv_maps[f"{subj}_Left"] = {
            'full': compute_cv(left_full),
            'subset': compute_cv(left_subset)
        }
    if right_full:
        subject_cv_maps[f"{subj}_Right"] = {
            'full': compute_cv(right_full),
            'subset': compute_cv(right_subset)
        }

# --- Save Output ---
save_path = os.path.join(data_dir, "subject_cv_maps_full_and_subset.pkl")
with open(save_path, 'wb') as f:
    pickle.dump(subject_cv_maps, f)

print(f"\n[SAVED] Dual CV maps saved to: {save_path}")



#%% Plots

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Subject-Independent Graphical Models\Subject-Independent-Graphical-Models\He\He_Dataset'
cv_path = os.path.join(data_dir, "subject_cv_maps_full_and_subset.pkl")

# Subset electrode labels
subset_labels = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7",
                 "C3", "CZ", "C4", "T8", "P7", "P3", "PZ", "P4",
                 "P8", "O1", "O2"]

# --- Load CV data ---
with open(cv_path, 'rb') as f:
    subject_cv_maps = pickle.load(f)

# --- Output directory for plots ---
output_dir = os.path.join(data_dir, "cv_visualisations")
os.makedirs(output_dir, exist_ok=True)

# --- Helper: Plot CV heatmap with labels ---
def plot_cv_matrix(matrix, title, labels, save_path, cmap):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(matrix, cmap=cmap, interpolation='nearest', vmin=0)
    plt.colorbar(im, label='Coefficient of Variation')
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90, fontsize=6)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=6)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# --- Collect for group averages ---
left_full_all, right_full_all = [], []
left_subset_all, right_subset_all = [], []

# --- Individual subject plots ---
for key in tqdm(subject_cv_maps.keys(), desc="Plotting individual CVs"):
    subj_id, class_name = key.split("_")
    subj_dir = os.path.join(output_dir, subj_id)
    os.makedirs(subj_dir, exist_ok=True)

    # --- Full CV ---
    full_cv = subject_cv_maps[key]["full"]
    full_labels = [f"Ch{i+1}" for i in range(full_cv.shape[0])]
    full_path = os.path.join(subj_dir, f"{class_name}_CV_full.png")
    plot_cv_matrix(full_cv, f"{subj_id} - {class_name} (Full)", full_labels, full_path, 'plasma')

    if class_name == "Left":
        left_full_all.append(full_cv)
    else:
        right_full_all.append(full_cv)

    # --- Subset CV ---
    subset_cv = subject_cv_maps[key]["subset"]
    subset_path = os.path.join(subj_dir, f"{class_name}_CV_subset.png")
    plot_cv_matrix(subset_cv, f"{subj_id} - {class_name} (Subset)", subset_labels, subset_path, 'plasma')

    if class_name == "Left":
        left_subset_all.append(subset_cv)
    else:
        right_subset_all.append(subset_cv)

# --- Group average plotting ---
def plot_group_average(cv_list, class_name, label_type, labels):
    if not cv_list:
        return
    avg = np.mean(np.stack(cv_list, axis=2), axis=2)
    title = f"Average CV - {class_name} ({label_type})"
    save_path = os.path.join(output_dir, f"average_cv_{class_name.lower()}_{label_type.lower()}.png")
    plot_cv_matrix(avg, title, labels, save_path, 'RdYlGn')
    print(f"[SAVED] {title}: {save_path}")

# --- Save all group averages ---
full_labels_dummy = [f"Ch{i+1}" for i in range(left_full_all[0].shape[0])] if left_full_all else []

plot_group_average(left_full_all, "Left", "Full", full_labels_dummy)
plot_group_average(right_full_all, "Right", "Full", full_labels_dummy)
plot_group_average(left_subset_all, "Left", "Subset", subset_labels)
plot_group_average(right_subset_all, "Right", "Subset", subset_labels)




#%% Without Parallel

import os
import pickle
import numpy as np
import scipy.signal as sig
from tqdm import tqdm

# --- Configuration ---
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Subject-Independent Graphical Models\Subject-Independent-Graphical-Models\He\He_Dataset'
subset_labels = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7",
                 "C3", "CZ", "C4", "T8", "P7", "P3", "PZ", "P4",
                 "P8", "O1", "O2"]

# --- PLV Calculation Function ---
def plvfcn(eegData):
    n_channels, n_times = eegData.shape
    plv = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase1 = np.angle(sig.hilbert(eegData[i]))
            phase2 = np.angle(sig.hilbert(eegData[j]))
            diff = phase2 - phase1
            val = np.abs(np.sum(np.exp(1j * diff)) / n_times)
            plv[i, j] = val
            plv[j, i] = val
    return plv

# --- Subset Extractor ---
def extract_subset(plv_matrix, labels, subset_labels):
    indices = [i for i, lbl in enumerate(labels) if lbl in subset_labels]
    return plv_matrix[np.ix_(indices, indices)], indices

# --- Initialize storage ---
subject_cv_maps = {}

# --- Find subject names ---
all_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
subjects = sorted(set(f.split('_')[0] for f in all_files))

# --- Process Each Subject ---
for subj in tqdm(subjects[:20], desc="Processing subjects"):
    left_full, right_full = [], []
    left_subset, right_subset = [], []
    chan_labels = None

    for sess in tqdm(range(1, 5), desc=f"  {subj} Sessions", leave=False):
        session_path = os.path.join(data_dir, f"{subj}_Session_{sess}.pkl")
        if not os.path.exists(session_path):
            continue

        with open(session_path, 'rb') as f:
            bci = pickle.load(f)

        eeg_trials = bci['data']
        trial_meta = bci['TrialData']
        chan_labels = [str(lbl) for lbl in bci['chaninfo']['label']]

        for trial, meta in tqdm(zip(eeg_trials, trial_meta), total=len(eeg_trials),
                                desc=f"    Session {sess} Trials", leave=False):
            label = meta.get('targetnumber')
            if label not in [1, 2]:
                continue

            plv_full = plvfcn(trial)
            plv_sub, _ = extract_subset(plv_full, chan_labels, subset_labels)

            if label == 1:
                left_full.append(plv_full)
                left_subset.append(plv_sub)
            elif label == 2:
                right_full.append(plv_full)
                right_subset.append(plv_sub)

    # --- Compute CV and store ---
    def compute_cv(plv_list):
        stack = np.stack(plv_list, axis=2)
        return np.std(stack, axis=2) / (np.mean(stack, axis=2) + 1e-8)

    if left_full:
        subject_cv_maps[f"{subj}_Left"] = {
            'full': compute_cv(left_full),
            'subset': compute_cv(left_subset)
        }
    if right_full:
        subject_cv_maps[f"{subj}_Right"] = {
            'full': compute_cv(right_full),
            'subset': compute_cv(right_subset)
        }

# --- Save Output ---
save_path = os.path.join(data_dir, "subject_cv_maps_full_and_subset.pkl")
with open(save_path, 'wb') as f:
    pickle.dump(subject_cv_maps, f)

print(f"\n[SAVED] Dual CV maps saved to: {save_path}")