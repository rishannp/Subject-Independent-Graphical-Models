#%% THE ANSWER IS, THIS COULD WORK FOR A DATASET TRANSFER LEARNING APPROACH. THE AVERAGE IS NOT SO CLOSE BUT YOU CAN SEE THE SIMILARITIES. 
# NOW GO AND CHECK THE CV OVER TIME AND PRODUCE THE AVERAGE CV PLOT PER SUBJECT FOR INVARIANT NODES
#%%

### INDIVIDUAL TRIALS OVER TIME
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# Load S1_Session_1
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Subject-Independent Graphical Models\Subject-Independent-Graphical-Models\He\He_Dataset'
file_path = os.path.join(data_dir, 'S1_Session_1.pkl')

with open(file_path, 'rb') as f:
    bci = pickle.load(f)

eeg_trials = bci['data']  # list of [channels x time] arrays
trial_meta = bci['TrialData']
channel_labels = bci['chaninfo']['label']

# --- PLV Function ---
def plvfcn(eegData):
    numElectrodes = eegData.shape[0]
    numTimeSteps = eegData.shape[1]
    plvMatrix = np.zeros((numElectrodes, numElectrodes))
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            phase1 = np.angle(sig.hilbert(eegData[electrode1, :]))
            phase2 = np.angle(sig.hilbert(eegData[electrode2, :]))
            phase_diff = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_diff)) / numTimeSteps)
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    return plvMatrix

# --- Label Mapping ---
label_map = {1: "Left", 2: "Right"}

# --- Plot all trials ---
for idx, (trial, meta) in enumerate(zip(eeg_trials, trial_meta)):
    label = meta.get('targetnumber')
    if label not in [1, 2]:
        continue

    plv = plvfcn(trial)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(plv, cmap='hot', interpolation='nearest')
    plt.colorbar(im, label='PLV')
    plt.title(f"Trial {idx} - Class: {label_map[label]}")

    # Use channel names as axis ticks
    plt.xticks(ticks=np.arange(len(channel_labels)), labels=channel_labels, rotation=90, fontsize=6)
    plt.yticks(ticks=np.arange(len(channel_labels)), labels=channel_labels, fontsize=6)

    plt.tight_layout()
    plt.show()

#%% Comparing ALS electrodes and Class Averages for less processing

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from tqdm import tqdm  

# --- Setup ---
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Subject-Independent Graphical Models\Subject-Independent-Graphical-Models\He\He_Dataset'
file_path = os.path.join(data_dir, 'S1_Session_1.pkl')

with open(file_path, 'rb') as f:
    bci = pickle.load(f)

eeg_trials = bci['data']
trial_meta = bci['TrialData']
all_labels = [str(lbl) for lbl in bci['chaninfo']['label']]

# --- Target subset of electrodes ---
subset_labels = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7",
                 "C3", "CZ", "C4", "T8", "P7", "P3", "PZ", "P4",
                 "P8", "O1", "O2"]

# --- PLV Function ---
def plvfcn(eegData):
    numElectrodes = eegData.shape[0]
    numTimeSteps = eegData.shape[1]
    plvMatrix = np.zeros((numElectrodes, numElectrodes))
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            phase1 = np.angle(sig.hilbert(eegData[electrode1, :]))
            phase2 = np.angle(sig.hilbert(eegData[electrode2, :]))
            phase_diff = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_diff)) / numTimeSteps)
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    return plvMatrix

# --- Collect PLVs by class ---
left_plvs, right_plvs = [], []

print("[INFO] Computing PLV for each trial...")
for trial, meta in tqdm(zip(eeg_trials, trial_meta), total=len(eeg_trials), desc="PLV per trial"):
    label = meta.get('targetnumber')
    if label not in [1, 2]:
        continue
    plv = plvfcn(trial)
    if label == 1:
        left_plvs.append(plv)
    elif label == 2:
        right_plvs.append(plv)

# --- Average PLVs ---
left_avg = np.mean(left_plvs, axis=0)
right_avg = np.mean(right_plvs, axis=0)

print(f"[INFO] Averaged PLVs: Left({len(left_plvs)} trials), Right({len(right_plvs)} trials)")

# --- Subset PLV using channel index mask ---
subset_indices = [i for i, ch in enumerate(all_labels) if ch in subset_labels]

def extract_subset(plv_matrix, indices):
    return plv_matrix[np.ix_(indices, indices)]

left_avg_subset = extract_subset(left_avg, subset_indices)
right_avg_subset = extract_subset(right_avg, subset_indices)
subset_names = [all_labels[i] for i in subset_indices]

# --- Plot Function ---
def plot_plv(plv_matrix, title, labels):
    plt.figure(figsize=(7, 6))
    im = plt.imshow(plv_matrix, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(im, label='PLV')
    plt.title(title)
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90, fontsize=6)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=6)
    plt.tight_layout()
    plt.show()

# --- Plot all-electrode averages ---
plot_plv(left_avg, "Average PLV - Left (All Channels)", all_labels)
plot_plv(right_avg, "Average PLV - Right (All Channels)", all_labels)

# --- Plot subset-electrode averages ---
plot_plv(left_avg_subset, "Average PLV - Left (Subset Channels)", subset_names)
plot_plv(right_avg_subset, "Average PLV - Right (Subset Channels)", subset_names)

