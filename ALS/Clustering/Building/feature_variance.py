import os
from os.path import join as pjoin
import numpy as np
import scipy.io as sio
import scipy.signal as sig
import networkx as nx
import torch
from torch_geometric.seed import seed_everything
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

# ---------------------------
# Electrode Labels (in order 1-19)
# ---------------------------
electrode_labels = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7",
                    "C3", "CZ", "C4", "T8", "P7", "P3", "PZ", "P4",
                    "P8", "O1", "O2"]

# ---------------------------
# Utility Functions (PLV and Graph Creation)
# ---------------------------
def plvfcn(eegData):
    # Use only the first 19 electrodes
    eegData = eegData[:, :19]
    numElectrodes = eegData.shape[1]
    numTimeSteps = eegData.shape[0]
    plvMatrix = np.zeros((numElectrodes, numElectrodes))
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            phase1 = np.angle(sig.hilbert(eegData[:, electrode1]))
            phase2 = np.angle(sig.hilbert(eegData[:, electrode2]))
            phase_difference = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_difference)) / numTimeSteps)
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    return plvMatrix

def compute_plv(subject_data):
    # Assumes subject_data has fields 'L' and 'R'
    idx = ['L', 'R']
    numElectrodes = 19
    # Create dictionary: each entry is an array of shape (numElectrodes, numElectrodes, num_trials)
    plv = {field: np.zeros((numElectrodes, numElectrodes, subject_data.shape[1])) for field in idx}
    for i, field in enumerate(idx):
        for j in range(subject_data.shape[1]):
            x = subject_data[field][0, j][:, :19]
            plv[field][:, :, j] = plvfcn(x)
    # For convenience, return the two class matrices separately
    return plv['L'], plv['R']

def create_graph_from_plv(plv_matrix, threshold=0.1):
    """
    Given a PLV matrix (numElectrodes x numElectrodes), create a NetworkX graph.
    Only include edges with weight >= threshold.
    """
    numElectrodes = plv_matrix.shape[0]
    G = nx.Graph()
    # Add nodes with electrode labels
    for i in range(numElectrodes):
        G.add_node(i, label=electrode_labels[i])
    for i in range(numElectrodes):
        for j in range(i+1, numElectrodes):
            weight = plv_matrix[i, j]
            if weight >= threshold:
                G.add_edge(i, j, weight=weight)
    return G

def interpolate_positions(pos1, pos2, steps=10):
    """
    Given two dictionaries of node positions (pos1, pos2), return a list of dictionaries 
    representing intermediate positions (linear interpolation).
    """
    interpolated = []
    nodes = pos1.keys()
    for s in range(1, steps+1):
        t = s / float(steps + 1)  # t in (0,1)
        pos_interp = {node: (1 - t) * np.array(pos1[node]) + t * np.array(pos2[node])
                      for node in nodes}
        interpolated.append(pos_interp)
    return interpolated

# ---------------------------
# Set Seed and Paths
# ---------------------------
seed_everything(12345)
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\OG_Full_Data'
output_root = r'./plv_analysis_outputs'
os.makedirs(output_root, exist_ok=True)
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]

# ---------------------------
# Processing and Visualization for Each Subject
# ---------------------------
for subject_number in subject_numbers:
    print(f'Processing Subject S{subject_number}')
    
    # Create output folders for this subject (one per class)
    subject_folder = pjoin(output_root, f'Subject_{subject_number}')
    os.makedirs(subject_folder, exist_ok=True)
    for cls in ['L', 'R']:
        os.makedirs(pjoin(subject_folder, cls, 'line_plots'), exist_ok=True)
        os.makedirs(pjoin(subject_folder, cls, 'heatmaps'), exist_ok=True)
        os.makedirs(pjoin(subject_folder, cls, 'animations'), exist_ok=True)
    
    # Load subject data
    mat_fname = pjoin(data_dir, f'S{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_raw = mat_contents[f'Subject{subject_number}']
    # Use all columns except the last one
    S1 = subject_raw[:, :-1]
    
    # Compute PLV matrices for left and right classes (shape: [19, 19, num_trials])
    plv_left, plv_right = compute_plv(S1)
    
    for cls, plv_all in zip(['L', 'R'], [plv_left, plv_right]):
        print(f"  Processing class {cls}")
        num_trials = plv_all.shape[2]
        numElectrodes = plv_all.shape[0]
        
        # --- For each unique electrode pair, extract time series and compute stability metrics ---
        # We'll store stability metrics in matrices (std and coefficient of variation, CV)
        std_matrix = np.zeros((numElectrodes, numElectrodes))
        cv_matrix = np.zeros((numElectrodes, numElectrodes))
        
        # Loop over unique pairs (i < j)
        for i in range(numElectrodes):
            for j in range(i+1, numElectrodes):
                time_series = plv_all[i, j, :]  # shape (num_trials,)
                std_val = np.std(time_series)
                mean_val = np.mean(time_series)
                cv_val = std_val / mean_val if mean_val != 0 else 0
                std_matrix[i, j] = std_val
                std_matrix[j, i] = std_val
                cv_matrix[i, j] = cv_val
                cv_matrix[j, i] = cv_val
                
                # --- Plot line plot for this electrode pair ---
                plt.figure(figsize=(8, 4))
                plt.plot(np.arange(1, num_trials+1), time_series, marker='o', linestyle='-')
                plt.ylim(0,1)
                plt.title(f'Subject {subject_number} Class {cls} - Electrode Pair ({electrode_labels[i]}, {electrode_labels[j]})')
                plt.xlabel('Trial')
                plt.ylabel('PLV')
                plt.grid(True)
                lineplot_path = pjoin(subject_folder, cls, 'line_plots', f'pair_{electrode_labels[i]}_{electrode_labels[j]}.png')
                plt.savefig(lineplot_path)
                plt.close()
        
        # --- Plot Heatmaps ---
        # Heatmap of standard deviation (with electrode labels on ticks)
        plt.figure(figsize=(10, 8))
        im = plt.imshow(std_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, label='Standard Deviation')
        plt.title(f'Subject {subject_number} Class {cls} - PLV Stability (STD)')
        plt.xticks(ticks=np.arange(numElectrodes), labels=electrode_labels, rotation=45)
        plt.yticks(ticks=np.arange(numElectrodes), labels=electrode_labels)
        heatmap_std_path = pjoin(subject_folder, cls, 'heatmaps', 'std_heatmap.png')
        plt.tight_layout()
        plt.savefig(heatmap_std_path)
        plt.close()
        
        # Heatmap of coefficient of variation (CV)
        plt.figure(figsize=(10, 8))
        im = plt.imshow(cv_matrix, cmap='plasma', interpolation='nearest')
        plt.colorbar(im, label='Coefficient of Variation')
        plt.title(f'Subject {subject_number} Class {cls} - PLV Stability (CV)')
        plt.xticks(ticks=np.arange(numElectrodes), labels=electrode_labels, rotation=45)
        plt.yticks(ticks=np.arange(numElectrodes), labels=electrode_labels)
        heatmap_cv_path = pjoin(subject_folder, cls, 'heatmaps', 'cv_heatmap.png')
        plt.tight_layout()
        plt.savefig(heatmap_cv_path)
        plt.close()
        
        # --- Create Animation with Smooth Transitions ---
        threshold = 0.1  # threshold for including edges
        
        # First, compute a spring layout for each trial's graph using a fixed seed for reproducibility.
        trial_layouts = []
        for trial in range(num_trials):
            plv_trial = plv_all[:, :, trial]
            G_trial = create_graph_from_plv(plv_trial, threshold=threshold)
            # Use spring layout (force-directed) for each trial.
            #layout = nx.spring_layout(G_trial, seed=42)
            layout = nx.spectral_layout(G_trial)
            trial_layouts.append(layout)
        
        # Now, interpolate between consecutive trial layouts to get smooth transitions.
        steps_between = 10  # number of intermediate frames between consecutive trials
        smooth_layouts = []
        for i in range(len(trial_layouts)-1):
            smooth_layouts.append(trial_layouts[i])
            intermediates = interpolate_positions(trial_layouts[i], trial_layouts[i+1], steps=steps_between)
            smooth_layouts.extend(intermediates)
        smooth_layouts.append(trial_layouts[-1])
        
        total_frames = len(smooth_layouts)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        def update(frame):
            ax.clear()
            # Determine the trial index from frame number:
            # For simplicity, use the corresponding trial's PLV graph if frame maps beyond a trial boundary.
            trial_idx = min(frame // (steps_between+1), num_trials-1)
            plv_trial = plv_all[:, :, trial_idx]
            # Use the smooth layout for this frame.
            pos = smooth_layouts[frame]
            # Create graph from current trial's PLV
            G = create_graph_from_plv(plv_trial, threshold=threshold)
            # Draw nodes, edges and labels.
            weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=500)
            nx.draw_networkx_edges(G, pos, ax=ax, width=weights, edge_color='gray')
            labels = {node: electrode_labels[node] for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10)
            # Calculate current trial number (approximate)
            trial_display = trial_idx + 1
            ax.set_title(f'Subject {subject_number} Class {cls} - Trial {trial_display}/{num_trials}')
            ax.axis('off')
        
        ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=100, repeat=True)
        anim_path = pjoin(subject_folder, cls, 'animations', 'graph_evolution.mp4')
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Your Name'), bitrate=1800)
        ani.save(anim_path, writer=writer)
        plt.close()
        
    print(f"Subject {subject_number} processing complete.\n")


#%% Trying to find the regions which correlate the most across participants.

import os
import numpy as np
import scipy.io as sio
import scipy.signal as sig
import torch
from torch_geometric.seed import seed_everything

# ---------------------------
# Electrode Labels (in order 1-19)
# ---------------------------
electrode_labels = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7",
                    "C3", "CZ", "C4", "T8", "P7", "P3", "PZ", "P4",
                    "P8", "O1", "O2"]

# ---------------------------
# Utility Functions (PLV computation)
# ---------------------------
def plvfcn(eegData):
    # Use only the first 19 electrodes
    eegData = eegData[:, :19]
    numElectrodes = eegData.shape[1]
    numTimeSteps = eegData.shape[0]
    plvMatrix = np.zeros((numElectrodes, numElectrodes))
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            phase1 = np.angle(sig.hilbert(eegData[:, electrode1]))
            phase2 = np.angle(sig.hilbert(eegData[:, electrode2]))
            phase_difference = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_difference)) / numTimeSteps)
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    return plvMatrix

def compute_plv(subject_data):
    # Assumes subject_data has fields 'L' and 'R'
    idx = ['L', 'R']
    numElectrodes = 19
    # Each entry will be an array of shape (numElectrodes, numElectrodes, num_trials)
    plv = {field: np.zeros((numElectrodes, numElectrodes, subject_data.shape[1])) for field in idx}
    for i, field in enumerate(idx):
        for j in range(subject_data.shape[1]):
            x = subject_data[field][0, j][:, :19]
            plv[field][:, :, j] = plvfcn(x)
    return plv['L'], plv['R']

# ---------------------------
# Set Seed and Define Data Directory/Subjects
# ---------------------------
seed_everything(12345)
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\OG_Full_Data'
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]

# Dictionary to store the stability matrices (STD and CV)
# Structure: stability_matrices[subject_number][class] = {'std': std_matrix, 'cv': cv_matrix}
stability_matrices = {}

# ---------------------------
# Processing for Each Subject
# ---------------------------
for subject_number in subject_numbers:
    print(f'Processing Subject S{subject_number}')
    
    # Initialize dictionary for this subject
    stability_matrices[subject_number] = {}
    
    # Load subject data
    mat_fname = os.path.join(data_dir, f'S{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_raw = mat_contents[f'Subject{subject_number}']
    # Use all columns except the last one
    S1 = subject_raw[:, :-1]
    
    # Compute PLV matrices for left and right classes (shape: [19, 19, num_trials])
    plv_left, plv_right = compute_plv(S1)
    
    for cls, plv_all in zip(['L', 'R'], [plv_left, plv_right]):
        print(f"  Processing class {cls}")
        num_trials = plv_all.shape[2]
        numElectrodes = plv_all.shape[0]
        
        # Initialize stability matrices for this class
        std_matrix = np.zeros((numElectrodes, numElectrodes))
        cv_matrix = np.zeros((numElectrodes, numElectrodes))
        
        # Loop over unique electrode pairs (i < j) to compute metrics
        for i in range(numElectrodes):
            for j in range(i+1, numElectrodes):
                time_series = plv_all[i, j, :]  # PLV values over trials for this pair
                std_val = np.std(time_series)
                mean_val = np.mean(time_series)
                cv_val = std_val / mean_val if mean_val != 0 else 0
                std_matrix[i, j] = std_val
                std_matrix[j, i] = std_val
                cv_matrix[i, j] = cv_val
                cv_matrix[j, i] = cv_val
        
        # Save the computed matrices in the dictionary for this subject and class
        stability_matrices[subject_number][cls] = {'std': std_matrix, 'cv': cv_matrix}
        
        print(f"    Completed class {cls}: STD and CV matrices computed.")

print("Processing complete. The stability_matrices dictionary contains the results.")

# For Subject 1, class 'L': stability_matrices[1]['L']['std'] and stability_matrices[1]['L']['cv']

#%% Averaging only tells me which electrode pairs are most stable but not necessarily across subjects.

import numpy as np

# Assume stability_matrices is a dictionary where:
# stability_matrices[subject][cls]['cv'] is a 19x19 numpy array.
# subject_numbers is a list of subject numbers (e.g., [1,2,5,9,21,31,34,39]).

# We'll compute the average CV matrix across subjects for each class:
avg_cv = {}  # Will store avg_cv['L'] and avg_cv['R']

for cls in ['L', 'R']:
    cv_accumulator = []
    for subj in subject_numbers:
        cv_accumulator.append(stability_matrices[subj][cls]['cv'])
    # Stack along a new axis and average over subjects
    avg_cv[cls] = np.mean(np.stack(cv_accumulator, axis=0), axis=0)

# Option 1: Look at each class separately.
def get_top_stable_pairs(cv_matrix, top_k=5):
    """
    Given a CV matrix (lower is better), return a sorted list of electrode pair indices (i,j)
    that have the lowest CV values. Only consider i < j to avoid duplicates.
    """
    numElectrodes = cv_matrix.shape[0]
    pairs = []
    for i in range(numElectrodes):
        for j in range(i+1, numElectrodes):
            pairs.append(((i, j), cv_matrix[i, j]))
    # Sort by CV value (ascending: lower is more stable)
    pairs.sort(key=lambda x: x[1])
    return pairs[:top_k]

print("Top stable electrode pairs for Class L (lowest CV):")
top_pairs_L = get_top_stable_pairs(avg_cv['L'], top_k=19)
for (i, j), cv_val in top_pairs_L:
    print(f"Electrode Pair ({electrode_labels[i]}, {electrode_labels[j]}) - Avg CV: {cv_val:.4f}")

print("\nTop stable electrode pairs for Class R (lowest CV):")
top_pairs_R = get_top_stable_pairs(avg_cv['R'], top_k=19)
for (i, j), cv_val in top_pairs_R:
    print(f"Electrode Pair ({electrode_labels[i]}, {electrode_labels[j]}) - Avg CV: {cv_val:.4f}")

# Option 2: Look for pairs that are stable in both classes.
# For this, we can average the CV matrices for the two classes.
avg_cv_both = (avg_cv['L'] + avg_cv['R']) / 2.0

#%% CV from each subject and getting the mean and std to find subject-invariant features.

import numpy as np
import matplotlib.pyplot as plt

# --- Assumptions ---
# stability_matrices[subject_number]['L']['cv'] and ['R']['cv'] are 19x19 CV arrays.
# subject_numbers is a list of subject numbers (e.g., [1,2,5,9,21,31,34,39]).
# electrode_labels is a list of 19 electrode labels.
numElectrodes = 19
num_subjects = len(subject_numbers)
top_k = 50  # Number of top pairs to highlight

def get_most_shared_pairs(mean_matrix, std_matrix, top_k=20):
    """
    Given the mean and std matrices for CV values across subjects,
    returns the top_k electrode pairs (considering only i < j) with the lowest std.
    Each pair is returned as a tuple: ((i, j), mean_cv, std_cv).
    """
    pairs = []
    for i in range(numElectrodes):
        for j in range(i+1, numElectrodes):
            pairs.append(((i, j), mean_matrix[i, j], std_matrix[i, j]))
    pairs.sort(key=lambda x: x[2])  # sort by std (lower means more consistent)
    return pairs[:top_k]

# Dictionaries to store overall average CV and top shared CV matrices for each class.
overall_mean_cv = {}
top_cv = {}

for cls in ['L', 'R']:
    # Initialize an array to hold CV values for each electrode pair across subjects.
    # Shape: (numElectrodes, numElectrodes, num_subjects)
    all_cv_values = np.zeros((numElectrodes, numElectrodes, num_subjects))
    for subj_idx, subj in enumerate(subject_numbers):
        all_cv_values[:, :, subj_idx] = stability_matrices[subj][cls]['cv']
    
    # Compute the overall mean and standard deviation across subjects for each electrode pair.
    mean_cv = np.mean(all_cv_values, axis=2)
    std_cv  = np.std(all_cv_values, axis=2)
    
    overall_mean_cv[cls] = mean_cv.copy()
    
    # Retrieve top shared electrode pairs (lowest inter-subject std)
    top_shared_pairs = get_most_shared_pairs(mean_cv, std_cv, top_k=top_k)
    
    print(f"Top shared electrode pairs for Class {cls}:")
    for (i, j), m_cv, s_cv in top_shared_pairs:
        print(f"Electrode Pair ({electrode_labels[i]}, {electrode_labels[j]}): Mean CV = {m_cv:.4f}, Std CV = {s_cv:.4f}")
    print("\n")
    
    # Build a 19x19 matrix that is zero everywhere except for the top shared pairs.
    top_matrix = np.zeros((numElectrodes, numElectrodes))
    for (i, j), m_cv, s_cv in top_shared_pairs:
        top_matrix[i, j] = m_cv
        top_matrix[j, i] = m_cv  # ensure symmetry
    top_cv[cls] = top_matrix

# --- Figure 1: 2-subplot figure for Top Shared Electrode Pairs for Class L and Class R ---
fig1, axs1 = plt.subplots(1, 2, figsize=(14, 6))
for idx, cls in enumerate(['L', 'R']):
    im = axs1[idx].imshow(top_cv[cls], cmap='viridis', interpolation='nearest')
    axs1[idx].set_title(f"Top Shared Electrode Pairs for Class {cls}")
    axs1[idx].set_xticks(np.arange(numElectrodes))
    axs1[idx].set_yticks(np.arange(numElectrodes))
    axs1[idx].set_xticklabels(electrode_labels, rotation=90)
    axs1[idx].set_yticklabels(electrode_labels)
    axs1[idx].set_xlabel("Electrodes")
    axs1[idx].set_ylabel("Electrodes")
    fig1.colorbar(im, ax=axs1[idx], fraction=0.046, pad=0.04, label='Mean CV')
fig1.tight_layout()
plt.show()

# --- Figure 2: Overall Average CV for Class L using the red-yellow-green colormap ---
fig2, ax2 = plt.subplots(figsize=(8, 6))
im2 = ax2.imshow(overall_mean_cv['L'], cmap='RdYlGn', interpolation='nearest')
ax2.set_title("Overall Average CV for Class L")
ax2.set_xticks(np.arange(numElectrodes))
ax2.set_yticks(np.arange(numElectrodes))
ax2.set_xticklabels(electrode_labels, rotation=90)
ax2.set_yticklabels(electrode_labels)
ax2.set_xlabel("Electrodes")
ax2.set_ylabel("Electrodes")
fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Mean CV')
fig2.tight_layout()
plt.show()

# --- Figure 3: Overall Average CV for Class R using the red-yellow-green colormap ---
fig3, ax3 = plt.subplots(figsize=(8, 6))
im3 = ax3.imshow(overall_mean_cv['R'], cmap='RdYlGn', interpolation='nearest')
ax3.set_title("Overall Average CV for Class R")
ax3.set_xticks(np.arange(numElectrodes))
ax3.set_yticks(np.arange(numElectrodes))
ax3.set_xticklabels(electrode_labels, rotation=90)
ax3.set_yticklabels(electrode_labels)
ax3.set_xlabel("Electrodes")
ax3.set_ylabel("Electrodes")
fig3.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Mean CV')
fig3.tight_layout()
plt.show()

