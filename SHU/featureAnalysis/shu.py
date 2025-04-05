import os
from os.path import join as pjoin
import numpy as np
from scipy.io import loadmat
import scipy.signal as sig
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, GraphNorm
from torch_geometric.seed import seed_everything
import matplotlib.pyplot as plt

# ---------------------------
# Set Seed for Reproducibility
# ---------------------------
seed_everything(12345)

# ---------------------------
# Data Loading and Preprocessing (SHU Dataset)
# ---------------------------
directory = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\SHU Dataset\MatFiles'
data_by_subject = {}

for filename in os.listdir(directory):
    if filename.endswith('.mat'):
        file_path = os.path.join(directory, filename)
        mat_data = loadmat(file_path)
        parts = filename.split('_')
        subject_id = parts[0]  # e.g., 'sub-001'
        if subject_id not in data_by_subject:
            data_by_subject[subject_id] = {'data': [], 'labels': []}
        data = mat_data['data']      # Adjust key as needed
        labels = mat_data['labels']  # Adjust key as needed
        data_by_subject[subject_id]['data'].append(data)
        data_by_subject[subject_id]['labels'].append(labels)

# Concatenate data for each subject
for subject_id in data_by_subject:
    concatenated_data = np.concatenate(data_by_subject[subject_id]['data'], axis=0)
    concatenated_labels = np.concatenate(data_by_subject[subject_id]['labels'], axis=1)
    data_by_subject[subject_id]['data'] = concatenated_data
    data_by_subject[subject_id]['labels'] = concatenated_labels

# ---------------------------
# Split Data by Label
# ---------------------------
def split_data_by_label(data_by_subject):
    data_split = {}
    for subject, data_dict in data_by_subject.items():
        data = data_dict['data']  # Trials x Channels x Samples
        labels = data_dict['labels']
        if labels.ndim == 2:
            labels = labels.flatten()
        data_L, data_R = [], []
        for i in range(data.shape[0]):
            if labels[i] == 1:
                data_L.append(data[i])
            elif labels[i] == 2:
                data_R.append(data[i])
        data_split[subject] = {'L': np.array(data_L), 'R': np.array(data_R)}
    return data_split

data_split = split_data_by_label(data_by_subject)

# ---------------------------
# Bandpass Filtering
# ---------------------------
fs = 250
def bandpass_filter_trials(data_split, low_freq, high_freq, sfreq):
    filtered_data_split = {}
    nyquist = 0.5 * sfreq
    low = low_freq / nyquist
    high = high_freq / nyquist
    from scipy.signal import butter, filtfilt
    b, a = butter(N=4, Wn=[low, high], btype='band')
    for subject in data_split:
        subject_data = data_split[subject]
        filtered_subject_data = {}
        for direction in ['L', 'R']:
            trials = subject_data[direction]  # Trials x Channels x Samples
            filtered_trials = []
            for trial in range(trials.shape[0]):
                trial_data = trials[trial]  # Channels x Samples
                filtered_trial_data = np.zeros_like(trial_data)
                for ch in range(trial_data.shape[0]):
                    filtered_trial_data[ch, :] = filtfilt(b, a, trial_data[ch, :])
                filtered_trials.append(filtered_trial_data)
            filtered_subject_data[direction] = np.array(filtered_trials)
        filtered_data_split[subject] = filtered_subject_data
    return filtered_data_split

filtered_data_split = bandpass_filter_trials(data_split, low_freq=8, high_freq=30, sfreq=fs)

# ---------------------------
# Merge Left and Right Trials per Subject
# ---------------------------
merged_data = {}
for subject in filtered_data_split:
    left_trials = filtered_data_split[subject]['L']
    right_trials = filtered_data_split[subject]['R']
    combined_trials = np.concatenate((left_trials, right_trials), axis=0)
    left_labels = np.zeros(left_trials.shape[0], dtype=int)  # Label 0 for left
    right_labels = np.ones(right_trials.shape[0], dtype=int)  # Label 1 for right
    combined_labels = np.concatenate((left_labels, right_labels), axis=0)
    merged_data[subject] = {'data': combined_trials, 'label': combined_labels}

# ---------------------------
# PLV Computation and Graph Construction
# ---------------------------
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

def create_graphs_and_edges(plv_matrices, threshold):
    graphs = []
    numElectrodes, _, numTrials = plv_matrices.shape
    adj_matrices = np.zeros((numElectrodes, numElectrodes, numTrials))
    edge_indices = []
    
    for i in range(numTrials):
        G = nx.Graph()
        G.add_nodes_from(range(numElectrodes))
        source_nodes = []
        target_nodes = []
        for u in range(numElectrodes):
            for v in range(u+1, numElectrodes):
                if plv_matrices[u, v, i] > threshold:
                    G.add_edge(u, v, weight=plv_matrices[u, v, i])
                    adj_matrices[u, v, i] = plv_matrices[u, v, i]
                    adj_matrices[v, u, i] = plv_matrices[u, v, i]
                    source_nodes.append(u)
                    target_nodes.append(v)
        graphs.append(G)
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_indices.append(edge_index)
        
    adj_matrices = torch.tensor(adj_matrices, dtype=torch.float32)
    edge_indices = torch.stack(edge_indices, dim=-1)
    return adj_matrices, edge_indices, graphs

def compute_plv(subject_data):
    data = subject_data['data']  # Trials x Channels x Samples
    labels = subject_data['label']  # Trials,
    numTrials, numElectrodes, _ = data.shape
    plv_matrices = np.zeros((numElectrodes, numElectrodes, numTrials))
    
    # Compute PLV matrix for each trial
    for trial_idx in range(numTrials):
        eeg_trial = data[trial_idx]
        plv_matrices[:, :, trial_idx] = plvfcn(eeg_trial)
        
    label_tensor = torch.tensor(labels, dtype=torch.long)
    # Set threshold (e.g., 0)
    adj_matrices, edge_indices, graphs = create_graphs_and_edges(plv_matrices, threshold=0)
    return {
        'plv_matrices': plv_matrices,
        'labels': label_tensor,
        'adj_matrices': adj_matrices,
        'edge_indices': edge_indices,
        'graphs': graphs
    }

subject_plv_data = {}
for subject_id, subject_data in merged_data.items():
    print(f"Processing subject: {subject_id}")
    subject_plv_data[subject_id] = compute_plv(subject_data)

# ---------------------------
# Build PyG Data Objects for GAT
# ---------------------------
all_data = []
for subject_key, subject_data in subject_plv_data.items():
    subject_int = int(subject_key.strip('sub-'))
    plv_matrices = subject_data['plv_matrices']
    labels = subject_data['labels']
    edge_indices = subject_data['edge_indices']
    num_trials = plv_matrices.shape[2]
    for i in range(num_trials):
        x = torch.tensor(plv_matrices[:, :, i], dtype=torch.float)  # Node features from the adjacency matrix.
        # Use edge_indices; if list, then index it, else assume proper tensor.
        if isinstance(edge_indices, list):
            edge_index = edge_indices[i]
        else:
            edge_index = edge_indices[:, :, i]
        y = torch.tensor(labels[i], dtype=torch.long) if not torch.is_tensor(labels) else labels[i]
        data = Data(x=x, edge_index=edge_index, y=y)
        data.subject = int(subject_int)
        all_data.append(data)

#%% Analysis (ALL ORIGINAL ELECTRODES)

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# List of 32 electrodes.
electrodes = ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'CZ',
              'C3', 'C4', 'T3', 'T4', 'A1', 'A2', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ', 'P3',
              'P4', 'T5', 'T6', 'PO3', 'PO4', 'OZ', 'O1', 'O2']

# Group data by subject and by class (assumed classes: 0 and 1).
subject_data = {}
for data in all_data:
    subj = data.subject
    # Assuming that the label is either a tensor or directly 0 or 1.
    label = data.y.item() if torch.is_tensor(data.y) else data.y  
    subject_data.setdefault(subj, {}).setdefault(label, []).append(data)

# Dictionary to hold CV matrices per subject for each class.
subject_cv = {}

for subj, classes in subject_data.items():
    subject_cv[subj] = {}
    for class_label, trials in classes.items():
        num_trials = len(trials)
        # Create a dictionary to store time series for each unique electrode pair.
        time_series = {}
        for i in range(len(electrodes)):
            for j in range(i+1, len(electrodes)):
                time_series[(i, j)] = []

        # Extract PLV values for each electrode pair over trials.
        for trial in trials:
            # Assuming trial.x is a tensor representing the PLV matrix.
            plv_matrix = trial.x.numpy()
            for i in range(len(electrodes)):
                for j in range(i+1, len(electrodes)):
                    time_series[(i, j)].append(plv_matrix[i, j])

        # Create folder structure for subject and class.
        subject_folder = f"subject_{subj}"
        class_folder = os.path.join(subject_folder, f"class_{class_label}")
        os.makedirs(class_folder, exist_ok=True)

        # Initialize a matrix to hold CV values.
        cv_matrix = np.zeros((len(electrodes), len(electrodes)))

        # Generate plots and compute CV for each electrode pair.
        for (i, j), values in time_series.items():
            values = np.array(values)
            # The following code for time series plots is commented out to speed up processing:
            # plt.figure()
            # plt.plot(range(num_trials), values, marker='o')
            # plt.title(f"Subject {subj} - Class {class_label}: {electrodes[i]} vs {electrodes[j]}")
            # plt.xlabel("Trial")
            # plt.ylabel("PLV")
            # plt.grid(True)
            #
            # # Create folder for this electrode pair.
            # pair_folder = os.path.join(class_folder, f"{electrodes[i]}_{electrodes[j]}")
            # os.makedirs(pair_folder, exist_ok=True)
            # plot_filename = os.path.join(pair_folder, "time_series.png")
            # plt.savefig(plot_filename)
            # plt.close()

            # Compute the coefficient of variation (CV).
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / mean_val if mean_val != 0 else np.nan
            cv_matrix[i, j] = cv
            cv_matrix[j, i] = cv  # Ensure symmetry.

        subject_cv[subj][class_label] = cv_matrix
        
        # Save the subject CV matrix figure for this subject and class.
        plt.figure(figsize=(10, 8))
        plt.imshow(cv_matrix, cmap='RdYlGn')
        plt.colorbar(label='CV')
        plt.xticks(range(len(electrodes)), electrodes, rotation=90)
        plt.yticks(range(len(electrodes)), electrodes)
        plt.title(f"Subject {subj} CV Matrix - Class {class_label}")
        plt.tight_layout()
        cv_plot_filename = os.path.join(class_folder, f"subject_cv_matrix_class_{class_label}.png")
        plt.savefig(cv_plot_filename)
        plt.close()

# Global CV calculation using a 3D array.

# Assuming 'subject_cv' is already computed as above, where subject_cv[subj][class_label] is the CV matrix.
subject_numbers = sorted(subject_cv.keys())

for class_label in [0, 1]:
    # Create a 3D array: shape (numElectrodes, numElectrodes, number of subjects)
    all_cv_values = np.stack([subject_cv[subj][class_label] for subj in subject_numbers], axis=2)
    
    # Compute the overall mean and standard deviation across subjects for each electrode pair.
    mean_cv = np.mean(all_cv_values, axis=2)
    std_cv  = np.std(all_cv_values, axis=2)
    
    # Plot the mean CV matrix as a heatmap.
    plt.figure(figsize=(10, 8))
    plt.imshow(mean_cv, cmap='RdYlGn')
    plt.colorbar(label='Mean CV')
    plt.xticks(range(len(electrodes)), electrodes, rotation=90)
    plt.yticks(range(len(electrodes)), electrodes)
    plt.title(f"Global Mean CV Matrix (3D Array) - Class {class_label}")
    plt.tight_layout()
    plt.savefig(f"global_cv_matrix_3d_class_{class_label}.png")
    plt.close()
    
    # Plot the standard deviation CV matrix as a heatmap.
    plt.figure(figsize=(10, 8))
    plt.imshow(std_cv, cmap='RdYlGn')
    plt.colorbar(label='Std CV')
    plt.xticks(range(len(electrodes)), electrodes, rotation=90)
    plt.yticks(range(len(electrodes)), electrodes)
    plt.title(f"Global Std CV Matrix (3D Array) - Class {class_label}")
    plt.tight_layout()
    plt.savefig(f"global_std_cv_matrix_3d_class_{class_label}.png")
    plt.close()


#%% ONLY ELECTRODES USED IN ALS DATASET

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Original 32 electrode labels.
original_electrodes = ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'CZ',
                       'C3', 'C4', 'T3', 'T4', 'A1', 'A2', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ', 'P3',
                       'P4', 'T5', 'T6', 'PO3', 'PO4', 'OZ', 'O1', 'O2']

# Desired ALS electrode labels.
ALS_electrode_labels = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FC1", "FC2", "FC5", "FC6", 
                        "C3", "CZ", "C4", "P3", "PZ", "P4",
                        "OZ", "O1", "O2"]

# Find indices of ALS electrodes that exist in the original list.
selected_indices = [original_electrodes.index(e) for e in ALS_electrode_labels if e in original_electrodes]
# Construct the new electrode list from the original using the selected indices.
subset_electrodes = [original_electrodes[i] for i in selected_indices]
print("Using subset electrodes:", subset_electrodes)

# Create an output folder specifically for ALS electrodes.
output_folder = "alselectrodes"
os.makedirs(output_folder, exist_ok=True)

# Group data by subject and by class (assumed classes: 0 and 1).
subject_data = {}
for data in all_data:
    subj = data.subject
    # Assuming that the label is either a tensor or directly 0 or 1.
    label = data.y.item() if torch.is_tensor(data.y) else data.y  
    subject_data.setdefault(subj, {}).setdefault(label, []).append(data)

# Dictionary to hold CV matrices per subject for each class.
subject_cv = {}

for subj, classes in subject_data.items():
    subject_cv[subj] = {}
    for class_label, trials in classes.items():
        num_trials = len(trials)
        # Create a dictionary to store time series for each unique electrode pair for the subset.
        time_series = {}
        for i in range(len(subset_electrodes)):
            for j in range(i+1, len(subset_electrodes)):
                time_series[(i, j)] = []

        # Process each trial.
        for trial in trials:
            # Assuming trial.x is a tensor representing the 32x32 PLV matrix.
            plv_matrix = trial.x.numpy()
            # Extract the subset corresponding to the ALS electrodes.
            plv_matrix_subset = plv_matrix[np.ix_(selected_indices, selected_indices)]
            # Collect PLV values for each electrode pair in the subset.
            for i in range(len(subset_electrodes)):
                for j in range(i+1, len(subset_electrodes)):
                    time_series[(i, j)].append(plv_matrix_subset[i, j])

        # Create folder structure for subject and class under the ALS output folder.
        subject_folder = os.path.join(output_folder, f"subject_{subj}")
        class_folder = os.path.join(subject_folder, f"class_{class_label}")
        os.makedirs(class_folder, exist_ok=True)

        # Initialize a matrix to hold CV values (size based on the subset).
        cv_matrix = np.zeros((len(subset_electrodes), len(subset_electrodes)))

        # Generate plots and compute CV for each electrode pair.
        for (i, j), values in time_series.items():
            values = np.array(values)
            # The following code for time series plots is commented out to speed up processing:
            # plt.figure()
            # plt.plot(range(num_trials), values, marker='o')
            # plt.title(f"Subject {subj} - Class {class_label}: {subset_electrodes[i]} vs {subset_electrodes[j]}")
            # plt.xlabel("Trial")
            # plt.ylabel("PLV")
            # plt.grid(True)
            # pair_folder = os.path.join(class_folder, f"{subset_electrodes[i]}_{subset_electrodes[j]}")
            # os.makedirs(pair_folder, exist_ok=True)
            # plot_filename = os.path.join(pair_folder, "time_series.png")
            # plt.savefig(plot_filename)
            # plt.close()

            # Compute the coefficient of variation (CV).
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / mean_val if mean_val != 0 else np.nan
            cv_matrix[i, j] = cv
            cv_matrix[j, i] = cv  # Ensure symmetry.

        subject_cv[subj][class_label] = cv_matrix

        # Save the subject CV matrix figure for this subject and class.
        plt.figure(figsize=(10, 8))
        plt.imshow(cv_matrix, cmap='RdYlGn')
        plt.colorbar(label='CV')
        plt.xticks(range(len(subset_electrodes)), subset_electrodes, rotation=90)
        plt.yticks(range(len(subset_electrodes)), subset_electrodes)
        plt.title(f"Subject {subj} CV Matrix - Class {class_label}")
        plt.tight_layout()
        cv_plot_filename = os.path.join(class_folder, f"subject_cv_matrix_class_{class_label}.png")
        plt.savefig(cv_plot_filename)
        plt.close()

# Assuming 'subject_cv' is already computed as above, where subject_cv[subj][class_label] is the CV matrix.
subject_numbers = sorted(subject_cv.keys())

for class_label in [0, 1]:
    # Create a 3D array: shape (numElectrodes, numElectrodes, number of subjects)
    all_cv_values = np.stack([subject_cv[subj][class_label] for subj in subject_numbers], axis=2)
    
    # Compute the overall mean and standard deviation across subjects for each electrode pair.
    mean_cv = np.mean(all_cv_values, axis=2)
    std_cv  = np.std(all_cv_values, axis=2)
    
    # Plot the mean CV matrix as a heatmap.
    plt.figure(figsize=(10, 8))
    plt.imshow(mean_cv, cmap='RdYlGn')
    plt.colorbar(label='Mean CV')
    plt.xticks(range(len(ALS_electrode_labels)), ALS_electrode_labels, rotation=90)
    plt.yticks(range(len(ALS_electrode_labels)), ALS_electrode_labels)
    plt.title(f"Global Mean CV Matrix (3D Array) - Class {class_label}")
    plt.tight_layout()
    plt.savefig(f"global_cv_matrix_3d_class_{class_label}.png")
    plt.close()
    
    # Plot the standard deviation CV matrix as a heatmap.
    plt.figure(figsize=(10, 8))
    plt.imshow(std_cv, cmap='RdYlGn')
    plt.colorbar(label='Std CV')
    plt.xticks(range(len(ALS_electrode_labels)), ALS_electrode_labels, rotation=90)
    plt.yticks(range(len(ALS_electrode_labels)), ALS_electrode_labels)
    plt.title(f"Global Std CV Matrix (3D Array) - Class {class_label}")
    plt.tight_layout()
    plt.savefig(f"global_std_cv_matrix_3d_class_{class_label}.png")
    plt.close()

# Global CV calculation using a 3D array.

# Assuming 'subject_cv' is already computed as above, where subject_cv[subj][class_label] is the CV matrix.
subject_numbers = sorted(subject_cv.keys())

for class_label in [0, 1]:
    # Create a 3D array: shape (numSubsetElectrodes, numSubsetElectrodes, number of subjects)
    all_cv_values = np.stack([subject_cv[subj][class_label] for subj in subject_numbers], axis=2)
    
    # Compute the overall mean and standard deviation across subjects for each electrode pair.
    mean_cv = np.mean(all_cv_values, axis=2)
    std_cv  = np.std(all_cv_values, axis=2)
    
    # Plot the mean CV matrix as a heatmap.
    plt.figure(figsize=(10, 8))
    plt.imshow(mean_cv, cmap='RdYlGn')
    plt.colorbar(label='Mean CV')
    plt.xticks(range(len(subset_electrodes)), subset_electrodes, rotation=90)
    plt.yticks(range(len(subset_electrodes)), subset_electrodes)
    plt.title(f"Global Mean CV Matrix (3D Array) - Class {class_label}")
    plt.tight_layout()
    global_mean_filename = os.path.join(output_folder, f"global_cv_matrix_3d_class_{class_label}.png")
    plt.savefig(global_mean_filename)
    plt.close()
    
    # Plot the standard deviation CV matrix as a heatmap.
    plt.figure(figsize=(10, 8))
    plt.imshow(std_cv, cmap='RdYlGn')
    plt.colorbar(label='Std CV')
    plt.xticks(range(len(subset_electrodes)), subset_electrodes, rotation=90)
    plt.yticks(range(len(subset_electrodes)), subset_electrodes)
    plt.title(f"Global Std CV Matrix (3D Array) - Class {class_label}")
    plt.tight_layout()
    global_std_filename = os.path.join(output_folder, f"global_std_cv_matrix_3d_class_{class_label}.png")
    plt.savefig(global_std_filename)
    plt.close()


#%%

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
from sklearn.decomposition import PCA

# ---------------------------
# Define the Simple GAT Model (Feature Extractor and Classifier)
# ---------------------------
class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(SimpleGAT, self).__init__()
        # First GAT layer: maps from in_channels to 32 features per head.
        self.conv1 = GATv2Conv(in_channels, 64, heads=num_heads, concat=True)
        self.gn1 = GraphNorm(64 * num_heads)
        
        # Second GAT layer: maps from 32*num_heads to 16 features per head.
        self.conv2 = GATv2Conv(64 * num_heads, 32, heads=num_heads, concat=True)
        self.gn2 = GraphNorm(32 * num_heads)
        
        # Third GAT layer: maps from 16*num_heads to 8 features.
        self.conv3 = GATv2Conv(32 * num_heads, 16, heads=num_heads, concat=False)
        self.gn3 = GraphNorm(16)
        
        # Final linear layer: maps the 8-dimensional representation to out_channels (number of classes)
        self.lin = nn.Linear(16, out_channels)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.gn1(self.conv1(x, edge_index)))
        x = F.relu(self.gn2(self.conv2(x, edge_index)))
        x = F.relu(self.gn3(self.conv3(x, edge_index)))
        
        # Global mean pooling to produce graph-level representation.
        x = global_mean_pool(x, batch)
        features = x  # Extracted feature vector.
        logits = self.lin(x)
        return logits, features

# ---------------------------
# Train on All Data for 100 Epochs and Output Feature Vectors for PCA
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 100

# Create DataLoader from all_data (assuming all_data is a list of data objects)
train_loader = DataLoader(all_data, batch_size=32, shuffle=True)

# Instantiate model, optimizer, and loss criterion.
model = SimpleGAT(in_channels=all_data[0].x.shape[1], hidden_channels=32, out_channels=2, num_heads=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training the model for 100 epochs on all_data...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits, _ = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# ---------------------------
# Extract Feature Vectors from All Data After Training
# ---------------------------
model.eval()
all_features = []
all_labels = []
with torch.no_grad():
    eval_loader = DataLoader(all_data, batch_size=32, shuffle=False)
    for batch in eval_loader:
        batch = batch.to(device)
        _, features = model(batch)
        all_features.append(features.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
all_features = np.concatenate(all_features, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# ---------------------------
# Perform PCA on the Extracted Feature Vectors
# ---------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_features)

# ---------------------------
# Plot the PCA Projection
# ---------------------------
plt.figure(figsize=(8, 6))
# Assume class 0 is left (red) and class 1 is right (blue)
for label, color in zip([0, 1], ['red', 'blue']):
    idx = all_labels == label
    plt.scatter(pca_result[idx, 0], pca_result[idx, 1], c=color, label=f'Class {label}', alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Feature Vectors (Red: Left, Blue: Right)")
plt.legend()
plt.tight_layout()
plt.savefig("pca_feature_vectors.png")
plt.show()
