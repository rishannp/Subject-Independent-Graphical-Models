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

# ---------------------------
# LOSO Split Function
# ---------------------------
def split_data_by_subject(data_list, test_subject):
    train_data = [d for d in data_list if d.subject != test_subject]
    test_data  = [d for d in data_list if d.subject == test_subject]
    return train_data, test_data

# ---------------------------
# Define the Simple GAT Model (Feature Extractor and Classifier)
# ---------------------------
class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(SimpleGAT, self).__init__()
        # First GAT layer: maps from in_channels to 64 features per head,
        # with concatenation output dimension 64*num_heads.
        self.conv1 = GATv2Conv(in_channels, 32, heads=num_heads, concat=True)
        self.gn1 = GraphNorm(32 * num_heads)
        
        # Second GAT layer: maps from 64*num_heads to 32 features per head,
        # with concatenation output dimension 32*num_heads.
        self.conv2 = GATv2Conv(32 * num_heads, 16, heads=num_heads, concat=True)
        self.gn2 = GraphNorm(16 * num_heads)
        
        # Third GAT layer: maps from 32*num_heads to 8 features,
        # using averaging over heads (concat=False) to yield a feature vector of dimension 8.
        self.conv3 = GATv2Conv(16 * num_heads, 8, heads=num_heads, concat=False)
        self.gn3 = GraphNorm(8)
        
        # Final linear layer: maps the 8-dimensional representation to out_channels (number of classes)
        self.lin = nn.Linear(8, out_channels)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.gn1(self.conv1(x, edge_index)))
        x = F.relu(self.gn2(self.conv2(x, edge_index)))
        x = F.relu(self.gn3(self.conv3(x, edge_index)))
        
        # Global mean pooling to produce graph-level representation.
        x = global_mean_pool(x, batch)
        features = x  # Extracted features.
        logits = self.lin(x)
        return logits, features

# ---------------------------
# LOSO Pipeline: Train GAT and Evaluate at Each Epoch
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 100

loso_results = {}

for test_subject in sorted([int(s.strip('sub-')) for s in subject_plv_data.keys()]):
    print(f"\n=== LOSO Fold: Test Subject {test_subject} ===")
    train_data, test_data = split_data_by_subject(all_data, test_subject)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)
    
    model = SimpleGAT(in_channels=all_data[0].x.shape[1], hidden_channels=32, out_channels=2, num_heads=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_test_acc = 0
    best_epoch = 0
    
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
        
        # Evaluate on test set at this epoch.
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits, _ = model(batch)
                preds = logits.argmax(dim=1)
                correct += (preds == batch.y).sum().item()
                total += batch.num_graphs
        test_acc = correct / total if total > 0 else 0
        print(f"Test Subject {test_subject}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Test Acc: {test_acc*100:.2f}%")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
    
    print(f"Best Test Accuracy for Subject {test_subject}: {best_test_acc*100:.2f}% at Epoch {best_epoch}")
    loso_results[test_subject] = (best_test_acc, best_epoch)

# ---------------------------
# Summary of LOSO Results
# ---------------------------
print("\nLOSO Summary (GAT Classification):")
for subj, (acc, epoch) in sorted(loso_results.items()):
    print(f"Subject {subj}: Best Test Accuracy = {acc*100:.2f}% at Epoch {epoch}")

avg_acc = np.mean([acc for acc, _ in loso_results.values()])
print(f"\nAverage Test Accuracy across LOSO folds: {avg_acc*100:.2f}%")
