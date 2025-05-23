import os
from os.path import join as pjoin
import numpy as np
import scipy.io as sio
import scipy.signal as sig
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import ChebConv, global_mean_pool, graclus, max_pool, BatchNorm
from torch_geometric.seed import seed_everything
import psutil
import matplotlib.pyplot as plt
from time import time

# ---------------------------
# Utility Functions
# ---------------------------
def plvfcn(eegData):
    """
    Compute the PLV matrix from a trial’s EEG data.
    Assumes eegData shape is (time_steps, numElectrodes) and uses only the first 19 electrodes.
    """
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
    # Set diagonal to zero (to mimic A = |PLV| - I)
    np.fill_diagonal(plvMatrix, 0)
    return plvMatrix

def compute_plv_trials(subject_data):
    """
    For a subject's data, compute the PLV matrix for each trial.
    Assumes subject_data has fields 'L' and 'R' (two conditions).
    Returns:
      - plv_trials: numpy array of shape (num_trials, 19, 19)
      - y: corresponding labels (0 for 'L', 1 for 'R')
    """
    idx = ['L', 'R']
    numElectrodes = 19
    plv_dict = {field: [] for field in idx}
    for field in idx:
        num_trials = subject_data[field].shape[1]
        for j in range(num_trials):
            trial_data = subject_data[field][0, j][:, :19]
            plv_matrix = plvfcn(trial_data)
            plv_dict[field].append(plv_matrix)
    
    plv_trials = np.concatenate([np.stack(plv_dict['L'], axis=0),
                                 np.stack(plv_dict['R'], axis=0)], axis=0)
    numL = len(plv_dict['L'])
    numR = len(plv_dict['R'])
    y = np.concatenate((np.zeros((numL, 1)), np.ones((numR, 1))), axis=0)
    y = torch.tensor(y, dtype=torch.long).squeeze()  # shape: (num_trials,)
    return plv_trials, y

def create_data_objects(plv_trials, y):
    """
    For each trial, build a PyG Data object.
    Uses the PLV matrix (19x19) as follows:
      - Each node’s feature is its corresponding row vector (dimension 19).
      - Edge indices and edge weights are derived from nonzero entries in the PLV matrix.
    """
    data_list = []
    num_trials = plv_trials.shape[0]
    numNodes = plv_trials.shape[1]  # 19 electrodes
    for i in range(num_trials):
        A = plv_trials[i]  # shape: (19, 19)
        x = torch.tensor(A, dtype=torch.float)  # Node features: each row (dim=19)
        source_nodes, target_nodes, weights = [], [], []
        for u in range(numNodes):
            for v in range(numNodes):
                if A[u, v] > 0:  # Adjust threshold if needed
                    source_nodes.append(u)
                    target_nodes.append(v)
                    weights.append(A[u, v])
        if len(source_nodes) == 0:
            # Fallback: add self-loops
            source_nodes = list(range(numNodes))
            target_nodes = list(range(numNodes))
            weights = [1.0] * numNodes
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y[i])
        data_list.append(data)
    return data_list

def get_model_memory(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.element_size() * p.numel() for p in model.parameters()) + \
                 sum(b.element_size() * b.numel() for b in model.buffers())
    return total_params, total_size / (1024 ** 2)  # in MB

def get_ram_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # in GB

# ---------------------------
# Set Seed for Reproducibility
# ---------------------------
seed_everything(12345)

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\OG_Full_Data'
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]

all_plvs = {}
for subject_number in subject_numbers:
    print(f'Processing Subject S{subject_number}')
    mat_fname = pjoin(data_dir, f'S{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_raw = mat_contents[f'Subject{subject_number}']
    S1 = subject_raw[:, :-1]
    plv_trials, y = compute_plv_trials(S1)
    data_list = create_data_objects(plv_trials, y)
    for data in data_list:
        data.subject = subject_number
    all_plvs[f'S{subject_number}'] = data_list

all_data = []
for data_list in all_plvs.values():
    all_data.extend(data_list)

# ---------------------------
# LOSO Split Function
# ---------------------------
def split_data_by_subject(data_list, test_subject):
    train_data = [d for d in data_list if d.subject != test_subject]
    test_data  = [d for d in data_list if d.subject == test_subject]
    return train_data, test_data

# ---------------------------
# GCNs-Net Model
# ---------------------------
class GCNsNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=2, dropout=0.5):
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K)
        self.bn1   = BatchNorm(hidden_channels)
        self.conv2 = ChebConv(hidden_channels, hidden_channels*2, K)
        self.bn2   = BatchNorm(hidden_channels*2)
        self.conv3 = ChebConv(hidden_channels*2, hidden_channels*2, K)
        self.bn3   = BatchNorm(hidden_channels*2)
        self.fc    = nn.Linear(hidden_channels*2, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        # Layer 1
        x = F.softplus(self.bn1(self.conv1(x, edge_index, edge_weight)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        cluster = graclus(edge_index, num_nodes=x.size(0))
        px = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        px = max_pool(cluster, px)
        x, edge_index, edge_weight, batch = px.x, px.edge_index, px.edge_weight, px.batch

        # Layer 2
        x = F.softplus(self.bn2(self.conv2(x, edge_index, edge_weight)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        cluster = graclus(edge_index, num_nodes=x.size(0))
        px = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        px = max_pool(cluster, px)
        x, edge_index, edge_weight, batch = px.x, px.edge_index, px.edge_weight, px.batch

        # Layer 3
        x = F.softplus(self.bn3(self.conv3(x, edge_index, edge_weight)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling + FC
        x = global_mean_pool(x, batch)
        return F.log_softmax(self.fc(x), dim=1)

# ---------------------------
# Evaluation
# ---------------------------
def evaluate(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.num_graphs
    return correct / total

# ---------------------------
# LOSO Training & Single Test
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 25

loso_results = {}

for test_subject in subject_numbers:
    print(f"\nLOSO: leaving subject {test_subject} out")
    train_data, test_data = split_data_by_subject(all_data, test_subject)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False)

    model = GCNsNet(in_channels=19, hidden_channels=32, out_channels=2, K=2, dropout=0.5)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train for all epochs
    start = time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Subject {test_subject}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    train_time = time() - start

    # Single evaluation after training
    test_acc = evaluate(test_loader, model, device)
    print(f"Subject {test_subject} test accuracy: {test_acc*100:.2f}%")

    loso_results[test_subject] = test_acc

    torch.cuda.empty_cache()

# ---------------------------
# Summary
# ---------------------------
print("\n===== LOSO Summary =====")
for subj, acc in loso_results.items():
    print(f"Subject {subj}: Test Accuracy = {acc*100:.2f}%")
avg_acc = np.mean(list(loso_results.values()))
print(f"\nAverage Test Accuracy: {avg_acc*100:.2f}%")
