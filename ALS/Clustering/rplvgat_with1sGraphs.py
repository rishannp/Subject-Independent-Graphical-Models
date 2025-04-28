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
from torch_geometric.nn import GATv2Conv, global_mean_pool, GraphNorm
from torch_geometric.seed import seed_everything

# ---------------------------
# Set Seed for Reproducibility
# ---------------------------
seed_everything(12345)

def plvfcn(eegData):
    # Limit to first 19 electrodes
    eegData = eegData[:, :19]
    numElectrodes = eegData.shape[1]
    numTimeSteps = eegData.shape[0]
    plvMatrix = np.zeros((numElectrodes, numElectrodes))
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            phase1 = np.angle(sig.hilbert(eegData[:, electrode1]))
            phase2 = np.angle(sig.hilbert(eegData[:, electrode2]))
            phase_diff = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_diff)) / numTimeSteps)
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    np.fill_diagonal(plvMatrix, 0)
    return plvMatrix

def compute_plv_segments(subject_data, fs):
    """
    Split each trial into 1-second (fs-sample) windows and compute PLV for each.
    Returns:
      img: np.array shape (19, 19, n_segments)
      y:   torch.LongTensor shape (n_segments,)
    """
    nch = 19
    all_plvs = []
    all_labels = []

    # Loop classes L=0, R=1
    for label, class_id in zip(['L','R'], [0,1]):
        trials = subject_data[label][0]
        for trial in trials:
            T = trial.shape[0]
            n_windows = T // fs
            for w in range(n_windows):
                start = w * fs
                seg = trial[start:start+fs, :nch]
                plv = plvfcn(seg)
                all_plvs.append(plv)
                all_labels.append(class_id)

    img = np.stack(all_plvs, axis=2)
    y = torch.tensor(all_labels, dtype=torch.long)
    return img, y

def create_graphs(plv, threshold):
    graphs = []
    for i in range(plv.shape[2]):
        G = nx.Graph()
        G.add_nodes_from(range(plv.shape[0]))
        for u in range(plv.shape[0]):
            for v in range(plv.shape[0]):
                if u != v and plv[u, v, i] > threshold:
                    G.add_edge(u, v, weight=plv[u, v, i])
        graphs.append(G)
    return graphs

# -------------------------------------
# Data Loading, PLV Segmentation & Graphs
# -------------------------------------

data_dir = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\OG_Full_Data'
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
fs = 256  # sampling frequency in Hz
all_plvs = {}

for subject_number in subject_numbers:
    print(f'Processing Subject S{subject_number}')
    mat_fname = pjoin(data_dir, f'S{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_raw = mat_contents[f'Subject{subject_number}']
    S1 = subject_raw[:, :-1]

    plv, y = compute_plv_segments(S1, fs)
    threshold = 0.1
    graphs = create_graphs(plv, threshold)

    numElectrodes = 19
    adj = np.zeros((numElectrodes, numElectrodes, len(graphs)))
    for i, G in enumerate(graphs):
        adj[:, :, i] = nx.to_numpy_array(G)
    adj = torch.tensor(adj, dtype=torch.float32)

    edge_indices = []
    for i in range(adj.shape[2]):
        src, tgt = [], []
        for u in range(adj.shape[0]):
            for v in range(adj.shape[1]):
                if adj[u, v, i] >= threshold:
                    src.append(u); tgt.append(v)
                else:
                    src.append(0); tgt.append(0)
        edge_indices.append(torch.tensor([src, tgt], dtype=torch.long))
    edge_indices = torch.stack(edge_indices, dim=-1)

    data_list = []
    for i in range(adj.shape[2]):
        data_list.append(Data(x=adj[:, :, i], edge_index=edge_indices[:, :, i], y=y[i]))

    half = len(data_list)//2
    combined = []
    for i in range(half):
        combined.extend([data_list[i], data_list[i+half]])
    all_plvs[f'S{subject_number}'] = combined

all_data = []
for subject, data_list in all_plvs.items():
    subj_id = int(subject.strip('S'))
    for d in data_list:
        d.subject = subj_id
        all_data.append(d)

def split_data_by_subject(data_list, test_subject):
    train = [d for d in data_list if d.subject != test_subject]
    test  = [d for d in data_list if d.subject == test_subject]
    return train, test

# --------------------------------
# Define GAT Model
# --------------------------------
class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, 32, heads=num_heads, concat=True)
        self.gn1   = GraphNorm(32 * num_heads)
        self.conv2 = GATv2Conv(32 * num_heads, 16, heads=num_heads, concat=True)
        self.gn2   = GraphNorm(16 * num_heads)
        self.conv3 = GATv2Conv(16 * num_heads, 8,  heads=num_heads, concat=False)
        self.gn3   = GraphNorm(8)
        self.lin   = nn.Linear(8, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.gn1(self.conv1(x, edge_index)))
        x = F.relu(self.gn2(self.conv2(x, edge_index)))
        x = F.relu(self.gn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.lin(x), x

# --------------------------------
# LOSO Training & Final Evaluation
# --------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 100
loso_results = {}

for test_subject in subject_numbers:
    print(f"\n=== LOSO Fold: Test Subject {test_subject} ===")
    train_data, test_data = split_data_by_subject(all_data, test_subject)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False)

    model     = SimpleGAT(in_channels=19, hidden_channels=32, out_channels=2, num_heads=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    for epoch in range(1, num_epochs+1):
        model.train()
        loss_sum = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        avg_loss = loss_sum / len(train_loader)
        print(f"Subject {test_subject}, Epoch {epoch}/{num_epochs}, Loss {avg_loss:.4f}")

    # After training, evaluate once on test set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            preds = logits.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total   += batch.num_graphs
    final_acc = correct / total if total > 0 else 0
    print(f"Subject {test_subject}: Final Test Accuracy = {final_acc*100:.2f}% (Best during training: {best_acc*100:.2f}% at epoch {best_epoch})")
    loso_results[test_subject] = (final_acc, best_epoch)

# Summary
avg = np.mean([a for a,_ in loso_results.values()])
print(f"\nAverage LOSO Acc: {avg*100:.2f}%")
