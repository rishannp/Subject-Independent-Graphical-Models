import os
from os.path import join as pjoin
import numpy as np
import scipy.io as sio
import scipy.signal as sig
import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import ChebConv, global_mean_pool, graclus, max_pool, BatchNorm
from torch_geometric.seed import seed_everything
from time import time

# ---------------------------
# Utility Functions
# ---------------------------

def pccfcn(eegData):
    """
    Compute the absolute Pearson Correlation Coefficient matrix
    for the first 19 EEG channels of a single trial.
    Returns a 19×19 adjacency matrix (abs(PCC)) with zero diagonal.
    """
    X = eegData[:, :19]
    pcc = np.corrcoef(X.T)
    abs_pcc = np.abs(pcc)
    np.fill_diagonal(abs_pcc, 0)
    return abs_pcc

def compute_pcc_trials(subject_data):
    """
    For each trial in 'L' and 'R' conditions, compute the abs(PCC) adjacency matrices.
    Returns:
      - adjs: numpy array (n_trials, 19, 19)
      - y: torch tensor of labels (0='L', 1='R')
    """
    labels = ['L', 'R']
    adj_dict = {lbl: [] for lbl in labels}

    for lbl in labels:
        for j in range(subject_data[lbl].shape[1]):
            trial = subject_data[lbl][0, j]
            adj = pccfcn(trial)
            adj_dict[lbl].append(adj)

    adjs = np.vstack([
        np.stack(adj_dict['L'], axis=0),
        np.stack(adj_dict['R'], axis=0)
    ])  # shape: (n_trials, 19, 19)

    nL = len(adj_dict['L'])
    nR = len(adj_dict['R'])
    y = np.concatenate([np.zeros(nL), np.ones(nR)])
    y = torch.tensor(y, dtype=torch.long)

    return adjs, y

def create_data_objects(adjs, y):
    """
    Turn each adjacency matrix into a PyG Data graph:
      - x: each node's feature is its row vector from the adjacency matrix (dim=19)
      - edge_index & edge_weight: from nonzero adjacency entries
    """
    data_list = []
    n_trials, N, _ = adjs.shape
    for i in range(n_trials):
        adj = adjs[i]
        x = torch.tensor(adj, dtype=torch.float)  # Node features: each row (19-dim)

        # Build edge list from nonzero adjacency entries
        src, dst = np.nonzero(adj)
        w = adj[src, dst]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.tensor(w, dtype=torch.float)

        if edge_index.size(1) == 0:
            # Safety: add self-loops if graph is empty
            src = np.arange(N)
            dst = np.arange(N)
            w = np.ones(N)
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_weight = torch.tensor(w, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index,
                    edge_weight=edge_weight,
                    y=y[i])
        data_list.append(data)
    return data_list

def get_model_memory(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size = (sum(p.element_size() * p.numel() for p in model.parameters()) +
                  sum(b.element_size() * b.numel() for b in model.buffers()))
    return total_params, total_size / (1024**2)  # MB

def get_ram_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)  # GB

# ---------------------------
# Reproducibility
# ---------------------------
seed_everything(12345)

# ---------------------------
# Data Loading & Preprocessing
# ---------------------------
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\OG_Full_Data'
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]

all_data = []
for s in subject_numbers:
    print(f'Processing Subject S{s}')
    mat = sio.loadmat(pjoin(data_dir, f'S{s}.mat'))
    raw = mat[f'Subject{s}']
    S1 = raw[:, :-1]  # drop last column if needed

    adjs, y = compute_pcc_trials(S1)
    data_list = create_data_objects(adjs, y)
    for d in data_list:
        d.subject = s
    all_data += data_list

# ---------------------------
# LOSO Split
# ---------------------------
def split_data(data_list, test_subj):
    train = [d for d in data_list if d.subject != test_subj]
    test = [d for d in data_list if d.subject == test_subj]
    return train, test

# ---------------------------
# GCNsNet Definition
# ---------------------------
class GCNsNet(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, K=2, dropout=0.5):
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden, K)
        self.bn1   = BatchNorm(hidden)
        self.conv2 = ChebConv(hidden, hidden*2, K)
        self.bn2   = BatchNorm(hidden*2)
        self.conv3 = ChebConv(hidden*2, hidden*2, K)
        self.bn3   = BatchNorm(hidden*2)
        self.fc    = nn.Linear(hidden*2, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        x = F.softplus(self.bn1(self.conv1(x, edge_index, edge_weight)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        cluster = graclus(edge_index, num_nodes=x.size(0))
        px = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        px = max_pool(cluster, px)
        x, edge_index, edge_weight, batch = px.x, px.edge_index, px.edge_weight, px.batch

        x = F.softplus(self.bn2(self.conv2(x, edge_index, edge_weight)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        cluster = graclus(edge_index, num_nodes=x.size(0))
        px = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        px = max_pool(cluster, px)
        x, edge_index, edge_weight, batch = px.x, px.edge_index, px.edge_weight, px.batch

        x = F.softplus(self.bn3(self.conv3(x, edge_index, edge_weight)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return F.log_softmax(self.fc(x), dim=1)

# ---------------------------
# Evaluation Helper
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
# LOSO Training & Testing
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 25
batch_size = 32

loso_results = {}

for test_s in subject_numbers:
    print(f"\nLOSO: leaving subject {test_s} out")
    train_list, test_list = split_data(all_data, test_s)
    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)

    model = GCNsNet(in_channels=19, hidden=32, out_channels=2, K=2, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    start = time()
    for epoch in range(1, num_epochs+1):
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
        print(f" Subj {test_s} | Epoch {epoch:02d}/{num_epochs} | Loss: {avg_loss:.4f}")
    train_time = time() - start

    acc = evaluate(test_loader, model, device)
    print(f" Subject {test_s} ⇒ Test Accuracy: {acc*100:.2f}% (Train time: {train_time:.1f}s)")
    loso_results[test_s] = acc
    torch.cuda.empty_cache()

# ---------------------------
# Summary
# ---------------------------
print("\n===== LOSO Summary =====")
for subj, acc in loso_results.items():
    print(f" Subject {subj}: {acc*100:.2f}%")
print(f"\n Average Test Accuracy: {np.mean(list(loso_results.values()))*100:.2f}%")
