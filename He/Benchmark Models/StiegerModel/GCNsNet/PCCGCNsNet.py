import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import (
    ChebConv, BatchNorm, graclus, max_pool,
    global_mean_pool
)
from torch_geometric.utils import dense_to_sparse
from torch_geometric.seed import seed_everything
from tqdm import tqdm
import scipy.io as sio

# ---------------------------
# CONFIG
# ---------------------------
server_dir = '/home/uceerjp/He/'
dataset_pkl_path = os.path.join(server_dir, 'eeg_trials_dataset.pkl')
pcc_graph_pkl_path = os.path.join(server_dir, 'pcc_graph_dataset.pkl')
results_path = os.path.join(server_dir, 'loso_results_pcc.pkl')

num_epochs = 25
batch_size = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n? Using device: {device}")
seed_everything(12345)

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def load_eeg_trials_dataset(pkl_path):
    """Load EEG trials dataset."""
    with open(pkl_path, 'rb') as f:
        all_data, subject_numbers = pickle.load(f)
    return all_data, subject_numbers

def compute_pcc(eeg_array):
    """Compute absolute Pearson Correlation Coefficient matrix."""
    pcc = np.corrcoef(eeg_array)
    abs_pcc = np.abs(pcc)
    np.fill_diagonal(abs_pcc, 0)
    return abs_pcc

def convert_trial_to_graph(trial):
    """Convert EEG trial (channels x timepoints) into a PyG Graph."""
    eeg_np = trial.x.numpy()  # shape: (channels, timepoints)

    adj = compute_pcc(eeg_np)

    x = torch.tensor(adj, dtype=torch.float)
    src, dst = np.nonzero(adj)
    w = adj[src, dst]

    if len(src) == 0:
        src = np.arange(adj.shape[0])
        dst = np.arange(adj.shape[0])
        w = np.ones(adj.shape[0])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=trial.y
    )
    data.subject = trial.subject
    return data

def split_data_by_subject(data_list, test_subj):
    train, test = [], []
    for graph in data_list:
        if graph.subject == test_subj:
            test.append(graph)
        else:
            train.append(graph)
    return train, test

# ---------------------------
# MODEL: GCNsNet
# ---------------------------
class GCNsNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=2, dropout=0.5):
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = ChebConv(hidden_channels, hidden_channels * 2, K)
        self.bn2 = BatchNorm(hidden_channels * 2)
        self.conv3 = ChebConv(hidden_channels * 2, hidden_channels * 2, K)
        self.bn3 = BatchNorm(hidden_channels * 2)
        self.fc = nn.Linear(hidden_channels * 2, out_channels)
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
# 1) Load or Generate Graph Data
# ---------------------------
if os.path.exists(pcc_graph_pkl_path):
    print(f"\n[CACHE] Loading graph data from: {pcc_graph_pkl_path}")
    with open(pcc_graph_pkl_path, 'rb') as f:
        all_data, subject_numbers = pickle.load(f)
else:
    print(f"\n[INFO] PCC graph file not found. Building graphs from EEG dataset...")
    all_trials, subject_numbers = load_eeg_trials_dataset(dataset_pkl_path)
    all_data = []
    for trial in tqdm(all_trials, desc="Building graphs"):
        graph = convert_trial_to_graph(trial)
        all_data.append(graph)
    with open(pcc_graph_pkl_path, 'wb') as f:
        pickle.dump((all_data, subject_numbers), f)
    print(f"[INFO] Graph data saved to {pcc_graph_pkl_path}")

# ---------------------------
# 2) Load Previous Results if Exist
# ---------------------------
if os.path.exists(results_path):
    with open(results_path, 'rb') as f:
        loso_results = pickle.load(f)
    print(f"\n[CHECKPOINT] Loaded {len(loso_results)} previous LOSO results.")
else:
    loso_results = {}

# ---------------------------
# 3) LOSO Training Loop
# ---------------------------
print("\n[INFO] Starting LOSO Training...\n")

for test_subject in tqdm(subject_numbers, desc="LOSO folds"):
    if test_subject in loso_results:
        print(f"[SKIP] Subject {test_subject} already done.")
        continue

    print(f"\n=== LOSO Fold: Test Subject {test_subject} ===")
    train_data, test_data = split_data_by_subject(all_data, test_subject)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = GCNsNet(
        in_channels=train_data[0].x.shape[1],
        hidden_channels=32,
        out_channels=2,
        K=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(num_epochs),
                      desc=f"  Epochs (Test S{test_subject})",
                      leave=False):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Subject {test_subject}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}", flush=True)

    # Evaluate after training
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch)
            preds = logits.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_graphs

    final_acc = correct / total if total > 0 else 0.0
    print(f"[RESULT] Subject {test_subject} - Final Accuracy: {final_acc * 100:.2f}%")

    loso_results[test_subject] = (final_acc, num_epochs)

    # Save interim results
    with open(results_path, 'wb') as f:
        pickle.dump(loso_results, f)

# ---------------------------
# 4) Print Summary
# ---------------------------
print("\n? LOSO Summary:")
for subj, (acc, ep) in sorted(loso_results.items()):
    print(f"Subject {subj}: Final Accuracy = {acc * 100:.2f}% after {ep} epochs")

avg_acc = np.mean([acc for acc, _ in loso_results.values()])
print(f"\n? Average LOSO Accuracy: {avg_acc * 100:.2f}%")
