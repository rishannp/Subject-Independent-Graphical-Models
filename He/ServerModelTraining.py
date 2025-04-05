import os
import pickle
import numpy as np
import scipy.signal as sig
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
from torch_geometric.utils import dense_to_sparse

# ---------------------------
# Server and dataset setup
# ---------------------------
server_dir = '/home/uceerjp/He/'
num_epochs = 100
all_data = []
subject_numbers = []

# ---------------------------
# Define the GAT model
# ---------------------------
class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(SimpleGAT, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.conv3 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        
        self.gn1 = GraphNorm(hidden_channels * num_heads)
        self.gn2 = GraphNorm(hidden_channels * num_heads)
        self.gn3 = GraphNorm(hidden_channels * num_heads)
        
        self.lin = nn.Linear(hidden_channels * num_heads, out_channels)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.gn1(self.conv1(x, edge_index)))
        x = F.relu(self.gn2(self.conv2(x, edge_index)))
        x = F.relu(self.gn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        features = x
        logits = self.lin(x)
        return logits, features

# ---------------------------
# Helper: compute PLV matrix
# ---------------------------
def compute_plv_matrix(eeg):
    n_channels, n_times = eeg.shape
    plv = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase1 = np.angle(sig.hilbert(eeg[i]))
            phase2 = np.angle(sig.hilbert(eeg[j]))
            diff = phase2 - phase1
            plv_val = np.abs(np.sum(np.exp(1j * diff)) / n_times)
            plv[i, j] = plv_val
            plv[j, i] = plv_val
    return plv

# ---------------------------
# Step 1: Load data and convert to PLV graphs
# ---------------------------
for filename in tqdm(sorted(os.listdir(server_dir)), desc="Loading trials"):
    if not filename.endswith('.pkl') or not filename.startswith('S'):
        continue
    
    subject_id = int(filename.split('_')[0][1:])
    with open(os.path.join(server_dir, filename), 'rb') as f:
        bci = pickle.load(f)

    labels = [str(lbl) for lbl in bci['chaninfo']['label']]
    eeg_data = bci['data']
    meta = bci['TrialData']

    for trial, trial_meta in zip(eeg_data, meta):
        label = trial_meta.get('targetnumber')
        if label not in [1, 2]:  # Only Left/Right
            continue

        eeg = np.array(trial)
        plv = compute_plv_matrix(eeg)
        edge_index, edge_attr = dense_to_sparse(torch.tensor(plv, dtype=torch.float32))
        node_count = eeg.shape[0]
        x = torch.eye(node_count, dtype=torch.float32)  # Identity features

        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(label - 1),  # 0 = Left, 1 = Right
            subject=subject_id
        )
        all_data.append(graph)
        subject_numbers.append(subject_id)

subject_numbers = sorted(set(subject_numbers))
print(f"[INFO] Loaded {len(all_data)} trials from {len(subject_numbers)} subjects.")

# ---------------------------
# Step 2: LOSubjectOCV training
# ---------------------------
def split_data_by_subject(data_list, test_subj):
    train, test = [], []
    for graph in data_list:
        if graph.subject == test_subj:
            test.append(graph)
        else:
            train.append(graph)
    return train, test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loso_results = {}

for test_subject in subject_numbers:
    print(f"\n=== LOSO Fold: Test Subject {test_subject} ===")
    train_data, test_data = split_data_by_subject(all_data, test_subject)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model = SimpleGAT(
        in_channels=train_data[0].x.shape[1],
        hidden_channels=32,
        out_channels=2,
        num_heads=8
    ).to(device)

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

        # --- Evaluation ---
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

        print(f"Subject {test_subject}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Test Acc: {test_acc*100:.2f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1

    print(f"Subject {test_subject} Best Accuracy: {best_test_acc*100:.2f}% at Epoch {best_epoch}")
    loso_results[test_subject] = (best_test_acc, best_epoch)

# ---------------------------
# Step 3: Print final results
# ---------------------------
print("\nLOSO Summary (GAT Classification):")
for subj, (acc, epoch) in sorted(loso_results.items()):
    print(f"Subject {subj}: Best Accuracy = {acc*100:.2f}% at Epoch {epoch}")

avg_acc = np.mean([acc for acc, _ in loso_results.values()])
print(f"\nAverage LOSO Accuracy: {avg_acc*100:.2f}%")
