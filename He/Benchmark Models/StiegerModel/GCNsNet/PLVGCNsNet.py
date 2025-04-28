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

# ---------------------------
# CONFIG
# ---------------------------
server_dir = '/home/uceerjp/He/'
num_epochs = 25
cache_path = os.path.join(server_dir, 'plv_graph_dataset.pkl')
results_path = os.path.join(server_dir, 'loso_results.pkl')
seed_everything(12345)

device = torch.device('cpu')
print(f"\n?? Forcing CPU execution. Using device: {device}")


# ---------------------------
# GCNs-Net Model (Chebyshev + Graclus + Pool)
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

        x = global_mean_pool(x, batch)
        return F.log_softmax(self.fc(x), dim=1)

# ---------------------------
# Load Graph Data
# ---------------------------
if os.path.exists(cache_path):
    print(f"\n[CACHE] Found graph data. Loading from: {cache_path}")
    with open(cache_path, 'rb') as f:
        all_data, subject_numbers = pickle.load(f)
else:
    raise FileNotFoundError(f"[ERROR] No data found at {cache_path}")

print(f"\n[INFO] Loaded {len(all_data)} trials from {len(subject_numbers)} subjects.")

# ---------------------------
# Helper to Split LOSO
# ---------------------------
def split_data_by_subject(data_list, test_subj):
    train, test = [], []
    for graph in data_list:
        if graph.subject == test_subj:
            test.append(graph)
        else:
            train.append(graph)
    return train, test

# ---------------------------
# Load Previous Results
# ---------------------------
if os.path.exists(results_path):
    with open(results_path, 'rb') as f:
        loso_results = pickle.load(f)
    print(f"\n[CHECKPOINT] Loaded {len(loso_results)} existing LOSO results.")
else:
    loso_results = {}

# ---------------------------
# LOSO Training Loop
# ---------------------------
print("\n[INFO] Starting LOSO Training...\n")

for test_subject in tqdm(subject_numbers, desc="LOSO folds"):
    if test_subject in loso_results:
        print(f"[SKIP] Subject {test_subject} already completed.")
        continue

    print(f"\n=== LOSO Fold: Test Subject {test_subject} ===")
    train_data, test_data = split_data_by_subject(all_data, test_subject)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

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
        print(f"Subject {test_subject}, Epoch {epoch+1}/{num_epochs}, "
              f"Loss: {avg_loss:.4f}", flush=True)

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

# ---------------------------
# Save All Results
# ---------------------------
with open(results_path, 'wb') as f:
    pickle.dump(loso_results, f)
print(f"\n? LOSO results saved to: {results_path}")

# ---------------------------
# Print Summary
# ---------------------------
print("\n?? LOSO Summary:")
for subj, (acc, ep) in sorted(loso_results.items()):
    print(f"Subject {subj}: Final Accuracy = {acc * 100:.2f}% after {ep} epochs")

avg_acc = np.mean([acc for acc, _ in loso_results.values()])
print(f"\n?? Average LOSO Accuracy: {avg_acc * 100:.2f}%")
