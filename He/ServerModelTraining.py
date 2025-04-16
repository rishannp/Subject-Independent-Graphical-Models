### WITH GPU

import os
import pickle
import numpy as np
import scipy.signal as sig
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
from torch_geometric.utils import dense_to_sparse
from torch_geometric.seed import seed_everything

# ---------------------------
# CONFIG
# ---------------------------
server_dir = '/home/uceerjp/He/'
num_epochs = 100
cache_path = os.path.join(server_dir, 'plv_graph_dataset.pkl')
results_path = os.path.join(server_dir, 'loso_results.pkl')
max_workers = 8  # Adjust based on available cores
seed_everything(12345)
# ---------------------------
# DEVICE SETUP
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✅ Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    # Uncomment to monitor memory usage
    # print(torch.cuda.memory_allocated(device) / 1e6, "MB allocated")
    # print(torch.cuda.memory_reserved(device) / 1e6, "MB reserved")

# ---------------------------
# GATv2 Model Definition
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

        self.dropout = nn.Dropout(0.3)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * num_heads, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_channels)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.gn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.gn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.gn3(self.conv3(x, edge_index)))
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        logits = self.mlp(x)
        return logits, x

# ---------------------------
# Step 1: Load or Generate Graphs
# ---------------------------
if os.path.exists(cache_path):
    print(f"\n[CACHE] Found existing PLV graph cache. Loading from: {cache_path}")
    with open(cache_path, 'rb') as f:
        all_data, subject_numbers = pickle.load(f)
else:
    raise FileNotFoundError(f"No data found at {cache_path}")

print(f"\n[INFO] Ready to train with {len(all_data)} trials from {len(subject_numbers)} subjects.")

# ---------------------------
# LOSOCV Training
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
# Load or Initialize Results
# ---------------------------
if os.path.exists(results_path):
    with open(results_path, 'rb') as f:
        loso_results = pickle.load(f)
    print(f"\n[CHECKPOINT] Loaded existing LOSO results with {len(loso_results)} subjects completed.")
else:
    loso_results = {}

print("\n[INFO] Starting LOSubjectOCV training...\n")

for test_subject in tqdm(subject_numbers, desc="LOSO folds"):
    print(f"\n=== LOSO Fold: Test Subject {test_subject} ===")
    train_data, test_data = split_data_by_subject(all_data, test_subject)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model = SimpleGAT(
        in_channels=train_data[0].x.shape[1],
        hidden_channels=128,
        out_channels=2,
        num_heads=8
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0
    best_epoch = 0

    for epoch in tqdm(range(num_epochs), desc=f"  Epochs (Test S{test_subject})", leave=False):
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

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Subject {test_subject}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Test Acc: {test_acc*100:.2f}%, LR: {current_lr:.6f}", flush=True)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1

    print(f"[RESULT] Subject {test_subject} - Best Accuracy: {best_test_acc*100:.2f}% at Epoch {best_epoch}")
    loso_results[test_subject] = (best_test_acc, best_epoch)

# ---------------------------
# Step 2: Save Results
# ---------------------------
with open(results_path, 'wb') as f:
    pickle.dump(loso_results, f)
print(f"\n✅ LOSO results saved to: {results_path}")

# ---------------------------
# Step 3: Print Summary
# ---------------------------
print("\n📊 LOSO Summary:")
for subj, (acc, epoch) in sorted(loso_results.items()):
    print(f"Subject {subj}: Best Accuracy = {acc*100:.2f}% at Epoch {epoch}")

avg_acc = np.mean([acc for acc, _ in loso_results.values()])
print(f"\n✅ Average LOSO Accuracy: {avg_acc*100:.2f}%")
