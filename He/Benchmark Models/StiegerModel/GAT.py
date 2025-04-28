import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n? Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ---------------------------
# Define the Simple GAT Model (Feature Extractor and Classifier)
# ---------------------------
class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, dropout=0.3):
        super(SimpleGAT, self).__init__()
        self.dropout = dropout

        self.conv1 = GATv2Conv(in_channels, 32, heads=num_heads, concat=True, dropout=dropout)
        self.gn1 = GraphNorm(32 * num_heads)

        self.conv2 = GATv2Conv(32 * num_heads, 16, heads=num_heads, concat=True, dropout=dropout)
        self.gn2 = GraphNorm(16 * num_heads)

        self.conv3 = GATv2Conv(16 * num_heads, 8, heads=num_heads, concat=False, dropout=dropout)
        self.gn3 = GraphNorm(8)

        self.lin = nn.Linear(8, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.gn1(self.conv1(x, edge_index)))
        # x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.gn2(self.conv2(x, edge_index)))
        # x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.gn3(self.conv3(x, edge_index)))
        # x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        features = x
        logits = self.lin(x)
        return logits, features

# ---------------------------
# Load or Generate Graphs
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

if os.path.exists(results_path):
    with open(results_path, 'rb') as f:
        loso_results = pickle.load(f)
    print(f"\n[CHECKPOINT] Loaded existing LOSO results with {len(loso_results)} subjects completed.")
else:
    loso_results = {}

print("\n[INFO] Starting LOSubjectOCV training...\n")

for test_subject in tqdm(subject_numbers, desc="LOSO folds"):
    if test_subject in loso_results:
        print(f"[SKIP] Subject {test_subject} already completed. Skipping.")
        continue

    print(f"\n=== LOSO Fold: Test Subject {test_subject} ===")
    train_data, test_data = split_data_by_subject(all_data, test_subject)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False)

    model = SimpleGAT(
        in_channels=train_data[0].x.shape[1],
        hidden_channels=32,
        out_channels=2,
        num_heads=8
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Optional: if you want to save the best model weights
    best_test_acc = 0.0
    best_state_dict = None

    # Train for num_epochs without testing each time 
    for epoch in tqdm(range(num_epochs),
                      desc=f"  Epochs (Test S{test_subject})",
                      leave=False):
        model.train()
        total_loss = 0.0

        # training loop
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits, _ = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        # I'm printing just the loss each epoch now
        print(f"Subject {test_subject}, Epoch {epoch+1}/{num_epochs}, "
              f"Loss: {avg_loss:.4f}", flush=True)

    # After all epochs, do a single test pass
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            preds = logits.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total   += batch.num_graphs

    final_test_acc = correct / total if total > 0 else 0.0

    print(f"[RESULT] Subject {test_subject} : Final Accuracy: "
          f"{final_test_acc*100:.2f}%", flush=True)

    # Optionally save best model weights if you want:
    # if final_test_acc > best_test_acc:
    #     best_test_acc = final_test_acc
    #     best_state_dict = model.state_dict()

    loso_results[test_subject] = (final_test_acc, num_epochs)

# ---------------------------
# Save Results
# ---------------------------
with open(results_path, 'wb') as f:
    pickle.dump(loso_results, f)
print(f"\n? LOSO results saved to: {results_path}")

# ---------------------------
# Print Summary
# ---------------------------
print("\n LOSO Summary:")
for subj, (acc, epoch) in sorted(loso_results.items()):
    print(f"Subject {subj}: Final Accuracy = {acc*100:.2f}% after {epoch} epochs")

avg_acc = np.mean([acc for acc, _ in loso_results.values()])
print(f"\n? Average LOSO Accuracy: {avg_acc*100:.2f}%")
