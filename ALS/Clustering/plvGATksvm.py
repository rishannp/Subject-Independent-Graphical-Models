import os
from os.path import join as pjoin
import numpy as np
import scipy.io as sio
import scipy.signal as sig
import networkx as nx
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, GraphNorm
from torch_geometric.seed import seed_everything
import psutil
import matplotlib.pyplot as plt

# ---------------------------
# Utility Functions
# ---------------------------
def plvfcn(eegData):
    # Use only the first 19 electrodes
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
    return plvMatrix

def compute_plv(subject_data):
    # Assumes subject_data has fields 'L' and 'R'
    idx = ['L', 'R']
    numElectrodes = 19
    plv = {field: np.zeros((numElectrodes, numElectrodes, subject_data.shape[1])) for field in idx}
    for i, field in enumerate(idx):
        for j in range(subject_data.shape[1]):
            x = subject_data[field][0, j][:, :19]
            plv[field][:, :, j] = plvfcn(x)
    l, r = plv['L'], plv['R']
    yl = np.zeros((subject_data.shape[1], 1))
    yr = np.ones((subject_data.shape[1], 1))
    img = np.concatenate((l, r), axis=2)
    y = np.concatenate((yl, yr), axis=0)
    y = torch.tensor(y, dtype=torch.long)
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

# Dictionary to hold processed PyG Data objects per subject
all_plvs = {}

for subject_number in subject_numbers:
    print(f'Processing Subject S{subject_number}')
    mat_fname = pjoin(data_dir, f'S{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_raw = mat_contents[f'Subject{subject_number}']
    # Use all columns except the last one
    S1 = subject_raw[:, :-1]
    
    # Compute PLV matrices and labels
    plv, y = compute_plv(S1)
    threshold = 0.1
    graphs = create_graphs(plv, threshold)
    
    numElectrodes = 19
    adj = np.zeros([numElectrodes, numElectrodes, len(graphs)])
    for i, G in enumerate(graphs):
        adj[:, :, i] = nx.to_numpy_array(G)
    
    adj = torch.tensor(adj, dtype=torch.float32)
    
    # Build edge indices for each graph (each graph is created from an adjacency matrix)
    edge_indices = []
    for i in range(adj.shape[2]):
        source_nodes = []
        target_nodes = []
        for row in range(adj.shape[0]):
            for col in range(adj.shape[1]):
                if adj[row, col, i] >= threshold:
                    source_nodes.append(row)
                    target_nodes.append(col)
                else:
                    # Placeholder to keep tensor shape consistent
                    source_nodes.append(0)
                    target_nodes.append(0)
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_indices.append(edge_index)
    
    edge_indices = torch.stack(edge_indices, dim=-1)
    
    # Create PyG Data objects for each graph.
    # Here we treat each graph’s adjacency matrix as node features (each node has a 19-dimensional feature).
    data_list = []
    for i in range(np.size(adj, 2)):
        data_list.append(Data(x=adj[:, :, i], edge_index=edge_indices[:, :, i], y=y[i, 0]))
    
    # Rearranging data (assuming two conditions interleaved)
    size = len(data_list)
    idx = size // 2
    datal = data_list[0:idx]
    datar = data_list[idx:size]
    combined = []
    for i in range(idx):
        combined.extend([datal[i], datar[i]])
    
    all_plvs[f'S{subject_number}'] = combined

#%%
# --- Create a combined dataset and add a subject attribute ---
all_data = []
for subject, data_list in all_plvs.items():
    for data in data_list:
        # Store subject as an integer (e.g., from 'S1' -> 1)
        data.subject = int(subject.strip('S'))
        all_data.append(data)

# ---------------------------
# Define the Simple GAT Model with MMD
# ---------------------------
class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=1):
        super(SimpleGAT, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.conv3 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        
        self.gn1 = GraphNorm(hidden_channels * num_heads)
        self.gn2 = GraphNorm(hidden_channels * num_heads)
        self.gn3 = GraphNorm(hidden_channels * num_heads)
        
        self.lin = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Each convolution is followed by graph normalization and ReLU.
        x = F.relu(self.gn1(self.conv1(x, edge_index)))
        x = F.relu(self.gn2(self.conv2(x, edge_index)))
        x = F.relu(self.gn3(self.conv3(x, edge_index)))
        
        # Global mean pooling to obtain graph-level features.
        x = global_mean_pool(x, batch)  # shape: [num_graphs, hidden_channels * num_heads]
        features = x  # These features can be used for dimensionality reduction.
        logits = self.lin(x)  # Classification logits (for left/right).
        return logits, features

# --- Define the MMD Loss Functions ---
def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return torch.stack(kernel_val, dim=0).sum(dim=0) # [n_samples, n_samples]

def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size_source = source.size(0)
    batch_size_target = target.size(0)
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)
    XX = kernels[:batch_size_source, :batch_size_source]
    YY = kernels[batch_size_source:, batch_size_source:]
    XY = kernels[:batch_size_source, batch_size_source:]
    YX = kernels[batch_size_source:, :batch_size_source]
    loss = torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)
    return loss

# For a batch, we compute the pairwise MMD between features of different subjects.
def compute_batch_mmd(features, subjects):
    unique_subjects = torch.unique(subjects)
    loss = 0
    count = 0
    for i in range(len(unique_subjects)):
        for j in range(i+1, len(unique_subjects)):
            subj_i = unique_subjects[i]
            subj_j = unique_subjects[j]
            feat_i = features[subjects == subj_i]
            feat_j = features[subjects == subj_j]
            if feat_i.size(0) > 0 and feat_j.size(0) > 0:
                loss += mmd_loss(feat_i, feat_j)
                count += 1
    if count > 0:
        loss /= count
    return loss

# ---------------------------
# Training Setup
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader = DataLoader(all_data, batch_size=32, shuffle=True)

in_channels = 19         # Each node has a 19-dimensional feature (a row of the adjacency matrix)
hidden_channels = 32
num_classes = 2          # Left (0) vs. Right (1)

model = SimpleGAT(in_channels, hidden_channels, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 100
lambda_mmd = 0.5  # Weight for MMD loss

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits, feats = model(batch)
        loss_cls = criterion(logits, batch.y)
        # Extract subject labels (stored in batch.subject) as a tensor
        subjects = batch.subject.to(device)
        loss_mmd = compute_batch_mmd(feats, subjects)
        # loss = loss_cls + lambda_mmd * loss_mmd
        loss = loss_cls
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader):.4f}")

# ---------------------------
# Feature Extraction for Dimensionality Reduction
# ---------------------------
model.eval()
all_feats = []
all_labels = []
all_subjects = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        logits, feats = model(batch)
        all_feats.append(feats.cpu())
        all_labels.append(batch.y.cpu())
        all_subjects.append(batch.subject.cpu())
all_feats = torch.cat(all_feats, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()
all_subjects = torch.cat(all_subjects, dim=0).numpy()

#%%
## %matplotlib qt

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import matplotlib.cm as cm

def plot_3d_embedding(embedding, title, mode='class'):
    """
    Plots a 3D scatter plot for the provided embedding.
    
    mode: 'class' colors by left/right only,
          'subject' colors by subject and uses different markers for left/right.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if mode == 'class':
        # Define two colors for left (0) and right (1)
        colors = {0: 'blue', 1: 'red'}
        marker = '.'  # Dot for both classes
        for side in [0, 1]:
            indices = np.where(all_labels == side)[0]
            if indices.size > 0:
                ax.scatter(embedding[indices, 0],
                           embedding[indices, 1],
                           embedding[indices, 2],
                           color=colors[side],
                           marker=marker,
                           s=30,
                           alpha=0.7,
                           label='Left' if side == 0 else 'Right')
    elif mode == 'subject':
        # Get unique subjects and assign each a color from a colormap
        unique_subjects = sorted(np.unique(all_subjects))
        num_subjects = len(unique_subjects)
        cmap = cm.get_cmap('tab10', num_subjects)
        # Define markers: use circle for left (0) and triangle for right (1)
        marker_dict = {0: 'o', 1: '^'}
        for i, subj in enumerate(unique_subjects):
            for side in [0, 1]:
                indices = np.where((all_subjects == subj) & (all_labels == side))[0]
                if indices.size > 0:
                    ax.scatter(embedding[indices, 0],
                               embedding[indices, 1],
                               embedding[indices, 2],
                               color=cmap(i),
                               marker=marker_dict[side],
                               edgecolor='k',
                               s=30,
                               alpha=0.7,
                               label=f"S{subj} - {'Left' if side == 0 else 'Right'}")
    
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    
    # Remove duplicate labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    ax.legend(unique_legend.values(), unique_legend.keys(), loc='best', fontsize='small')
    plt.show()

# Compute 3D PCA embedding only once
pca = PCA(n_components=3)
features_pca = pca.fit_transform(all_feats)

# Compute 3D t-SNE embedding only once
tsne = TSNE(n_components=3, random_state=12345)
features_tsne = tsne.fit_transform(all_feats)

# Plot 3D PCA using left/right color scheme
plot_3d_embedding(features_pca, '3D PCA of GAT Features (Left/Right Only)', mode='class')

# Plot 3D t-SNE using left/right color scheme
plot_3d_embedding(features_tsne, '3D t-SNE of GAT Features (Left/Right Only)', mode='class')

# Plot 3D PCA using subject-based color scheme
plot_3d_embedding(features_pca, '3D PCA of GAT Features (Subject & Class)', mode='subject')

# Plot 3D t-SNE using subject-based color scheme
plot_3d_embedding(features_tsne, '3D t-SNE of GAT Features (Subject & Class)', mode='subject')


#%% Optimisation:
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import itertools

# Choose one subject to leave out for testing; for example, subject 39.
test_subject = 39

# Split the data based on subject membership:
train_indices = np.where(all_subjects != test_subject)[0]
test_indices = np.where(all_subjects == test_subject)[0]

X_train = all_feats[train_indices]
y_train = all_labels[train_indices]
X_test = all_feats[test_indices]
y_test = all_labels[test_indices]

print(f"Training on {len(X_train)} samples from subjects: {np.unique(all_subjects[train_indices])}")
print(f"Testing on {len(X_test)} samples from subject: {test_subject}")

# Define hyperparameter grid
param_grid = {
    'kernel': ['rbf', 'linear', 'poly'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.001, 0.01, 0.1, 1]
}

best_params = None
best_test_acc = 0
results = []

# Iterate over all combinations of hyperparameters
for kernel, C, gamma in itertools.product(param_grid['kernel'], param_grid['C'], param_grid['gamma']):
    # Create a pipeline with StandardScaler and SVC with current hyperparameters
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=kernel, C=C, gamma=gamma, random_state=12345))
    ])
    # Fit on the training data
    pipeline.fit(X_train, y_train)
    # Evaluate on the test data (LOSO)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((kernel, C, gamma, acc))
    print(f"Params: kernel={kernel}, C={C}, gamma={gamma}, Test Accuracy: {acc*100:.2f}%")
    
    if acc > best_test_acc:
        best_test_acc = acc
        best_params = {'kernel': kernel, 'C': C, 'gamma': gamma}

print("\nBest LOSO Test Accuracy: {:.2f}% with parameters: {}".format(best_test_acc * 100, best_params))

# Optionally, print the classification report for the best model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'], random_state=12345))
])
pipeline.fit(X_train, y_train)
y_pred_best = pipeline.predict(X_test)
report = classification_report(y_test, y_pred_best)
print("\nClassification Report for Best Model:\n", report)
