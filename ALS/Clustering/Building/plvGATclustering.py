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
from torch_geometric.nn import GATv2Conv, global_mean_pool
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

# --- Create a combined dataset and add a subject attribute ---
all_data = []
for subject, data_list in all_plvs.items():
    for data in data_list:
        # Store subject as an integer (e.g., from 'S1' -> 1)
        data.subject = int(subject.strip('S'))
        all_data.append(data)
#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D

# ------------------------------
# Feature Extraction from all_data
# ------------------------------
# For each graph, we extract a simple graph-level feature by taking the mean of its node features.
# This is a simple example; in practice, you might compute more complex graph descriptors.
graph_features = []
graph_labels = []
graph_subjects = []

for data in all_data:
    # data.x is assumed to be a tensor or a numpy array of shape [num_nodes, num_features]
    # We'll take the mean across nodes as a simple graph-level feature.
    if hasattr(data.x, 'cpu'):
        feat = data.x.cpu().numpy()  # if it's a torch tensor, move to CPU and convert to numpy
    else:
        feat = data.x  # if it's already a numpy array
    feat_mean = np.mean(feat, axis=0)
    graph_features.append(feat_mean)
    
    # data.y is assumed to be the class label (0 for Left, 1 for Right)
    # data.subject is an integer representing the subject ID.
    if hasattr(data.y, 'item'):
        graph_labels.append(data.y.item())
    else:
        graph_labels.append(data.y)
    graph_subjects.append(data.subject)

graph_features = np.vstack(graph_features)
graph_labels = np.array(graph_labels)
graph_subjects = np.array(graph_subjects)

# ------------------------------
# Dimensionality Reduction
# ------------------------------
# Compute PCA (2 components)
pca = PCA(n_components=2)
features_pca = pca.fit_transform(graph_features)

# Compute t-SNE (2 components)
tsne = TSNE(n_components=2, random_state=12345)
features_tsne = tsne.fit_transform(graph_features)

# ------------------------------
# Utility: Plotting functions
# ------------------------------
def plot_class_embedding(embedding, title):
    # Define fixed colors for classes: 0 (Left) = blue, 1 (Right) = red
    class_color = {0: 'blue', 1: 'red'}
    # Define a list of marker styles and assign each subject a unique marker
    marker_list = ['o', 's', '^', 'v', 'p', '*', 'D', 'X', 'h']
    unique_subjects = np.unique(graph_subjects)
    subject_marker = {subj: marker_list[i % len(marker_list)] for i, subj in enumerate(unique_subjects)}

    plt.figure(figsize=(8, 6))
    for i in range(embedding.shape[0]):
        subj = graph_subjects[i]
        label = graph_labels[i]
        plt.scatter(embedding[i, 0], embedding[i, 1],
                    color=class_color[label],
                    marker=subject_marker[subj],
                    edgecolor='k',
                    s=50,
                    alpha=0.8)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    # Create custom legend for classes
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Left', markerfacecolor='blue', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Right', markerfacecolor='red', markersize=8, markeredgecolor='k')
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.show()

def plot_subject_embedding(embedding, title):
    # Define a colormap to assign each subject a unique color (using tab10)
    unique_subjects = np.unique(graph_subjects)
    subject_colors = {subj: cm.get_cmap('tab10')(i % 10) for i, subj in enumerate(unique_subjects)}
    
    plt.figure(figsize=(8, 6))
    for i in range(embedding.shape[0]):
        subj = graph_subjects[i]
        plt.scatter(embedding[i, 0], embedding[i, 1],
                    color=subject_colors[subj],
                    marker='o',  # same marker for all points here
                    edgecolor='k',
                    s=50,
                    alpha=0.8)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    # Create legend for subjects
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f"Subject {subj}",
                              markerfacecolor=subject_colors[subj], markersize=8, markeredgecolor='k')
                       for subj in unique_subjects]
    plt.legend(handles=legend_elements, loc='best', title="Subjects", fontsize='small')
    plt.show()

# ------------------------------
# Plotting PCA Embeddings
# ------------------------------
plot_class_embedding(features_pca, "PCA: Left vs Right Classification with Subject Markers\n(Blue=Left, Red=Right)")
plot_subject_embedding(features_pca, "PCA: Each Subject in Its Own Color")

# ------------------------------
# Plotting t-SNE Embeddings
# ------------------------------
plot_class_embedding(features_tsne, "t-SNE: Left vs Right Classification with Subject Markers\n(Blue=Left, Red=Right)")
plot_subject_embedding(features_tsne, "t-SNE: Each Subject in Its Own Color")

#%% After training a GAT model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
from torch_geometric.loader import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D

# ---------------------------
# Define the Simple GAT Model (Feature Extractor and Classifier)
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
        
        self.lin = nn.Linear(hidden_channels * num_heads, out_channels)
    
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
# Hyperparameters and Setup
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = 19        # Adjust based on your node feature dimensionality
hidden_channels = 32
num_classes = 2         # Left vs Right
num_heads = 8

# Create a DataLoader from your all_data dataset (assumed to be defined already)
loader = DataLoader(all_data, batch_size=32, shuffle=True)

# Instantiate model, optimizer, and loss criterion.
model = SimpleGAT(in_channels, hidden_channels, num_classes, num_heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ---------------------------
# Training Loop for 100 Epochs
# ---------------------------
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits, _ = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader):.4f}")

# ---------------------------
# Feature Extraction After Training
# ---------------------------
model.eval()
all_feats = []
all_labels = []
all_subjects = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        _, feats = model(batch)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
        all_subjects.append(batch.subject.cpu().numpy())
all_feats = np.concatenate(all_feats, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
all_subjects = np.concatenate(all_subjects, axis=0)

# ---------------------------
# Dimensionality Reduction
# ---------------------------
# PCA (2 components)
pca = PCA(n_components=2)
features_pca = pca.fit_transform(all_feats)

# t-SNE (2 components)
tsne = TSNE(n_components=2, random_state=12345)
features_tsne = tsne.fit_transform(all_feats)

# ---------------------------
# Plotting Functions
# ---------------------------
def plot_class_embedding(embedding, title):
    # Colors for classes: 0 (Left) = blue, 1 (Right) = red.
    class_color = {0: 'blue', 1: 'red'}
    # Unique marker for each subject.
    marker_list = ['o', 's', '^', 'v', 'p', '*', 'D', 'X', 'h']
    unique_subjects = np.unique(all_subjects)
    subject_marker = {subj: marker_list[i % len(marker_list)] for i, subj in enumerate(unique_subjects)}

    plt.figure(figsize=(8, 6))
    for i in range(embedding.shape[0]):
        subj = all_subjects[i]
        label = all_labels[i]
        plt.scatter(embedding[i, 0], embedding[i, 1],
                    color=class_color[label],
                    marker=subject_marker[subj],
                    edgecolor='k',
                    s=50,
                    alpha=0.8)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    # Legend for classes.
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Left', markerfacecolor='blue', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Right', markerfacecolor='red', markersize=8, markeredgecolor='k')
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.show()

def plot_subject_embedding(embedding, title):
    # Assign a unique color to each subject using a colormap.
    unique_subjects = np.unique(all_subjects)
    subject_colors = {subj: cm.get_cmap('tab10')(i % 10) for i, subj in enumerate(unique_subjects)}
    
    plt.figure(figsize=(8, 6))
    for i in range(embedding.shape[0]):
        subj = all_subjects[i]
        plt.scatter(embedding[i, 0], embedding[i, 1],
                    color=subject_colors[subj],
                    marker='o',  # same marker for all points
                    edgecolor='k',
                    s=50,
                    alpha=0.8)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    # Legend for subjects.
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f"Subject {subj}",
                              markerfacecolor=subject_colors[subj], markersize=8, markeredgecolor='k')
                       for subj in unique_subjects]
    plt.legend(handles=legend_elements, loc='best', title="Subjects", fontsize='small')
    plt.show()

# ---------------------------
# Plotting the Embeddings
# ---------------------------
# PCA plots.
plot_class_embedding(features_pca, "PCA: Left vs Right Classification with Subject Markers\n(Blue=Left, Red=Right)")
plot_subject_embedding(features_pca, "PCA: Each Subject in Its Own Color")

# t-SNE plots.
plot_class_embedding(features_tsne, "t-SNE: Left vs Right Classification with Subject Markers\n(Blue=Left, Red=Right)")
plot_subject_embedding(features_tsne, "t-SNE: Each Subject in Its Own Color")

#%% After training a GAT model that reduces features over time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
from torch_geometric.loader import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D

# ---------------------------
# Define the Simple GAT Model (Feature Extractor and Classifier)
# ---------------------------
class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(SimpleGAT, self).__init__()
        # First GAT layer: 19 -> 32 per head, concatenated output -> 32*num_heads
        self.conv1 = GATv2Conv(in_channels, 32, heads=num_heads, concat=True)
        self.gn1 = GraphNorm(32 * num_heads)
        
        # Second GAT layer: reduce from 32*num_heads to 16 per head, concatenated -> 16*num_heads
        self.conv2 = GATv2Conv(32 * num_heads, 16, heads=num_heads, concat=True)
        self.gn2 = GraphNorm(16 * num_heads)
        
        # Third GAT layer: reduce from 16*num_heads to 8, without concatenation (averaging heads)
        self.conv3 = GATv2Conv(16 * num_heads, 8, heads=num_heads, concat=False)
        self.gn3 = GraphNorm(8)
        
        # Final linear layer: maps from 8 to the desired number of output classes
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
# Hyperparameters and Setup
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = 19        # Adjust based on your node feature dimensionality
hidden_channels = 32
num_classes = 2         # Left vs Right
num_heads = 8

# Create a DataLoader from your all_data dataset (assumed to be defined already)
loader = DataLoader(all_data, batch_size=32, shuffle=True)

# Instantiate model, optimizer, and loss criterion.
model = SimpleGAT(in_channels, hidden_channels, num_classes, num_heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ---------------------------
# Training Loop for 100 Epochs
# ---------------------------
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits, _ = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader):.4f}")

# ---------------------------
# Feature Extraction After Training
# ---------------------------
model.eval()
all_feats = []
all_labels = []
all_subjects = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        _, feats = model(batch)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
        all_subjects.append(batch.subject.cpu().numpy())
all_feats = np.concatenate(all_feats, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
all_subjects = np.concatenate(all_subjects, axis=0)

# ---------------------------
# Dimensionality Reduction
# ---------------------------
# PCA (2 components)
pca = PCA(n_components=2)
features_pca = pca.fit_transform(all_feats)

# t-SNE (2 components)
tsne = TSNE(n_components=2, random_state=12345)
features_tsne = tsne.fit_transform(all_feats)

# ---------------------------
# Plotting Functions
# ---------------------------
def plot_class_embedding(embedding, title):
    # Colors for classes: 0 (Left) = blue, 1 (Right) = red.
    class_color = {0: 'blue', 1: 'red'}
    # Unique marker for each subject.
    marker_list = ['o', 's', '^', 'v', 'p', '*', 'D', 'X', 'h']
    unique_subjects = np.unique(all_subjects)
    subject_marker = {subj: marker_list[i % len(marker_list)] for i, subj in enumerate(unique_subjects)}

    plt.figure(figsize=(8, 6))
    for i in range(embedding.shape[0]):
        subj = all_subjects[i]
        label = all_labels[i]
        plt.scatter(embedding[i, 0], embedding[i, 1],
                    color=class_color[label],
                    marker=subject_marker[subj],
                    edgecolor='k',
                    s=50,
                    alpha=0.8)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    # Legend for classes.
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Left', markerfacecolor='blue', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Right', markerfacecolor='red', markersize=8, markeredgecolor='k')
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.show()

def plot_subject_embedding(embedding, title):
    # Assign a unique color to each subject using a colormap.
    unique_subjects = np.unique(all_subjects)
    subject_colors = {subj: cm.get_cmap('tab10')(i % 10) for i, subj in enumerate(unique_subjects)}
    
    plt.figure(figsize=(8, 6))
    for i in range(embedding.shape[0]):
        subj = all_subjects[i]
        plt.scatter(embedding[i, 0], embedding[i, 1],
                    color=subject_colors[subj],
                    marker='o',  # same marker for all points
                    edgecolor='k',
                    s=50,
                    alpha=0.8)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    # Legend for subjects.
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f"Subject {subj}",
                              markerfacecolor=subject_colors[subj], markersize=8, markeredgecolor='k')
                       for subj in unique_subjects]
    plt.legend(handles=legend_elements, loc='best', title="Subjects", fontsize='small')
    plt.show()

# ---------------------------
# Plotting the Embeddings
# ---------------------------
# PCA plots.
plot_class_embedding(features_pca, "PCA: Left vs Right Classification with Subject Markers\n(Blue=Left, Red=Right)")
plot_subject_embedding(features_pca, "PCA: Each Subject in Its Own Color")

# t-SNE plots.
plot_class_embedding(features_tsne, "t-SNE: Left vs Right Classification with Subject Markers\n(Blue=Left, Red=Right)")
plot_subject_embedding(features_tsne, "t-SNE: Each Subject in Its Own Color")


