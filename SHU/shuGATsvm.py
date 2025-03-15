import os
from os.path import join as pjoin
import numpy as np
from scipy.io import loadmat
import scipy.signal as sig
import networkx as nx
import torch
from torch_geometric.data import Data
from scipy.signal import butter, filtfilt
from torch_geometric.seed import seed_everything

# ---------------------------
# Set Seed for Reproducibility
# ---------------------------
seed_everything(12345)



# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
directory = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\SHU Dataset\MatFiles'
data_by_subject = {}

for filename in os.listdir(directory):
    if filename.endswith('.mat'):
        file_path = os.path.join(directory, filename)
        mat_data = loadmat(file_path)
        parts = filename.split('_')
        subject_id = parts[0]  # e.g., 'sub-001'
        # session_id = parts[1]  # Not used anymore
        if subject_id not in data_by_subject:
            data_by_subject[subject_id] = {'data': [], 'labels': []}
        data = mat_data['data']      # Adjust key as needed
        labels = mat_data['labels']  # Adjust key as needed
        data_by_subject[subject_id]['data'].append(data)
        data_by_subject[subject_id]['labels'].append(labels)

# Concatenate data for each subject
for subject_id in data_by_subject:
    concatenated_data = np.concatenate(data_by_subject[subject_id]['data'], axis=0)
    concatenated_labels = np.concatenate(data_by_subject[subject_id]['labels'], axis=1)
    data_by_subject[subject_id]['data'] = concatenated_data
    data_by_subject[subject_id]['labels'] = concatenated_labels

# ---------------------------
# Split Data by Label
# ---------------------------
def split_data_by_label(data_by_subject):
    data_split = {}
    for subject, data_dict in data_by_subject.items():
        data = data_dict['data']  # Trials x Channels x Samples
        labels = data_dict['labels']
        if labels.ndim == 2:
            labels = labels.flatten()
        data_L, data_R = [], []
        for i in range(data.shape[0]):
            if labels[i] == 1:
                data_L.append(data[i])
            elif labels[i] == 2:
                data_R.append(data[i])
        data_split[subject] = {'L': np.array(data_L), 'R': np.array(data_R)}
    return data_split

data_split = split_data_by_label(data_by_subject)

# ---------------------------
# Bandpass Filtering
# ---------------------------
fs = 250
def bandpass_filter_trials(data_split, low_freq, high_freq, sfreq):
    filtered_data_split = {}
    nyquist = 0.5 * sfreq
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='band')
    for subject in data_split:
        subject_data = data_split[subject]
        filtered_subject_data = {}
        for direction in ['L', 'R']:
            trials = subject_data[direction]  # Trials x Channels x Samples
            filtered_trials = []
            for trial in range(trials.shape[0]):
                trial_data = trials[trial]  # Channels x Samples
                filtered_trial_data = np.zeros_like(trial_data)
                for ch in range(trial_data.shape[0]):
                    filtered_trial_data[ch, :] = filtfilt(b, a, trial_data[ch, :])
                filtered_trials.append(filtered_trial_data)
            filtered_subject_data[direction] = np.array(filtered_trials)
        filtered_data_split[subject] = filtered_subject_data
    return filtered_data_split

filtered_data_split = bandpass_filter_trials(data_split, low_freq=8, high_freq=30, sfreq=fs)

# ---------------------------
# Merge Left and Right Trials per Subject
# ---------------------------
# For each subject, combine L and R trials into one dataset and create corresponding labels.
merged_data = {}
for subject in filtered_data_split:
    left_trials = filtered_data_split[subject]['L']
    right_trials = filtered_data_split[subject]['R']
    combined_trials = np.concatenate((left_trials, right_trials), axis=0)
    left_labels = np.zeros(left_trials.shape[0], dtype=int)  # Label 0 for left
    right_labels = np.ones(right_trials.shape[0], dtype=int)  # Label 1 for right
    combined_labels = np.concatenate((left_labels, right_labels), axis=0)
    merged_data[subject] = {'data': combined_trials, 'label': combined_labels}

# ---------------------------
# PLV Computation and Graph Construction
# ---------------------------
def plvfcn(eegData):
    numElectrodes = eegData.shape[0]
    numTimeSteps = eegData.shape[1]
    plvMatrix = np.zeros((numElectrodes, numElectrodes))
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            phase1 = np.angle(sig.hilbert(eegData[electrode1, :]))
            phase2 = np.angle(sig.hilbert(eegData[electrode2, :]))
            phase_diff = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_diff)) / numTimeSteps)
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    return plvMatrix

def create_graphs_and_edges(plv_matrices, threshold):
    graphs = []
    numElectrodes, _, numTrials = plv_matrices.shape
    adj_matrices = np.zeros((numElectrodes, numElectrodes, numTrials))
    edge_indices = []
    
    for i in range(numTrials):
        G = nx.Graph()
        G.add_nodes_from(range(numElectrodes))
        source_nodes = []
        target_nodes = []
        for u in range(numElectrodes):
            for v in range(u+1, numElectrodes):
                if plv_matrices[u, v, i] > threshold:
                    G.add_edge(u, v, weight=plv_matrices[u, v, i])
                    adj_matrices[u, v, i] = plv_matrices[u, v, i]
                    adj_matrices[v, u, i] = plv_matrices[u, v, i]
                    source_nodes.append(u)
                    target_nodes.append(v)
        graphs.append(G)
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_indices.append(edge_index)
        
    adj_matrices = torch.tensor(adj_matrices, dtype=torch.float32)
    edge_indices = torch.stack(edge_indices, dim=-1)
    return adj_matrices, edge_indices, graphs

def compute_plv(subject_data):
    data = subject_data['data']  # Trials x Channels x TimeSteps
    labels = subject_data['label']  # Trials,
    numTrials, numElectrodes, _ = data.shape
    plv_matrices = np.zeros((numElectrodes, numElectrodes, numTrials))
    
    # Compute PLV matrix for each trial
    for trial_idx in range(numTrials):
        eeg_trial = data[trial_idx]
        plv_matrices[:, :, trial_idx] = plvfcn(eeg_trial)
        
    label_tensor = torch.tensor(labels, dtype=torch.long)
    # Set your desired threshold (e.g., 0.1)
    adj_matrices, edge_indices, graphs = create_graphs_and_edges(plv_matrices, threshold=0)
    
    # Return a dictionary with the computed data for this subject
    return {
        'plv_matrices': plv_matrices,
        'labels': label_tensor,
        'adj_matrices': adj_matrices,
        'edge_indices': edge_indices,
        'graphs': graphs
    }

# Process each subject
subject_plv_data = {}
for subject_id, subject_data in merged_data.items():
    print(f"Processing subject: {subject_id}")
    subject_plv_data[subject_id] = compute_plv(subject_data)

# subject_plv_data now contains the PLV matrices, graph data, and labels for each subject.

#%% Apply GAT model

from torch_geometric.data import Data
import torch

all_data = []

for subject_key, subject_data in subject_plv_data.items():
    subject_int = int(subject_key.strip('sub-'))
    plv_matrices = subject_data['plv_matrices']  
    labels = subject_data['labels']
    edge_indices = subject_data['edge_indices']
    
    num_trials = plv_matrices.shape[2]
    
    for i in range(num_trials):
        # Convert the adjacency matrix (node features) to a torch tensor.
        x = torch.tensor(plv_matrices[:, :, i], dtype=torch.float)  # Shape: [32, 32]
        
        if isinstance(edge_indices, list):
            edge_index = edge_indices[i]
        else:
            edge_index = edge_indices[:, :, i]
        
        # Make sure the label is a torch tensor
        y = torch.tensor(labels[i], dtype=torch.long) if not torch.is_tensor(labels) else labels[i]
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data.subject = torch.tensor(subject_int, dtype=torch.long)
        all_data.append(data)

#%%

# --- Define the Simple GAT Model with MMD Loss ---
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
from torch import nn
from torch_geometric.data import Data, DataLoader

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

# --- Training Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader = DataLoader(all_data, batch_size=32, shuffle=True)

# Here in_channels should match the number of node features.
# In our case, since we used each row of the adjacency matrix and our adjacencies are square matrices of size [num_nodes, num_nodes],
# in_channels equals num_nodes. For example, if num_nodes is 19, then in_channels=19.
in_channels = all_data[0].x.shape[1]  # e.g., 19
hidden_channels = 32
num_classes = 2  # For left (0) vs. right (1)

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
        subjects = batch.subject.to(device)
        loss_mmd = compute_batch_mmd(feats, subjects)
        # Uncomment the line below if you wish to include the MMD loss:
        # loss = loss_cls + lambda_mmd * loss_mmd
        loss = loss_cls
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader):.4f}")

# --- Feature Extraction for Dimensionality Reduction ---
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
tsne = TSNE(n_components=3, random_state=12345, verbose=1)
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

# Hyperparameter grid
param_grid = {
    'kernel': ['rbf', 'linear'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.001, 0.01, 0.1, 1]
}

# Dictionary to store results for each test subject
all_results = {}

# Loop over test subjects from 1 to 25
for test_subject in range(1, 26):
    # Split the data based on subject membership
    train_indices = np.where(all_subjects != test_subject)[0]
    test_indices = np.where(all_subjects == test_subject)[0]
    
    X_train = all_feats[train_indices]
    y_train = all_labels[train_indices]
    X_test = all_feats[test_indices]
    y_test = all_labels[test_indices]
    
    print(f"\nTesting on subject {test_subject}")
    print(f"Training on {len(X_train)} samples from subjects: {np.unique(all_subjects[train_indices])}")
    print(f"Testing on {len(X_test)} samples from subject: {test_subject}")
    
    best_params = None
    best_test_acc = 0
    results = []
    
    # Iterate over all combinations of hyperparameters
    for kernel, C, gamma in itertools.product(param_grid['kernel'], param_grid['C'], param_grid['gamma']):
        # Create a pipeline with StandardScaler and SVC using current hyperparameters
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=kernel, C=C, gamma=gamma, random_state=12345))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append((kernel, C, gamma, acc))
        print(f"Params: kernel={kernel}, C={C}, gamma={gamma}, Test Accuracy: {acc*100:.2f}%")
        
        if acc > best_test_acc:
            best_test_acc = acc
            best_params = {'kernel': kernel, 'C': C, 'gamma': gamma}
    
    print(f"\nBest Test Accuracy for subject {test_subject}: {best_test_acc*100:.2f}% with parameters: {best_params}")
    
    # Fit best model and print classification report
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'], random_state=12345))
    ])
    pipeline.fit(X_train, y_train)
    y_pred_best = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred_best)
    print(f"\nClassification Report for subject {test_subject}:\n{report}")
    
    # Save results for this subject
    all_results[test_subject] = {'best_accuracy': best_test_acc,
                                 'best_params': best_params,
                                 'report': report,
                                 'results': results}

# Optionally, you can print a summary of results for all subjects
print("\nSummary of best accuracies for each test subject:")
for subj in sorted(all_results.keys()):
    print(f"Subject {subj}: {all_results[subj]['best_accuracy']*100:.2f}% with {all_results[subj]['best_params']}")
