"""
LOSO AutoGL Graph Classification for EEG Motor Imagery Data

This script loads EEG data, creates graphs from PLV matrices,
wraps them into a custom PyG-compatible dataset, and performs
Leave-One-Subject-Out (LOSO) cross-validation using AutoGraphClassifier.
"""

import sys
import os
import argparse
import random
import copy
import numpy as np
import scipy.io as sio
import scipy.signal as sig
import networkx as nx
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.seed import seed_everything
from autogl.solver import AutoGraphClassifier
from sklearn.metrics import accuracy_score

# ---------------------------
# Functions for PLV Computation & Graph Creation
# ---------------------------
def plvfcn(eegData):
    """
    Compute phase-locking value (PLV) from EEG signals.
    """
    eegData = eegData[:, :19]  # use only first 19 electrodes
    numElectrodes = eegData.shape[1]
    numTimeSteps = eegData.shape[0]
    plvMatrix = np.zeros((numElectrodes, numElectrodes))
    for i in range(numElectrodes):
        for j in range(i + 1, numElectrodes):
            phase1 = np.angle(sig.hilbert(eegData[:, i]))
            phase2 = np.angle(sig.hilbert(eegData[:, j]))
            phase_diff = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_diff)) / numTimeSteps)
            plvMatrix[i, j] = plv
            plvMatrix[j, i] = plv
    np.fill_diagonal(plvMatrix, 0)
    return plvMatrix

def compute_plv(subject_data):
    """
    Compute PLV matrices for 'L' and 'R' conditions and return 
    concatenated PLV tensor with corresponding labels.
    """
    idx = ['L', 'R']
    numElectrodes = 19
    trials = subject_data.shape[1]
    plv = {k: np.zeros((numElectrodes, numElectrodes, trials)) for k in idx}
    for k in idx:
        for j in range(trials):
            plv[k][:, :, j] = plvfcn(subject_data[k][0, j][:, :19])
    # Concatenate left and right trials; labels 0 for L, 1 for R.
    X = np.concatenate((plv['L'], plv['R']), axis=2)
    y = np.concatenate((np.zeros((trials, 1)), np.ones((trials, 1))), axis=0)
    return X, torch.tensor(y, dtype=torch.long)

def create_graphs(plv_tensor, threshold):
    """
    Create a list of NetworkX graph objects from the PLV tensor.
    """
    graphs = []
    for i in range(plv_tensor.shape[2]):
        G = nx.Graph()
        G.add_nodes_from(range(plv_tensor.shape[0]))
        for u in range(plv_tensor.shape[0]):
            for v in range(plv_tensor.shape[0]):
                if u != v and plv_tensor[u, v, i] > threshold:
                    G.add_edge(u, v, weight=plv_tensor[u, v, i])
        graphs.append(G)
    return graphs

# ---------------------------
# Custom Dataset Class
# ---------------------------
class MyDataset(Dataset):
    """
    A simple Dataset wrapper for a list of torch_geometric.data.Data objects.
    This version avoids the caching issues of InMemoryDataset.
    """
    def __init__(self, data_list):
        super(MyDataset, self).__init__()
        self.data_list = data_list
        # These splits will be defined externally.
        self.train_split = None
        self.val_split = None
        self.test_split = None

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    @property
    def num_classes(self):
        return 2  # binary classification (L vs R)

    @property
    def num_node_features(self):
        return self.data_list[0].x.size(1)

def wrap_dataset_safe(data_list):
    """
    Clone every Data object to avoid caching issues.
    """
    return MyDataset([d.clone() for d in data_list])

def split_data_by_subject(data_list, test_subject):
    """
    Split data into training and test lists based on subject id.
    """
    train = [d for d in data_list if d.subject != test_subject]
    test  = [d for d in data_list if d.subject == test_subject]
    return train, test

# ---------------------------
# Main Function
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "LOSO AutoGL EEG Graph Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\uceerjp\Desktop\PhD\Multi-session Data\OG_Full_Data",
                        help="Directory containing EEG .mat files")
    parser.add_argument("--subjects", type=str, default="1,2,5,9,21,31,34,39",
                        help="Comma-separated subject numbers to include")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--max_evals", type=int, default=20, help="Maximum evaluations for HPO")
    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Parse subject numbers
    subject_numbers = [int(s) for s in args.subjects.split(",")]

    # ---------------------------
    # Data Loading: Process each subject's data
    # ---------------------------
    threshold = 0.1
    all_data = []

    for subject_number in subject_numbers:
        print(f"Processing Subject S{subject_number}")
        mat_fname = os.path.join(args.data_dir, f"S{subject_number}.mat")
        mat_contents = sio.loadmat(mat_fname)
        # Use all columns except the last one in the MATLAB file
        subject_raw = mat_contents[f"Subject{subject_number}"][:, :-1]
        plv, y = compute_plv(subject_raw)
        graphs = create_graphs(plv, threshold)
        for i, G in enumerate(graphs):
            edge_index = torch.tensor(list(G.edges)).t().contiguous()
            edge_attr = torch.tensor(
                [G[u][v]['weight'] for u, v in G.edges], dtype=torch.float32
            )
            # Use identity features as a placeholder (replace if needed)
            x = torch.eye(G.number_of_nodes(), dtype=torch.float32)
            d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y[i])
            d.subject = subject_number  # Attach subject information
            all_data.append(d)

    # ---------------------------
    # Define Hyperparameter Space for GAT (No activation function)
    # ---------------------------
    gat_hp_space = [
        {
            "parameterName": "heads",
            "type": "INT_EXP",
            "minValue": 1,
            "maxValue": 8,
            "scalingType": "LOG",
        },
        {
            "parameterName": "hidden",
            "type": "CATEGORICAL",
            "feasiblePoints": [[16], [32], [64], [128]],  # Hidden dims as list (required)
        },
        {
            "parameterName": "dropout",
            "type": "DOUBLE",
            "minValue": 0.1,
            "maxValue": 0.6,
            "scalingType": "LINEAR",
        },
        {
            "parameterName": "lr",
            "type": "DOUBLE",
            "minValue": 1e-4,
            "maxValue": 5e-2,
            "scalingType": "LOG",
        },
        {
            "parameterName": "weight_decay",
            "type": "DOUBLE",
            "minValue": 1e-6,
            "maxValue": 1e-2,
            "scalingType": "LOG",
        },
        {
            "parameterName": "num_layers",
            "type": "INT",
            "minValue": 1,
            "maxValue": 3,
            "scalingType": "LINEAR",
        }
    ]

    # ---------------------------
    # LOSO Cross-Validation Loop
    # ---------------------------
    results = {}

    for test_subject in subject_numbers:
        print(f"\n=== LOSO Fold: Test on Subject {test_subject} ===")
        # Split data into train and test sets
        train_list, test_list = split_data_by_subject(all_data, test_subject)
        # Wrap safely to avoid caching issues
        train_dataset = wrap_dataset_safe(train_list)
        test_dataset = wrap_dataset_safe(test_list)
        combined_dataset = wrap_dataset_safe(train_list + test_list)
        num_train = len(train_dataset)
        num_total = len(combined_dataset)
        # Create a validation split (20% of training set)
        val_count = int(0.2 * num_train)
        combined_dataset.train_split = list(range(0, num_train - val_count))
        combined_dataset.val_split = list(range(num_train - val_count, num_train))
        combined_dataset.test_split = list(range(num_train, num_total))
        
        # Initialize AutoGraphClassifier with HPO options
        model = AutoGraphClassifier(
            graph_models=["gat"],
            hpo_module="random",
            max_evals=args.max_evals,
            model_hp_spaces=[gat_hp_space],
            default_trainer="GraphClassificationFull",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Fit the model (AutoGL will internally use the defined train/val splits)
        model.fit(combined_dataset, train_split=combined_dataset.train_split, val_split=combined_dataset.val_split, seed=args.seed)
        # Get predictions based on the test split
        preds = model.predict(combined_dataset)
        # Extract ground truth labels for the test set
        true = [d.y.item() for d in test_dataset]
        acc = accuracy_score(true, preds)
        print(f"Accuracy on Subject {test_subject}: {acc:.4f}")
        results[test_subject] = acc

    # ---------------------------
    # Print Summary Results
    # ---------------------------
    print("\n=== LOSO Summary ===")
    for s, acc in results.items():
        print(f"Subject {s}: {acc:.4f}")
    print(f"\nAverage Accuracy: {np.mean(list(results.values())):.4f} ~ Std: {np.std(list(results.values())):.4f}")
