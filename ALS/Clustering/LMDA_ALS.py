import os
import numpy as np
import scipy.io as sio
import scipy.signal as sig
from scipy.stats import zscore
from os.path import join as pjoin
from time import time
import psutil
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.seed import seed_everything
seed_everything(12345)

#%% Utility Functions

def get_ram_usage():
    """Returns RAM usage in MB."""
    return psutil.virtual_memory().used / (1024 ** 2)

def bandpass(data: np.ndarray, edges: list, sample_rate: float, poles: int = 5):
    sos = sig.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = sig.sosfiltfilt(sos, data, axis=0)
    return filtered_data

def aggregate_eeg_data(S1, chunk_size=1024):
    """
    Processes subject data S1 (assumed to have keys 'L' and 'R').
    For each trial, extracts a middle chunk (of length chunk_size) from the first 19 electrodes,
    normalizes it (zscore), and assigns a label: 0 for 'L', 1 for 'R'.
    Returns:
      data: array of shape (trials, samples, electrodes)
      labels: array of shape (trials,)
    """
    numElectrodes = 19  # Keep only the first 19 electrodes
    data_list = []
    labels_list = []

    for i in range(S1['L'].shape[1]):
        # Process left condition
        l_trial = S1['L'][0, i]
        l_num_samples = l_trial.shape[0]
        if l_num_samples >= chunk_size:
            l_start = (l_num_samples - chunk_size) // 2
            l_end = l_start + chunk_size
            l_chunk = l_trial[l_start:l_end, :numElectrodes]
            l_chunk = zscore(l_chunk, axis=0)
            data_list.append(l_chunk)
            labels_list.append(0)

        # Process right condition
        r_trial = S1['R'][0, i]
        r_num_samples = r_trial.shape[0]
        if r_num_samples >= chunk_size:
            r_start = (r_num_samples - chunk_size) // 2
            r_end = r_start + chunk_size
            r_chunk = r_trial[r_start:r_end, :numElectrodes]
            r_chunk = zscore(r_chunk, axis=0)
            data_list.append(r_chunk)
            labels_list.append(1)

    data = np.stack(data_list, axis=0)  # shape: (trials, samples, electrodes)
    labels = np.array(labels_list)
    return data, labels

def load_and_preprocess_subject_LMDA(subject_number, data_dir, fs, chunk_duration_sec=3):
    """
    Loads a subject's data from a .mat file, applies bandpass filtering,
    and aggregates EEG data.
    For LMDA, we need the input shape to be (trials, 1, channels, samples).
    Returns:
      data: numpy array of shape (trials, 1, channels, samples)
      labels: one-hot encoded labels of shape (trials, nb_classes)
    """
    mat_fname = os.path.join(data_dir, f'S{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    # Load subject data (removing the last column as before)
    S1 = mat_contents[f'Subject{subject_number}'][:, :-1]
    
    # Apply bandpass filtering to both conditions
    for key in ['L', 'R']:
        for i in range(S1.shape[1]):
            S1[key][0, i] = bandpass(S1[key][0, i], [8, 30], fs)
    
    chunk_size = int(fs * chunk_duration_sec)
    data, labels = aggregate_eeg_data(S1, chunk_size=chunk_size)
    
    # For LMDA, we want the data in shape (trials, 1, channels, samples).
    # Our data from aggregate_eeg_data is (trials, samples, channels).
    # First, transpose to (trials, channels, samples)...
    data = data.transpose(0, 2, 1)
    # ... then add a singleton dimension at index 1.
    data = data[:, np.newaxis, :, :]  # Now shape is (trials, 1, channels, samples)
    
    # One-hot encode labels
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    labels = encoder.fit_transform(labels.reshape(-1, 1))
    
    return data, labels

#%% LMDA Model Definition (using PyTorch)
# (The LMDA model code is as provided.)

class EEGDepthAttention(nn.Module):
    """
    Build EEG Depth Attention module.
    :arg
    C: number of channels
    W: number of time samples
    k: learnable kernel size
    """
    def __init__(self, W, C, k=7):
        super(EEGDepthAttention, self).__init__()
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        x_pool = self.adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = self.conv(x_transpose)
        y = self.softmax(y)
        y = y.transpose(-2, -3)
        return y * self.C * x

class LMDA(nn.Module):
    """
    LMDA-Net for the paper
    """
    def __init__(self, chans=19, samples=768, num_classes=2, depth=9, kernel=75, channel_depth1=24, channel_depth2=9,
                ave_depth=1, avepool=5):
        super(LMDA, self).__init__()
        self.ave_depth = ave_depth
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel),
                      groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )
        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(chans, 1),
                      groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )
        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            nn.Dropout(p=0.65),
        )

        # A dummy forward pass to determine classifier input size
        out = torch.ones((1, 1, chans, samples))
        out = torch.einsum('bdcw, hdc->bhcw', out, self.channel_weight)
        out = self.time_conv(out)
        N, C, H, W = out.size()
        self.depthAttention = EEGDepthAttention(W, C, k=7)
        out = self.chanel_conv(out)
        out = self.norm(out)
        n_out_shape = out.size()
        print('LMDA output shape: ', n_out_shape)
        self.classifier = nn.Linear(n_out_shape[1]*n_out_shape[2]*n_out_shape[3], num_classes)

        # Weight initialization for all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: shape (batch, 1, chans, samples)
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)  # Apply channel weight
        x_time = self.time_conv(x)
        x_time = self.depthAttention(x_time)
        x = self.chanel_conv(x_time)
        x = self.norm(x)
        features = torch.flatten(x, 1)
        cls = self.classifier(features)
        return cls

#%% LOSO LMDA Training (100 Epochs, No Session Splitting)

from torch.utils.data import TensorDataset, DataLoader

data_dir = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\OG_Full_Data'
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
fs = 256

results = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 32
num_epochs = 25       # ← train only 25 epochs now
learning_rate = 0.001

for test_subject in subject_numbers:
    print(f"\n===== LOSO Iteration: Test Subject {test_subject} =====")
    train_data_list, train_labels_list = [], []
    
    # Gather training data (all except test_subject)
    for subj in subject_numbers:
        if subj == test_subject:
            continue
        data_subj, labels_subj = load_and_preprocess_subject_LMDA(
            subj, data_dir, fs, chunk_duration_sec=3
        )
        train_data_list.append(data_subj)
        train_labels_list.append(labels_subj)
    
    train_data_np = np.concatenate(train_data_list, axis=0)
    train_labels_np = np.concatenate(train_labels_list, axis=0)
    
    # Load test subject's data
    test_data_np, test_labels_np = load_and_preprocess_subject_LMDA(
        test_subject, data_dir, fs, chunk_duration_sec=3
    )
    
    print(f"Training on {train_data_np.shape[0]} trials from subjects "
          f"{[s for s in subject_numbers if s != test_subject]}")
    print(f"Testing on {test_data_np.shape[0]} trials from subject {test_subject}")
    
    # Convert to tensors
    train_data_tensor = torch.tensor(train_data_np, dtype=torch.float32)
    test_data_tensor  = torch.tensor(test_data_np,  dtype=torch.float32)
    train_labels_idx  = torch.tensor(train_labels_np.argmax(axis=1), dtype=torch.long)
    test_labels_idx   = torch.tensor(test_labels_np.argmax(axis=1),  dtype=torch.long)
    
    train_ds = TensorDataset(train_data_tensor, train_labels_idx)
    test_ds  = TensorDataset(test_data_tensor,  test_labels_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    
    # RAM usage before model
    ram_before = get_ram_usage()
    
    # Init LMDA (expects input shape [batch, 1, chans, samples])
    _, _, chans, samples = train_data_tensor.shape
    model = LMDA(chans=chans, samples=samples, num_classes=2).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    num_model_parameters = sum(p.numel() for p in model.parameters())
    ram_after = get_ram_usage()
    ram_model_usage = ram_after - ram_before
    
    # ----- Training -----
    print(f"Training LMDA for {num_epochs} epochs...")
    start_train = time()
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)
        avg_loss = running_loss / len(train_ds)
        print(f"  Epoch {epoch+1}/{num_epochs}, Loss = {avg_loss:.4f}")
    train_time = time() - start_train
    
    # ----- Single Evaluation -----
    model.eval()
    correct = 0
    total = 0
    inf_start = time()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            _, preds = outputs.max(1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    inf_time = time() - inf_start
    
    test_acc = correct / total
    inf_time_per_trial = inf_time / total
    
    # Store results
    results[f"Subject_{test_subject}"] = {
        "train_time": train_time,
        "test_accuracy": test_acc,
        "inference_time_per_trial": inf_time_per_trial,
        "model_parameters": num_model_parameters,
        "ram_model_usage": ram_model_usage,
    }
    
    print(f"Test Accuracy = {test_acc*100:.2f}%")
    print(f"Inference Time per trial = {inf_time_per_trial:.4f} sec")
    print(f"Training Time = {train_time:.2f} sec, "
          f"Params = {num_model_parameters}, RAM = {ram_model_usage:.2f} MB")
    
    # Clean up
    del model, optimizer, train_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

# ----- Summary -----
print("\n===== LOSO Summary =====")
for subj, r in results.items():
    print(f"{subj}:")
    print(f"  Training Time: {r['train_time']:.2f} sec")
    print(f"  Test Accuracy: {r['test_accuracy']*100:.2f}%")
    print(f"  Model Params: {r['model_parameters']}")
    print(f"  RAM Model Usage: {r['ram_model_usage']:.2f} MB")
    print(f"  Inference Time per trial: {r['inference_time_per_trial']:.4f} sec")