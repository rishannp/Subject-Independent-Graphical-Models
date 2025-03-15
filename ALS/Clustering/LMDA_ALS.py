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

data_dir = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\OG_Full_Data'
#data_dir = r'/home/uceerjp/Multi-sessionData/OG_Full_Data' ##  Server Directory

subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
fs = 256

results = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 32
num_epochs = 100
learning_rate = 0.001

for test_subject in subject_numbers:
    print(f"\n===== LOSO Iteration: Test Subject {test_subject} =====")
    train_data_list = []
    train_labels_list = []
    
    # Gather training data from all subjects except the test subject
    for subj in subject_numbers:
        if subj == test_subject:
            continue
        data_subj, labels_subj = load_and_preprocess_subject_LMDA(subj, data_dir, fs, chunk_duration_sec=3)
        train_data_list.append(data_subj)
        train_labels_list.append(labels_subj)
    
    # Concatenate training data across subjects
    train_data_np = np.concatenate(train_data_list, axis=0)
    train_labels_np = np.concatenate(train_labels_list, axis=0)
    
    # Load test subject's data (entire dataset)
    test_data_np, test_labels_np = load_and_preprocess_subject_LMDA(test_subject, data_dir, fs, chunk_duration_sec=3)
    
    print(f"Training on {train_data_np.shape[0]} trials from subjects: {[s for s in subject_numbers if s != test_subject]}")
    print(f"Testing on {test_data_np.shape[0]} trials from subject: {test_subject}")
    
    # Convert to torch tensors
    train_data_tensor = torch.tensor(train_data_np, dtype=torch.float32)
    test_data_tensor = torch.tensor(test_data_np, dtype=torch.float32)
    # Convert one-hot labels to class indices
    train_labels_indices = torch.tensor(np.argmax(train_labels_np, axis=1), dtype=torch.long)
    test_labels_indices = torch.tensor(np.argmax(test_labels_np, axis=1), dtype=torch.long)
    
    # Create DataLoaders with batch size 32
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(train_data_tensor, train_labels_indices)
    test_dataset = TensorDataset(test_data_tensor, test_labels_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Track RAM usage before model creation
    ram_before = get_ram_usage()
    
    # Determine input dimensions for LMDA: (batch, 1, chans, samples)
    _, _, chans, samples = train_data_tensor.shape
    
    # Initialize LMDA model
    model = LMDA(chans=chans, samples=samples, num_classes=2).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    num_model_parameters = sum(p.numel() for p in model.parameters())
    ram_after = get_ram_usage()
    ram_model_usage = ram_after - ram_before
    
    best_accuracy = 0
    best_epoch = 0
    total_inf_time = 0.0
    start_train_time = time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_data.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        
        # Evaluation after each epoch
        model.eval()
        correct = 0
        total = 0
        epoch_inf_time = 0.0
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                start_inf = time()
                outputs = model(batch_data)
                epoch_inf_time += time() - start_inf
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        test_acc = correct / total
        total_inf_time += epoch_inf_time
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch + 1
        
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss:.4f}, Test Accuracy = {test_acc*100:.2f}%, " 
              f"Inference Time per trial = {epoch_inf_time/total:.4f} sec")
    
    total_train_time = time() - start_train_time
    results[f"Subject_{test_subject}"] = {
        "train_time": total_train_time,
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "inference_time_per_trial": total_inf_time / total,
        "model_parameters": num_model_parameters,
        "ram_model_usage": ram_model_usage,
    }
    
    print(f"LOSO iteration for Test Subject {test_subject} complete.")
    print(f"Train Time: {total_train_time:.2f} sec, Best Accuracy: {best_accuracy*100:.2f}% at Epoch {best_epoch}, "
          f"Model Params: {num_model_parameters}, RAM Usage: {ram_model_usage:.2f} MB.")
    
    # Clean up for next iteration
    del model, optimizer, train_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

# Print summary of results
print("\n===== LOSO Summary =====")
for subj, res in results.items():
    print(f"{subj}:")
    print(f"  Training Time: {res['train_time']:.2f} sec")
    print(f"  Best Accuracy: {res['best_accuracy']*100:.2f}% at Epoch {res['best_epoch']}")
    print(f"  Model Params: {res['model_parameters']}")
    print(f"  RAM Model Usage: {res['ram_model_usage']:.2f} MB")
    print(f"  Inference Time per trial: {res['inference_time_per_trial']:.4f} sec")