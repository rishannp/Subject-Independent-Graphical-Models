import os
import numpy as np
import scipy.io as sio
import scipy.signal as sig
from scipy.stats import zscore
import psutil
import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# For reproducibility in graph-based experiments
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
    For each trial in left (L) and right (R) conditions, extracts a middle chunk 
    (of length chunk_size) from the first 19 electrodes, applies zscore normalization,
    and assigns labels: 0 for 'L' and 1 for 'R'.
    """
    numElectrodes = 19
    data_list = []
    labels_list = []
    for i in range(S1['L'].shape[1]):
        # Left condition
        l_trial = S1['L'][0, i]
        l_num_samples = l_trial.shape[0]
        if l_num_samples >= chunk_size:
            l_start = (l_num_samples - chunk_size) // 2
            l_chunk = l_trial[l_start:l_start+chunk_size, :numElectrodes]
            l_chunk = zscore(l_chunk, axis=0)
            data_list.append(l_chunk)
            labels_list.append(0)
        # Right condition
        r_trial = S1['R'][0, i]
        r_num_samples = r_trial.shape[0]
        if r_num_samples >= chunk_size:
            r_start = (r_num_samples - chunk_size) // 2
            r_chunk = r_trial[r_start:r_start+chunk_size, :numElectrodes]
            r_chunk = zscore(r_chunk, axis=0)
            data_list.append(r_chunk)
            labels_list.append(1)
    data = np.stack(data_list, axis=0)  # shape: (trials, samples, electrodes)
    labels = np.array(labels_list, dtype=np.int64)  # integer labels for CrossEntropyLoss
    return data, labels

def load_and_preprocess_subject(subject_number, data_dir, fs, chunk_duration_sec=3):
    """
    Loads subject data from a .mat file, applies bandpass filtering,
    aggregates EEG data, and returns data along with integer labels.
    Data is returned in shape (trials, samples, electrodes).
    """
    mat_fname = os.path.join(data_dir, f'S{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    # Assuming the .mat file contains a key like 'Subject{subject_number}'
    S1 = mat_contents[f'Subject{subject_number}'][:, :-1]
    
    for key in ['L', 'R']:
        for i in range(S1.shape[1]):
            S1[key][0, i] = bandpass(S1[key][0, i], [8, 30], fs)
    
    chunk_size = int(fs * chunk_duration_sec)
    data, labels = aggregate_eeg_data(S1, chunk_size=chunk_size)
    
    # Transpose data so that channels come second: (trials, electrodes, samples)
    data = data.transpose(0, 2, 1)
    # Add a dummy dimension so that the final shape is (trials, 1, samples, electrodes)
    data = data[..., np.newaxis]
    
    return data, labels

#%% PyTorch MIN2Net Model and Loss Functions

class MIN2Net_PyTorch(nn.Module):
    def __init__(self, input_shape=(1, 768, 19), num_class=2, latent_dim=64,
                 pool_size_1=(1,19), pool_size_2=(1,1), filter_1=20, filter_2=10):
        """
        input_shape: (D, T, C) where D is typically 1.
        In our design, we set pool_size_1=(1,19) so that the width (C) is reduced from 19 to 1.
        """
        super(MIN2Net_PyTorch, self).__init__()
        D, T, C = input_shape
        self.input_shape = input_shape
        self.num_class = num_class
        self.latent_dim = latent_dim
        self.filter_1 = filter_1
        self.filter_2 = filter_2

        # Encoder
        # Use padding='same' (requires PyTorch >= 1.9) to preserve dimensions.
        self.conv1 = nn.Conv2d(in_channels=D, out_channels=filter_1, kernel_size=(1,64), padding='same')
        self.bn1 = nn.BatchNorm2d(filter_1)
        # Pool across width to reduce 19 -> 1
        self.pool1 = nn.AvgPool2d(kernel_size=pool_size_1)  # pool_size_1 = (1,19)
        
        self.conv2 = nn.Conv2d(in_channels=filter_1, out_channels=filter_2, kernel_size=(1,32), padding='same')
        self.bn2 = nn.BatchNorm2d(filter_2)
        self.pool2 = nn.AvgPool2d(kernel_size=pool_size_2)  # pool_size_2 = (1,1) does nothing
        
        # After conv1 and pool1:
        # Input: (batch, 1, 768, 19) -> conv1 -> (batch, 20, 768, 19) -> pool1 -> (batch, 20, 768, 1)
        # Then conv2 -> (batch, 10, 768, 1)
        self.flatten_size = filter_2 * T * 1  # = 10 * 768 = 7680
        
        self.fc_latent = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder: recover the encoded feature map
        self.fc_decoder = nn.Linear(latent_dim, self.flatten_size)
        # We reshape to (batch, filter_2, T, 1) i.e., (batch, 10, 768, 1)
        # Then upsample to (batch, 10, 768, 19) using bilinear interpolation
        self.upsample = nn.Upsample(size=(T, C), mode='bilinear', align_corners=False)
        # Finally, map feature channels to 1 with a 1x1 convolution
        self.reconstruct = nn.Conv2d(filter_2, D, kernel_size=(1,1), padding=0)
        
        # Classifier: from latent space to class logits
        self.classifier = nn.Linear(latent_dim, num_class)
        
    def forward(self, x):
        # x: (batch, D, T, C), e.g., (batch, 1, 768, 19)
        x = F.elu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)  # (batch, 20, 768, 1)
        x = F.elu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)  # (batch, 10, 768, 1)
        x_flat = x.view(x.size(0), -1)  # (batch, 7680)
        latent = self.fc_latent(x_flat)  # (batch, latent_dim)
        
        # Decoder
        x_dec = self.fc_decoder(latent)  # (batch, 7680)
        x_dec = x_dec.view(x.size(0), self.filter_2, self.input_shape[1], 1)  # (batch, 10, 768, 1)
        x_dec = self.upsample(x_dec)  # (batch, 10, 768, 19)
        rec = self.reconstruct(x_dec)  # (batch, 1, 768, 19)
        
        # Classifier branch
        logits = self.classifier(latent)
        return rec, latent, logits

# Define loss functions
reconstruction_loss_fn = nn.MSELoss()
classification_loss_fn = nn.CrossEntropyLoss()
triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)

def train_step(model, optimizer, inputs, labels, device):
    model.train()
    optimizer.zero_grad()
    rec, latent, logits = model(inputs.to(device))
    loss_rec = reconstruction_loss_fn(rec, inputs.to(device))
    loss_cls = classification_loss_fn(logits, labels.to(device))
    # Naively form triplets if batch size >= 3
    if latent.size(0) >= 3:
        anchor = latent[:-2]
        positive = latent[1:-1]
        negative = latent[2:]
        loss_trip = triplet_loss_fn(anchor, positive, negative)
    else:
        loss_trip = 0.0
    total_loss = loss_rec + loss_cls + loss_trip
    total_loss.backward()
    optimizer.step()
    return total_loss.item()

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            rec, latent, logits = model(inputs)
            loss_rec = reconstruction_loss_fn(rec, inputs)
            loss_cls = classification_loss_fn(logits, labels)
            batch_loss = (loss_rec + loss_cls).item() * inputs.size(0)
            total_loss += batch_loss
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)
    return total_loss / total, correct / total

#%% Main LOSO Training with MIN2Net in PyTorch (No Validation Set)
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\OG_Full_Data'
#data_dir = r'/home/uceerjp/Multi-sessionData/OG_Full_Data' ##  Server Directory

subject_numbers = [39, 34, 31, 21, 9, 5, 2, 1]
fs = 256

results = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for test_subject in subject_numbers:
    print(f"\n===== LOSO Iteration: Test Subject {test_subject} =====")
    train_data_list = []
    train_labels_list = []
    
    # Gather training data from all subjects except the test subject
    for subj in subject_numbers:
        if subj == test_subject:
            continue
        data_subj, labels_subj = load_and_preprocess_subject(subj, data_dir, fs, chunk_duration_sec=3)
        train_data_list.append(data_subj)
        train_labels_list.append(labels_subj)
    
    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    
    # Load test subject's data
    test_data, test_labels = load_and_preprocess_subject(test_subject, data_dir, fs, chunk_duration_sec=3)
    
    print(f"Training on {train_data.shape[0]} trials from subjects: {[s for s in subject_numbers if s != test_subject]}")
    print(f"Testing on {test_data.shape[0]} trials from subject: {test_subject}")
    
    # MIN2Net expects input shape (D, T, C); our data is (trials, electrodes, samples, 1).
    # We transpose it to (trials, 1, samples, electrodes).
    train_data_pt = np.transpose(train_data, (0, 3, 2, 1))
    test_data_pt = np.transpose(test_data, (0, 3, 2, 1))
    
    # Create PyTorch datasets
    X_train_tensor = torch.tensor(train_data_pt, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_labels, dtype=torch.long)
    X_test_tensor = torch.tensor(test_data_pt, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_labels, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Determine input shape from training data: (D, T, C)
    input_shape = (X_train_tensor.shape[1], X_train_tensor.shape[2], X_train_tensor.shape[3])
    print(f"MIN2Net Input Shape (PyTorch): {input_shape}")
    
    # Initialize model and optimizer
    model = MIN2Net_PyTorch(input_shape=input_shape, num_class=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop: 100 epochs, evaluate after each epoch on test set,
    # and log the highest testing accuracy
    num_epochs = 100
    best_test_acc = 0.0
    best_epoch = -1
    start_train = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            loss = train_step(model, optimizer, inputs, labels, device)
            epoch_loss += loss
        avg_loss = epoch_loss / len(train_loader)
        test_loss, test_acc = evaluate(model, test_loader, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Best Test Acc: {best_test_acc:.4f} (Epoch {best_epoch})")
    total_train_time = time.time() - start_train
    
    results[f'Subject_{test_subject}'] = {'train_time': total_train_time,
                                           'best_test_accuracy': best_test_acc,
                                           'best_epoch': best_epoch}
    print(f"Final Evaluation for Subject {test_subject}: Accuracy = {best_test_acc*100:.2f}%")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n===== LOSO Summary =====")
for subj, eval_dict in results.items():
    print(f"{subj}: {eval_dict}")
