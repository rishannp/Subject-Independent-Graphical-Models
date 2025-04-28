#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOSO cross-validation for LMDA-Net on your 62-channel EEG pickle dataset.
Saves checkpointed results so you can resume if it crashes.
"""
import os
import pickle
import numpy as np
import psutil
import gc
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.seed import seed_everything

# Reproducibility
seed_everything(12345)
np.random.seed(12345)
torch.manual_seed(12345)

# --- Utilities ---
def get_ram_usage():
    """Returns RAM usage in MB."""
    return psutil.virtual_memory().used / (1024 ** 2)

def load_pickle_dataset(pkl_path):
    """Load trials list and subject IDs from pickle."""
    with open(pkl_path, 'rb') as f:
        all_data, subject_numbers = pickle.load(f)
    return all_data, subject_numbers

def center_crop_trials(trials, target_samples):
    """Center-crop each trial to `target_samples` timepoints."""
    for trial in trials:
        eeg = trial.x  # torch.Tensor shape (channels, times)
        chans, times = eeg.shape
        if times > target_samples:
            start = (times - target_samples) // 2
            trial.x = eeg[:, start:start+target_samples]
    return trials

def trial_to_numpy(trial):
    """Convert single trial to EEG numpy array and integer label."""
    arr = trial.x.numpy()           # (channels, samples)
    label = int(trial.y.numpy())    # integer class
    return arr.astype(np.float32), label

#%% LMDA Modules
class EEGDepthAttention(nn.Module):
    def __init__(self, W, C, k=7):
        super().__init__()
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k//2, 0), bias=True)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        x_pool = self.adaptive_pool(x)             # (B,1,C,W)
        x_t = x_pool.transpose(-2, -3)             # (B,C,1,W)
        y = self.conv(x_t)                         # (B,1,1,W)
        y = self.softmax(y)
        y = y.transpose(-2, -3)
        return y * self.C * x

class LMDA(nn.Module):
    def __init__(self, chans, samples, num_classes=2,
                 depth=9, channel_depth1=24, channel_depth2=9,
                 kernel=75, avepool=5):
        super().__init__()
        # learnable channel weighting
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans))
        nn.init.xavier_uniform_(self.channel_weight)

        # temporal convolution
        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1,
                      kernel_size=(1, kernel), groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )
        # spatial convolution (depthwise conv over channels)
        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2,
                      kernel_size=(chans,1), groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )
        # pooling and dropout
        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1,1,avepool)),
            nn.Dropout(p=0.65),
        )
        # depth attention
        self.depthAttention = EEGDepthAttention(samples, channel_depth1, k=7)

        # compute classification input dim
        H2 = 1  # after channel conv over 'chans'
        W2 = samples // avepool  # after time pooling
        flat_dim = channel_depth2 * H2 * W2
        self.classifier = nn.Linear(flat_dim, num_classes)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B,1,chans,samples)
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)
        x = self.time_conv(x)
        x = self.depthAttention(x)
        x = self.chanel_conv(x)
        x = self.norm(x)
        features = x.flatten(1)
        return self.classifier(features)

#%% Training Script with LOSO and checkpointing
if __name__ == '__main__':
    print("[INFO] Starting LMDA LOSO Training...")
    dataset_pkl = '/home/uceerjp/He/eeg_trials_dataset.pkl'
    save_path   = '/home/uceerjp/StiegerModel/LMDA/LMDA_results.pkl'

    all_data, subjects = load_pickle_dataset(dataset_pkl)
    print(f"Loaded {len(all_data)} trials from subjects: {subjects}")
    min_len = min(t.x.shape[1] for t in all_data)
    all_data = center_crop_trials(all_data, min_len)

    # group by subject
    by_subj = {}
    for t in all_data:
        by_subj.setdefault(t.subject, []).append(t)
    subjects = sorted(by_subj)

    # load checkpoint
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Resuming from {len(results)} completed subjects.")
    else:
        results = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 32
    num_epochs = 25
    lr = 1e-3

    for test_subj in subjects:
        key = f'Subject_{test_subj}'
        if key in results:
            print(f"{key} done, skipping.")
            continue
        print(f"\n--- Testing Subject {test_subj} ---")

        # prepare data
        train_trials = [t for s,ts in by_subj.items() if s!=test_subj for t in ts]
        test_trials  = by_subj[test_subj]
        X_train, y_train = [], []
        for tr in train_trials:
            arr, lab = trial_to_numpy(tr)
            X_train.append(arr.T[np.newaxis,:,:])
            y_train.append(lab)
        X_test, y_test = [], []
        for tr in test_trials:
            arr, lab = trial_to_numpy(tr)
            X_test.append(arr.T[np.newaxis,:,:])
            y_test.append(lab)
        X_train = np.stack(X_train, axis=0)
        y_train = np.array(y_train, dtype=np.int64)
        X_test  = np.stack(X_test, axis=0)
        y_test  = np.array(y_test, dtype=np.int64)

        # datasets/loaders
        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        # model
        chans, samples = X_train.shape[2], X_train.shape[3]
        model = LMDA(chans=chans, samples=samples).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        ram_before = get_ram_usage()

        # train
        print(f"Training for {num_epochs} epochs...")
        start = time()
        model.train()
        for epoch in range(1, num_epochs+1):
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            print(f" Epoch {epoch}/{num_epochs} loss={total_loss/len(train_ds):.4f}")
        train_time = time() - start

        # evaluate
        model.eval()
        correct = 0; total = 0
        inf_start = time()
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds==yb).sum().item()
                total += yb.size(0)
        inf_time = time() - inf_start
        acc = correct/total
        ram_used = get_ram_usage() - ram_before

        # checkpoint
        results[key] = {
            'accuracy': acc,
            'train_time': train_time,
            'inf_per_trial': inf_time/total,
            'params': sum(p.numel() for p in model.parameters()),
            'ram_usage': ram_used
        }
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved results for {key}.")

        # cleanup
        del model, optimizer, train_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    # summary
    print("\n===== LOSO Summary =====")
    for k, v in results.items():
        print(f"{k}: acc={v['accuracy']*100:.2f}% | train={v['train_time']:.2f}s | params={v['params']} | ram={v['ram_usage']:.2f}MB | inf={v['inf_per_trial']:.4f}s")
