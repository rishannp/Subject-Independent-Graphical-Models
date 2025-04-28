#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU-enabled FTL (Federated Transfer Learning) training on EEG covariance matrices.
Full LOSO Cross-Validation with checkpointing after each subject.
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
from scipy.signal import butter, sosfiltfilt
from scipy.stats import zscore

# Set seeds for reproducibility
seed_everything(12345)
torch.manual_seed(12345)
np.random.seed(12345)

# ------------------ Utility Functions ------------------

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
        eeg = trial.x
        chans, times = eeg.shape
        if times > target_samples:
            start_idx = (times - target_samples) // 2
            trial.x = eeg[:, start_idx:start_idx+target_samples]
    return trials

def compute_covariance(trials):
    """Compute covariance matrix of each trial."""
    covs = []
    labels = []
    for trial in trials:
        eeg = trial.x.numpy()  # (channels, samples)
        cov = np.cov(eeg)
        cov += np.eye(cov.shape[0]) * 1e-6  # regularization
        covs.append(cov)
        labels.append(int(trial.y.numpy()))
    return np.stack(covs, axis=0), np.array(labels, dtype=np.int64)

def shuffle_data(x, y):
    """Shuffle x and y together."""
    idx = np.random.permutation(len(y))
    return x[idx], y[idx]

def split_class_feat(feat, target):
    """Split features into positive and negative class features."""
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    pos = feat[target==1]
    neg = feat[target==0]
    return pos, neg

# ------------------ Riemannian Functions ------------------

class RecFunction(torch.autograd.Function):
    """Rectified matrix (clamp singular values)."""
    @staticmethod
    def forward(ctx, input):
        Us = torch.zeros_like(input)
        Ss = torch.zeros((input.size(0), input.size(1)), dtype=input.dtype, device=input.device)
        max_Ss = torch.zeros_like(input)
        max_Ids = torch.zeros_like(input)
        for i in range(input.size(0)):
            U, S, V = torch.svd(input[i])
            eps = 1e-4
            S_clamped = torch.clamp(S, min=eps)
            id_mask = (S >= eps).float()
            Ss[i] = S
            Us[i] = U
            max_Ss[i] = torch.diag(S_clamped)
            max_Ids[i] = torch.diag(id_mask)
        result = Us @ max_Ss @ Us.transpose(1,2)
        ctx.save_for_backward(input, Us, Ss, max_Ss, max_Ids)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, Us, Ss, max_Ss, max_Ids = ctx.saved_tensors
        Ks = torch.zeros_like(grad_output)
        dLdC = 0.5 * (grad_output + grad_output.transpose(1,2))
        Ut = Us.transpose(1,2)
        dLdV = 2 * (dLdC @ Us) @ max_Ss
        dLdS1 = Ut @ dLdC @ Us
        dLdS = max_Ids @ dLdS1
        diag_dLdS = torch.zeros_like(grad_output)
        for i in range(dLdS.size(0)):
            s = Ss[i]
            vs1 = s.view(-1,1)
            vs2 = s.view(1,-1)
            K = 1.0 / (vs1 - vs2)
            K[K==float('inf')] = 0
            Ks[i] = K
            diag_dLdS[i] = torch.diag(torch.diag(dLdS[i]))
        tmp = Ut @ dLdV
        tmp = 0.5 * (Ks.transpose(1,2) * tmp + (Ks.transpose(1,2) * tmp).transpose(1,2)) + diag_dLdS
        grad = Us @ tmp @ Ut
        return grad

class LogFunction(torch.autograd.Function):
    """Logarithmic matrix (take log of singular values)."""
    @staticmethod
    def forward(ctx, input):
        Us = torch.zeros_like(input)
        Ss = torch.zeros((input.size(0), input.size(1)), dtype=input.dtype, device=input.device)
        logSs = torch.zeros_like(input)
        invSs = torch.zeros_like(input)
        for i in range(input.size(0)):
            U, S, V = torch.svd(input[i])
            Ss[i] = S
            Us[i] = U
            logSs[i] = torch.diag(torch.log(S))
            invSs[i] = torch.diag(1.0/S)
        result = Us @ logSs @ Us.transpose(1,2)
        ctx.save_for_backward(input, Us, Ss, logSs, invSs)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, Us, Ss, logSs, invSs = ctx.saved_tensors
        dLdC = 0.5 * (grad_output + grad_output.transpose(1,2))
        Ut = Us.transpose(1,2)
        dLdV = 2 * dLdC @ (Us @ logSs)
        dLdS1 = Ut @ dLdC @ Us
        dLdS = invSs @ dLdS1
        diag_dLdS = torch.zeros_like(grad_output)
        Ks = torch.zeros_like(grad_output)
        for i in range(dLdS.size(0)):
            s = Ss[i]
            vs1 = s.view(-1,1)
            vs2 = s.view(1,-1)
            K = 1.0 / (vs1 - vs2)
            K[K==float('inf')] = 0
            Ks[i] = K
            diag_dLdS[i] = torch.diag(torch.diag(dLdS[i]))
        tmp = Ut @ dLdV
        tmp = 0.5 * (Ks.transpose(1,2) * tmp + (Ks.transpose(1,2) * tmp).transpose(1,2)) + diag_dLdS
        grad = Us @ tmp @ Ut
        return grad

def rec_mat(X): return RecFunction.apply(X)
def log_mat(X): return LogFunction.apply(X)

# ------------------ MMD Loss ------------------

class MMD(nn.Module):
    """Maximum Mean Discrepancy."""
    def forward(self, source, target):
        if source.size(0) == 0 or target.size(0) == 0:
            return torch.tensor(0.0, device=source.device)
        n = source.size(0)
        kernels = gaussian_kernel(source, target)
        XX = kernels[:n,:n].mean()
        YY = kernels[n:,n:].mean()
        XY = kernels[:n,n:].mean()
        YX = kernels[n:,:n].mean()
        return XX + YY - XY - YX

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Gaussian kernel for MMD."""
    n_samples = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2 = ((total0 - total1)**2).sum(2)
    bandwidth = fix_sigma or (L2.sum() / (n_samples**2 - n_samples))
    bandwidth /= kernel_mul**(kernel_num // 2)
    sigmas = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    return sum(torch.exp(-L2 / s) for s in sigmas)

# ------------------ SPDNet ------------------

class SPDNetwork(nn.Module):
    """SPDNet (62->8->8)."""
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(62,8, dtype=torch.double))
        self.w2 = nn.Parameter(torch.randn(8,8, dtype=torch.double))
        self.fc = nn.Parameter(torch.randn(8*8,2, dtype=torch.double))

    def forward(self, X):
        out = rec_mat(self.w1.t().unsqueeze(0) @ X @ self.w1.unsqueeze(0))
        out = rec_mat(self.w2.t().unsqueeze(0) @ out @ self.w2.unsqueeze(0))
        out = log_mat(out)
        feat = out.view(out.size(0), -1)
        logits = feat @ self.fc
        return F.log_softmax(logits, dim=1), feat

    def update_federated_layer(self, lr, avg_grad):
        with torch.no_grad():
            self.fc -= lr * avg_grad

# ------------------ FTL Transfer Function (Minibatched) ------------------

def transfer_SPD_batch(cov1, cov2, lab1, lab2, batch_size=32):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cov1, lab1 = shuffle_data(cov1, lab1)
    cov2, lab2 = shuffle_data(cov2, lab2)

    n2 = cov2.shape[0]
    n_train2 = int(np.floor(0.2 * n2))

    data_train_1 = torch.from_numpy(cov1).double()
    data_train_2 = torch.from_numpy(cov2[:n_train2]).double()
    data_test_2  = torch.from_numpy(cov2[n_train2:]).double()
    y1 = torch.LongTensor(lab1)
    y2 = torch.LongTensor(lab2[:n_train2])
    y_test = torch.LongTensor(lab2[n_train2:])

    model1, model2 = SPDNetwork(), SPDNetwork()
    mmd = MMD()
    opt1 = optim.Adam(model1.parameters(), lr=1e-3)
    opt2 = optim.Adam(model2.parameters(), lr=1e-3)

    train_ds1 = TensorDataset(data_train_1, y1)
    train_ds2 = TensorDataset(data_train_2, y2)
    loader1 = DataLoader(train_ds1, batch_size=batch_size, shuffle=True)
    loader2 = DataLoader(train_ds2, batch_size=batch_size, shuffle=True)

    for it in range(25):
        model1.train(); model2.train()
        for (xb1, yb1), (xb2, yb2) in zip(loader1, loader2):
            opt1.zero_grad(); opt2.zero_grad()
            o1, f1 = model1(xb1)
            o2, f2 = model2(xb2)
            loss = F.nll_loss(o1, yb1) + F.nll_loss(o2, yb2)
            f1_pos, f1_neg = split_class_feat(f1, yb1)
            f2_pos, f2_neg = split_class_feat(f2, yb2)
            loss += mmd(f1_pos, f2_pos) + mmd(f1_neg, f2_neg)
            loss.backward()
            avg_grad = (model1.fc.grad + model2.fc.grad) / 2
            model1.update_federated_layer(1e-3, avg_grad)
            model2.update_federated_layer(1e-3, avg_grad)
            opt1.step(); opt2.step()
        print(f"Iter {it+1}/25 done.")

    model2.eval()
    with torch.no_grad():
        logits, _ = model2(data_test_2)
        preds = logits.argmax(dim=1)
        acc = (preds == y_test).float().mean().item()

    print(f"Target Test Acc: {acc:.4f}")
    return acc

# ------------------ Main LOSO Training Loop ------------------

if __name__ == '__main__':
    #print("[INFO] Starting GPU-enabled FTL training...")
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f"Using device: {device}")

    dataset_pkl = '/home/uceerjp/He/eeg_trials_dataset.pkl'
    save_path = '/home/uceerjp/StiegerModel/FTL/FTL_results.pkl'

    all_data, subjects = load_pickle_dataset(dataset_pkl)
    min_len = min(t.x.shape[1] for t in all_data)
    all_data = center_crop_trials(all_data, min_len)

    by_subj = {}
    for t in all_data:
        by_subj.setdefault(t.subject, []).append(t)

    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Resuming from {len(results)} completed subjects.")
    else:
        results = {}

    for test_subj in sorted(by_subj):
        key = f'Subject_{test_subj}'
        if key in results: continue
        print(f"\n--- Test Subject {test_subj} ---")
        src = [t for s, ts in by_subj.items() if s != test_subj for t in ts]
        tgt = by_subj[test_subj]
        cov1, lab1 = compute_covariance(src)
        cov2, lab2 = compute_covariance(tgt)
        acc = transfer_SPD_batch(cov1, cov2, lab1, lab2, batch_size=32)
        results[key] = acc
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Checkpoint saved for {key}.")
        gc.collect()

    print("\n===== Final LOSO Summary =====")
    for k,v in results.items():
        print(f"{k}: {v*100:.2f}%")
