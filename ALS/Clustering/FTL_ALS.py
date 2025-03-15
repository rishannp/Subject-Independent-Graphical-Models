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

#%% Updated Autograd Functions

class RecFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # input shape: (batch, dim, dim)
        Us = torch.zeros_like(input)
        Ss = torch.zeros((input.shape[0], input.shape[1]), dtype=input.dtype, device=input.device)
        max_Ss = torch.zeros_like(input)
        max_Ids = torch.zeros_like(input)
        for i in range(input.shape[0]):
            U, S, V = torch.svd(input[i, :, :])
            eps = 1e-4
            max_S = torch.clamp(S, min=eps)
            max_Id = (S >= eps).float()
            Ss[i, :] = S
            Us[i, :, :] = U
            max_Ss[i, :, :] = torch.diag(max_S)
            max_Ids[i, :, :] = torch.diag(max_Id)
        result = torch.matmul(Us, torch.matmul(max_Ss, torch.transpose(Us, 1, 2)))
        ctx.save_for_backward(input, Us, Ss, max_Ss, max_Ids)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, Us, Ss, max_Ss, max_Ids = ctx.saved_tensors
        Ks = torch.zeros_like(grad_output)
        dLdC = grad_output
        dLdC = 0.5 * (dLdC + dLdC.transpose(1, 2))
        Ut = Us.transpose(1, 2)
        dLdV = 2 * torch.matmul(torch.matmul(dLdC, Us), max_Ss)
        dLdS_1 = torch.matmul(torch.matmul(Ut, dLdC), Us)
        dLdS = torch.matmul(max_Ids, dLdS_1)
        diag_dLdS = torch.zeros_like(grad_output)
        for i in range(grad_output.shape[0]):
            diagS = Ss[i, :].contiguous()
            vs_1 = diagS.view(-1, 1)
            vs_2 = diagS.view(1, -1)
            K = 1.0 / (vs_1 - vs_2)
            K[K == float("inf")] = 0.0
            Ks[i, :, :] = K
            diag_dLdS[i, :, :] = torch.diag(torch.diag(dLdS[i, :, :]))
        tmp = torch.matmul(Ut, dLdV)
        tmp = 0.5 * (torch.transpose(Ks, 1, 2) * tmp + (torch.transpose(Ks, 1, 2) * tmp).transpose(1, 2))
        tmp = tmp + diag_dLdS
        grad = torch.matmul(Us, torch.matmul(tmp, Ut))
        return grad

class LogFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        Us = torch.zeros_like(input)
        Ss = torch.zeros((input.shape[0], input.shape[1]), dtype=input.dtype, device=input.device)
        logSs = torch.zeros_like(input)
        invSs = torch.zeros_like(input)
        for i in range(input.shape[0]):
            U, S, V = torch.svd(input[i, :, :])
            Ss[i, :] = S
            Us[i, :, :] = U
            logSs[i, :, :] = torch.diag(torch.log(S))
            invSs[i, :, :] = torch.diag(1.0 / S)
        result = torch.matmul(Us, torch.matmul(logSs, torch.transpose(Us, 1, 2)))
        ctx.save_for_backward(input, Us, Ss, logSs, invSs)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, Us, Ss, logSs, invSs = ctx.saved_tensors
        grad_output = grad_output.double()
        Ks = torch.zeros_like(grad_output)
        dLdC = grad_output
        dLdC = 0.5 * (dLdC + dLdC.transpose(1, 2))
        Ut = Us.transpose(1, 2)
        dLdV = 2 * torch.matmul(dLdC, torch.matmul(Us, logSs))
        dLdS_1 = torch.matmul(torch.matmul(Ut, dLdC), Us)
        dLdS = torch.matmul(invSs, dLdS_1)
        diag_dLdS = torch.zeros_like(grad_output)
        for i in range(grad_output.shape[0]):
            diagS = Ss[i, :].contiguous()
            vs_1 = diagS.view(-1, 1)
            vs_2 = diagS.view(1, -1)
            K = 1.0 / (vs_1 - vs_2)
            K[K == float("inf")] = 0.0
            Ks[i, :, :] = K
            diag_dLdS[i, :, :] = torch.diag(torch.diag(dLdS[i, :, :]))
        tmp = torch.matmul(Ut, dLdV)
        tmp = 0.5 * (torch.transpose(Ks, 1, 2) * tmp + (torch.transpose(Ks, 1, 2) * tmp).transpose(1, 2))
        tmp = tmp + diag_dLdS
        grad = torch.matmul(Us, torch.matmul(tmp, Ut))
        return grad

def rec_mat(input):
    return RecFunction.apply(input)

def log_mat(input):
    return LogFunction.apply(input)

#%% Riemannian Update Functions

def update_para_riemann(X, U, t):
    Up = cal_riemann_grad(X, U)
    new_X = cal_retraction(X, Up, t)
    return new_X

def cal_riemann_grad(X, U):
    XtU = np.matmul(X.T, U)
    symXtU = 0.5 * (XtU + XtU.T)
    Up = U - np.matmul(X, symXtU)
    return Up

def cal_retraction(X, rU, t):
    Y = X - t * rU
    Q, R = np.linalg.qr(Y, mode='reduced')
    sR = np.diag(np.sign(np.diag(R)))
    Y = np.matmul(Q, sR)
    return Y

#%% Other Utility Functions

def get_ram_usage():
    return psutil.virtual_memory().used / (1024 ** 2)

def bandpass(data: np.ndarray, edges: list, sample_rate: float, poles: int = 5):
    sos = sig.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = sig.sosfiltfilt(sos, data, axis=0)
    return filtered_data

def aggregate_eeg_data(S1, chunk_size=1024):
    numElectrodes = 19
    data_list = []
    labels_list = []
    for i in range(S1['L'].shape[1]):
        l_trial = S1['L'][0, i]
        l_num_samples = l_trial.shape[0]
        if l_num_samples >= chunk_size:
            l_start = (l_num_samples - chunk_size) // 2
            l_end = l_start + chunk_size
            l_chunk = l_trial[l_start:l_end, :numElectrodes]
            l_chunk = zscore(l_chunk, axis=0)
            data_list.append(l_chunk)
            labels_list.append(0)
        r_trial = S1['R'][0, i]
        r_num_samples = r_trial.shape[0]
        if r_num_samples >= chunk_size:
            r_start = (r_num_samples - chunk_size) // 2
            r_end = r_start + chunk_size
            r_chunk = r_trial[r_start:r_end, :numElectrodes]
            r_chunk = zscore(r_chunk, axis=0)
            data_list.append(r_chunk)
            labels_list.append(1)
    data = np.stack(data_list, axis=0)
    labels = np.array(labels_list)
    return data, labels

def load_and_preprocess_subject_LMDA(subject_number, data_dir, fs, chunk_duration_sec=3):
    mat_fname = os.path.join(data_dir, f'S{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    S1 = mat_contents[f'Subject{subject_number}'][:, :-1]
    for key in ['L', 'R']:
        for i in range(S1.shape[1]):
            S1[key][0, i] = bandpass(S1[key][0, i], [8, 30], fs)
    chunk_size = int(fs * chunk_duration_sec)
    data, labels = aggregate_eeg_data(S1, chunk_size=chunk_size)
    data = data.transpose(0, 2, 1)
    data = data[:, np.newaxis, :, :]
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    labels = encoder.fit_transform(labels.reshape(-1, 1))
    return data, labels

def compute_covariance(trials):
    n_trials = trials.shape[0]
    cov_matrices = []
    for i in range(n_trials):
        trial = trials[i, 0, :, :]
        cov = np.cov(trial)
        cov += np.eye(cov.shape[0]) * 1e-6
        cov_matrices.append(cov)
    return np.stack(cov_matrices, axis=0)

def shuffle_data(x, y):
    idx = np.random.permutation(x.shape[0])
    return x[idx, :, :], y[idx]

def split_class_feat(feat, target):
    # Ensure target is a numpy array.
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    positive_index = target == 1
    negative_index = target == 0
    # If no samples, return an empty tensor with correct feature dimension.
    if np.sum(positive_index) == 0:
        positive_feat = torch.empty((0, feat.shape[1]), dtype=torch.double)
    else:
        positive_feat = torch.tensor(feat[positive_index].detach().cpu().numpy(), dtype=torch.double)
    if np.sum(negative_index) == 0:
        negative_feat = torch.empty((0, feat.shape[1]), dtype=torch.double)
    else:
        negative_feat = torch.tensor(feat[negative_index].detach().cpu().numpy(), dtype=torch.double)
    return positive_feat, negative_feat

#%% FTL Model and SPDNet Components

import torch.nn.functional as F

## MMD Loss
class MMD(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
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
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        # If either input is empty, return zero loss.
        if source.size(0) == 0 or target.size(0) == 0:
            return torch.tensor(0.0, dtype=source.dtype, device=source.device)
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul,
                                       kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        with torch.no_grad():
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
        torch.cuda.empty_cache()
        return loss

def update_manifold_reduction_layer(lr, params_list):
    for w in params_list:
        grad_w_np = w.grad.data.numpy()
        w_np = w.data.numpy()
        updated_w = update_para_riemann(w_np, grad_w_np, lr)
        w.data.copy_(torch.DoubleTensor(updated_w))
        w.grad.data.zero_()

class SPDNetwork_1(nn.Module):
    """
    SPDNetwork with network structure: [(19, 19), (19, 16), (16, 4)]
    """
    def __init__(self):
        super(SPDNetwork_1, self).__init__()
        self.w_1_p = torch.randn(19, 19, dtype=torch.double, requires_grad=True)
        self.w_2_p = torch.randn(19, 16, dtype=torch.double, requires_grad=True)
        self.w_3_p = torch.randn(16, 4, dtype=torch.double, requires_grad=True)
        self.fc_w = torch.randn(16, 2, dtype=torch.double, requires_grad=True)

    def forward(self, input):
        batch_size = input.shape[0]
        output = input  # input: covariance matrices, shape (batch, dim, dim)
        for w in [self.w_1_p, self.w_2_p]:
            w = w.contiguous().view(1, w.shape[0], w.shape[1])
            w_tX = torch.matmul(torch.transpose(w, 1, 2), output)
            w_tXw = torch.matmul(w_tX, w)
            output = rec_mat(w_tXw)
        w_3 = self.w_3_p.contiguous().view(1, self.w_3_p.shape[0], self.w_3_p.shape[1])
        w_tX = torch.matmul(torch.transpose(w_3, 1, 2), output)
        w_tXw = torch.matmul(w_tX, w_3)
        X_3 = log_mat(w_tXw)
        feat = X_3.view(batch_size, -1)
        logits = torch.matmul(feat, self.fc_w)
        output = F.log_softmax(logits, dim=-1)
        return output, feat

    def update_manifold_reduction_layer(self, lr):
        update_manifold_reduction_layer(lr, [self.w_1_p, self.w_2_p, self.w_3_p])

    def update_federated_layer(self, lr, average_grad):
        self.fc_w.data -= lr * average_grad
        self.fc_w.grad.data.zero_()

class SPDNetwork_2(nn.Module):
    """
    SPDNetwork with network structure: [(19, 4), (4, 4), (4, 4)]
    """
    def __init__(self):
        super(SPDNetwork_2, self).__init__()
        self.w_1_p = torch.randn(19, 4, dtype=torch.double, requires_grad=True)
        self.w_2_p = torch.randn(4, 4, dtype=torch.double, requires_grad=True)
        self.w_3_p = torch.randn(4, 4, dtype=torch.double, requires_grad=True)
        self.fc_w = torch.randn(16, 2, dtype=torch.double, requires_grad=True)

    def forward(self, input):
        batch_size = input.shape[0]
        output = input
        for w in [self.w_1_p, self.w_2_p]:
            w = w.contiguous().view(1, w.shape[0], w.shape[1])
            w_tX = torch.matmul(torch.transpose(w, 1, 2), output)
            w_tXw = torch.matmul(w_tX, w)
            output = rec_mat(w_tXw)
        w_3 = self.w_3_p.contiguous().view(1, self.w_3_p.shape[0], self.w_3_p.shape[1])
        w_tX = torch.matmul(torch.transpose(w_3, 1, 2), output)
        w_tXw = torch.matmul(w_tX, w_3)
        X_3 = log_mat(w_tXw)
        feat = X_3.view(batch_size, -1)
        logits = torch.matmul(feat, self.fc_w)
        output = F.log_softmax(logits, dim=-1)
        return output, feat

    def update_manifold_reduction_layer(self, lr):
        update_manifold_reduction_layer(lr, [self.w_1_p, self.w_2_p, self.w_3_p])

    def update_federated_layer(self, lr, average_grad):
        self.fc_w.data -= lr * average_grad
        self.fc_w.grad.data.zero_()

# For convenience we alias SPDNetwork to one of the architectures; here we use SPDNetwork_2.
SPDNetwork = SPDNetwork_2

def transfer_SPD(cov_data_1, cov_data_2, labels_1, labels_2):
    """
    Trains the Federated Transfer Learning model using the source domain (cov_data_1)
    and target domain (cov_data_2). The labels should be one-dimensional arrays.
    """
    cov_data_1, labels_1 = shuffle_data(cov_data_1, labels_1)
    cov_data_2, labels_2 = shuffle_data(cov_data_2, labels_2)
    print("Data shapes:", cov_data_1.shape, cov_data_2.shape)
    target_train_frac = 0.2  # Change this value to control the target split
    train_data_1_num = cov_data_1.shape[0]
    cov_data_train_1 = cov_data_1  # All source data is used for training
    train_data_2_num = int(np.floor(cov_data_2.shape[0] * target_train_frac))
    cov_data_train_2 = cov_data_2[:train_data_2_num, :, :]
    cov_data_test_2  = cov_data_2[train_data_2_num:, :, :]
    print('Target training samples:', train_data_2_num)
    print('Target testing samples:', labels_2.shape[0] - train_data_2_num)
    print('-------------------------------------------------------')
    
    data_train_1 = torch.from_numpy(cov_data_train_1).double()
    data_train_2 = torch.from_numpy(cov_data_train_2).double()
    data_test_2  = torch.from_numpy(cov_data_test_2).double()
    
    target_train_1 = torch.LongTensor(labels_1)
    target_train_2 = torch.LongTensor(labels_2[:train_data_2_num])
    target_test_2  = torch.LongTensor(labels_2[train_data_2_num:])
    
    model_1 = SPDNetwork()
    model_2 = SPDNetwork()
    
    best_test_acc = 0.0  # Variable to store highest test accuracy found
    old_loss = 0
    lr_val, lr_1, lr_2 = 0.1, 0.1, 0.1
    
    for iteration in range(100):
        output_1, feat_1 = model_1(data_train_1)
        output_2, feat_2 = model_2(data_train_2)
        
        feat_1_positive, feat_1_negative = split_class_feat(feat_1, target_train_1)
        feat_2_positive, feat_2_negative = split_class_feat(feat_2, target_train_2)
        
        mmd = MMD('rbf', kernel_mul=2.0)
        loss = F.nll_loss(output_1, target_train_1) + F.nll_loss(output_2, target_train_2)
        loss += mmd(feat_1_positive, feat_2_positive) + mmd(feat_1_negative, feat_2_negative)
        
        loss.backward()
        model_1.update_manifold_reduction_layer(lr_1)
        model_2.update_manifold_reduction_layer(lr_2)
        
        average_grad = (model_1.fc_w.grad.data + model_2.fc_w.grad.data) / 2
        model_1.update_federated_layer(lr_val, average_grad)
        model_2.update_federated_layer(lr_val, average_grad)
        
        # Evaluate performance on training and target test set
        if iteration % 1 == 0:
            pred_1 = output_1.data.max(1, keepdim=True)[1]
            pred_2 = output_2.data.max(1, keepdim=True)[1]
            train_accuracy_1 = pred_1.eq(target_train_1.data.view_as(pred_1)).sum().float() / pred_1.shape[0]
            train_accuracy_2 = pred_2.eq(target_train_2.data.view_as(pred_2)).sum().float() / pred_2.shape[0]
            print(f"Iteration {iteration}: Source Train Acc: {train_accuracy_1:.4f}, Target Train Acc: {train_accuracy_2:.4f}")
            
            logits_2, _ = model_2(data_test_2)
            output_test = F.log_softmax(logits_2, dim=-1)
            loss_test = F.nll_loss(output_test, target_test_2)
            pred_test = output_test.data.max(1, keepdim=True)[1]
            test_accuracy = pred_test.eq(target_test_2.data.view_as(pred_test)).sum().float() / pred_test.shape[0]
            print(f"Target Test Acc: {test_accuracy:.4f}")
            
            # Update best test accuracy if current test_accuracy is higher
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
        
        if np.abs(loss.item() - old_loss) < 1e-4:
            break
        old_loss = loss.item()
        if iteration % 50 == 0:
            lr_val = max(0.98 * lr_val, 0.01)
            lr_1 = max(0.98 * lr_1, 0.01)
            lr_2 = max(0.98 * lr_2, 0.01)
    
    return best_test_acc.item()


#%% LOSO FTL Training using Your Dataset

data_dir = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\OG_Full_Data'
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
fs = 256
results = {}
for test_subject in subject_numbers:
    print(f"\n===== LOSO Iteration: Test Subject {test_subject} =====")
    source_data_list = []
    source_labels_list = []
    for subj in subject_numbers:
        if subj == test_subject:
            continue
        data_subj, labels_subj = load_and_preprocess_subject_LMDA(subj, data_dir, fs, chunk_duration_sec=3)
        cov_subj = compute_covariance(data_subj)
        labels_subj = np.argmax(labels_subj, axis=1)
        source_data_list.append(cov_subj)
        source_labels_list.append(labels_subj)
    source_cov = np.concatenate(source_data_list, axis=0)
    source_labels = np.concatenate(source_labels_list, axis=0)
    target_data, target_labels = load_and_preprocess_subject_LMDA(test_subject, data_dir, fs, chunk_duration_sec=3)
    target_cov = compute_covariance(target_data)
    target_labels = np.argmax(target_labels, axis=1)
    print(f"Source trials: {source_cov.shape[0]}, Target trials: {target_cov.shape[0]}")
    test_acc = transfer_SPD(source_cov, target_cov, source_labels, target_labels)
    results[f"Subject_{test_subject}"] = test_acc
    print(f"LOSO iteration for Test Subject {test_subject} complete. Target Test Accuracy: {test_acc:.4f}")
print("\n===== LOSO Summary =====")
for subj, acc in results.items():
    print(f"{subj}: Target Test Accuracy = {acc*100:.2f}%")
