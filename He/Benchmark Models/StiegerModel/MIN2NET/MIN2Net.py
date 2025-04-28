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
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.seed import seed_everything

# Reproducibility
seed_everything(12345)
np.random.seed(12345)
torch.manual_seed(12345)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(12345)

# Utilities
def get_ram_usage():
    return psutil.virtual_memory().used / (1024 ** 2)

def load_pickle_dataset(pkl_path):
    with open(pkl_path, 'rb') as f:
        all_data, subject_numbers = pickle.load(f)
    return all_data, subject_numbers

def center_crop_trials(trials, target_samples):
    for trial in trials:
        eeg = trial.x
        chans, times = eeg.shape
        if times > target_samples:
            start_idx = (times - target_samples) // 2
            trial.x = eeg[:, start_idx:start_idx+target_samples]
    return trials

def trial_to_numpy(trial):
    eeg = trial.x.numpy()  # (channels, samples)
    eeg = eeg.T            # (samples, channels)
    label = int(trial.y.numpy())
    return eeg.astype(np.float32), label

# MIN2Net Model
class MIN2Net_PyTorch(nn.Module):
    def __init__(self, input_shape=(1, 768, 62), num_class=2, latent_dim=64,
                 pool_size_1=(1,62), pool_size_2=(1,1), filter_1=20, filter_2=10):
        super(MIN2Net_PyTorch, self).__init__()
        D, T, C = input_shape
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(D, filter_1, kernel_size=(1,64), padding='same')
        self.bn1 = nn.BatchNorm2d(filter_1)
        self.pool1 = nn.AvgPool2d(kernel_size=pool_size_1)

        self.conv2 = nn.Conv2d(filter_1, filter_2, kernel_size=(1,32), padding='same')
        self.bn2 = nn.BatchNorm2d(filter_2)
        self.pool2 = nn.AvgPool2d(kernel_size=pool_size_2)

        self.flatten_size = filter_2 * T * 1
        self.fc_latent = nn.Linear(self.flatten_size, latent_dim)

        self.fc_decoder = nn.Linear(latent_dim, self.flatten_size)
        self.upsample = nn.Upsample(size=(T,C), mode='bilinear', align_corners=False)
        self.reconstruct = nn.Conv2d(filter_2, D, kernel_size=(1,1))

        self.classifier = nn.Linear(latent_dim, num_class)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)

        x = F.elu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        x_flat = x.view(x.size(0), -1)
        latent = self.fc_latent(x_flat)

        x_dec = self.fc_decoder(latent)
        x_dec = x_dec.view(x.size(0), 10, self.input_shape[1], 1)
        x_dec = self.upsample(x_dec)
        rec = self.reconstruct(x_dec)

        logits = self.classifier(latent)
        return rec, latent, logits

# Losses
reconstruction_loss_fn = nn.MSELoss()
classification_loss_fn = nn.CrossEntropyLoss()
triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)

def train_step(model, optimizer, inputs, labels, device):
    model.train()
    optimizer.zero_grad()
    rec, latent, logits = model(inputs.to(device))
    loss_rec = reconstruction_loss_fn(rec, inputs.to(device))
    loss_cls = classification_loss_fn(logits, labels.to(device))
    loss_trip = 0
    if latent.size(0) >= 3:
        anchor, positive, negative = latent[:-2], latent[1:-1], latent[2:]
        loss_trip = triplet_loss_fn(anchor, positive, negative)
    total_loss = loss_rec + loss_cls + loss_trip
    total_loss.backward()
    optimizer.step()
    return total_loss.item()

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            rec, latent, logits = model(inputs)
            loss_rec = reconstruction_loss_fn(rec, inputs)
            loss_cls = classification_loss_fn(logits, labels)
            batch_loss = (loss_rec + loss_cls).item() * inputs.size(0)
            total_loss += batch_loss
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)
    return total_loss/total, correct/total

# Training Script
if __name__ == '__main__':
    print("[INFO] Starting MIN2Net LOSO Training...")

    dataset_pkl = '/home/uceerjp/He/eeg_trials_dataset.pkl'
    save_path = '/home/uceerjp/StiegerModel/MIN2NET/Min2Net_results.pkl'

    all_data, subjects = load_pickle_dataset(dataset_pkl)
    print(f"[INFO] Loaded {len(all_data)} trials from {len(subjects)} subjects.")

    min_len = min(t.x.shape[1] for t in all_data)
    all_data = center_crop_trials(all_data, min_len)

    by_subj = {}
    for t in all_data:
        by_subj.setdefault(t.subject, []).append(t)
    subjects = sorted(by_subj)

    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            results = pickle.load(f)
        print(f"[INFO] Resuming from {len(results)} completed subjects.")
    else:
        results = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for test_subj in subjects:
        subj_key = f'Subject_{test_subj}'
        if subj_key in results:
            print(f"[INFO] {subj_key} already done. Skipping...")
            continue

        print(f"\n===== LOSO: Test Subject {test_subj} =====")
        train_trials = [t for s, ts in by_subj.items() if s != test_subj for t in ts]
        test_trials = by_subj[test_subj]

        # Prepare training data
        X_train, y_train = zip(*[trial_to_numpy(t) for t in train_trials])
        X_train = np.stack(X_train)
        y_train = np.array(y_train, dtype=np.int64)

        X_test, y_test = zip(*[trial_to_numpy(t) for t in test_trials])
        X_test = np.stack(X_test)
        y_test = np.array(y_test, dtype=np.int64)

        # Add dummy dimension and swap axes to (batch, 1, T, C)
        X_train = np.expand_dims(X_train, axis=1).transpose(0,1,2,3)
        X_test = np.expand_dims(X_test, axis=1).transpose(0,1,2,3)

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.long))
        test_dataset  = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                      torch.tensor(y_test, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

        input_shape = (1, X_train.shape[2], X_train.shape[3])
        model = MIN2Net_PyTorch(input_shape=input_shape, num_class=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        start_train = time()
        for epoch in range(25):
            epoch_loss = 0
            for inputs, labels in train_loader:
                loss = train_step(model, optimizer, inputs, labels, device)
                epoch_loss += loss
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/25, Loss: {avg_loss:.4f}")

        total_train_time = time() - start_train

        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"[RESULT] {subj_key}: Accuracy={test_acc*100:.2f}%")

        results[subj_key] = {
            'accuracy': test_acc,
            'train_time': total_train_time,
            'test_loss': test_loss
        }

        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

        torch.cuda.empty_cache()
        gc.collect()

    print("\n===== Final LOSO Summary =====")
    for subj, r in results.items():
        print(f"{subj}: acc={r['accuracy']*100:.2f}%, train={r['train_time']:.2f}s, loss={r['test_loss']:.4f}")
