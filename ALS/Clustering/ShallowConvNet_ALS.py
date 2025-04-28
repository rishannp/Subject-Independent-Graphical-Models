import os
import numpy as np
import scipy.io as sio
import scipy.signal as sig
from scipy.stats import zscore
from os.path import join as pjoin
from time import time
import psutil
import gc

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder

# EEGNet-specific imports (adjust to import ShallowConvNet)
from EEGModels import ShallowConvNet

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

def load_and_preprocess_subject(subject_number, data_dir, fs, chunk_duration_sec=3):
    """
    Loads a subject's data from a .mat file, applies bandpass filtering,
    and aggregates EEG data.
    Returns:
      data: array of shape (trials, samples, electrodes)
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
    
    # For ShallowConvNet, data should be shaped as (trials, channels, samples, 1)
    # Our data is (trials, samples, channels) so we transpose.
    data = data.transpose(0, 2, 1)  # Now (trials, channels, samples)
    data = data[..., np.newaxis]   # Add singleton dimension: (trials, channels, samples, 1)
    
    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    labels = encoder.fit_transform(labels.reshape(-1, 1))
    
    return data, labels

#%% LOSO ShallowConvNet Training (100 Epochs, No Session Splits)
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\OG_Full_Data'
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
fs = 256

results = {}

# Loop over each subject as the test subject (LOSO)
for test_subject in subject_numbers:
    print(f"\n===== LOSO Iteration: Test Subject {test_subject} =====")
    train_data_list, train_labels_list = [], []
    
    # Collect training data from all except the test subject
    for subj in subject_numbers:
        if subj == test_subject:
            continue
        data_subj, labels_subj = load_and_preprocess_subject(
            subj, data_dir, fs, chunk_duration_sec=3
        )
        train_data_list.append(data_subj)
        train_labels_list.append(labels_subj)
    
    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    
    # Load test subject's full data
    test_data, test_labels = load_and_preprocess_subject(
        test_subject, data_dir, fs, chunk_duration_sec=3
    )
    
    print(f"Training on {train_data.shape[0]} trials from subjects "
          f"{[s for s in subject_numbers if s != test_subject]}")
    print(f"Testing on {test_data.shape[0]} trials from subject {test_subject}")
    
    # RAM before model
    ram_before = get_ram_usage()
    
    # Model input dims
    _, chans, samples, _ = train_data.shape
    
    # Initialize and compile ShallowConvNet
    model = ShallowConvNet(
        nb_classes=2,
        Chans=chans,
        Samples=samples,
        dropoutRate=0.5
    )
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    num_model_parameters = model.count_params()
    ram_after = get_ram_usage()
    ram_model_usage = ram_after - ram_before
    
    print(f"Training ShallowConvNet for LOSO iteration with Test Subject {test_subject}...")
    start_train_time = time()
    
    # Train all epochs in one go
    epochs = 25
    model.fit(
        train_data,
        train_labels,
        batch_size=32,
        epochs=epochs,
        verbose=2
    )
    
    total_train_time = time() - start_train_time
    
    # Single evaluation after training
    start_inf = time()
    probs = model.predict(test_data, verbose=0)
    inf_time = time() - start_inf
    preds = probs.argmax(axis=-1)
    test_acc = np.mean(preds == test_labels.argmax(axis=-1))
    
    # Store results
    results[f'Subject_{test_subject}'] = {
        'train_time': total_train_time,
        'test_accuracy': test_acc,
        'inference_time_per_trial': inf_time / test_data.shape[0],
        'model_parameters': num_model_parameters,
        'ram_model_usage': ram_model_usage
    }
    
    print(f"Test Accuracy = {test_acc:.4f}")
    print(f"Inference Time per trial = {inf_time / test_data.shape[0]:.4f} sec")
    print(f"Training Time: {total_train_time:.2f} sec, "
          f"Model Params: {num_model_parameters}, RAM Usage: {ram_model_usage:.2f} MB.")
    
    # Clear session & free memory
    tf.keras.backend.clear_session()
    gc.collect()

# Optionally save
np.save(os.path.join(os.getcwd(), 'ShallowConvNet_LOSO_results.npy'), results)

# Summary
print("\n===== LOSO Summary =====")
for subj, res in results.items():
    print(f"{subj}:")
    print(f"  Training Time: {res['train_time']:.2f} sec")
    print(f"  Test Accuracy: {res['test_accuracy']*100:.2f}%")
    print(f"  Model Params: {res['model_parameters']}")
    print(f"  RAM Model Usage: {res['ram_model_usage']:.2f} MB")
    print(f"  Inference Time per trial: {res['inference_time_per_trial']:.4f} sec")
