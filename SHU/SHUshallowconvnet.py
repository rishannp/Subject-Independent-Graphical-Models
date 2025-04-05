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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report

# Import ShallowConvNet from EEGModels instead of EEGNet
from EEGModels import ShallowConvNet  # Ensure ShallowConvNet is defined in EEGModels.py

from torch_geometric.seed import seed_everything
seed_everything(12345)

#%% Utility functions

def get_ram_usage():
    """Returns RAM usage in MB."""
    return psutil.virtual_memory().used / (1024 ** 2)

def bandpass(data: np.ndarray, edges: list, sample_rate: float, poles: int = 5):
    sos = sig.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = sig.sosfiltfilt(sos, data, axis=0)
    return filtered_data

def aggregate_eeg_data_full(S1):
    """
    Processes subject data S1 (assumed to have keys 'L' and 'R').
    For each trial, uses the entire trial from all electrodes,
    normalizes it (zscore) along the sample axis, and assigns a label:
      0 for left (key 'L') and 1 for right (key 'R').
    Returns:
      data: array of shape (trials, samples, electrodes)
      labels: array of shape (trials,)
    """
    data_list = []
    labels_list = []
    num_trials = S1['L'].shape[1]  # Assuming equal number of trials in 'L' and 'R'
    for i in range(num_trials):
        # Process left condition trial
        l_trial = S1['L'][0, i]  # shape: (samples, electrodes)
        l_trial = zscore(l_trial, axis=0)
        data_list.append(l_trial)
        labels_list.append(0)

        # Process right condition trial
        r_trial = S1['R'][0, i]
        r_trial = zscore(r_trial, axis=0)
        data_list.append(r_trial)
        labels_list.append(1)
    data = np.stack(data_list, axis=0)  # shape: (trials, samples, electrodes)
    labels = np.array(labels_list)
    return data, labels

def load_and_preprocess_subject(subject_number, data_dir, fs):
    """
    Loads a subject's data from a .mat file, applies bandpass filtering,
    and aggregates EEG data using the entire trial from all electrodes.
    Returns:
      data: array of shape (trials, channels, samples, 1)
      labels: one-hot encoded labels of shape (trials, nb_classes)
    """
    mat_fname = os.path.join(data_dir, f'S{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    # Remove the last column as before.
    S1 = mat_contents[f'Subject{subject_number}'][:, :-1]
    
    # Apply bandpass filtering to both conditions
    for key in ['L', 'R']:
        for i in range(S1.shape[1]):
            S1[key][0, i] = bandpass(S1[key][0, i], [8, 30], fs)
    
    # Use the entire trial for each condition and all electrodes
    data, labels = aggregate_eeg_data_full(S1)
    
    # For ShallowConvNet, data should be shaped as (trials, channels, samples, 1)
    # Currently data is (trials, samples, electrodes) so we transpose it.
    data = data.transpose(0, 2, 1)  # Now (trials, electrodes, samples)
    data = data[..., np.newaxis]   # Add channel dimension: (trials, electrodes, samples, 1)
    
    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    labels = encoder.fit_transform(labels.reshape(-1, 1))
    
    return data, labels

#%% LOSO ShallowConvNet Training without Session Splitting

data_dir = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\OG_Full_Data'
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
fs = 256

results = {}

# Loop over each subject as test subject (LOSO)
for test_subject in subject_numbers:
    print(f"\n===== LOSO Iteration: Test Subject {test_subject} =====")
    train_data_list = []
    train_labels_list = []
    
    # Gather training data from all subjects except the test subject
    for subj in subject_numbers:
        if subj == test_subject:
            continue
        data_subj, labels_subj = load_and_preprocess_subject(subj, data_dir, fs)
        train_data_list.append(data_subj)
        train_labels_list.append(labels_subj)
    
    # Concatenate training data across subjects
    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    
    # Load test subject's data (use entire trial)
    test_data, test_labels = load_and_preprocess_subject(test_subject, data_dir, fs)
    
    print(f"Training on {train_data.shape[0]} trials from subjects: {[s for s in subject_numbers if s != test_subject]}")
    print(f"Testing on {test_data.shape[0]} trials from subject: {test_subject}")
    
    # Track RAM usage before model creation
    ram_before = get_ram_usage()
    
    # Determine input dimensions for EEGNet
    _, chans, samples, _ = train_data.shape
    
    # Initialize ShallowConvNet model
    model = ShallowConvNet(nb_classes=2, Chans=chans, Samples=samples, dropoutRate=0.5)
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    num_model_parameters = model.count_params()
    ram_after = get_ram_usage()
    ram_model_usage = ram_after - ram_before
    
    best_accuracy = 0
    subject_results = {'train_time': 0, 'best_accuracy': 0, 'inference_time_per_trial': 0,
                       'model_parameters': num_model_parameters, 'ram_model_usage': ram_model_usage}
    
    print(f"Training ShallowConvNet for LOSO iteration with Test Subject {test_subject}...")
    start_train_time = time()
    
    epochs = 100
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.fit(train_data, train_labels, batch_size=32, epochs=1, verbose=2)
        
        # Evaluate every epoch
        start_inf = time()
        probs = model.predict(test_data)
        inf_time = time() - start_inf
        preds = probs.argmax(axis=-1)
        test_acc = np.mean(preds == test_labels.argmax(axis=-1))
        if test_acc > best_accuracy:
            best_accuracy = test_acc
        print(f"Epoch {epoch+1}: Test Accuracy = {test_acc:.4f}, Inference Time per trial = {inf_time / test_data.shape[0]:.4f} sec")
    
    total_train_time = time() - start_train_time
    subject_results['train_time'] = total_train_time
    subject_results['best_accuracy'] = best_accuracy
    subject_results['inference_time_per_trial'] = inf_time / test_data.shape[0]
    
    results[f'Subject_{test_subject}'] = subject_results
    
    print(f"LOSO iteration for Test Subject {test_subject} complete.")
    print(f"Train Time: {total_train_time:.2f} sec, Best Accuracy: {best_accuracy*100:.2f}%, "
          f"Model Params: {num_model_parameters}, RAM Usage: {ram_model_usage:.2f} MB.")
    
    # Clear session and garbage collect to free up memory between iterations.
    tf.keras.backend.clear_session()
    gc.collect()

# Print summary of results
print("\n===== LOSO Summary =====")
for subj, res in results.items():
    print(f"{subj}:")
    print(f"  Training Time: {res['train_time']:.2f} sec")
    print(f"  Best Accuracy: {res['best_accuracy']*100:.2f}%")
    print(f"  Model Params: {res['model_parameters']}")
    print(f"  RAM Model Usage: {res['ram_model_usage']:.2f} MB")
    print(f"  Inference Time per trial: {res['inference_time_per_trial']:.4f} sec")
