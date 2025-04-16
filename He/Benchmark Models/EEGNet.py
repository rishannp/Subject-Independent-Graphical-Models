#!/usr/bin/env python3
import os
import pickle
import numpy as np
from time import time
import psutil
import gc

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report

# EEGNet-specific imports (assumes EEGNet is defined in EEGModels.py)
from EEGModels import EEGNet

# Set random seeds (if needed)
tf.random.set_seed(12345)
np.random.seed(12345)

# Utility functions
def get_ram_usage():
    """Returns RAM usage in MB."""
    return psutil.virtual_memory().used / (1024 ** 2)

def load_eeg_trials_dataset(pkl_path):
    """Load the EEG trials dataset from pickle file."""
    with open(pkl_path, 'rb') as f:
        all_data, subject_numbers = pickle.load(f)
    return all_data, subject_numbers

def convert_trial_to_numpy(trial):
    """
    Convert a PyTorch Geometric Data trial into a NumPy array.
    Expected trial.x shape: (channels, times)
    Returns EEG data reshaped to (channels, times, 1) and the label.
    """
    eeg = trial.x.numpy()             # (channels, times)
    # Optionally: perform per-trial channel normalization (z-score) if desired:
    # from scipy.stats import zscore
    # eeg = zscore(eeg, axis=1)
    eeg = eeg[..., np.newaxis]          # add a channel dimension -> (channels, times, 1)
    label = trial.y.item()              # get scalar label (0 or 1)
    return eeg, label

def prepare_data_for_eegnet(trials):
    """
    Given a list of Data objects from the pickle file, convert them into a training-ready dataset:
      - data: NumPy array of shape (trials, channels, samples, 1)
      - labels: NumPy array of labels (integer)
    """
    data_list = []
    label_list = []
    for trial in trials:
        eeg_arr, label = convert_trial_to_numpy(trial)
        data_list.append(eeg_arr)
        label_list.append(label)
    data = np.stack(data_list, axis=0)
    labels = np.array(label_list)
    return data, labels

#%% LOSO EEGNet Training Using the Pickle Dataset

# Path to your saved dataset (make sure this path is correct)
dataset_pkl_path = '/home/uceerjp/He/eeg_trials_dataset.pkl'

print("Loading EEG trials dataset ...")
all_data, subject_numbers = load_eeg_trials_dataset(dataset_pkl_path)
print(f"Loaded dataset with {len(all_data)} trials from subjects: {subject_numbers}")

# Convert the list of trials into a list-of-data; note each trial has attribute .subject.
# We'll use the subject attribute to create LOSO splits.
trials_by_subject = {}
for trial in all_data:
    subj = trial.subject
    if subj not in trials_by_subject:
        trials_by_subject[subj] = []
    trials_by_subject[subj].append(trial)

# Get sorted list of subject numbers from dataset
subject_numbers = sorted(trials_by_subject.keys())

results = {}

# Loop over each subject as test subject (LOSO)
for test_subject in subject_numbers:
    print(f"\n===== LOSO Iteration: Test Subject {test_subject} =====")
    
    # Gather training trials from all subjects except the test subject
    train_trials = []
    for subj in subject_numbers:
        if subj == test_subject:
            continue
        train_trials.extend(trials_by_subject[subj])
    
    # Get test trials
    test_trials = trials_by_subject[test_subject]
    
    # Convert trials to NumPy arrays
    train_data, train_labels = prepare_data_for_eegnet(train_trials)
    test_data, test_labels   = prepare_data_for_eegnet(test_trials)
    
    # For EEGNet, the expected shape is (trials, channels, samples, 1)
    # If necessary, check data dimensions:
    print(f"Training on {train_data.shape[0]} trials from subjects: {[s for s in subject_numbers if s != test_subject]}")
    print(f"Testing on {test_data.shape[0]} trials from subject: {test_subject}")

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    train_labels_oh = encoder.fit_transform(train_labels.reshape(-1, 1))
    test_labels_oh  = encoder.transform(test_labels.reshape(-1, 1))
    
    # Get input dimensions for EEGNet
    _, chans, samples, _ = train_data.shape
    
    # Record current RAM usage before model creation
    ram_before = get_ram_usage()
    
    # Initialize EEGNet model (adjust hyperparameters as needed)
    model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
                   dropoutRate=0.5, kernLength=500, F1=8, D=2, F2=16,
                   dropoutType='Dropout')
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    num_model_parameters = model.count_params()
    ram_after = get_ram_usage()
    ram_model_usage = ram_after - ram_before
    
    best_accuracy = 0
    subject_results = {'train_time': 0, 'best_accuracy': 0, 
                       'inference_time_per_trial': 0, 'model_parameters': num_model_parameters,
                       'ram_model_usage': ram_model_usage}
    
    print(f"Training EEGNet for LOSO iteration with Test Subject {test_subject}...")
    start_train_time = time()
    
    epochs = 100
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.fit(train_data, train_labels_oh, batch_size=32, epochs=1, verbose=2)
        
        # Evaluate after each epoch
        start_inf = time()
        probs = model.predict(test_data)
        inf_time = time() - start_inf
        preds = probs.argmax(axis=-1)
        current_accuracy = np.mean(preds == test_labels_oh.argmax(axis=-1))
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
        print(f"Epoch {epoch+1}: Test Accuracy = {current_accuracy:.4f}, Inference Time per trial = {inf_time / test_data.shape[0]:.4f} sec")
    
    total_train_time = time() - start_train_time
    subject_results['train_time'] = total_train_time
    subject_results['best_accuracy'] = best_accuracy
    subject_results['inference_time_per_trial'] = inf_time / test_data.shape[0]
    
    results[f'Subject_{test_subject}'] = subject_results
    
    print(f"LOSO iteration for Test Subject {test_subject} complete.")
    print(f"Train Time: {total_train_time:.2f} sec, Best Accuracy: {best_accuracy*100:.2f}%, "
          f"Model Params: {num_model_parameters}, RAM Usage: {ram_model_usage:.2f} MB.")
    
    # Clear session and collect garbage to free memory
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
