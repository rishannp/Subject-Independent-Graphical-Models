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

from torch_geometric.seed import seed_everything
seed_everything(12345)

fs= 256
#%% New Model: HopefullNet (Modified to output 2 classes for binary classification)
class HopefullNet(tf.keras.Model):
    """
    HopefullNet modified for binary classification.
    Expected input shape: (640, 2)
    """
    def __init__(self, inp_shape=(640,2)):
        super(HopefullNet, self).__init__()
        self.inp_shape = inp_shape

        self.kernel_size_0 = 20
        self.kernel_size_1 = 6
        self.drop_rate = 0.5

        self.conv1 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding="same",
                                            input_shape=self.inp_shape)
        self.batch_n_1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding="valid")
        self.batch_n_2 = tf.keras.layers.BatchNormalization()
        self.spatial_drop_1 = tf.keras.layers.SpatialDropout1D(self.drop_rate)
        self.conv3 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding="valid")
        self.avg_pool1 = tf.keras.layers.AvgPool1D(pool_size=2)
        self.conv4 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding="valid")
        self.spatial_drop_2 = tf.keras.layers.SpatialDropout1D(self.drop_rate)
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(296, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(self.drop_rate)
        self.dense2 = tf.keras.layers.Dense(148, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(self.drop_rate)
        self.dense3 = tf.keras.layers.Dense(74, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(self.drop_rate)
        # Modified output layer for 2 classes instead of 5
        self.out = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, input_tensor):
        conv1 = self.conv1(input_tensor)
        batch_n_1 = self.batch_n_1(conv1)
        conv2 = self.conv2(batch_n_1)
        batch_n_2 = self.batch_n_2(conv2)
        spatial_drop_1 = self.spatial_drop_1(batch_n_2)
        conv3 = self.conv3(spatial_drop_1)
        avg_pool1 = self.avg_pool1(conv3)
        conv4 = self.conv4(avg_pool1)
        spatial_drop_2 = self.spatial_drop_2(conv4)
        flat = self.flat(spatial_drop_2)
        dense1 = self.dense1(flat)
        dropout1 = self.dropout1(dense1)
        dense2 = self.dense2(dropout1)
        dropout2 = self.dropout2(dense2)
        return self.out(dropout2)

#%% Utility functions

def get_ram_usage():
    """Returns RAM usage in MB."""
    return psutil.virtual_memory().used / (1024 ** 2)

def bandpass(data: np.ndarray, edges: list, sample_rate: float, poles: int = 5):
    sos = sig.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = sig.sosfiltfilt(sos, data, axis=0)
    return filtered_data

def aggregate_eeg_data(S1, chunk_size=640):
    """
    Processes subject data S1 (assumed to have keys 'L' and 'R').
    For each trial, extracts a middle chunk (of length chunk_size) from the first 2 electrodes,
    normalizes it (zscore), and assigns a label: 0 for 'L', 1 for 'R'.
    Returns:
      data: array of shape (trials, samples, channels)  e.g. (trials, 640, 2)
      labels: array of shape (trials,)
    """
    numElectrodes = 2  # Use only the first 2 electrodes
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

    data = np.stack(data_list, axis=0)  # shape: (trials, samples, channels)
    labels = np.array(labels_list)
    return data, labels


def load_and_preprocess_subject(subject_number, data_dir, fs=256, chunk_duration_sec=640/fs):
    """
    Loads a subject's data from a .mat file, applies bandpass filtering,
    and aggregates EEG data.
    Returns:
      data: array of shape (trials, samples, channels)  e.g. (trials, 640, 2)
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
    
    chunk_size = int(fs * chunk_duration_sec)  # Should be 640 if fs and chunk_duration_sec are chosen appropriately.
    data, labels = aggregate_eeg_data(S1, chunk_size=chunk_size)
    
    # For HopefullNet, data should be (trials, samples, channels) with shape (trials, 640, 2)
    # So no additional transposition or channel-dimension expansion is needed.
    
    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    labels = encoder.fit_transform(labels.reshape(-1, 1))
    
    return data, labels

#%% LOSO Training with HopefullNet

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
        data_subj, labels_subj = load_and_preprocess_subject(subj, data_dir, fs, chunk_duration_sec=640/fs)
        train_data_list.append(data_subj)
        train_labels_list.append(labels_subj)
    
    # Concatenate training data across subjects
    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    
    # Load test subject's data (use entire data)
    test_data, test_labels = load_and_preprocess_subject(test_subject, data_dir, fs, chunk_duration_sec=640/fs)
    
    print(f"Training on {train_data.shape[0]} trials from subjects: {[s for s in subject_numbers if s != test_subject]}")
    print(f"Testing on {test_data.shape[0]} trials from subject: {test_subject}")
    
    # Track RAM usage before model creation
    ram_before = get_ram_usage()
    
    # For HopefullNet, the expected input shape is (samples, channels) i.e. (640, 2)
    inp_shape = (train_data.shape[1], train_data.shape[2])
    
    # Initialize HopefullNet
    model = HopefullNet(inp_shape=inp_shape)
    model.build((None, inp_shape[0], inp_shape[1]))  # Build the model explicitly
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    num_model_parameters = model.count_params()  # Now this works without error

    num_model_parameters = model.count_params()
    ram_after = get_ram_usage()
    ram_model_usage = ram_after - ram_before
    
    best_accuracy = 0
    subject_results = {'train_time': 0, 'best_accuracy': 0, 'inference_time_per_trial': 0,
                       'model_parameters': num_model_parameters, 'ram_model_usage': ram_model_usage}
    
    print(f"Training HopefullNet for LOSO iteration with Test Subject {test_subject}...")
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
