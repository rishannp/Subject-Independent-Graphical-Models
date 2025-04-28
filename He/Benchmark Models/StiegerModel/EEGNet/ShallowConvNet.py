#!/usr/bin/env python3
import os
import pickle
import numpy as np
import psutil
import gc
from time import time

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder

from EEGModels import ShallowConvNet

# Set random seeds
tf.random.set_seed(12345)
np.random.seed(12345)

# Enable GPU memory growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# --- 1) CHECK HOST RAM ---
total_bytes = psutil.virtual_memory().total
used_bytes = psutil.virtual_memory().used
print(f"Host RAM: {total_bytes / (1024**3):.2f} GB total, {used_bytes / (1024**3):.2f} GB used")

# Utility functions
def get_ram_usage():
    return psutil.virtual_memory().used / (1024 ** 2)

def load_eeg_trials_dataset(pkl_path):
    with open(pkl_path, 'rb') as f:
        all_data, subject_numbers = pickle.load(f)
    return all_data, subject_numbers

def prepare_data_for_model(trials, encoder, batch_size=32, shuffle=True):
    data_list = []
    label_list = []
    for trial in trials:
        eeg_arr = trial.x.numpy()
        label = trial.y.item()
        data_list.append(eeg_arr[..., np.newaxis])
        label_list.append(label)

    data = np.stack(data_list, axis=0)
    labels = np.array(label_list)
    labels_oh = encoder.transform(labels.reshape(-1, 1))

    dataset = tf.data.Dataset.from_tensor_slices((data, labels_oh))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(data), 1000))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

#%% LOSO ShallowConvNet Training

print("Starting Script...")
dataset_pkl_path = '/home/uceerjp/He/eeg_trials_dataset.pkl'

print("Loading EEG trials dataset ...")
all_data, subject_numbers = load_eeg_trials_dataset(dataset_pkl_path)
print(f"Loaded dataset with {len(all_data)} trials from subjects: {subject_numbers}")

# Center-Crop All Trials to Minimum Length
min_times = min(trial.x.shape[1] for trial in all_data)
print(f"[INFO] Minimum time length found: {min_times}")

for trial in all_data:
    eeg = trial.x
    chans, times = eeg.shape
    if times > min_times:
        start_idx = (times - min_times) // 2
        end_idx = start_idx + min_times
        trial.x = eeg[:, start_idx:end_idx]

print("[INFO] All trials cropped to minimum time length.")

trials_by_subject = {}
for trial in all_data:
    subj = trial.subject
    if subj not in trials_by_subject:
        trials_by_subject[subj] = []
    trials_by_subject[subj].append(trial)

subject_numbers = sorted(trials_by_subject.keys())
results = {}

# LOSO Cross-Validation
for test_subject in subject_numbers:
    print(f"\n===== LOSO Iteration: Test Subject {test_subject} =====")

    train_trials = []
    for subj in subject_numbers:
        if subj == test_subject:
            continue
        train_trials.extend(trials_by_subject[subj])
    
    test_trials = trials_by_subject[test_subject]

    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(np.array([trial.y.item() for trial in all_data]).reshape(-1,1))

    train_dataset = prepare_data_for_model(train_trials, encoder, batch_size=8, shuffle=True)
    test_dataset = prepare_data_for_model(test_trials, encoder, batch_size=8, shuffle=False)

    chans, samples = train_trials[0].x.shape

    ram_before = get_ram_usage()

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = ShallowConvNet(nb_classes=2, Chans=chans, Samples=samples, dropoutRate=0.5)
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

    epochs = 25
    history = model.fit(
        train_dataset,
        epochs=epochs,
        verbose=2
    )

    print(f"Evaluating on Test Subject {test_subject}...")
    start_inf = time()
    eval_result = model.evaluate(test_dataset, verbose=0)
    inf_time = time() - start_inf

    test_accuracy = eval_result[1]  # accuracy

    total_train_time = time() - start_train_time

    results[f'Subject_{test_subject}'] = {
        'train_time': total_train_time,
        'best_accuracy': test_accuracy,
        'inference_time_per_trial': inf_time / len(test_trials),
        'model_parameters': num_model_parameters,
        'ram_model_usage': ram_model_usage
    }

    print(
        f"Train Time: {total_train_time:.2f} sec, "
        f"Test Accuracy: {test_accuracy*100:.2f}%, "
        f"Model Params: {num_model_parameters}, "
        f"RAM Usage: {ram_model_usage:.2f} MB."
    )

    tf.keras.backend.clear_session()
    gc.collect()

print("\n===== LOSO Summary =====")
for subj, res in results.items():
    print(f"{subj}:")
    print(f"  Training Time: {res['train_time']:.2f} sec")
    print(f"  Best Accuracy: {res['best_accuracy']*100:.2f}%")
    print(f"  Model Params: {res['model_parameters']}")
    print(f"  RAM Model Usage: {res['ram_model_usage']:.2f} MB")
    print(f"  Inference Time per trial: {res['inference_time_per_trial']:.4f} sec")
