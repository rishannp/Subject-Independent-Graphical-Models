#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from time import time
import psutil
import gc

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.seed import seed_everything

from EEGModels import EEGNet  # EEGNet model definition

# ------ Helper functions ------
def get_ram_usage():
    """Returns RAM usage in MB."""
    return psutil.virtual_memory().used / (1024 ** 2)

def load_eeg_trials_dataset(pkl_path):
    """Load the EEG trials dataset from pickle file."""
    with open(pkl_path, 'rb') as f:
        all_data, subject_numbers = pickle.load(f)
    return all_data, subject_numbers

def convert_trial_to_numpy(trial):
    """Convert a PyG trial into (eeg_array, label)."""
    eeg = trial.x.numpy()           # (channels, times)
    eeg = eeg[..., np.newaxis]      # -> (channels, times, 1)
    label = trial.y.item()          # scalar label
    return eeg, label

def prepare_data_for_eegnet(trials):
    """Stack list of trials into (N, chans, samples, 1) & labels."""
    data_list, label_list = [], []
    for trial in trials:
        eeg_arr, lbl = convert_trial_to_numpy(trial)
        data_list.append(eeg_arr)
        label_list.append(lbl)
    return np.stack(data_list, axis=0), np.array(label_list)

# ------ Paths & seeds ------
dataset_pkl_path  = '/home/uceerjp/He/eeg_trials_dataset.pkl'
results_pkl_path  = '/home/uceerjp/He/lso_results.pkl'
tf.random.set_seed(12345)
np.random.seed(12345)
seed_everything(12345)

# ------ Enable GPU memory growth ------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ------ 1) LOAD (OR INIT) RESULTS ------
if os.path.exists(results_pkl_path):
    with open(results_pkl_path, 'rb') as f:
        results = pickle.load(f)
    print(f"Loaded existing results for {len(results)} subjects; will skip them.")
else:
    results = {}
    print("No existing results found; starting fresh.")

# ------ 2) LOAD DATASET ------
print("Loading EEG trials dataset ...")
all_data, subject_numbers = load_eeg_trials_dataset(dataset_pkl_path)
print(f"Dataset contains {len(all_data)} trials from subjects: {subject_numbers}")

# ------ 3) CENTER-CROP TO MINIMUM LENGTH ------
min_times = min(tr.x.shape[1] for tr in all_data)
print(f"[INFO] Cropping all trials to {min_times} timepoints.")
for tr in all_data:
    chans, times = tr.x.shape
    if times > min_times:
        start = (times - min_times) // 2
        tr.x = tr.x[:, start:start + min_times]

# ------ 4) GROUP BY SUBJECT ------
trials_by_subject = {}
for tr in all_data:
    trials_by_subject.setdefault(tr.subject, []).append(tr)
subject_numbers = sorted(trials_by_subject.keys())

# ------ 5) LOSO LOOP ------
for test_subj in subject_numbers:
    key = f"Subject_{test_subj}"
    if key in results:
        print(f"Skipping {key} (already done).")
        continue

    print(f"\n===== LOSO: Test Subject {test_subj} =====")
    # prepare train/test splits
    train_trials = [
        tr for subj, trials in trials_by_subject.items()
        if subj != test_subj for tr in trials
    ]
    test_trials = trials_by_subject[test_subj]

    X_train, y_train = prepare_data_for_eegnet(train_trials)
    X_test,  y_test  = prepare_data_for_eegnet(test_trials)

    print(f"Train: {X_train.shape[0]} trials (subs != {test_subj}), "
          f"Test: {X_test.shape[0]} trials (sub {test_subj})")

    # one-hot encode
    enc = OneHotEncoder(sparse_output=False)
    y_train_oh = enc.fit_transform(y_train.reshape(-1, 1))
    y_test_oh  = enc.transform(y_test.reshape(-1, 1))

    # model setup under MirroredStrategy
    ram_before = get_ram_usage()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        nb_chans = X_train.shape[1]
        nb_samps = X_train.shape[2]
        model = EEGNet(
            nb_classes=2, Chans=nb_chans, Samples=nb_samps,
            dropoutRate=0.5, kernLength=32, F1=16, D=2, F2=32,
            dropoutType='Dropout'
        )
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(1e-3),
            metrics=['accuracy']
        )
    ram_model = get_ram_usage() - ram_before

    # train
    print("Training ...")
    t0 = time()
    hist = model.fit(X_train, y_train_oh, batch_size=32, epochs=25, verbose=2)
    train_time = time() - t0

    # test
    print("Evaluating ...")
    t1 = time()
    preds = model.predict(X_test).argmax(axis=-1)
    inf_time = time() - t1
    acc = (preds == y_test).mean()

    # record results
    results[key] = {
        "train_time": train_time,
        "best_accuracy": acc,
        "inference_time_per_trial": inf_time / X_test.shape[0],
        "model_parameters": model.count_params(),
        "ram_model_usage": ram_model
    }
    print(f"{key}: acc={acc:.4f}, train={train_time:.1f}s, "
          f"inf/trial={inf_time/X_test.shape[0]:.3f}s")

    # clear session & GC
    tf.keras.backend.clear_session()
    gc.collect()

    # save intermediate results
    with open(results_pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {results_pkl_path} "
          f"(completed {len(results)}/{len(subject_numbers)})")

# ------ 6) FINAL SUMMARY ------
print("\n===== LOSO SUMMARY =====")
for subj_key in sorted(results):
    res = results[subj_key]
    print(
        f"{subj_key}: train_time={res['train_time']:.1f}s, "
        f"acc={res['best_accuracy']:.2%}, "
        f"params={res['model_parameters']}, "
        f"ram={res['ram_model_usage']:.1f}MB, "
        f"inference_per_trial={res['inference_time_per_trial']:.3f}s"
    )
