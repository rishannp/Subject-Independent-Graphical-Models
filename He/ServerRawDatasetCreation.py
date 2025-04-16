#!/usr/bin/env python3

import os
import pickle
import numpy as np
from tqdm import tqdm
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from torch_geometric.data import Data

# ---------------------------
# CONFIG
# ---------------------------
server_dir = '/home/uceerjp/He/'
cache_path = os.path.join(server_dir, 'eeg_trials_dataset.pkl')
max_workers = 4  # Adjust based on available cores

# ---------------------------
# Parallel EEG Trials Dataset Builder
# ---------------------------
def process_pkl_file(filepath):
    filename = os.path.basename(filepath)
    # Assume filenames follow the format e.g., "S01_xyz.pkl"
    subject_id = int(filename.split('_')[0][1:])
    print(f"[THREAD] Processing {filename}")

    try:
        with open(filepath, 'rb') as f:
            bci = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {filename}: {e}")
        return []

    trials = []
    eeg_data = bci['data']
    meta = bci['TrialData']  # Contains metadata for each trial

    for i in range(len(eeg_data)):
        label = meta[i].get('targetnumber')
        if label not in [1, 2]:
            continue

        # Get the raw EEG time series for this trial (n_channels x n_times)
        eeg = np.array(eeg_data[i])
        
        # Create a Data object to store the trial, its metadata, label and subject id.
        trial = Data(
            x=torch.tensor(eeg, dtype=torch.float32),
            y=torch.tensor(label - 1),  # Adjust label if needed
            meta=meta[i],               # Store the whole metadata dictionary
            subject=subject_id
        )
        trials.append(trial)

    return trials

# ---------------------------
# Main: Load or Generate EEG Trials Dataset
# ---------------------------
if os.path.exists(cache_path):
    print(f"\n[CACHE] Found existing EEG trials dataset. Loading from: {cache_path}")
    with open(cache_path, 'rb') as f:
        all_data, subject_numbers = pickle.load(f)
else:
    print(f"\n[INFO] Generating EEG trials dataset in parallel...")
    all_files = sorted([f for f in os.listdir(server_dir) if f.endswith('.pkl') and f.startswith('S')])
    all_paths = [os.path.join(server_dir, f) for f in all_files]

    all_data = []
    subject_numbers = set()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pkl_file, path): path for path in all_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel EEG trials generation"):
            filename = os.path.basename(futures[future])
            try:
                result = future.result()
                all_data.extend(result)
                if result:
                    subject_numbers.add(result[0].subject)
            except Exception as e:
                print(f"[ERROR] Error processing {filename}: {e}")
                traceback.print_exc()

    subject_numbers = sorted(subject_numbers)
    with open(cache_path, 'wb') as f:
        pickle.dump((all_data, subject_numbers), f)
    print(f"\n✅ Saved {len(all_data)} EEG trials from {len(subject_numbers)} subjects to {cache_path}")

print(f"\n[INFO] Ready to train with {len(all_data)} EEG trials from {len(subject_numbers)} subjects.")
