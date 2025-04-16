import os
import pickle
import numpy as np
from tqdm import tqdm

# ---------------------------
# CONFIG
# ---------------------------
server_dir = r'C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Subject-Independent Graphical Models\Subject-Independent-Graphical-Models\He\He_Dataset'
cache_path = os.path.join(server_dir, 'plv_graph_dataset.pkl')

# ---------------------------
# Load Graph Dataset
# ---------------------------
if os.path.exists(cache_path):
    print(f"\n[CACHE] Found PLV graph dataset at: {cache_path}")
    with open(cache_path, 'rb') as f:
        all_data, subject_numbers = pickle.load(f)
else:
    raise FileNotFoundError(f"[ERROR] PLV graph dataset not found at: {cache_path}")

print(f"\n[INFO] Dataset loaded: {len(all_data)} trials from {len(subject_numbers)} subjects.")
