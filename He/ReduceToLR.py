import os
import numpy as np
import scipy.io
import pickle

data_dir = r'C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Subject-Independent Graphical Models\Subject-Independent-Graphical-Models\He\He_Dataset'

# --- Helper: convert MATLAB struct to dict
def matlab_struct_to_dict(mat_struct):
    result = {}
    for field in mat_struct._fieldnames:
        val = getattr(mat_struct, field)
        if isinstance(val, np.ndarray) and val.dtype == np.object_:
            result[field] = [matlab_struct_to_dict(v) if hasattr(v, '_fieldnames') else v for v in val]
        elif hasattr(val, '_fieldnames'):
            result[field] = matlab_struct_to_dict(val)
        else:
            result[field] = val
    return result

# --- Helper: filter by indices
def select_by_indices(array_like, indices):
    if isinstance(array_like, list):
        return [array_like[i] for i in indices]
    elif isinstance(array_like, np.ndarray):
        return array_like[indices]
    else:
        raise TypeError("Unsupported data type for indexing.")

# --- Main loop over .mat files
mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
print(f"[INFO] Found {len(mat_files)} .mat files to process.")

for mat_filename in mat_files:
    try:
        file_path = os.path.join(data_dir, mat_filename)
        print(f"\n[PROCESSING] {mat_filename}")

        mat_data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        bci_data = mat_data['BCI']

        if isinstance(bci_data, np.ndarray):
            bci_list = [matlab_struct_to_dict(entry) for entry in bci_data]
        elif hasattr(bci_data, '_fieldnames'):
            bci_list = [matlab_struct_to_dict(bci_data)]
        else:
            print(f"[SKIP] Unexpected BCI structure in: {mat_filename}")
            continue

        bci = bci_list[0]
        trial_data = bci['TrialData']
        left_right_indices = [i for i, trial in enumerate(trial_data) if trial.get('targetnumber') in [1, 2]]

        if len(left_right_indices) == 0:
            print(f"[SKIP] No Left/Right MI trials in: {mat_filename}")
            continue

        filtered_data = select_by_indices(bci['data'], left_right_indices)
        filtered_positionx = select_by_indices(bci['positionx'], left_right_indices)
        filtered_positiony = select_by_indices(bci['positiony'], left_right_indices)
        filtered_time = select_by_indices(bci['time'], left_right_indices)
        filtered_trialdata = [trial_data[i] for i in left_right_indices]

        bci_filtered = {
            'data': filtered_data,
            'positionx': filtered_positionx,
            'positiony': filtered_positiony,
            'time': filtered_time,
            'TrialData': filtered_trialdata,
            'chaninfo': bci['chaninfo'],
            'SRATE': bci['SRATE'],
            'metadata': bci['metadata']
        }

        save_path = file_path.replace('.mat', '.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(bci_filtered, f)

        print(f"[SAVED] Filtered file saved as: {os.path.basename(save_path)}")

        # Now delete the original .mat
        os.remove(file_path)
        print(f"[DELETED] Original .mat file deleted: {mat_filename}")

    except Exception as e:
        print(f"[ERROR] Failed to process {mat_filename}: {e}")
