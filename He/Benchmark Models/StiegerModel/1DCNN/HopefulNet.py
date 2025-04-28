import os
import pickle
import numpy as np
import psutil
import gc
from time import time

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.seed import seed_everything

# Set seeds for reproducibility
tf.random.set_seed(12345)
np.random.seed(12345)
seed_everything(12345)

# GPU memory growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# --- Utilities ---
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
    eeg = trial.x.numpy().T  # (samples, channels)
    label = int(trial.y.numpy())
    return eeg.astype(np.float32), label

def data_generator(trials, encoder):
    for trial in trials:
        eeg_arr, label = trial_to_numpy(trial)
        label_oh = encoder.transform([[label]])[0]
        yield eeg_arr, label_oh

def prepare_tf_dataset(trials, encoder, batch_size, shuffle=True, channels=None):
    if channels is None:
        first_eeg, _ = trial_to_numpy(trials[0])
        channels = first_eeg.shape[1]

    ds = tf.data.Dataset.from_generator(
        lambda: data_generator(trials, encoder),
        output_signature=(
            tf.TensorSpec(shape=(None, channels), dtype=tf.float32),
            tf.TensorSpec(shape=(2,), dtype=tf.float32)
        )
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(trials))
    return ds.batch(batch_size)

class HopefullNet(tf.keras.Model):
    def __init__(self, inp_shape=(640, 2)):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(32, 20, activation='relu', padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv1D(32, 20, activation='relu', padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.sdrop1 = tf.keras.layers.SpatialDropout1D(0.5)

        self.conv3 = tf.keras.layers.Conv1D(32, 6, activation='relu', padding='same')
        self.pool1 = tf.keras.layers.AvgPool1D(2)

        self.conv4 = tf.keras.layers.Conv1D(32, 6, activation='relu', padding='same')
        self.sdrop2 = tf.keras.layers.SpatialDropout1D(0.5)

        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()

        self.d1 = tf.keras.layers.Dense(296, activation='relu')
        self.do1 = tf.keras.layers.Dropout(0.5)

        self.d2 = tf.keras.layers.Dense(148, activation='relu')
        self.do2 = tf.keras.layers.Dropout(0.5)

        self.d3 = tf.keras.layers.Dense(74, activation='relu')
        self.do3 = tf.keras.layers.Dropout(0.5)

        self.out = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.sdrop1(x, training=training)

        x = self.conv3(x)
        x = self.pool1(x)

        x = self.conv4(x)
        x = self.sdrop2(x, training=training)

        x = self.global_pool(x)

        x = self.d1(x)
        x = self.do1(x, training=training)

        x = self.d2(x)
        x = self.do2(x, training=training)

        x = self.d3(x)
        x = self.do3(x, training=training)

        return self.out(x)

# --- Training Script ---
if __name__ == '__main__':
    print("[INFO] Starting Script...")

    dataset_pkl = '/home/uceerjp/He/eeg_trials_dataset.pkl'
    save_path = '/home/uceerjp/StiegerModel/1DCNN/1DCNN_results.pkl'

    all_data, subjects = load_pickle_dataset(dataset_pkl)
    print(f"[INFO] Loaded {len(all_data)} trials from subjects: {subjects}")

    min_len = min(t.x.shape[1] for t in all_data)
    print(f"[INFO] Center cropping to {min_len} samples.")
    all_data = center_crop_trials(all_data, min_len)

    by_subj = {}
    for t in all_data:
        by_subj.setdefault(t.subject, []).append(t)
    subjects = sorted(by_subj)

    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            results = pickle.load(f)
        print(f"[INFO] Loaded existing results with {len(results)} subjects.")
    else:
        results = {}

    for test_subj in subjects:
        subj_key = f'Subject_{test_subj}'
        if subj_key in results:
            print(f"[INFO] {subj_key} already completed. Skipping...")
            continue

        print(f"\n===== LOSO: Test Subject {test_subj} =====")
        train_trials = [t for s, ts in by_subj.items() if s != test_subj for t in ts]
        test_trials = by_subj[test_subj]

        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit([[0], [1]])

        channels = train_trials[0].x.shape[0]
        samples = train_trials[0].x.shape[1]

        train_ds = prepare_tf_dataset(train_trials, encoder, batch_size=32, shuffle=True, channels=channels)
        train_ds = train_ds.repeat()

        # True labels are extracted independently
        true_labels = np.array([int(trial.y.numpy()) for trial in test_trials])

        test_ds = prepare_tf_dataset(test_trials, encoder, batch_size=32, shuffle=False, channels=channels)

        ram_before = get_ram_usage()
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = HopefullNet(inp_shape=(samples, channels))
            _ = model(tf.zeros((1, samples, channels)))
            model.compile(optimizer=Adam(1e-3),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        params = model.count_params()
        ram_used = get_ram_usage() - ram_before

        print(f"[INFO] Training on subject {test_subj}...")
        start_t = time()
        steps_per_epoch = len(train_trials) // 32
        if len(train_trials) % 32 != 0:
            steps_per_epoch += 1

        model.fit(train_ds, epochs=25, steps_per_epoch=steps_per_epoch, verbose=2)
        train_time = time() - start_t

        print(f"[INFO] Evaluating subject {test_subj}...")
        start_i = time()
        probs = model.predict(test_ds, verbose=0)
        inf_time = time() - start_i

        preds = np.argmax(probs, axis=-1)
        assert preds.shape == true_labels.shape, f"Shape mismatch: preds {preds.shape}, true {true_labels.shape}"

        acc = np.mean(preds == true_labels)

        results[subj_key] = {
            'accuracy': acc,
            'train_time': train_time,
            'inf_per_trial': inf_time / len(true_labels),
            'params': params,
            'ram_usage': ram_used
        }

        print(f"[RESULT] {subj_key}: Acc: {acc:.4f}, TrainTime: {train_time:.2f}s, RAM: {ram_used:.2f}MB")

        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"[INFO] Saved checkpoint after {subj_key}.")

        tf.keras.backend.clear_session()
        gc.collect()

    print("\n===== LOSO Summary =====")
    for subj, r in results.items():
        print(f"{subj}: acc={r['accuracy']*100:.2f}% | train={r['train_time']:.2f}s | "
              f"params={r['params']} | ram={r['ram_usage']:.2f}MB | "
              f"inf_trial={r['inf_per_trial']:.4f}s")
