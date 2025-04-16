#%% ALS DATASET

import numpy as np
import matplotlib.pyplot as plt
from mne.viz import Brain
from mne.datasets import fetch_fsaverage
from matplotlib.cm import coolwarm
from matplotlib.colors import Normalize
import os
import pyvista as pv

# -----------------------------
# Helper Functions
# -----------------------------

def normalize_and_project(coords, scale=95, push_out=15):
    """Normalize coords to a sphere, then push them outward slightly."""
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    unit = coords / norms
    return unit * scale + unit * push_out

def make_arc(start, end, curvature=20):
    """Generate a 3-point arc (quadratic Bezier-style) between electrodes."""
    midpoint = (start + end) / 2
    direction = midpoint / np.linalg.norm(midpoint)
    arc_point = midpoint + direction * curvature
    return np.array([start, arc_point, end])

def rotate_coords(coords, angle_deg=90):
    """Rotate coordinates around Z-axis (to fix orientation)."""
    angle_rad = np.deg2rad(angle_deg)
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    return coords @ R.T  # Apply rotation


# -----------------------------
# Step 1: Load data
# -----------------------------

als_plv = np.load("alsplvmatrix.npy")

electrode_labels = [
    'FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'O2'
]

xyz_als = np.array([
    [0.950, 0.309, -0.0349], [0.950, -0.309, -0.0349], [0.587, 0.809, -0.0349],
    [0.673, 0.545, 0.500], [0.719, 0.000, 0.695], [0.673, -0.545, 0.500],
    [0.587, -0.809, -0.0349], [6.120e-17, 0.999, -0.0349], [4.400e-17, 0.719, 0.695],
    [0.000, 0.000, 1.000], [4.400e-17, -0.719, 0.695], [6.120e-17, -0.999, -0.0349],
    [-0.587, 0.809, -0.0349], [-0.673, 0.545, 0.500], [-0.719, -8.810e-17, 0.695],
    [-0.673, -0.545, 0.500], [-0.587, -0.809, -0.0349], [-0.950, 0.309, -0.0349],
    [-0.950, -0.309, -0.0349]
])


# -----------------------------
# Step 3: Load fsaverage brain
# -----------------------------

fs_dir = fetch_fsaverage()
subjects_dir = os.path.dirname(fs_dir)

brain = Brain('fsaverage', 'both', 'pial',
              subjects_dir=subjects_dir, background='white',
              cortex='classic', size=(1200, 900))

# -----------------------------
# Step 4: Plot electrodes and labels
# -----------------------------

rotated_xyz = rotate_coords(xyz_als, angle_deg=90)  # rotate counter-clockwise
projected_coords = normalize_and_project(rotated_xyz)
plotter = brain._renderer.plotter

# Use a fixed color or based on anatomical region (optional enhancement)
node_color = 'dodgerblue'

for i, (label, coord) in enumerate(zip(electrode_labels, projected_coords)):
    hemi = 'lh' if coord[0] < 0 else 'rh'
    
    # Add node sphere
    brain.add_foci(coord, coords_as_verts=False, hemi=hemi,
                   color=node_color, scale_factor=1.5,
                   name=f"elec_{i}")
    
    # Add floating label
    label_pos = coord + np.array([0, 0, 5])
    plotter.add_point_labels(
        points=np.array([label_pos]),
        labels=[label],
        font_size=10,
        text_color='black',
        point_color=None,
        shape=None,
        always_visible=True
    )

# -----------------------------
# Step 5: Draw PLV Arcs
# -----------------------------
# Define colormap range from actual PLV values, not the thresholded one
norm = Normalize(vmin=np.min(als_plv[np.triu_indices_from(als_plv, k=1)]),
                 vmax=np.max(als_plv[np.triu_indices_from(als_plv, k=1)]))
colormap = coolwarm.reversed()

# Show only weakest 50% (bottom half of PLV strengths)
low_percent = 100
strength_thresh = np.percentile(als_plv[np.triu_indices_from(als_plv, k=1)], low_percent)

for i in range(len(projected_coords)):
    for j in range(i + 1, len(projected_coords)):
        strength = als_plv[i, j]
        if strength <= strength_thresh:
            arc_pts = make_arc(projected_coords[i], projected_coords[j])
            spline = pv.Spline(arc_pts, n_points=100)
            color_rgb = colormap(norm(strength))[:3]
            plotter.add_mesh(spline, color=color_rgb, line_width=6.0)


# -----------------------------
# Step 6: Save
# -----------------------------

brain.show_view('dorsal')  # Try 'lateral' or 'medial' for alternatives
brain_image_path = "als_plv_brain_arcs_labeled.png"
brain.save_image(brain_image_path)

print(f"\n✅ Saved brain visualization with labeled nodes and arcs to: {brain_image_path}")


#%% STIEGER DATASET
import numpy as np
import matplotlib.pyplot as plt
from mne.viz import Brain
from mne.datasets import fetch_fsaverage
from matplotlib.cm import coolwarm
from matplotlib.colors import Normalize
import os
import pyvista as pv
import mne

# -----------------------------
# Helper Functions
# -----------------------------

def normalize_and_project(coords, scale=95, push_out=15):
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    unit = coords / norms
    return unit * scale + unit * push_out

def make_arc(start, end, curvature=20):
    midpoint = (start + end) / 2
    direction = midpoint / np.linalg.norm(midpoint)
    arc_point = midpoint + direction * curvature
    return np.array([start, arc_point, end])

def rotate_coords(coords, angle_deg=90, axis='z'):
    angle_rad = np.deg2rad(angle_deg)
    if axis == 'z':
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    return coords @ R.T

# -----------------------------
# Use MNE montage and patch missing coordinates
# -----------------------------
labels = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2'
]

manual_coords = {
    'FP1': [0, 85, 50], 'FPZ': [0, 90, 60], 'FP2': [0, 85, -50], 'FZ': [20, 60, 0],
    'FCZ': [30, 40, 0], 'CZ': [40, 0, 0], 'CPZ': [30, -40, 0], 'PZ': [20, -60, 0],
    'POZ': [10, -75, 0], 'CB1': [-10, -85, 10], 'OZ': [0, -90, 30]
}

montage = mne.channels.make_standard_montage('standard_1005')
info = mne.create_info(ch_names=labels, sfreq=256, ch_types='eeg')
info.set_montage(montage, on_missing='ignore')

mne_label_lookup = {ch.lower(): ch for ch in montage.ch_names}
existing_coords = {
    ch['ch_name'].upper(): ch['loc'][:3] * 1000
    for ch in info['chs'] if ch['ch_name'].lower() in mne_label_lookup
}
manual_coords_upper = {k.upper(): v for k, v in manual_coords.items()}

full_coords = []
for label in labels:
    label_up = label.upper()
    if label_up in existing_coords:
        full_coords.append(existing_coords[label_up])
    elif label_up in manual_coords_upper:
        full_coords.append(manual_coords_upper[label_up])
    else:
        print(f"⚠️ Missing coordinate for label: {label}")
        full_coords.append([np.nan, np.nan, np.nan])

full_coords = np.array(full_coords)

# -----------------------------
# Load PLV and Rotate
# -----------------------------
ste_plv = np.load("stiegerplvmatrix.npy")
ste_plv = ste_plv[:60, :60]
projected_coords = normalize_and_project(full_coords)

# -----------------------------
# Plot on Brain Surface
# -----------------------------
fs_dir = fetch_fsaverage()
subjects_dir = os.path.dirname(fs_dir)
brain = Brain('fsaverage', 'both', 'pial', subjects_dir=subjects_dir,
              background='white', cortex='classic', size=(1200, 900))
plotter = brain._renderer.plotter

for i, (label, coord) in enumerate(zip(labels[:60], projected_coords)):
    hemi = 'lh' if coord[0] < 0 else 'rh'
    brain.add_foci(coord, coords_as_verts=False, hemi=hemi,
                   color='dodgerblue', scale_factor=1.2,
                   name=f"elec_{i}")
    label_pos = coord + np.array([0, 0, 5])
    plotter.add_point_labels([label_pos], [label],
                             font_size=9, text_color='black',
                             point_color=None, shape=None,
                             always_visible=True)

# -----------------------------
# Draw Arcs for Weakest 50%
# -----------------------------
colormap = coolwarm.reversed()
triu = np.triu_indices(60, k=1)
norm = Normalize(vmin=np.min(ste_plv[triu]), vmax=np.max(ste_plv[triu]))
threshold = np.percentile(ste_plv[triu], 100)

for i in range(60):
    for j in range(i + 1, 60):
        strength = ste_plv[i, j]
        if strength <= threshold:
            arc_pts = make_arc(projected_coords[i], projected_coords[j])
            spline = pv.Spline(arc_pts, n_points=100)
            plotter.add_mesh(spline, color=colormap(norm(strength))[:3], line_width=4.0)

# -----------------------------
# Final View
# -----------------------------
brain.show_view('dorsal')
brain.save_image("stieger_brain_weak_connectivity.png")
print("\n✅ Saved: stieger_brain_weak_connectivity.png")
