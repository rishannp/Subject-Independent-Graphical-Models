# rPLV-GAT: Calibration-Free Motor Imagery BCI via Graph-Based Transfer Learning

![License](https://img.shields.io/badge/license-MIT-blue.svg)

This repository accompanies the IEEE EMBS NER 2025 paper:

**"Towards Calibration-Free Motor Imagery BCIs via Graph-Based Transfer Learning"**  
*R. Patel, Z. Zhu, B. Bryson, T. Carlson, D. Jiang, A. Demosthenous*

---

## 🧠 Overview

EEG-based Brain–Computer Interfaces (BCIs) remain limited by non-stationary signals across time and users. This work proposes a calibration-free, subject-independent framework for motor imagery classification using:

- **Phase-Locking Value (PLV)** connectivity matrices
- **Graph Attention Networks (GAT)**
- **Leave-One-Subject-Out Cross-Validation (LOSOCV)** evaluation

We demonstrate competitive or superior performance to seven baseline models on two longitudinal datasets (ALS and healthy participants) — without any subject-specific calibration.

---


---

## 🧪 Methodology

- **Graph Construction**: PLV matrices computed per trial across EEG channels
- **Node Features**: Pairwise synchrony across trials, quantified using Coefficient of Variation (CV)
- **Model**: 3-layer GATv2 with global pooling and fully connected classification
- **Training**: LOSOCV protocol with no target subject calibration

---

## 📊 Datasets

### ✅ Dataset 1: ALS Cohort (Patel et al. https://doi.org/10.5522/04/28156016.v1)
- 8 participants, 4 sessions each
- 19-channel EEG
- Cursor control MI task

### ✅ Dataset 2: Healthy Cohort ([Stieger et al.](https://physionet.org/content/mindful-mi-eeg/1.0.0/))
- 62 participants (mindfulness + control)
- 64-channel EEG, 6–10 sessions
- Used only left/right MI trials

> 📎 *Note*: Due to licensing, users must download raw datasets from original sources. Scripts to preprocess into PLV format are provided.

---

If you use this work, please cite: 
@inproceedings{patel2025,
  title={Towards Calibration-Free Motor Imagery BCIs via Graph-Based Transfer Learning},
  author={Patel, Rishan and Zhu, Ziyue and Bryson, Barney and Carlson, Tom and Jiang, Dai and Demosthenous, Andreas},
  booktitle={IEEE Engineering in Medicine and Biology Society Neural Engineering Conference (NER)},
  year={2025}
}

