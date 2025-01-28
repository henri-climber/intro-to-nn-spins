# Spins Project: Classifying Snapshots of Spins with CNNs

Einführung in neuronale Netzwerke für Studierende der Physik

## Project Overview

This project explores the application of Convolutional Neural Networks (CNNs) to classify high-resolution snapshots of
spins in a 2D optical lattice, obtained from experiments simulating the Fermi-Hubbard model. The objective is to
identify the theoretical model that best matches experimental data at varying doping levels.

Key Results:

- Combining experimental data and string theories improved classification accuracy.
- Experimental data aligns more closely with geometric string theory than π-flux theory.

---

## Dataset

Snapshots are 10x10 grayscale images representing spatial configurations of spins and atoms:

- Classes: Experimental data, String Theory, π-Flux
- Labels: -1, 0, 1
- Dataset distribution: Train (70%), Validation (10%), Test (20%)

---

## Key Files

- **`DataLoader.py`**: Handles loading and preprocessing of the dataset.
- **`ModelBuilder.py`**: Defines the CNN architectures for training and experimentation.
- **`ModelTrainer.py`**: Includes training loops, hyperparameter tuning, and evaluation.
- **`check_out_snapshots.py`**: Provides utilities for visualizing the data snapshots.
- **`main.py`**: The main entry point for running experiments.
- **Notebooks**:
    - `edit_neptune_data.ipynb`: Neptune logging utilities.
    - `eval*.ipynb`: Evaluation and result visualization scripts.
    - `model_testing.ipynb`: Architecture testing and experimentation.
    - `test.ipynb`: Additional exploratory analysis.

---

## Results

- Approach 1: Three-class classification achieved 47% accuracy (challenges distinguishing string theory and experimental
  data).
- Approach 2: Combined "Experimental + String Theory" vs. π-Flux classification achieved ~70% accuracy.
- Approach 3: Classifying experimental data using trained String Theory and π-Flux models showed a bias towards String
  Theory (60%).

---

## How to Run

1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/your-repo/spins-project.git
   cd spins-project
   pip install -r requirements.txt