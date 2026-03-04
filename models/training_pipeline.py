# models/training_pipeline.py

import os
import numpy as np
import torch
from preprocessing.uci_loader import UCILoader
from models.triplet_loss import TripletTrainer


class TrainingPipeline:

    def __init__(self,
                 uci_path="data/raw/uci/UCI HAR Dataset",
                 synthetic_path="data/raw/synthetic/generated_windows",
                 model_save_path="models/gait_embedding_model.pth",
                 norm_save_path="models/normalization_params.npz"):

        self.uci_path = uci_path
        self.synthetic_path = synthetic_path
        self.model_save_path = model_save_path
        self.norm_save_path = norm_save_path

        # Ensure models directory exists
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

    # ----------------------------------
    # LOAD SYNTHETIC DATA
    # ----------------------------------
    def load_synthetic_data(self):

        windows = []
        subjects = []

        if not os.path.exists(self.synthetic_path):
            print("Synthetic folder not found.")
            return np.empty((0, 128, 6)), np.empty((0,))

        synthetic_files = os.listdir(self.synthetic_path)
        synthetic_id_counter = 1000

        for file in synthetic_files:
            if file.endswith(".npy"):

                data = np.load(os.path.join(self.synthetic_path, file))

                identity_id = synthetic_id_counter
                synthetic_id_counter += 1

                for window in data:
                    windows.append(window)
                    subjects.append(identity_id)

        if len(windows) == 0:
            return np.empty((0, 128, 6)), np.empty((0,))

        return np.array(windows, dtype=np.float32), np.array(subjects)

    # ----------------------------------
    # GLOBAL NORMALIZATION
    # ----------------------------------
    def compute_global_normalization(self, windows):

        # windows shape = (N, 128, 6)

        mean = np.mean(windows, axis=(0, 1)).astype(np.float32)
        std = np.std(windows, axis=(0, 1)).astype(np.float32) + 1e-8

        # Save normalization parameters
        np.savez(self.norm_save_path, mean=mean, std=std)

        print("\nNormalization Parameters:")
        print("Mean:", mean)
        print("Std:", std)

        print("\nSaved normalization parameters to:", self.norm_save_path)

        return mean, std

    def apply_normalization(self, windows, mean, std):
        return ((windows - mean) / std).astype(np.float32)

    # ----------------------------------
    # RUN TRAINING
    # ----------------------------------
    def run(self, epochs=20):

        print("\nLoading UCI data...")
        loader = UCILoader(self.uci_path)
        uci_windows, uci_subjects = loader.load_all()

        uci_windows = uci_windows.astype(np.float32)

        print("UCI windows:", uci_windows.shape)

        print("\nLoading Synthetic data...")
        syn_windows, syn_subjects = self.load_synthetic_data()

        print("Synthetic windows:", syn_windows.shape)

        # Combine datasets safely
        if len(syn_windows) > 0:
            all_windows = np.concatenate([uci_windows, syn_windows], axis=0)
            all_subjects = np.concatenate([uci_subjects, syn_subjects], axis=0)
        else:
            all_windows = uci_windows
            all_subjects = uci_subjects

        print("\nTotal windows:", all_windows.shape)

        # ----------------------------------
        # GLOBAL NORMALIZATION
        # ----------------------------------
        mean, std = self.compute_global_normalization(all_windows)

        all_windows = self.apply_normalization(all_windows, mean, std)

        # ----------------------------------
        # SHUFFLE DATASET (important for triplet training)
        # ----------------------------------
        indices = np.random.permutation(len(all_windows))
        all_windows = all_windows[indices]
        all_subjects = all_subjects[indices]

        # ----------------------------------
        # TRAIN EMBEDDING MODEL
        # ----------------------------------
        trainer = TripletTrainer()

        model = trainer.train(
            all_windows,
            all_subjects,
            epochs=epochs
        )

        torch.save(model.state_dict(), self.model_save_path)

        print("\nModel saved to:", self.model_save_path)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run(epochs=20)