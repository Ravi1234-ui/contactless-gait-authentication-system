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
                 model_save_path="models/gait_embedding_model.pth"):

        self.uci_path = uci_path
        self.synthetic_path = synthetic_path
        self.model_save_path = model_save_path

    def load_synthetic_data(self):
        """
        Load all synthetic identity windows
        """
        windows = []
        subjects = []

        synthetic_files = os.listdir(self.synthetic_path)

        synthetic_id_counter = 1000  # Start synthetic IDs from 1000+

        for file in synthetic_files:
            if file.endswith(".npy"):
                data = np.load(os.path.join(self.synthetic_path, file))

                identity_id = synthetic_id_counter
                synthetic_id_counter += 1

                for window in data:
                    windows.append(window)
                    subjects.append(identity_id)

        return np.array(windows), np.array(subjects)

    def run(self, epochs=5):

        print("Loading UCI data...")
        loader = UCILoader(self.uci_path)
        uci_windows, uci_subjects = loader.load_all()

        print("UCI windows:", uci_windows.shape)

        print("Loading Synthetic data...")
        syn_windows, syn_subjects = self.load_synthetic_data()

        print("Synthetic windows:", syn_windows.shape)

        # Combine
        all_windows = np.concatenate([uci_windows, syn_windows], axis=0)
        all_subjects = np.concatenate([uci_subjects, syn_subjects], axis=0)

        print("Total windows:", all_windows.shape)

        # Train embedding model
        trainer = TripletTrainer()
        model = trainer.train(all_windows, all_subjects, epochs=epochs)

        # Save model
        torch.save(model.state_dict(), self.model_save_path)

        print(f"Model saved to {self.model_save_path}")


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run(epochs=20)