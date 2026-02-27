# models/evaluation.py

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from preprocessing.uci_loader import UCILoader
import os


class ModelEvaluator:

    def __init__(self,
                 model_path="models/gait_embedding_model.pth",
                 uci_path="data/raw/uci/UCI HAR Dataset",
                 synthetic_path="data/raw/synthetic/generated_windows"):

        self.model_path = model_path
        self.uci_path = uci_path
        self.synthetic_path = synthetic_path

    def load_synthetic_data(self):
        windows = []
        subjects = []

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

        return np.array(windows), np.array(subjects)

    def evaluate(self, embedding_model):

        print("Loading evaluation data...")

        loader = UCILoader(self.uci_path)
        uci_windows, uci_subjects = loader.load_all()

        syn_windows, syn_subjects = self.load_synthetic_data()

        all_windows = np.concatenate([uci_windows, syn_windows], axis=0)
        all_subjects = np.concatenate([uci_subjects, syn_subjects], axis=0)

        print("Total identities:", len(np.unique(all_subjects)))

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            all_windows, all_subjects, test_size=0.3, random_state=42
        )

        # Generate embeddings
        embedding_model.eval()

        def get_embeddings(data):
            embeddings = []
            with torch.no_grad():
                for sample in data:
                    sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
                    emb = embedding_model(sample)
                    embeddings.append(emb.squeeze(0).numpy())
            return np.array(embeddings)

        train_embeddings = get_embeddings(X_train)
        test_embeddings = get_embeddings(X_test)

        # Simple KNN classifier for identity accuracy
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(train_embeddings, y_train)

        predictions = knn.predict(test_embeddings)

        acc = accuracy_score(y_test, predictions)

        print("Identification Accuracy:", round(acc * 100, 2), "%")

        return acc


if __name__ == "__main__":
    from models.embedding_model import GaitEmbeddingModel

    model = GaitEmbeddingModel()
    model.load_state_dict(torch.load("models/gait_embedding_model.pth"))

    evaluator = ModelEvaluator()
    evaluator.evaluate(model)