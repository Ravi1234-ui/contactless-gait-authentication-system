# authentication/register_and_authenticate.py

import torch
import numpy as np
from models.embedding_model import GaitEmbeddingModel
from authentication.similarity_engine import SimilarityEngine
from preprocessing.real_data_loader import RealWorldDataLoader


class GaitAuthenticator:

    def __init__(self,
                 model_path="models/gait_embedding_model.pth",
                 threshold=0.75):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load trained model
        self.model = GaitEmbeddingModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Similarity engine
        self.engine = SimilarityEngine(threshold=threshold)

        # Real-world loader
        self.loader = RealWorldDataLoader()

    # ----------------------------
    # Generate embedding from windows
    # ----------------------------
    def generate_embedding(self, windows):

        embeddings = []

        with torch.no_grad():
            for window in windows:
                tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
                emb = self.model(tensor)
                embeddings.append(emb.squeeze(0).cpu().numpy())

        embeddings = np.array(embeddings)

        # Average embedding across windows
        mean_embedding = np.mean(embeddings, axis=0)

        # Normalize again
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

        return torch.tensor(mean_embedding, dtype=torch.float32)

    # ----------------------------
    # Register all EMP_* folders
    # ----------------------------
    def register_employees(self):

        windows, labels = self.loader.load_all()

        unique_people = set(labels)

        for person in unique_people:
            if person.startswith("EMP_"):

                person_windows = windows[labels == person]

                embedding = self.generate_embedding(person_windows)

                self.engine.save_employee_embedding(person, embedding)

                print(f"Registered {person}")

        self.engine.load_registry()

    # ----------------------------
    # Authenticate UNKNOWN_TEST
    # ----------------------------
    def authenticate_unknown(self):

        windows, labels = self.loader.load_all()

        if "UNKNOWN_TEST" not in labels:
            print("No UNKNOWN_TEST folder found.")
            return

        unknown_windows = windows[labels == "UNKNOWN_TEST"]

        embedding = self.generate_embedding(unknown_windows)

        match_id, score, status = self.engine.authenticate(embedding)

        print("\n===== AUTHENTICATION RESULT =====")
        print("Best Match:", match_id)
        print("Similarity Score:", round(score, 4))
        print("Decision:", status)
        print("=================================")


# ----------------------------
# RUN SCRIPT
# ----------------------------
if __name__ == "__main__":

    authenticator = GaitAuthenticator(threshold=0.75)

    print("\nRegistering Employees...\n")
    authenticator.register_employees()

    print("\nAuthenticating UNKNOWN_TEST...\n")
    authenticator.authenticate_unknown()