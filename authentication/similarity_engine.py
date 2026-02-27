# authentication/similarity_engine.py

import torch
import numpy as np
import os


class SimilarityEngine:
    """
    Handles:
    - Storing employee embeddings
    - Comparing unknown embeddings
    - Access decision
    """

    def __init__(self, threshold=0.75, storage_path="data/processed/embeddings"):
        self.threshold = threshold
        self.storage_path = storage_path

        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

        self.registry = {}  # employee_id -> embedding tensor

    def save_employee_embedding(self, employee_id, embedding):
        """
        Save employee embedding to disk
        """
        path = os.path.join(self.storage_path, f"{employee_id}.pt")
        torch.save(embedding, path)

        self.registry[employee_id] = embedding

    def load_registry(self):
        """
        Load all saved employee embeddings
        """
        for file in os.listdir(self.storage_path):
            if file.endswith(".pt"):
                employee_id = file.replace(".pt", "")
                embedding = torch.load(
                    os.path.join(self.storage_path, file)
                )
                self.registry[employee_id] = embedding

    def cosine_similarity(self, emb1, emb2):
        return torch.nn.functional.cosine_similarity(
            emb1, emb2, dim=0
        ).item()

    def authenticate(self, unknown_embedding):
        """
        Compare unknown embedding with all employees.
        Returns:
            (matched_id, similarity, access_status)
        """

        if not self.registry:
            return None, 0.0, "NO REGISTERED EMPLOYEES"

        best_match = None
        best_score = -1

        for employee_id, stored_embedding in self.registry.items():
            score = self.cosine_similarity(
                unknown_embedding, stored_embedding
            )

            if score > best_score:
                best_score = score
                best_match = employee_id

        if best_score >= self.threshold:
            return best_match, best_score, "ACCESS GRANTED"
        else:
            return None, best_score, "ACCESS DENIED"


if __name__ == "__main__":
    # Quick test

    engine = SimilarityEngine(threshold=0.7)

    emb1 = torch.randn(128)
    emb1 = torch.nn.functional.normalize(emb1, dim=0)

    emb2 = torch.randn(128)
    emb2 = torch.nn.functional.normalize(emb2, dim=0)

    engine.save_employee_embedding("EMP001", emb1)
    engine.load_registry()

    match_id, score, status = engine.authenticate(emb1)

    print("Match:", match_id)
    print("Score:", score)
    print("Status:", status)