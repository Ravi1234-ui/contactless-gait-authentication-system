# models/triplet_loss.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from models.embedding_model import GaitEmbeddingModel


class TripletGaitDataset(Dataset):
    """
    Creates triplets:
    (anchor, positive, negative)
    """

    def __init__(self, windows, subjects):
        self.windows = windows
        self.subjects = subjects
        self.unique_subjects = np.unique(subjects)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        anchor = self.windows[idx]
        anchor_label = self.subjects[idx]

        # Positive sample (same subject)
        pos_indices = np.where(self.subjects == anchor_label)[0]
        pos_idx = np.random.choice(pos_indices)
        positive = self.windows[pos_idx]

        # Negative sample (different subject)
        neg_subject = np.random.choice(
            self.unique_subjects[self.unique_subjects != anchor_label]
        )
        neg_indices = np.where(self.subjects == neg_subject)[0]
        neg_idx = np.random.choice(neg_indices)
        negative = self.windows[neg_idx]

        return (
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
            torch.tensor(negative, dtype=torch.float32),
        )


class TripletTrainer:
    def __init__(self, embedding_dim=128, margin=1.0, lr=0.001, device=None):
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = GaitEmbeddingModel(embedding_dim).to(self.device)
        self.criterion = nn.TripletMarginLoss(margin=margin)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, windows, subjects, epochs=10, batch_size=32):

        dataset = TripletGaitDataset(windows, subjects)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0

            for anchor, positive, negative in loader:
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                negative_emb = self.model(negative)

                loss = self.criterion(anchor_emb, positive_emb, negative_emb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        return self.model


if __name__ == "__main__":
    # Quick dummy test

    # Fake data
    windows = np.random.randn(200, 128, 6)
    subjects = np.random.randint(1, 6, size=200)

    trainer = TripletTrainer()
    trainer.train(windows, subjects, epochs=2)