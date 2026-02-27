# models/embedding_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaitEmbeddingModel(nn.Module):
    """
    CNN + BiLSTM model that produces
    128-dimensional gait embedding.

    Input shape: (batch_size, 128, 6)
    Output shape: (batch_size, 128)
    """

    def __init__(self, embedding_dim=128):
        super(GaitEmbeddingModel, self).__init__()

        # Convolution layers (extract motion patterns)
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)

        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(128)

        # BiLSTM for temporal gait rhythm
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layer to embedding
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        """
        x: (batch_size, 128, 6)
        """

        # Convert to (batch, channels, time)
        x = x.permute(0, 2, 1)

        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))

        # Convert back to (batch, time, features)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Take last timestep
        final_feature = lstm_out[:, -1, :]

        embedding = self.fc(final_feature)

        # Normalize embedding (important for cosine similarity)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


if __name__ == "__main__":
    # Quick test

    model = GaitEmbeddingModel()

    dummy_input = torch.randn(8, 128, 6)
    output = model(dummy_input)

    print("Output shape:", output.shape)