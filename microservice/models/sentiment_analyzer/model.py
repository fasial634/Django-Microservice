import torch
import torch.nn as nn
from .config import VOC_SIZE, EMBED_DIM, HIDDEN_DIM, MAX_LENGTH

class LSTMModel(nn.Module):
    def __init__(self, voc_size: int, embed_dim: int, hidden_dim: int, max_length: int):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=voc_size + 1, embedding_dim=embed_dim)
        self.lstm1 = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :])  # Use the last time step
        x = self.sigmoid(x)
        return x