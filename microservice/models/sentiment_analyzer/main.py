import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
from collections import Counter
import pickle

voc_size = 20000
max_length = 100
embed_dim = 64
hidden_dim = 32
batch_size = 64
device = "cpu"
# Convert texts to sequences of indices
def text_to_sequence(texts, vocab, max_length):
    sequences = []
    for text in texts:
        tokens = text.lower().split()
        sequence = [vocab.get(token, 0) for token in tokens]  # 0 for out-of-vocabulary tokens
        sequences.append(sequence)
    return sequences


def pad_sequences(sequences, max_length, pad_value=0):
    padded = []
    for seq in sequences:
        if len(seq) < max_length:
            padded_seq = seq + [pad_value] * (max_length - len(seq))  # Pad with zeros
        else:
            padded_seq = seq[:max_length]  # Truncate if longer than max_length
        padded.append(padded_seq)
    return padded



# Define the PyTorch model
class LSTMModel(nn.Module):
    def __init__(self, voc_size, embed_dim, hidden_dim, max_length):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=voc_size + 1, embedding_dim=embed_dim)
        self.lstm1 = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :])  # Use the last time step
        x = self.sigmoid(x)
        return x




model = LSTMModel(voc_size=20000, embed_dim=64, hidden_dim=32, max_length=100)
model.load_state_dict(torch.load('saved_models/model_epoch_6_loss0.10_accuracy0.96_.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()



sample_text = ["it is not wonderful experience using the product"]

# Load the saved vocabulary
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

sequence = text_to_sequence(sample_text, vocab, max_length)
sequence_padded = pad_sequences(sequence, 100)
sequence_tensor = torch.tensor(sequence_padded, dtype=torch.long).to(device)

with torch.no_grad():
    output = model(sequence_tensor)
    print("Probability:", output.item())
    print("Label:", "Positive" if output.item() >= 0.5 else "Negative")
