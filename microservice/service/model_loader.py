import torch
from models.sentiment_analyzer.model import LSTMModel
from models.sentiment_analyzer.config import VOC_SIZE, EMBED_DIM, HIDDEN_DIM, MAX_LENGTH, DEVICE, MODEL_PATH, VOCAB_PATH
from models.sentiment_analyzer.utils import text_to_sequence, pad_sequences, load_vocab


def _load_model():   
    try:
        model = LSTMModel(voc_size=VOC_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, max_length=MAX_LENGTH)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load the model: {e}")


def _load_vocab() -> dict:
    try:
        return load_vocab(VOCAB_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load the vocabulary: {e}")


# Singleton instances of the model and vocabulary
model = _load_model()
vocab = _load_vocab()



def predict_sentiment(text: str):
    try:
        # Preprocess the input text
        sequence = text_to_sequence([text], vocab, max_length=MAX_LENGTH)
        sequence_padded = pad_sequences(sequence, max_length=MAX_LENGTH)
        sequence_tensor = torch.tensor(sequence_padded, dtype=torch.long).to(DEVICE)

        # Make a prediction
        with torch.no_grad():
            output = model(sequence_tensor)
            probability = output.item()
            label = "Positive" if probability >= 0.5 else "Negative"

        return probability, label
    except Exception as e:
        raise RuntimeError(f"Failed to make a prediction: {e}")





