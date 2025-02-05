from pathlib import Path

# Hyperparameters
VOC_SIZE = 20000
MAX_LENGTH = 100
EMBED_DIM = 64
HIDDEN_DIM = 32
BATCH_SIZE = 64
DEVICE = "cpu"

# Paths
MODEL_PATH = Path("/home/app/microservice/models/sentiment_analyzer/saved_models/model_epoch_6_loss0.10_accuracy0.96_.pth")
VOCAB_PATH = Path("/home/app/microservice/models/sentiment_analyzer/vocab.pkl")