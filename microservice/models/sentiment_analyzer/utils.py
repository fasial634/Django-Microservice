from typing import List, Dict
import torch
import pickle
from pathlib import Path
from .config import MAX_LENGTH, VOCAB_PATH

def text_to_sequence(texts: List[str], vocab: Dict[str, int], max_length: int) -> List[List[int]]:
    sequences = []
    for text in texts:
        tokens = text.lower().split()
        sequence = [vocab.get(token, 0) for token in tokens]  # 0 for out-of-vocabulary tokens
        sequences.append(sequence)
    return sequences

def pad_sequences(sequences: List[List[int]], max_length: int, pad_value: int = 0) -> List[List[int]]:
    padded = []
    for seq in sequences:
        if len(seq) < max_length:
            padded_seq = seq + [pad_value] * (max_length - len(seq))  # Pad with zeros
        else:
            padded_seq = seq[:max_length]  # Truncate if longer than max_length
        padded.append(padded_seq)
    return padded

def load_vocab(vocab_path: Path) -> Dict[str, int]:
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab