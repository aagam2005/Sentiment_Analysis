"""
predict.py
Loads the trained model + BPE tokenizer and exposes a predict_emotion()
function used by app.py. Can also be run standalone from the CLI for quick
tests. Uses the SAME normalization (text_utils.normalize_text) and the SAME
BPE tokenizer that data_gen.py used to build the training data, so
train/inference stay consistent.
"""

import os
import pickle

import numpy as np
import tensorflow as tf
from tokenizers import Tokenizer

from text_utils import normalize_text

MODEL_DIR = "model"

_model = None
_tokenizer = None
_labels = None
_max_len = None


def load_artifacts():
    global _model, _tokenizer, _labels, _max_len
    if _model is None:
        model_path = os.path.join(MODEL_DIR, "emotion_model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Trained model not found. Run 'python train.py' first."
            )
        _model = tf.keras.models.load_model(model_path)

    if _tokenizer is None:
        tok_path = os.path.join(MODEL_DIR, "bpe_tokenizer.json")
        if not os.path.exists(tok_path):
            raise FileNotFoundError(
                "Tokenizer not found. Run 'python data_gen.py' first."
            )
        _tokenizer = Tokenizer.from_file(tok_path)

    if _labels is None:
        with open(os.path.join(MODEL_DIR, "labels.pickle"), "rb") as f:
            _labels = pickle.load(f)

    if _max_len is None:
        with open(os.path.join(MODEL_DIR, "config.pickle"), "rb") as f:
            _max_len = pickle.load(f)["max_len"]

    return _model, _tokenizer, _labels


def predict_emotion(text: str) -> dict:
    model, tokenizer, labels = load_artifacts()

    cleaned = normalize_text(text)
    encoding = tokenizer.encode(cleaned)  # tokenizer already pads/truncates to max_len
    padded = np.array([encoding.ids], dtype=np.int32)

    probs = model.predict(padded, verbose=0)[0]
    top_idx = int(np.argmax(probs))

    return {
        "text": text,
        "emotion": labels[top_idx],
        "confidence": float(probs[top_idx]),
        "all_scores": {labels[i]: float(probs[i]) for i in range(len(labels))},
    }


if __name__ == "__main__":
    while True:
        text = input("\nEnter text (or 'quit'): ")
        if text.lower() == "quit":
            break
        result = predict_emotion(text)
        print(f"Emotion: {result['emotion']}  (confidence: {result['confidence']:.2%})")
