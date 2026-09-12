"""
data_gen.py
Builds the training data from TWO sources merged together:
  1. dair-ai/emotion (unsplit, ~416k Twitter-style short texts)
  2. GoEmotions (Reddit-based, ~58k, mapped from 27 fine-grained emotions
     down to our 6 classes)
Text is normalized (text_utils.normalize_text) before tokenizing. A BPE
subword tokenizer is trained on the combined corpus (instead of whole-word
tokenization) so unseen words/typos degrade gracefully into known subword
pieces rather than collapsing into a single opaque <UNK> token.
"""

import os
import pickle

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from text_utils import normalize_text

MODEL_DIR = "model"
VOCAB_SIZE = 8000
MAX_LEN = 40

os.makedirs(MODEL_DIR, exist_ok=True)

# Canonical label order used across the whole project
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABEL_NAMES)}

# GoEmotions' 27 fine-grained emotions (+neutral, excluded) mapped down to
# our 6 classes. Rows with zero or multiple labels, or a label not in this
# map (i.e. 'neutral'), are dropped to keep training signal clean.
GOEMOTIONS_MAP = {
    "sadness": "sadness", "grief": "sadness", "remorse": "sadness", "disappointment": "sadness",
    "joy": "joy", "amusement": "joy", "excitement": "joy", "gratitude": "joy",
    "admiration": "joy", "approval": "joy", "optimism": "joy", "pride": "joy",
    "relief": "joy", "caring": "joy",
    "love": "love", "desire": "love",
    "anger": "anger", "annoyance": "anger", "disapproval": "anger", "disgust": "anger",
    "fear": "fear", "nervousness": "fear", "embarrassment": "fear",
    "surprise": "surprise", "curiosity": "surprise", "confusion": "surprise", "realization": "surprise",
}


def load_dair_ai():
    print("Downloading dair-ai/emotion (unsplit, ~416k)...")
    ds = load_dataset("dair-ai/emotion", "unsplit")
    df = pd.DataFrame(ds["train"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    print(f"  -> {len(df)} examples")
    return df[["text", "label"]]


def load_goemotions():
    print("Downloading GoEmotions (~58k, simplified)...")
    ds = load_dataset("go_emotions", "simplified")
    ge_label_names = ds["train"].features["labels"].feature.names

    rows = []
    for split in ("train", "validation", "test"):
        for ex in ds[split]:
            if len(ex["labels"]) != 1:
                continue  # keep only single-label examples for clean signal
            ge_name = ge_label_names[ex["labels"][0]]
            mapped = GOEMOTIONS_MAP.get(ge_name)
            if mapped is None:
                continue  # drops 'neutral' and any unmapped category
            rows.append({"text": ex["text"], "label": LABEL_TO_IDX[mapped]})

    df = pd.DataFrame(rows)
    print(f"  -> {len(df)} examples after filtering to single-label + mapped classes")
    return df


def train_bpe_tokenizer(texts):
    print(f"Training BPE tokenizer (vocab_size={VOCAB_SIZE})...")
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<PAD>", "<UNK>"],  # <PAD> gets id 0, required for mask_zero
    )
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<PAD>"), pad_token="<PAD>", length=MAX_LEN)
    tokenizer.enable_truncation(max_length=MAX_LEN)
    return tokenizer


def encode_texts(tokenizer, texts, batch_size=2000):
    all_ids = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encodings = tokenizer.encode_batch(batch)
        all_ids.extend([e.ids for e in encodings])
    return np.array(all_ids, dtype=np.int32)


def load_and_prepare():
    dair_df = load_dair_ai()
    goemo_df = load_goemotions()

    full_df = pd.concat([dair_df, goemo_df], ignore_index=True)
    print(f"Combined dataset: {len(full_df)} examples")

    print("Normalizing text...")
    full_df["text"] = full_df["text"].apply(normalize_text)
    full_df = full_df[full_df["text"].str.len() > 0].reset_index(drop=True)

    print("Class distribution:")
    print(full_df["label"].map(lambda i: LABEL_NAMES[i]).value_counts())

    train_df, temp_df = train_test_split(
        full_df, test_size=0.10, stratify=full_df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )
    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    tokenizer = train_bpe_tokenizer(train_df["text"].tolist())

    X_train = encode_texts(tokenizer, train_df["text"].tolist())
    X_val = encode_texts(tokenizer, val_df["text"].tolist())
    X_test = encode_texts(tokenizer, test_df["text"].tolist())

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    np.savez(
        os.path.join(MODEL_DIR, "dataset.npz"),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )

    tokenizer.save(os.path.join(MODEL_DIR, "bpe_tokenizer.json"))

    with open(os.path.join(MODEL_DIR, "labels.pickle"), "wb") as f:
        pickle.dump(LABEL_NAMES, f)

    config = {"vocab_size": tokenizer.get_vocab_size(), "max_len": MAX_LEN}
    with open(os.path.join(MODEL_DIR, "config.pickle"), "wb") as f:
        pickle.dump(config, f)

    print(f"Saved dataset.npz, bpe_tokenizer.json, labels.pickle, config.pickle to '{MODEL_DIR}/'")
    print(f"Actual vocab size: {tokenizer.get_vocab_size()}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


if __name__ == "__main__":
    load_and_prepare()
