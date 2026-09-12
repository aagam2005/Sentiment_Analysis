"""
train.py
Builds and trains a Bidirectional LSTM (RNN) for text emotion classification
using the arrays produced by data_gen.py, then saves the trained model.
"""

import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, callbacks

MODEL_DIR = "model"
EMBED_DIM = 128
BATCH_SIZE = 64
EPOCHS = 15


def build_model(num_classes: int, vocab_size: int, max_len: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, mask_zero=True),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    data_path = os.path.join(MODEL_DIR, "dataset.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "dataset.npz not found. Run 'python data_gen.py' first."
        )

    data = np.load(data_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    with open(os.path.join(MODEL_DIR, "labels.pickle"), "rb") as f:
        label_names = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "config.pickle"), "rb") as f:
        config = pickle.load(f)

    print("GPU available:", tf.config.list_physical_devices("GPU"))
    print(f"Vocab size: {config['vocab_size']}, Max len: {config['max_len']}")

    model = build_model(
        num_classes=len(label_names),
        vocab_size=config["vocab_size"],
        max_len=config["max_len"],
    )
    model.summary()

    # Counter class imbalance (e.g. 'love'/'surprise' are much rarer than
    # 'joy'/'sadness' in this dataset) so rare classes aren't ignored.
    class_weights_arr = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = {i: w for i, w in enumerate(class_weights_arr)}
    print("Class weights:", {label_names[i]: round(w, 2) for i, w in class_weights.items()})

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    checkpoint = callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "emotion_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint],
        class_weight=class_weights,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    # Final save (in case checkpoint didn't trigger, e.g. very short run)
    model.save(os.path.join(MODEL_DIR, "emotion_model.keras"))
    print("Model saved to model/emotion_model.keras")


if __name__ == "__main__":
    main()
