import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

# ======================
# LOAD + PREPROCESS DATA
# ======================

df = pd.read_csv("track_data_final.csv")

threshold = df["track_popularity"].quantile(0.80)
df["popular"] = (df["track_popularity"] >= threshold).astype(int)

df["album_release_year"] = pd.to_datetime(
    df["album_release_date"], errors="coerce"
).dt.year

df["track_duration_ms"] = np.log1p(df["track_duration_ms"])
df["artist_followers"] = np.log1p(df["artist_followers"])
df["explicit"] = df["explicit"].astype(int)

FEATURES = [
    "track_number",
    "track_duration_ms",
    "explicit",
    "artist_popularity",
    "artist_followers",
    "album_total_tracks",
    "album_release_year",
    "album_type",
    "artist_genres"
]

X = df[FEATURES]
y = df["popular"]

numeric_features = [
    "track_number",
    "track_duration_ms",
    "explicit",
    "artist_popularity",
    "artist_followers",
    "album_total_tracks",
    "album_release_year"
]

categorical_features = ["album_type", "artist_genres"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_processed, y, test_size=0.2, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
# =========================
# Build Best Model Function
# =========================
def build_best_model(input_dim, learning_rate):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(20, activation="relu", kernel_initializer="glorot_normal"),
        Dense(20, activation="relu", kernel_initializer="glorot_normal"),
        Dense(20, activation="relu", kernel_initializer="glorot_normal"),
        Dense(20, activation="relu", kernel_initializer="glorot_normal"),
        Dense(1, activation="sigmoid")
    ])

    optimizer = SGD(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR")
        ]
    )

    return model

# =========================
# Learning Rates to Test
# =========================
learning_rates = [0.1, 0.05, 0.01, 0.005]

histories = {}

# =========================
# Train Models
# =========================
for lr in learning_rates:
    print(f"\nTraining SGD with learning rate = {lr}")

    model = build_best_model(
        input_dim=X_train.shape[1],
        learning_rate=lr
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1,
        shuffle=True
    )

    histories[lr] = history

# =========================
# Plot Loss Curves
# =========================
for lr, history in histories.items():
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title(f"SGD Learning Rate = {lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.legend()
    plt.grid(True)

    filename = f"sgd_lr_{lr}_loss.png"
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"Saved plot: {filename}")
