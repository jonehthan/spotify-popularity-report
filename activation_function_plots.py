# ============================================================
# Compare ReLU vs Tanh vs ELU
# Best model hyperparameters FIXED except activation
# Separate plots + saved to disk
# ============================================================

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD

# ---------------------------
# Load data
# ---------------------------
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

categorical_features = [
    "album_type",
    "artist_genres"
]

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

X_processed = preprocessor.fit_transform(X).toarray()

X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------
# MLP builder (BEST CONFIG)
# ---------------------------
def build_best_mlp(input_dim, activation):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(5, activation=activation, kernel_initializer="glorot_normal"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ---------------------------
# Train & plot for each activation
# ---------------------------
activations = ["relu"]

for act in activations:
    print(f"\nTraining model with activation = {act.upper()}")

    model = build_best_mlp(
        input_dim=X_train.shape[1],
        activation=act
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        shuffle=True,
        verbose=1
    )

    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title(f"Training vs Validation Loss ({act.upper()})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f"underfit.png"
    plt.savefig(filename)
    plt.show()

    print(f"Saved plot to {filename}")
