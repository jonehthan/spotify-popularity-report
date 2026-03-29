# ============================================================
# Compare weight initializers: glorot_uniform, glorot_normal, normal, uniform
# Best model hyperparameters FIXED except initializer
# Subplots + saved to disk
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
from tensorflow.keras.layers import Dense, Input
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
# MLP builder
# ---------------------------
def build_best_mlp(input_dim, initializer):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(20, activation="relu", kernel_initializer=initializer),
        Dense(20, activation="relu", kernel_initializer=initializer),
        Dense(20, activation="relu", kernel_initializer=initializer),
        Dense(20, activation="relu", kernel_initializer=initializer),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ---------------------------
# Train & plot all initializers in subplots
# ---------------------------
initializers = ["glorot_uniform", "glorot_normal", "normal", "uniform"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, init in enumerate(initializers):
    print(f"\nTraining model with initializer = {init.upper()}")

    model = build_best_mlp(
        input_dim=X_train.shape[1],
        initializer=init
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        shuffle=True,
        verbose=0  # suppress per-epoch output for clean display
    )

    ax = axes[i]
    ax.plot(history.history["loss"], label="Training Loss")
    ax.plot(history.history["val_loss"], label="Validation Loss")
    ax.set_title(f"{init.upper()}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.suptitle("Training vs Validation Loss by Initializer", fontsize=16, y=1.02)
plt.savefig("mlp_loss_comparison_initializers.png")
plt.show()

print("Saved combined subplot figure as mlp_loss_comparison_initializers.png")
