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
from tensorflow.keras.optimizers.schedules import ExponentialDecay, InverseTimeDecay

# =========================
# DATA LOADING
# =========================
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

X_train, X_val, y_train, y_val = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# MODEL BUILDER
# =========================
def build_model(input_dim, optimizer):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(20, activation="relu", kernel_initializer="glorot_normal"),
        Dense(20, activation="relu", kernel_initializer="glorot_normal"),
        Dense(20, activation="relu", kernel_initializer="glorot_normal"),
        Dense(20, activation="relu", kernel_initializer="glorot_normal"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =========================
# LEARNING RATE SCHEDULES
# =========================
exp_decay = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)

inv_time_decay = InverseTimeDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.5,
    staircase=False
)

optimizers = {
    "Exponential Decay": SGD(learning_rate=exp_decay),
    "Inverse Time Decay": SGD(learning_rate=inv_time_decay)
}

histories = {}

# =========================
# TRAINING
# =========================
for name, opt in optimizers.items():
    print(f"\nTraining with {name} schedule...")
    model = build_model(X_train.shape[1], opt)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1,
        shuffle=True
    )
    histories[name] = history

# =========================
# PLOTTING
# =========================
for name, history in histories.items():
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{name} Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name.replace(' ', '_').lower()}_lr_schedule.png")
    plt.show()
