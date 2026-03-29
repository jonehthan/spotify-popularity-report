import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)

# =========================
# DATA LOAD & PREP
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
    "album_release_year"
]

X = df[FEATURES].fillna(df[FEATURES].median())
y = df["popular"]

# =========================
# TRAIN / VAL / TEST SPLIT
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

input_dim = X_train.shape[1]   # ✅ NOW DEFINED

# =========================
# MODEL BUILDER
# =========================
def build_best_model(loss_fn, input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(20, activation="relu", kernel_initializer="glorot_normal"),
        Dense(20, activation="relu", kernel_initializer="glorot_normal"),
        Dense(20, activation="relu", kernel_initializer="glorot_normal"),
        Dense(20, activation="relu", kernel_initializer="glorot_normal"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    return model

# =========================
# LOSS FUNCTIONS TO TEST
# =========================
loss_functions = {
    "Binary Cross-Entropy": "binary_crossentropy",
    "Hinge Loss": "hinge",
    "Squared Hinge": "squared_hinge"
}

# =========================
# TRAIN & EVALUATE
# =========================
for name, loss_fn in loss_functions.items():
    print(f"\n===== Training with {name} =====")

    model = build_best_model(loss_fn, input_dim)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=0,
        shuffle=True
    )

    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC:    {average_precision_score(y_test, y_prob):.4f}")
